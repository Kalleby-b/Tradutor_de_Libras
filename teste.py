import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib
import numpy as np

# 1. Carregar o Modelo e o Encoder que você treinou
model = joblib.load('modelo_libras.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# 2. Configurar o MediaPipe Tasks (o mesmo arquivo .task de antes)
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

print("Iniciando tradução... Pressione 'q' para sair.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Inverter a imagem para parecer um espelho
    frame = cv2.flip(frame, 1)
    
    # Converter para o formato do MediaPipe
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Detecção
    detection_result = detector.detect(mp_image)

    if detection_result.hand_landmarks:
        landmarks = []
        for lm in detection_result.hand_landmarks[0]:
            landmarks.extend([lm.x, lm.y, lm.z])
        
        # Predição da IA
        prediction = model.predict([landmarks])
        label = label_encoder.inverse_transform(prediction)[0]
        prob = np.max(model.predict_proba([landmarks]))

        # Mostrar o resultado na tela
        cv2.putText(frame, f'{label} ({prob*100:.1f}%)', (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Tradutor de LIBRAS - Kalleby', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()