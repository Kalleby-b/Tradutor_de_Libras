import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import pandas as pd

base_options = python.BaseOptions(model_asset_path='C:\\Users\\Kalleby\\Documents\\GitHub\\Tradutor_de_Libras\\hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

data = []
dataset_path = 'C:\\Users\\Kalleby\\Documents\\GitHub\\Tradutor_de_Libras\\Dataset' 

for label in os.listdir(dataset_path):
    label_path = os.path.join(dataset_path, label)
    if not os.path.isdir(label_path): continue
    
    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)
        
        # O MediaPipe Tasks precisa de um objeto Image específico
        image = mp.Image.create_from_file(img_path)
        
        # Detectar pontos
        detection_result = detector.detect(image)
        
        if detection_result.hand_landmarks:
            landmarks = []

            # 1. Pegar pontos da mão
            for lm in detection_result.hand_landmarks[0]:
                landmarks.extend([lm.x, lm.y, lm.z])

            # 2. Normalizar com base no primeiro ponto (base da mão)
            base_x = landmarks[0]
            base_y = landmarks[1]
            base_z = landmarks[2]

            normalized = []

            for i in range(0, len(landmarks), 3):
                normalized.append(landmarks[i] - base_x)
                normalized.append(landmarks[i+1] - base_y)
                normalized.append(landmarks[i+2] - base_z)

            # 3. Substituir
            landmarks = normalized
            import numpy as np

            landmarks_array = np.array(landmarks)

            # evitar divisão por zero
            max_value = np.max(np.abs(landmarks_array))
            if max_value != 0:
                landmarks_array = landmarks_array / max_value

            landmarks = landmarks_array.tolist()
            
            landmarks.append(label)
            data.append(landmarks)

df = pd.DataFrame(data)
df.to_csv('dataset_libras_completo.csv', index=False)
print("Finalizado com sucesso!")