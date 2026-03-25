import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import pandas as pd

# 1. Configuração do Detector (Baixe o arquivo 'hand_landmarker.task' no site do MediaPipe se necessário)
base_options = python.BaseOptions(model_asset_path='C:\\Users\\Kalleby\\Documents\\GitHub\\Tradutor_de_Libras\\hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

data = []
dataset_path = 'C:\\Users\\Kalleby\\Documents\\GitHub\\Tradutor_de_Libras\\Dataset\\images' 

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
            # Pegamos a primeira mão detectada
            for lm in detection_result.hand_landmarks[0]:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            landmarks.append(label)
            data.append(landmarks)

df = pd.DataFrame(data)
df.to_csv('dataset_libras_final.csv', index=False)
print("Finalizado com sucesso!")