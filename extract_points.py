import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)

def extract_landmarks(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    

    if results.multi_hand_landmarks:
        res = np.array([[res.x, res.y, res.z] for res in results.multi_hand_landmarks[0].landmark]).flatten()
        return res
    return np.zeros(21*3)

# Aqui você faria um loop pelas suas pastas e salvaria os arrays
# Exemplo: np.save('sequencia_oi_1.npy', lista_de_landmarks)