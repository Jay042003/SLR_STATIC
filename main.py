import cv2 as cv 
import mediapipe as mp
from utils import mediapipe_utils
import time

cap = cv.VideoCapture(0)
with mp.solutions.hands.Hands(static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.3) as model:
    while True:
        suc, img = cap.read()
        img, results = mediapipe_utils.detect(img, model)
        mediapipe_utils.visualise_hand_landmarks(img,results)
        cv.imshow('img', img)
        if cv.waitKey(15) & 0xFF ==ord('q'):
            break
