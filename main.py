import cv2 as cv
import pickle
import numpy as np
import mediapipe as mp
from utils import mediapipe_utils


with open('svm_model.pkl', 'rb') as f:
    loaded_svm_model, loaded_scaler = pickle.load(f)

cap = cv.VideoCapture(0) 
with mp.solutions.hands.Hands(static_image_mode=True,
                              max_num_hands=2,
                              min_detection_confidence=0.3) as model:
    counter = 0
    text = ''
    while True:
        suc, img = cap.read()
        img, results = mediapipe_utils.detect(img, model)
        mediapipe_utils.visualise_hand_landmarks(img, results)
        cv.imshow('img', img)
        if counter % 20 == 0:
            x = []
            if results.multi_hand_landmarks is not None:
                for landmarks in results.multi_hand_landmarks:
                    for landmark in landmarks.landmark:
                        x.append(float(landmark.x))
                        x.append(float(landmark.y))
                        x.append(float(landmark.z))
            if len(x) == 63:
                x = np.array(x).reshape(1, -1)
                x = loaded_scaler.transform(x)
                y = loaded_svm_model.predict(x)
                if y[0] is not None:
                    if y[0] == 'space':
                        text += '_'
                    elif y[0] == 'del':
                        if len(y) != 0:
                            text = text.rstrip(text[-1])
                    else:
                        text += y[0]
                else: 
                    continue

        counter += 1
        if text[-4:-1] == 'QQQ':
            break
        cv.putText(img, text, (50, 50),
                   cv.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255))
        cv.imshow('img', img)
        if cv.waitKey(15) & 0xFF == ord('q'):
            break
