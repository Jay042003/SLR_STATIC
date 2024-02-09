import cv2 as cv 
import numpy as np
import mediapipe as mp

def detect(img, model)->tuple:
    '''Process an input image using a Mediapipe model for holistic (hand, pose, face) detection.
    
    :param img: Input image (BGR format).
    :param model: Mediapipe holistic(mp.holistic.Holistc()) detection model.
    :return: Tuple containing the processed image (BGR format) and the results from the detection model.
'''
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB) # model only supports rgb image
    img.flags.writeable = False # to increase the speed
    results = model.process(img_rgb) # using mp.holistic.Holistic() model for hand, pose and face detection
    img.flags.writeable = True
    img = cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR)
    return img, results

def visualise_hand_landmarks(img, results)->None:
    '''visulaise_hand_landmarks(img, results):
    Draw landmarks and connections on the input image based on the hand landmarks detected by Mediapipe.

    :param img: Input image (BGR format).
    :param results: Results obtained from the Mediapipe holistic detection model.
    :return: None (The function modifies the input image in place).
'''
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils # using inbuilt drawing_utils to draw the landmarks and lines on hand

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_draw.DrawingSpec(
                    color=(0, 0, 0), thickness=5, circle_radius=3
                ),
                connection_drawing_spec=mp_draw.DrawingSpec(
                    color=(161, 161, 161), thickness=5, circle_radius=3
                ),
            )

    return

def extract_features(results):
    mp_hands = mp.solutions.hands
    coloumn = []
    if results.multi_hand_landmarks is not None:
        for landmarks in results.multi_hand_landmarks:
            for landmark in landmarks.landmark:
                coloumn.append(float(landmark.x))
                coloumn.append(float(landmark.y))
                coloumn.append(float(landmark.z))        
    if len(coloumn) == 64:
        print(coloumn)
        return coloumn 
    else: 
        return None
