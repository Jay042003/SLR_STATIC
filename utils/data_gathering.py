import os
import cv2 as cv
import pandas as pd
import numpy as np
import mediapipe as mp
import mediapipe_utils as mptools
from multiprocessing import Pool

# Define a function to process each character


def process_character(character, directory):
    """This program access each image and extract it's features

    Args:
        character (str): what character is extracted
        directory (str): directory of the class of character

    Returns:
        vector: input vector for svm
    """
    data_array = []
    with mp.solutions.hands.Hands(static_image_mode=True,
                                  max_num_hands=2,
                                  min_detection_confidence=0.3) as model:
        path = directory + '/' + character
        for filename in os.listdir(path):
            coloumn = [character]
            if filename.endswith(".jpg") or filename.endswith(".png"):
                img = cv.imread(path + '/' + filename)
                img, results = mptools.detect(img, model)
                if results.multi_hand_landmarks is not None:
                    for landmarks in results.multi_hand_landmarks:
                        for landmark in landmarks.landmark:
                            coloumn.append(float(landmark.x))
                            coloumn.append(float(landmark.y))
                            coloumn.append(float(landmark.z))
                if len(coloumn) == 64:
                    data_array.append(coloumn)
    return data_array


if __name__ == '__main__':
    directory = '/Users/jaykadel/Downloads/ASL_Alphabet_Dataset/asl_alphabet_train'
    characters = [chr(i) for i in range(ord('A'), ord('Z')+1)]
    characters.append('del')
    characters.append('space')

    # Using Pool for parallel processing
    with Pool() as p:
        results = p.starmap(process_character, [
                            (char, directory) for char in characters])

    # Concatenate results from all processes
    data_array = [item for sublist in results for item in sublist]

    # Create DataFrame and save to CSV
    df = pd.DataFrame(data_array)
    df.to_csv('data.csv')
