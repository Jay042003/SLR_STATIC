# Real-Time Sign Language Detection on Edge Devices

## Overview

This project enables real-time sign language detection on edge devices using Support Vector Machine (SVM) for classification and MediaPipe for feature extraction. It is designed to recognize hand gestures corresponding to letters in American Sign Language (ASL). The classifier has been trained on a dataset of labeled images of ASL gestures. YouTube Link:- https://youtu.be/2uwYzzfZ8RM

## Features

- Real-time sign language detection on edge devices.
- Classification of hand gestures representing letters in ASL.
- Feature extraction using MediaPipe for robust representation of hand gestures.
- Easy deployment on Raspberry Pi 3B for real-time inference.
- High accuracy and fast inference time.

## Built With

- [![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
- [![OpenCV](https://img.shields.io/badge/-OpenCV-008000?style=for-the-badge&logo=opencv&logoColor=ffdd54)](https://opencv.org/)
- [![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
- [![MediaPipe](https://img.shields.io/badge/MediaPipe-82CAFF?style=for-the-badge&logo=mediapipe&logoColor=black)](https://mediapipe.dev/)

## Usage

1. **Training the Classifier**:

   - Train the SVM classifier using the provided dataset or your own dataset.
   - Use `model.py` script to train the classifier.

2. **Deployment on Raspberry Pi**:

   - Transfer the trained model to Raspberry Pi.
   - Run `main.py` script on Raspberry Pi.

3. **Testing**:
   - Evaluate the classifier using test images or live camera feed.

## How to Run

1. Clone the repository:

```sh
https://github.com/Jay042003/SLR_STATIC.git
```

2. Install dependencies:

```sh
pip install -r requirements.txt
```

3. Train the classifier:

```sh
python model.py
```

4. Deploy on Raspberry Pi:

- Transfer the trained model to Raspberry Pi.
- Run `main.py` script on Raspberry Pi.

## Author

- [Jay Kadel](https://github.com/author1)
- [Dibyam Jalan](https://github.com/dibyam-jalan27)
- [Akash Kumar Singh](https://github.com/author3)
