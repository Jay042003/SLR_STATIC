# Real-Time Sign Language Detection on Edge Devices

## Overview

This project enables real-time sign language detection on edge devices using Support Vector Machine (SVM) for classification and MediaPipe for feature extraction. It is designed to recognize hand gestures corresponding to letters in American Sign Language (ASL). The classifier has been trained on a dataset of labeled images of ASL gestures.

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

## Steps

1. **Download the Dataset from Kaggle**:

   - Go to [Kaggle](https://www.kaggle.com/), find the ["ASL Dataset"](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset) dataset, and download it to your local machine.

2. **Clone the Repository**:
   ```
   git clone https://github.com/Jay042003/SLR_STATIC
   cd SLR_STATIC
   ```
3. **Install Required Libraries**:

   - Install the required dependencies by using

   ```
   pip install -r requirements.txt
   ```

4. **Preprocess the Dataset**:

   - Change the directory to the path of your dataset in `data_gathering.py`.
   - Run the `data_gathering.py` script in the `utils` folder with the path to the downloaded dataset:

   ```
   python utils/data_gathering.py
   ```

5. **Train the Model**:

   - Open the `model.ipynb` notebook and follow the instructions to train the SVM classifier using the generated CSV file from the previous step.
   - Make sure to adjust any parameters or configurations as needed.

6. **Get the Pickle File**:

   - After training, the notebook will generate a pickle file containing the trained SVM model.
   - This [pickle file](https://drive.google.com/file/d/1laIP-rHnH3zDud8LnVoTO2gxbMF1ensM/view?usp=sharing) (`svm_model.pkl`) will be used for inference in the next step.

7. **Deploy on Raspberry Pi**:
   - Transfer the `svm_model.pkl` file to your Raspberry Pi.
   - Run the `main.py` script on the Raspberry Pi to perform real-time classification using the trained model.

## Note

- Feel free to customize the code and experiment with different datasets or classification tasks.
- Ensure that you have sufficient computational resources for training the model, especially for larger datasets.

## Author

- [Jay Kadel](https://github.com/author1)
- [Dibyam Jalan](https://github.com/dibyam-jalan27)
- [Akash Kumar Singh](https://github.com/author3)
