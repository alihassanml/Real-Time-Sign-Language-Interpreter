# Real-Time Sign Language Interpreter

This project is a real-time sign language interpreter that utilizes a Convolutional Neural Network (CNN) to recognize hand gestures based on the American Sign Language (ASL) dataset. The application captures hand images using a webcam and processes them to predict the corresponding sign language gesture using a pre-trained model.

## Features
- Real-time hand detection using OpenCV and MediaPipe.
- Gesture recognition using a pre-trained CNN model.
- Predicts American Sign Language (ASL) gestures.
- Provides a visual display of the detected hand and gesture label on the webcam feed.

## Technologies Used
- **TensorFlow/Keras**: For building and loading the CNN model.
- **OpenCV**: For video capture and real-time image processing.
- **MediaPipe**: For efficient hand landmark detection.
- **NumPy**: For handling data arrays.
- **Pickle**: For loading gesture class labels.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/alihassanml/Real-Time-Sign-Language-Interpreter.git
   cd Real-Time-Sign-Language-Interpreter
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have the trained model (`model.h5`) and the class label file (`class.pkl`) in the project directory.

## Usage
1. Run the script to start the real-time sign language interpreter:
   ```bash
   python main.py
   ```

2. The application will start capturing the video feed from your webcam. Detected hand gestures will be displayed along with the predicted sign language gesture.

3. Press the `ESC` key to exit the program.

## File Structure
- `main.py`: Main script for running the real-time interpreter.
- `model.h5`: Pre-trained CNN model.
- `class.pkl`: Pickle file containing class labels for the gestures.
- `requirements.txt`: List of dependencies required to run the project.

## How it Works
1. The application captures video frames from the webcam.
2. MediaPipe is used to detect hands and extract hand landmarks.
3. The detected hand region is cropped and preprocessed to match the input size expected by the CNN model.
4. The pre-trained model predicts the gesture class based on the hand image.
5. The predicted gesture is displayed on the video feed.

## Requirements
- Python 3.x
- TensorFlow/Keras
- OpenCV
- MediaPipe
- NumPy
- Pickle

## Demo
A demo video of the real-time sign language interpreter in action will be available soon!

## License
This project is licensed under the MIT License.

## Acknowledgments
- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands) for efficient hand tracking.
- TensorFlow/Keras for providing the framework to build and train the model.
- OpenCV for real-time video processing.

---

**GitHub Repository**: [Real-Time Sign Language Interpreter](https://github.com/alihassanml/Real-Time-Sign-Language-Interpreter.git)
