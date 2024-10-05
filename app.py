import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model # type: ignore

# Loading Mdoel
model = load_model('model.h5')

#Pickle Classes
import pickle
with open('class.pkl','rb')as f:
    classes = pickle.load(f)

padding = 30

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hand_detection = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)


box_color = (0, 0, 0)
cap = cv2.VideoCapture(0)

# Image preprocessing function
def image_preprocess(hand_img):
    hand_img = cv2.resize(hand_img, (128, 128))
    hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
    hand_img = hand_img.astype('float32') / 255.0
    hand_img = np.expand_dims(hand_img, axis=0)
    predict = model.predict(hand_img)
    predicted_class = np.argmax(predict, axis=-1)
    predicted_class_label = classes[int(predicted_class)]
    return predicted_class_label

# Start the webcam feed
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hand_detection.process(frame_rgb)

    # Check for hand landmarks
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Calculate the bounding box for the hand
            x_min, y_min = frame.shape[1], frame.shape[0]
            x_max, y_max = 0, 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                x_min, y_min = min(x, x_min), min(y, y_min)
                x_max, y_max = max(x, x_max), max(y, y_max)

            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(frame.shape[1], x_max + padding)
            y_max = min(frame.shape[0], y_max + padding)

            # Crop the hand region
            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size == 0:  # Check for empty crop in case of any issues
                continue

            # Predict the class for the cropped hand image
            predicted_class_label = image_preprocess(hand_img)

            # Draw bounding box and display the prediction
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)
            cv2.putText(frame, predicted_class_label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
            )

    # Show the frame
    cv2.imshow("Hand Detection with MediaPipe", frame)
    if cv2.waitKey(1) == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()
