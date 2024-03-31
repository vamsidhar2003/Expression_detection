import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the pre-trained model and define class labels
path = r"V:\vscode\visual_codes\symposium\Emotion_Detection.h5"
classifier = load_model(path)
class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Define face classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class EmotionDetection(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Emotion Detection")
        self.video_size = (640, 480)

        self.video_label = QLabel(self)
        self.video_label.resize(self.video_size[0], self.video_size[1])

        self.emotion_label = QLabel(self)
        self.emotion_label.setText("Detected Emotions:")
        self.emotion_label.setStyleSheet("font-size: 16px;")

        self.start_button = QPushButton('Start Detection', self)
        self.start_button.clicked.connect(self.start_detection)

        self.stop_button = QPushButton('Stop Detection', self)
        self.stop_button.setEnabled(False)  # Disabled initially
        self.stop_button.clicked.connect(self.stop_detection)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.emotion_label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)

        self.setLayout(layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.detect_emotion)
        self.cap = cv2.VideoCapture(0)

        # Initialize emotion tracking variables
        self.emotion_data = {'Angry': 0, 'Happy': 0, 'Neutral': 0, 'Sad': 0, 'Surprise': 0}
        self.total_frames = 0

    def start_detection(self):
        if not self.timer.isActive():
            self.timer.start(30)  # Update every 30 milliseconds
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
        else:
            self.timer.stop()
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)

    def stop_detection(self):
        self.timer.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.plot_emotion_data()  # Generate and display pie chart

    def detect_emotion(self):
        ret, frame = self.cap.read()
        if not ret:
            QMessageBox.warning(self, 'Error', 'Failed to open webcam.')
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        detected_emotions = []  # List to store detected emotions for each face

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                preds = classifier.predict(roi)[0]
                label = class_labels[preds.argmax()]
                detected_emotions.append(label)  # Add detected emotion to the list

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Update emotion data
                self.emotion_data[label] += 1
                self.total_frames += 1

        # Update the emotion label with all detected emotions
        if detected_emotions:
            emotion_text = "Detected Emotions: " + ", ".join(detected_emotions)
        else:
            emotion_text = "No Faces Detected"
        self.emotion_label.setText(emotion_text)

        frame = cv2.resize(frame, self.video_size)
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.video_label.setPixmap(pixmap)

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

    def plot_emotion_data(self):
        df = pd.DataFrame(list(self.emotion_data.items()), columns=['Emotion', 'Count'])
        plt.figure(figsize=(8, 6))
        plt.pie(df['Count'], labels=df['Emotion'], autopct='%1.1f%%')
        plt.title('Emotion Distribution')
        plt.axis('equal')
        plt.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = EmotionDetection()
    window.show()
    sys.exit(app.exec_())
