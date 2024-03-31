This Python script implements real-time emotion detection using a pre-trained deep learning model and displays the results in a PyQt5 GUI application. Here's a breakdown of the code:

1. **Importing Libraries**:
   - `sys`: Provides access to some variables used or maintained by the Python interpreter and to functions that interact strongly with the interpreter.
   - `cv2`: OpenCV library for computer vision tasks.
   - `PyQt5`: PyQt5 library for GUI development.
   - `keras`: Deep learning library for building and training models.
   - `numpy`: Library for numerical computations.
   - `pandas`: Library for data manipulation and analysis.
   - `matplotlib.pyplot`: Library for creating plots and visualizations.

2. **Loading Pre-Trained Model and Classifier**:
   - Loads a pre-trained emotion detection model (`Emotion_Detection.h5`) using Keras.
   - Defines class labels for different emotions.
   - Initializes the Haar cascade classifier for face detection using OpenCV.

3. **Defining the GUI Application**:
   - Creates a PyQt5 QWidget subclass for the emotion detection application.
   - Sets up GUI elements such as labels, buttons, layout, and video display area.

4. **Initializing Video Capture and Emotion Tracking Variables**:
   - Sets up a video capture object to access the webcam feed.
   - Initializes variables to track detected emotions and total frames processed.

5. **Methods for Emotion Detection and GUI Interaction**:
   - `start_detection`: Starts the emotion detection process and updates the GUI accordingly.
   - `stop_detection`: Stops the emotion detection process, displays the detected emotions, and plots the emotion distribution.
   - `detect_emotion`: Performs real-time emotion detection on video frames captured from the webcam feed.
   - `plot_emotion_data`: Generates and displays a pie chart showing the distribution of detected emotions.

6. **Main Execution**:
   - Initializes the PyQt5 application (`QApplication`), creates an instance of the `EmotionDetection` class, and starts the event loop.

To run this script:
1. Install the required libraries (`opencv-python`, `PyQt5`, `keras`, `numpy`, `pandas`, `matplotlib`) if you haven't already.
2. Ensure you have the pre-trained model file (`Emotion_Detection.h5`) and the Haar cascade XML file (`haarcascade_frontalface_default.xml`) in the specified paths or update the paths accordingly.
3. Run the script, and it will open a GUI window displaying real-time emotion detection from the webcam feed.
4. Click the "Start Detection" button to begin detection and "Stop Detection" to stop and display the results.

Note: Make sure your webcam is connected and accessible to the application. Adjustments may be needed based on your system configuration and file paths.
