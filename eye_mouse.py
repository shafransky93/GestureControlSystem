import argparse
import cv2
import dlib
import pyautogui
from screeninfo import get_monitors

pyautogui.FAILSAFE = False

# Initialize the face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Constants for the eye landmarks (adjust these based on your landmark numbering)
LEFT_EYE_LANDMARK_START = 36
RIGHT_EYE_LANDMARK_START = 42
EYE_LANDMARK_COUNT = 6  # Number of landmarks for each eye

# Function to calculate the midpoint between two points
def midpoint(point1, point2):
    return int((point1.x + point2.x) / 2), int((point1.y + point2.y) / 2)

# Function to move the mouse cursor based on the eye position with increased speed
def move_mouse(left_eye, right_eye, sensitivity=10.0, tilt_degrees=10):
    screen_width, screen_height = pyautogui.size()
    
    # Calculate the midpoint between the left and right eyes
    eye_midpoint = midpoint(left_eye, right_eye)
    
    # Get the eye position
    eye_x = eye_midpoint[0]
    eye_y = eye_midpoint[1]
    
    # Normalize the eye position to the range [0, 1]
    eye_x_normalized = eye_x / screen_width
    eye_y_normalized = eye_y / screen_height

    # Calculate the buffer size as 10% of the screen size
    buffer_width = int(screen_width/2 * 0.9)
    buffer_height = int(screen_height/2 * 0.9)

    # Adjust the screen width and height to include the buffer
    screen_width += buffer_width
    screen_height += buffer_height
    
    # Scale the eye position to move the cursor faster
    eye_x_scaled = eye_x_normalized * sensitivity
    
    # Adjust for the tilt by decreasing the vertical position
    # You can modify the tilt_degrees value as needed
    eye_y_scaled = (eye_y_normalized - (tilt_degrees / 90)) * sensitivity
    
    # Move the mouse cursor to the scaled position within the screen
    pyautogui.moveTo(eye_x_scaled * screen_width, eye_y_scaled * screen_height)

# Function to draw landmarks for the eyes
def draw_eye_landmarks(frame, landmarks):
    for idx in range(LEFT_EYE_LANDMARK_START, LEFT_EYE_LANDMARK_START + EYE_LANDMARK_COUNT):
        point = landmarks.part(idx)
        x, y = point.x, point.y
        cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)  # Draw a red circle at each landmark point
    for idx in range(RIGHT_EYE_LANDMARK_START, RIGHT_EYE_LANDMARK_START + EYE_LANDMARK_COUNT):
        point = landmarks.part(idx)
        x, y = point.x, point.y
        cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)  # Draw a red circle at each landmark point

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', nargs='*', help='Input video device (number or path), defaults to 0', default=['0'])
    args = parser.parse_args()
    
    if len(args.input) == 1:
        INPUT = int(args.input[0]) if args.input[0].isdigit() else args.input[0]
    else:
        return print("Wrong number of values for 'input' argument; should be 0, 1, or 4.")

    cap = cv2.VideoCapture(INPUT)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = detector(gray)

        for face in faces:
            # Get the facial landmarks for the detected face
            landmarks = predictor(gray, face)

            # Draw the eye landmarks on the frame
            draw_eye_landmarks(frame, landmarks)

            # Get the left and right eye landmarks for eye tracking
            left_eye = landmarks.part(LEFT_EYE_LANDMARK_START + 2)
            right_eye = landmarks.part(RIGHT_EYE_LANDMARK_START + 2)

            # Move the mouse cursor based on the eye position
            move_mouse(left_eye, right_eye, sensitivity=3)

        # Display the frame with eye landmarks and eye tracking
        cv2.imshow('Facial Landmarks with Eye Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
