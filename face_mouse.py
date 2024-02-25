import argparse
import cv2
import dlib
import pyautogui
from screeninfo import get_monitors

pyautogui.FAILSAFE = False

# Initialize the face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Constants for the nose landmarks (adjust this based on your landmark numbering)
NOSE_LANDMARK_INDEX = 30

# Function to calculate the midpoint between two points
def midpoint(point1, point2):
    return int((point1.x + point2.x) / 2), int((point1.y + point2.y) / 2)

# Function to move the mouse cursor based on the nose position with increased speed
def move_mouse(nose, sensitivity=5.0, tilt_degrees=10):
    screen_width, screen_height = pyautogui.size()
    
    # Get the nose position
    nose_x = nose.x
    nose_y = nose.y
    
    # Normalize the nose position to the range [0, 1]
    nose_x_normalized = nose_x / screen_width
    nose_y_normalized = nose_y / screen_height

    # Calculate the buffer size as 10% of the screen size
    buffer_width = int(screen_width * 0.2)
    buffer_height = int(screen_height * 0.2)

    # Adjust the screen width and height to include the buffer
    screen_width += buffer_width
    screen_height += buffer_height
    
    # Scale the nose position to move the cursor faster
    nose_x_scaled = nose_x_normalized * sensitivity
    
    # Adjust for the tilt by decreasing the vertical position
    # You can modify the tilt_degrees value as needed
    nose_y_scaled = (nose_y_normalized - (tilt_degrees / 90)) * sensitivity
    
    # Move the mouse cursor to the scaled position within the screen
    pyautogui.moveTo(nose_x_scaled * screen_width, nose_y_scaled * screen_height)


# Function to draw landmarks and landmark numbers
def draw_landmarks(frame, landmarks):
    for idx, point in enumerate(landmarks.parts()):
        x, y = point.x, point.y
        cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)  # Draw a red circle at each landmark point
        cv2.putText(frame, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

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

            # Get the nose landmark
            nose = landmarks.part(NOSE_LANDMARK_INDEX)

            # Draw the landmarks and numbers on the frame
            draw_landmarks(frame, landmarks)

            # Move the mouse cursor based on the nose position
            move_mouse(nose, sensitivity=3)

        # Display the frame with landmarks and nose tracking
        cv2.imshow('Facial Landmarks with Nose Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
