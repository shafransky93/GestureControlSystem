import argparse
import cv2
import mediapipe as mp
import pyautogui
from math import sqrt

pyautogui.FAILSAFE = False

# Global variables to store the last known hand position and a smoothing factor
last_known_x = None
last_known_y = None
smoothing_factor = 0.7
mode = "None"  # Initialize mode as "None"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', nargs='*', help='Input video device (number or path), defaults to 0', default=['0'])
    args = parser.parse_args()

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    if len(args.input) == 1:
        INPUT = int(args.input[0]) if args.input[0].isdigit() else args.input[0]
    else:
        return print("Wrong number of values for 'input' argument; should be 0, 1, or 4.")

    frame_width = 640  # Set the desired video frame width
    frame_height = 480  # Set the desired video frame height

    cap = cv2.VideoCapture(INPUT)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv2.CAP_PROP_FPS, 120)

    global last_known_x, last_known_y, mode

    screen_width, screen_height = pyautogui.size()

    # Calculate the buffer size as 10% of the video frame size
    buffer_width = int(frame_width * 0.1)
    buffer_height = int(frame_height * 0.1)

    # Adjust the screen width and height to include the buffer
    screen_width += buffer_width
    screen_height += buffer_height

    center_x = screen_width // 2
    center_y = screen_height // 2
    pyautogui.moveTo(center_x, center_y)

    # Variable to keep track of previous pinky finger position
    prev_pinky_y = None

    # Define the dead zone size in pixels
    dead_zone = 20

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                # Draw lines connecting finger landmarks
                connections = mp_hands.HAND_CONNECTIONS
                for connection in connections:
                    x1, y1 = int(hand_landmarks.landmark[connection[0]].x * frame_width), int(hand_landmarks.landmark[connection[0]].y * frame_height)
                    x2, y2 = int(hand_landmarks.landmark[connection[1]].x * frame_width), int(hand_landmarks.landmark[connection[1]].y * frame_height)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            hand_landmarks = results.multi_hand_landmarks[0]

            thumb_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            pinky_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            thumb_x, thumb_y = int(thumb_landmark.x * frame_width), int(thumb_landmark.y * frame_height)
            index_finger_x, index_finger_y = int(index_finger_landmark.x * frame_width), int(index_finger_landmark.y * frame_height)
            pinky_x, pinky_y = int(pinky_landmark.x * frame_width), int(pinky_landmark.y * frame_height)

            distance = sqrt((thumb_x - index_finger_x)**2 + (thumb_y - index_finger_y)**2)

            # Adjust this threshold as needed for pinch sensitivity
            pinch_threshold = 15

            # Check if all five fingers are detected
            if len(hand_landmarks.landmark) == 21:
                if distance < pinch_threshold:
                    # Perform a left-click action
                    pyautogui.click()
                    mode = "Click Mode"
                else:
                    mode = "None"  # Reset mode if not pinching
            else:
                mode = "None"  # Reset mode if not all fingers detected

            # Check if pinky is extended (scroll mode)
            if pinky_y < index_finger_y:
                mode = "Scroll Mode"

                if prev_pinky_y is not None:
                    dy = pinky_y - prev_pinky_y
                    pyautogui.scroll(3*dy)

                prev_pinky_y = pinky_y

            inverted_index_finger_x = int(index_finger_x * screen_width / frame_width)

            # Check if only the index finger is extended (move mouse mode)
            if len(hand_landmarks.landmark) == 21:
                index_finger_tip = hand_landmarks.landmark[8]  # Assuming index finger tip landmark is at index 8
                index_finger_x, index_finger_y = int(index_finger_tip.x * frame_width), int(index_finger_tip.y * frame_height)

                if index_finger_y < pinky_y:
                    mode = "Move Mouse Mode"

                    if last_known_x is not None and last_known_y is not None:
                        dx = int((index_finger_x * screen_width / frame_width) - last_known_x)
                        dy = int((index_finger_y * screen_height / frame_height) - last_known_y)

                        # Apply the dead zone to prevent small movements
                        if abs(dx) <= dead_zone:
                            dx = 0
                        if abs(dy) <= dead_zone:
                            dy = 0

                        dx_smoothed = int(smoothing_factor * dx)
                        dy_smoothed = int(smoothing_factor * dy)

                        current_x, current_y = pyautogui.position()
                        pyautogui.moveTo(current_x + dx_smoothed, current_y + dy_smoothed, duration=0.1)

                    last_known_x = int(index_finger_x * screen_width / frame_width)
                    last_known_y = int(index_finger_y * screen_height / frame_height)
                else:
                    mode = "None"  # Reset mode if the index finger is not extended

        else:
            # Hand not detected, so keep the last known position
            if last_known_x is not None and last_known_y is not None:
                current_x, current_y = pyautogui.position()

        # Display the mode on the frame
        cv2.putText(frame, f"Mode: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Hand Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
