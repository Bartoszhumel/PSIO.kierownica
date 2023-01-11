import cv2
import numpy as np


# Function to calculate the angle of turn
def calculate_angle(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dy = y2 - y1
    dx = x2 - x1
    angle = np.arctan2(dy, dx)
    return angle


# Load video
video = cv2.VideoCapture("wideo.mp4")

# Read the first frame
_, prev_frame = video.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_obraz_fil = cv2.GaussianBlur(prev_gray, (15, 15), 0)

while True:
    # Read the next frame
    _, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    obraz_fil = cv2.GaussianBlur(gray, (15, 15), 0)

    if gray is None:
        break

    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(prev_obraz_fil, obraz_fil, None, 0.5, 3, 5, 3, 5, 1.2, 0)

    # Find the position of the steering wheel
    x, y = None, None
    for row in range(flow.shape[0]):
        for col in range(flow.shape[1]):
            # Set a threshold for the magnitude of the optical flow
            if flow[row, col, 0] > 20:
                x, y = col, row
                break
        if x is not None and y is not None:
            break

    # Draw the position of the steering wheel on the frame
    if x is not None and y is not None:
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        # Calculate the angle of turn
        angle = calculate_angle((x, y), (x + flow[y, x, 0], y + flow[y, x, 1]))
        print("Angle of turn: {:.2f} radians".format(angle))

    # Display the frame
    cv2.imshow("Steering Wheel Tracking", frame)
    key = cv2.waitKey(1)
    if key == 27:  # Press 'ESC' to exit
        break

    prev_gray = gray

video.release()
cv2.destroyAllWindows()