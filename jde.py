import cv2
import torch

# Initialize the JDE tracker
from pytracking import JDETracker

tracker = JDETracker(model_path='jde.pth', frame_rate=30)

# Open the video file
cap = cv2.VideoCapture('path/to/your/video.mp4')

# Loop through the video frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Update the tracker with the new frame
    tracks = tracker.track(frame)

    # Draw the bounding boxes on the frame
    for track in tracks:
        bbox = track.to_tlbr()  # Get the bounding box coordinates (top-left, bottom-right)
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('JDE Tracker', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()