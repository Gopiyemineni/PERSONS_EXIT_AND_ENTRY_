import cv2

# Load the pre-trained HOG human detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Create a CSRT tracker
tracker = cv2.legacy.TrackerCSRT_create()

# Open video file or camera stream
video_path = R"C:\Users\sbyog\Desktop\people counter project\People-Counting-in-Real-Time-master\utils\data\tests\test_3.mp4"  # Replace with your video file path or 0 for camera stream
cap = cv2.VideoCapture(video_path)

# Check if video file or camera stream opened successfully
if not cap.isOpened():
    print("Error: Couldn't open video file or camera stream.")
    exit()

# Read the first frame
ret, frame = cap.read()
if not ret:
    print("Error: Couldn't read the first frame.")
    exit()

# Initialize bounding box coordinates for the tracker
bbox = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
tracker.init(frame, bbox)

# Loop through the video frames
while cap.isOpened():
    # Read a new frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect humans in the frame using HOG
    # The parameters may need adjustment based on your specific scenario
    humans, _ = hog.detectMultiScale(frame, winStride=(8, 8), padding=(16, 16), scale=1.05)

    # Track the detected humans
    for (x, y, w, h) in humans:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Update the tracker
    ret, bbox = tracker.update(frame)
    if ret:
        # Tracking successful, draw the bounding box
        (x, y, w, h) = [int(i) for i in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("Human Detection and Tracking", frame)

    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
