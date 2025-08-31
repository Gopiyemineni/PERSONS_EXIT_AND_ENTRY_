import cv2

# Variables to store starting and ending points of the line
start_point = None
end_point = None
drawing = False
line_draw=0
# Mouse callback function to capture mouse events
def draw_line(event, x, y, flags, param):
    global start_point, end_point, drawing, line_draw

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        end_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        line_draw=1
        end_point = (x, y)

        cv2.line(frame, start_point, end_point, (255, 0, 0), 2)

# Open video file or camera stream
video_path = R"C:\Users\sbyog\Desktop\people counter project\People-Counting-in-Real-Time-master\utils\data\tests\test_3.mp4"  
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

# Create a window and set the mouse callback function
cv2.namedWindow("Draw Line and Select ROI")
cv2.setMouseCallback("Draw Line and Select ROI", draw_line)


while cap.isOpened():

    frame_copy = frame.copy()
    # print('in while')
    if drawing:
        cv2.line(frame_copy, start_point, end_point, (255, 0, 0), 2)

    cv2.imshow("Draw Line and Select ROI", frame_copy)
    cv2.waitKey(1)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    # Exit loop if 'q' key is pressed
    if line_draw:
        break

bbox = cv2.selectROI("Draw Line and Select ROI", frame, fromCenter=False, showCrosshair=False)
cap.release()
cv2.destroyAllWindows()
