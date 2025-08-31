import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time
from deep_sort_realtime.deepsort_tracker import DeepSort
from torchvision import models, transforms

# Initialize YOLOv8 model
print("Initializing YOLOv8 model...")
model = YOLO("yolov8l.pt")
print("YOLOv8 model initialized.")

# Initialize DeepSORT
print("Initializing DeepSORT tracker...")
tracker = DeepSort(max_age=1500, n_init=10, nms_max_overlap=0.5, nn_budget=100)
print("DeepSORT tracker initialized.")

# Initialize ResNet50 for feature extraction
print("Initializing ResNet50 for feature extraction...")
feature_extractor = models.resnet50(pretrained=True)
feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-1])
feature_extractor.eval()
if torch.cuda.is_available():
    feature_extractor = feature_extractor.cuda()
print("ResNet50 initialized.")

# Define image transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def draw_boxes(frame, bbox_tlwh, identities, labels, fps):
    for bbox, identity, label in zip(bbox_tlwh, identities, labels):
        x, y, w, h = [int(i) for i in bbox]
        
        color1 = (0, 255, 0)  # Bounding box color
        color2 = (0, 0, 255)  # Text color
        cv2.rectangle(frame, (x, y), (x + w, y + h), color1, 2)
        cv2.putText(frame, f'{identity}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color2, 2)
    
    # Display FPS
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return frame

def extract_features(frame, bbox):
    x, y, w, h = [int(i) for i in bbox]
    crop = frame[y:y+h, x:x+w]
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = transform(crop).unsqueeze(0)
    
    if torch.cuda.is_available():
        crop = crop.cuda()
    
    with torch.no_grad():
        features = feature_extractor(crop)
    
    return features.cpu().numpy().flatten()

# Read the video file
video_path = r'C:\Users\91879\OneDrive\Desktop\entry_exit_counter\video_ex8.mp4'
print(f"Loading video from {video_path}...")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

print("Video loaded successfully.")

frame_count = 0
start_time = time.time()
fps = 0

# Variables to track total frames and total time
total_frames = 0
total_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot fetch frame.")
        break
    
    frame_count += 1
    total_frames += 1
    
    # Calculate FPS every second
    if (time.time() - start_time) > 1:
        fps = frame_count / (time.time() - start_time)
        frame_count = 0
        total_time += (time.time() - start_time)
        start_time = time.time()
    
    # YOLOv8 inference
    results = model(frame, classes=[0])  # Only detect persons
    
    # Process YOLOv8 results
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf.cpu().numpy()[0]
            cls = box.cls.cpu().numpy()[0]
            w, h = x2 - x1, y2 - y1
            detections.append(([x1, y1, w, h], conf, cls))
    
    # Prepare detections for DeepSORT
    raw_detections = []
    for bbox, confidence, cls in detections:
        features = extract_features(frame, bbox)
        raw_detections.append((bbox, confidence, features))
    
    # Update DeepSORT tracker
    tracks = tracker.update_tracks(raw_detections=raw_detections, frame=frame)
    
    bbox_tlwh = []
    identities = []
    labels = []
    
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlwh()
        identity = track.track_id
        
        bbox_tlwh.append(bbox)
        identities.append(identity)
        labels.append('person')
    
    # Draw boxes with consistent IDs
    frame = draw_boxes(frame, bbox_tlwh, identities, labels, fps)
    
    # Display the frame with bounding boxes and counts
    cv2.imshow("Person Tracker", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Calculate and print the average FPS
if total_time > 0:
    average_fps = total_frames / total_time
    print(f"Average FPS: {average_fps:.2f}")
else:
    print("No frames processed.")
