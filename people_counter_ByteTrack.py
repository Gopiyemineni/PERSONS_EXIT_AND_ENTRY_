import argparse
import csv
import datetime
import json
import logging
import time
from itertools import zip_longest
from utils.mailer import Mailer
from utils.thread import ThreadingClass

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from imutils.video import FPS
from imutils.video import VideoStream
from yolov5 import YOLOv5

from tracker.centroidtracker import CentroidTracker
from tracker.trackableobject import TrackableObject

# execution start time
start_time = time.time()
# setup logger
logging.basicConfig(level=logging.INFO, format="[INFO] %(message)s")
logger = logging.getLogger(__name__)
# initiate features config.
with open("utils/config.json", "r") as file:
    config = json.load(file)

# Variables to store starting and ending points of the line
start_point = None
end_point = None
drawing = False
line_draw = 0

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
        line_draw = 1
        end_point = (x, y)

        cv2.line(frame, start_point, end_point, (255, 0, 0), 2)

def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required=False,
                    help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True,
                    help="path to Caffe pre-trained model")
    ap.add_argument("-i", "--input", type=str,
                    help="path to optional input video file")
    ap.add_argument("-o", "--output", type=str,
                    help="path to optional output video file")
    ap.add_argument("-c", "--confidence", type=float, default=0.4,
                    help="minimum probability to filter weak detections")
    ap.add_argument("-s", "--skip-frames", type=int, default=2,
                    help="# of skip frames between detections")
    args = vars(ap.parse_args())
    return args

def send_mail():
    Mailer().send(config["Email_Receive"])

def log_data(move_in, in_time, move_out, out_time):
    data = [move_in, in_time, move_out, out_time]
    export_data = zip_longest(*data, fillvalue='')

    with open('utils/data/logs/counting_data.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        if myfile.tell() == 0:
            wr.writerow(("Move In", "In Time", "Move Out", "Out Time"))
            wr.writerows(export_data)

def people_counter():
    args = parse_arguments()
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    yolo_model = YOLOv5("yolov5s.pt")
    tracker = DeepSort(max_age=30, nn_budget=70)

    if not args.get("input", False):
        logger.info("Starting the live stream..")
        vs = VideoStream(config["url"]).start()
        time.sleep(2.0)
    else:
        logger.info("Starting the video..")
        vs = cv2.VideoCapture(args["input"])

    writer = None
    W = None
    H = None

    trackableObjects = {}
    totalFrames = 0

    # Initialize FPS tracker
    fps_tracker = FPS().start()

    if config["Thread"]:
        vs = ThreadingClass(config["url"])

    # New variables for FPS calculation
    frame_count = 0
    start_time = time.time()

    while True:
        frame = vs.read()

        if frame[1] is None:
            break

        frame = cv2.resize(frame[1], (704, 576))
        frame = frame if args.get("input", False) else frame

        if args["input"] is not None and frame is None:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        if args["output"] is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)

        status = "Waiting"

        results = yolo_model.predict(frame)

        bboxes = results.xyxy[0][:, :4].cpu().numpy()
        confidences = results.xyxy[0][:, 4].cpu().numpy()
        class_ids = results.xyxy[0][:, 5].cpu().numpy().astype(int)

        mask = class_ids == 0
        bboxes = bboxes[mask]
        confidences = confidences[mask]

        detections = []
        for bbox, confidence in zip(bboxes, confidences):
            x1, y1, x2, y2 = bbox
            left = x1
            top = y1
            width = x2 - x1
            height = y2 - y1
            detections.append(([left, top, width, height], confidence, "person"))

        tracks = tracker.update_tracks(raw_detections=detections, frame=rgb)

        rects = []
        for track in tracks:
            status = "Tracking"
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            left, top, width, height = track.to_tlwh()

            startX, startY, endX, endY = int(left), int(top), int(left + width), int(top + height)
            rects.append((startX, startY, endX, endY))

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            text = f"ID {track_id}"

            centroid = ((startX + endX) // 2, (startY + endY) // 2)

            cv2.putText(frame, text, (centroid[0] + 10, centroid[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(frame, centroid, 4, (255, 255, 255), -1)

        info_status = [
            ("Status", status),
        ]

        for (i, (k, v)) in enumerate(info_status):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        if writer is not None:
            writer.write(frame)

        # Update FPS tracker
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps_value = frame_count / elapsed_time

        # Display FPS on the frame
        cv2.putText(frame, f"FPS: {fps_value:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Real-Time Monitoring/Analysis Window", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        totalFrames += 1
        fps_tracker.update()

        if config["Timer"]:
            end_time = time.time()
            num_seconds = (end_time - start_time)
            if num_seconds > 28800:
                break

    fps_tracker.stop()
    logger.info("Elapsed time: {:.2f}".format(fps_tracker.elapsed()))
    logger.info("Approx. FPS: {:.2f}".format(fps_tracker.fps()))

    if config["Thread"]:
        vs.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    people_counter()
