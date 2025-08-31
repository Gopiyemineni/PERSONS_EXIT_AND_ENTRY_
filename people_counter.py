import argparse
import csv
import datetime
import json
import logging
import time
from itertools import zip_longest

import cv2
import dlib
import numpy as np
from imutils.video import FPS
from imutils.video import VideoStream

from tracker.centroidtracker import CentroidTracker
from tracker.trackableobject import TrackableObject
from utils import thread
from utils.mailer import Mailer

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

    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

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

    ct = CentroidTracker(maxDisappeared=100, maxDistance=75)
    trackers = []
    trackableObjects = {}

    totalFrames = 0
    fps_list = []  # List to store FPS values for each frame

    fps = FPS().start()

    if config["Thread"]:
        vs = thread.ThreadingClass(config["url"])

    while True:
        frame_start_time = time.time()  # Start time for FPS calculation
        
        frame = vs.read()
        frame = frame[1] if args.get("input", False) else frame

        if frame is None:
            break

        frame = cv2.resize(frame, (704, 576))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        if args["output"] is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)

        status = "Waiting"
        rects = []

        if totalFrames % args["skip_frames"] == 0:
            status = "Detecting"
            trackers = []

            blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
            net.setInput(blob)
            detections = net.forward()

            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > args["confidence"]:
                    idx = int(detections[0, 0, i, 1])

                    if CLASSES[idx] != "person":
                        continue

                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype("int")
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)

                    trackers.append(tracker)
        else:
            for tracker in trackers:
                status = "Tracking"

                tracker.update(rgb)
                pos = tracker.get_position()

                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

                rects.append((startX, startY, endX, endY))

        objects = ct.update(rects)

        for (objectID, centroid) in objects.items():
            to = trackableObjects.get(objectID, None)

            if to is None:
                to = TrackableObject(objectID, centroid)

            trackableObjects[objectID] = to

            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

        info_status = [
            ("Status", status),
        ]

        for (i, (k, v)) in enumerate(info_status):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        frame_time = time.time() - frame_start_time
        fps_value = 1 / frame_time
        fps_list.append(fps_value)
        cv2.putText(frame, f"FPS: {fps_value:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if writer is not None:
            writer.write(frame)

        cv2.imshow("Real-Time Monitoring/Analysis Window", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        totalFrames += 1
        fps.update()

        if config["Timer"]:
            end_time = time.time()
            num_seconds = (end_time - start_time)
            if num_seconds > 28800:
                break

    fps.stop()
    logger.info("Elapsed time: {:.2f}".format(fps.elapsed()))
    logger.info("Approx. FPS: {:.2f}".format(fps.fps()))

    # Calculate and log the average FPS
    if fps_list:
        average_fps = sum(fps_list) / len(fps_list)
        logger.info("Average FPS: {:.2f}".format(average_fps))

    if config["Thread"]:
        vs.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    people_counter()
