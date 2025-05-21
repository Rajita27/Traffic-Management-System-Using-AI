import torch
import cv2
import warnings
import numpy as np
from collections import OrderedDict

warnings.filterwarnings("ignore")

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5m')

video_path = 'media/video.mp4'
cap = cv2.VideoCapture(video_path)

confidence_threshold = 0.6
vehicle_classes = [2, 3, 1, 7, 5]

box_c7olor = (0, 255, 0)
text_color = (255, 255, 255)
text_background_color = (0, 0, 0)

# Move red line lower (e.g. 550 instead of 500)
line_y_position = 650
vehicle_count = 0

# Simple Centroid Tracker
class CentroidTracker:
    def __init__(self, max_distance=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.counted = set()
        self.max_distance = max_distance

    def update(self, rects):
        new_objects = OrderedDict()
        for rect in rects:
            x1, y1, x2, y2 = rect
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            matched = False
            for obj_id, existing_centroid in self.objects.items():
                distance = np.linalg.norm(np.array(existing_centroid) - np.array(centroid))
                if distance < self.max_distance:
                    new_objects[obj_id] = centroid
                    matched = True
                    break
            if not matched:
                new_objects[self.nextObjectID] = centroid
                self.nextObjectID += 1
        self.objects = new_objects
        return self.objects

tracker = CentroidTracker()

frame_skip = 5  # <== Skip every 2nd frame to simulate fast-forward

frame_id = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    if frame_id % frame_skip != 0:
        continue  # Skip this frame

    results = model(frame)
    filtered_boxes = []

    for *box, conf, cls in results.xywh[0]:
        if conf >= confidence_threshold and int(cls) in vehicle_classes:
            x_center, y_center, width, height = [int(i) for i in box]
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            filtered_boxes.append((x1, y1, x2, y2))

    objects = tracker.update(filtered_boxes)

    for obj_id, centroid in objects.items():
        cx, cy = centroid
        cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
        cv2.putText(frame, f"ID {obj_id}", (cx - 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

        if obj_id not in tracker.counted and cy > line_y_position:
            vehicle_count += 1
            tracker.counted.add(obj_id)

    cv2.line(frame, (0, line_y_position), (frame.shape[1], line_y_position), (0, 0, 255), 2)
    cv2.putText(frame, f"Vehicle Count: {vehicle_count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow("YOLO + Vehicle Counting", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
