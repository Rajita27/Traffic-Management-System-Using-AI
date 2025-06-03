from django.http import StreamingHttpResponse, JsonResponse
from django.shortcuts import render
import torch
import cv2
import warnings
import numpy as np
from collections import OrderedDict
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt


warnings.filterwarnings("ignore")

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5m')

video_path = 'combined dataset/sample_video.mp4'
confidence_threshold = 0.6
vehicle_classes = [2, 3, 1, 7, 5]
box_color = (0, 255, 0)
text_color = (255, 255, 255)
text_background_color = (0, 0, 0)
line_y_position = 650
vehicle_count = 0
frame_skip = 5

# Control flags
detection_active = False
reset_requested = False

# Tracker class
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

# Renders your index.html
def dashboard(request):
    return render(request, 'index.html')

# AJAX call to get current vehicle count
def get_vehicle_count(request):
    return JsonResponse({'count': vehicle_count})

# Video stream
def video_feed(request):
    def generate_video():
        print("video_feed started")
        global vehicle_count, detection_active, reset_requested, tracker

        cap = cv2.VideoCapture(video_path)
        frame_id = 0

        while True:
            if not detection_active:
                cv2.waitKey(30)
                continue  # skip loop until detection is active

            if reset_requested:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind video
                frame_id = 0
                reset_requested = False

            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            if frame_id % frame_skip != 0:
                continue

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

            for obj_id, (cx, cy) in objects.items():
                cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
                cv2.putText(frame, f"ID {obj_id}", (cx - 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

                if obj_id not in tracker.counted and cy > line_y_position:
                    vehicle_count += 1
                    tracker.counted.add(obj_id)

            #cv2.putText(frame, f"Vehicle Count: {vehicle_count}", (30, 50),
                        #cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

        cap.release()

    return StreamingHttpResponse(generate_video(), content_type='multipart/x-mixed-replace; boundary=frame')

@csrf_exempt
def start_detection(request):
    global detection_active
    detection_active = True
    print("=== START DETECTION CALLED ===")  # <--- Add this line
    return JsonResponse({'status': 'started'})


@csrf_exempt
def stop_detection(request):
    global detection_active
    detection_active = False
    return JsonResponse({'status': 'stopped'})

@csrf_exempt
def reset_detection(request):
    global detection_active, tracker, vehicle_count, reset_requested
    detection_active = True
    vehicle_count = 0
    tracker = CentroidTracker()
    reset_requested = True
    return JsonResponse({'status': 'reset'})
