import cv2
from ultralytics import YOLO
import numpy as np
import time
from collections import deque

# Load YOLOv8 model
model = YOLO('models/yolov8n.pt')  # Choose the appropriate YOLOv8 model based on your requirements

# Load class names
with open("data/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define different colors for different classes
class_colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Capture video from the webcam
cap = cv2.VideoCapture(0)

# Initialize variables for object tracking
object_tracker = {}  # Dictionary to store object tracking information
object_id_counter = 0  # Counter to assign unique IDs to detected objects

def calculate_distance(w, width):
    distance = (width * 20) / (w + 10)  # Adjust this formula based on your observations
    return distance

def calculate_speed(tracker, object_id, new_position, current_time):
    if object_id in tracker and len(tracker[object_id]) > 1:
        old_position, old_time = tracker[object_id][-1]  # Get the last known position and time
        distance_moved = np.linalg.norm(np.array(new_position) - np.array(old_position))
        time_elapsed = current_time - old_time
        if time_elapsed > 0:
            speed = distance_moved / time_elapsed  # Speed in pixels per second
            return speed
    return 0

# Set confidence threshold and minimum bounding box area
confidence_threshold = 0.5
min_bbox_area = 500  # Adjust this value based on your requirements

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Perform object detection
    results = model(frame)

    # List to store current detections
    detections = []

    for result in results:
        for r in result.boxes:
            conf = r.conf.item()
            if conf < confidence_threshold:
                continue

            x1, y1, x2, y2 = map(int, r.xyxy[0])
            cls = int(r.cls.item())
            label = f"{classes[cls]} {conf:.2f}"

            # Calculate bounding box width and height
            w = x2 - x1
            h = y2 - y1

            # Filter out small bounding boxes
            if w * h < min_bbox_area:
                continue

            # Calculate distance and speed
            distance = calculate_distance(w, width)
            object_id = cls  # Use class index as object ID (assuming unique per class)
            current_time = time.time()
            speed = calculate_speed(object_tracker, object_id, (x1 + w // 2, y1 + h // 2), current_time)

            display_label = f"{label} : {distance:.2f}m : {speed:.2f} px/s"

            # Update object tracker with current position and time
            if object_id not in object_tracker:
                object_tracker[object_id] = deque(maxlen=2)
            object_tracker[object_id].append(((x1 + w // 2, y1 + h // 2), current_time))

            # Draw bounding box with rounded corners
            cv2.rectangle(frame, (x1, y1), (x2, y2), class_colors[cls], 2, cv2.LINE_AA)

            # Draw label text with background
            text_size, _ = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            text_x = x1
            text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
            cv2.rectangle(frame, (text_x, text_y - text_size[1] - 4), (text_x + text_size[0], text_y), class_colors[cls], -1)
            cv2.putText(frame, display_label, (text_x, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    # Resize the frame for display
    frame = cv2.resize(frame, (width * 2, height * 2))  # Double the size of the frame

    cv2.imshow("YOLOv8 Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
