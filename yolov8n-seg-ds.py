import cv2
from ultralytics import YOLO
import numpy as np
import random
import time
import math

# Load YOLOv8 segmentation model (you can use a smaller model like yolov8n-seg.pt for faster performance)
model = YOLO('models/yolov8n-seg.pt')

# Load class names
with open("data/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Generate different colors for each class
class_colors = {cls: [random.randint(0, 255) for _ in range(3)] for cls in range(len(classes))}

# Capture video from the webcam
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Reduce frame size for faster processing
frame_width = 640
frame_height = 480

# Initialize variables for speed calculation
previous_centroids = {}
previous_times = {}

# Known distance in meters corresponding to a known width in pixels
known_distance_meters = 0.2  # Example: 2 meters
known_width_pixels = 100  # Example: width of object in pixels corresponding to known_distance_meters

def calculate_distance(w, frame_width):
    # Distance calculation (example formula, needs calibration)
    distance = (frame_width * known_distance_meters) / (w + 10)
    return distance

def calculate_speed(centroid, previous_centroid, time_elapsed):
    if previous_centroid is None or time_elapsed == 0:
        return 0
    distance_moved = np.linalg.norm(np.array(centroid) - np.array(previous_centroid))
    speed_pixels_per_second = distance_moved / time_elapsed
    
    # Convert speed from pixels per second to meters per second
    speed_meters_per_second = (speed_pixels_per_second * known_distance_meters * 10) / known_width_pixels
    return speed_meters_per_second

def calculate_angle(centroid, frame_width):
    # Calculate the angle with respect to the center of the frame
    frame_center_x = frame_width // 2
    object_center_x = centroid[0]
    angle_degrees = math.degrees(math.atan2(object_center_x - frame_center_x, frame_width))
    return angle_degrees

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster processing
    frame = cv2.resize(frame, (frame_width, frame_height))

    # Perform detection and segmentation
    results = model(frame)[0]

    # Process masks
    masks = results.masks.data.cpu().numpy()

    # Process bounding boxes and labels
    for i, r in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, r.xyxy[0].tolist())
        w = x2 - x1
        h = y2 - y1
        centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
        conf = r.conf[0].item()
        cls = int(r.cls[0].item())
        label = classes[cls]
        color = class_colors[cls]

        # Calculate distance
        distance = calculate_distance(w, frame_width)

        # Calculate speed
        current_time = time.time()
        time_elapsed = current_time - previous_times.get(i, current_time)
        speed = calculate_speed(centroid, previous_centroids.get(i), time_elapsed)

        # Update previous centroids and times
        previous_centroids[i] = centroid
        previous_times[i] = current_time

        # Calculate angle
        angle_degrees = calculate_angle(centroid, frame_width)

        # Create a colored mask for the current object
        mask = masks[i]
        colored_mask = np.zeros_like(frame, dtype=np.uint8)
        for c in range(3):
            colored_mask[:, :, c] = mask * color[c]

        # Blend the colored mask with the original frame
        frame = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # Draw label text with background
        label_text = f"{label}:{conf:.2f}|{distance:.2f}m|{speed:.2f}m/s|{angle_degrees:.2f}deg"
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x = x1
        text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
        cv2.rectangle(frame, (text_x, text_y - text_size[1] - 4), (text_x + text_size[0], text_y), color, -1)
        cv2.putText(frame, label_text, (text_x, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Calculate and display FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Resize the frame for display (increase the size)
    display_frame = cv2.resize(frame, (frame_width * 2, frame_height * 2))  # Double the size of the frame

    cv2.imshow("YOLOv8 Segmentation", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
