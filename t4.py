import cv2
import numpy as np
import random
import time
from collections import deque

# Thresholds to detect object
confidence_threshold = 0.4  # Higher confidence threshold for better accuracy
nms_threshold = 0.3        # Non-maximum suppression threshold

# Load class names
classNames = []
classFile = "data/coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# Generate random colors for each class
class_colors = {className: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for className in classNames}

# Load the neural network model
configPath = "cfg/ssd_mobilenet_v3_large_coco.pbtxt"
weightsPath = "weights/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Object tracker
object_tracker = {}
next_object_id = 0

# Function to calculate distance
def calculate_distance(w, width):
    distance = (width * 20) / (w + 10)  # Adjust this formula based on your observations
    return distance

# Function to calculate speed
def calculate_speed(tracker, object_id, new_position, current_time, pixel_to_meter_ratio, min_movement=2, min_time_interval=0.1):
    if object_id in tracker and len(tracker[object_id]) > 1:
        old_position, old_time = tracker[object_id][-1]  # Get the last known position and time
        distance_moved_pixels = np.linalg.norm(np.array(new_position) - np.array(old_position))
        if distance_moved_pixels < min_movement:
            return 0  # Ignore very small movements
        distance_moved_meters = distance_moved_pixels * pixel_to_meter_ratio  # Convert pixels to meters
        time_elapsed = current_time - old_time
        if time_elapsed < min_time_interval:
            return 0  # Ignore very short time intervals
        if time_elapsed > 0:
            speed = distance_moved_meters / time_elapsed  # Speed in meters per second
            return speed
    return 0

def getObjects(img, confidence_threshold, nms_threshold, draw=True, objects=[]):
    global next_object_id  # Declare next_object_id as global

    classIds, confs, bbox = net.detect(img, confThreshold=confidence_threshold, nmsThreshold=nms_threshold)
    if len(objects) == 0:
        objects = classNames

    objectInfo = []
    height, width, _ = img.shape
    current_time = time.time()
    pixel_to_meter_ratio = 0.01  # Adjust this based on your setup

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            if 0 < classId <= len(classNames):
                className = classNames[classId - 1]
                if className in objects:
                    objectInfo.append([box, className])
                    if draw:
                        color = class_colors[className]
                        cv2.rectangle(img, box, color=color, thickness=2)
                        # Calculate bounding box width and height
                        w = box[2]
                        h = box[3]

                        # Calculate the center of the bounding box
                        center_x, center_y = box[0] + w // 2, box[1] + h // 2

                        # Try to match this detection to an existing object
                        matched_object_id = None
                        min_distance = float('inf')

                        for object_id, positions in object_tracker.items():
                            if len(positions) > 0:
                                last_position, _ = positions[-1]
                                distance = np.linalg.norm(np.array((center_x, center_y)) - np.array(last_position))
                                if distance < min_distance:
                                    min_distance = distance
                                    matched_object_id = object_id

                        # If no match found, assign a new object ID
                        if matched_object_id is None or min_distance > 50:  # Distance threshold for matching
                            matched_object_id = next_object_id
                            next_object_id += 1

                        # Calculate distance and speed
                        distance = calculate_distance(w, width)
                        speed = calculate_speed(object_tracker, matched_object_id, (center_x, center_y), current_time, pixel_to_meter_ratio)

                        display_label = f"{className} : {distance:.2f}cm : {speed:.2f} m/s"

                        # Update object tracker with current position and time
                        if matched_object_id not in object_tracker:
                            object_tracker[matched_object_id] = deque(maxlen=2)
                        object_tracker[matched_object_id].append(((center_x, center_y), current_time))

                        # Draw label text with background
                        text_size, _ = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        text_x = box[0]
                        text_y = box[1] - 10 if box[1] - 10 > 10 else box[1] + 10
                        cv2.rectangle(img, (text_x, text_y - text_size[1] - 4), (text_x + text_size[0], text_y), color, -1)
                        cv2.putText(img, display_label, (text_x, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return img, objectInfo

# Main loop to process video frames
cap = cv2.VideoCapture("traffic.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection and get annotated image with object info
    result_frame, object_info = getObjects(frame, confidence_threshold, nms_threshold)

    # Display the annotated frame
    cv2.imshow("Object Detection", result_frame)

    # Exit the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
