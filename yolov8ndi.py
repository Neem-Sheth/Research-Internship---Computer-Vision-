import cv2
from ultralytics import YOLO
import numpy as np

# Load YOLOv8 model
model = YOLO('models/yolov8n.pt')  # You can use yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt based on your need

# Load class names
with open("data/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load object heights and widths from the file
object_dimensions = {}
with open("data/object_dimensions.txt", "r") as f:
    for line in f:
        class_name, height, width = line.strip().split()
        object_dimensions[class_name] = (float(height), float(width))  # Ensure the dimensions are in meters

# Define different colors for different classes
class_colors = np.random.uniform(0, 255, size=(len(classes), 3)).astype(int)

# Load the image
input_image_path = 'images/input/8.png'
output_image_path = 'images/output/8.png'
frame = cv2.imread(input_image_path)

def calculate_distance(object_height, bbox_height, object_width, bbox_width, frame_height, frame_width):
    # Adjusted focal length estimation
    sensor_height_mm = 99.2  # Typical height of a kitti camera sensor in mm
    sensor_width_mm = 328.6  # Typical width of a kitti camera sensor in mm
    focal_length_mm = 190.8  # Typical focal length of a kitti camera in mm

    focal_length_px_h = (focal_length_mm / sensor_height_mm) * frame_height  # Focal length in pixels (height)
    focal_length_px_w = (focal_length_mm / sensor_width_mm) * frame_width  # Focal length in pixels (width)

    # Distance calculation using height and width
    distance_h = (object_height * focal_length_px_h) / bbox_height
    distance_w = (object_width * focal_length_px_w) / bbox_width

    # Average the distances from height and width calculations
    distance_m = (distance_h + distance_w) / 2


    # if object_height-(bbox_height/100)>1:
    # if object_width-(bbox_width/100)>1:
    # if object_width-(bbox_width/100)>1 or object_height-(bbox_height/100)>1:
    #     distance_f = distance_m - (bbox_width/100)
    # else:
    #     distance_f = distance_m 

    distance_f = distance_m + (object_width-bbox_width/100)

    print(object_height, bbox_height, object_width, bbox_width, frame_height, frame_width, distance_m, distance_f)

    return distance_f # Return distance in meters

height, width, channels = frame.shape

# Perform detection
results = model(frame)[0]
detections = []

for r in results.boxes:
    x1, y1, x2, y2 = map(int, r.xyxy[0].tolist())
    conf = r.conf[0].item()
    cls = int(r.cls[0].item())
    label = classes[cls]

    detections.append({
        'label': f"{label} {conf:.2f}",
        'bbox': [x1, y1, x2 - x1, y2 - y1],
        'color': tuple(class_colors[cls].tolist())
    })

for detection in detections:
    x, y, w, h = detection['bbox']
    image = detection['label'].split()[0]
    color = detection['color']

    # Use the dimensions from the file to calculate distance
    if image in object_dimensions:
        object_height, object_width = object_dimensions[image]
        distance = calculate_distance(object_height, h, object_width, w, height, width)
    else:
        distance = 0  # Default distance if object dimensions are not known

    label = f"{image} : {distance:.2f}m"

    # Draw bounding box with rounded corners
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)

    # Draw label text with background
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    text_x = x
    text_y = y - 10 if y - 10 > 10 else y + 10
    cv2.rectangle(frame, (text_x, text_y - text_size[1] - 4), (text_x + text_size[0], text_y), color, -1)
    cv2.putText(frame, label, (text_x, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

# Save the processed image
cv2.imwrite(output_image_path, frame)

# Optionally display the frame
cv2.imshow("YOLOv8 Object Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
