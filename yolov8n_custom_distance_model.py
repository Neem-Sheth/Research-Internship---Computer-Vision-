import cv2
from ultralytics import YOLO
import numpy as np
import torch
from torch import nn

# Load YOLOv8 model
yolo_model = YOLO('models/yolov8n.pt')  # You can use yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt based on your need

# Load class names
with open("data/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define different colors for different classes
class_colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Load the distance prediction model
class DistancePredictor(nn.Module): # 0.71
    def __init__(self):
        super(DistancePredictor, self).__init__()
        self.fc1 = nn.Linear(6, 64, True)
        self.fc2 = nn.Linear(64, 64, True)
        self.fc3 = nn.Linear(64, 1, True)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x): 
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
distance_model = DistancePredictor().to(device)
distance_model.load_state_dict(torch.load('models/best_model.pth'))
distance_model.eval()

# Read object dimensions from file
object_dimensions = {}
with open("data/coco_object_dimensions.txt", "r") as f:
    for line in f:
        class_name, height, width = line.strip().split()
        class_name = class_name.replace('_', ' ')
        object_dimensions[class_name] = (float(height), float(width))  # Ensure the dimensions are in meters

# Capture video from the webcam
cap = cv2.VideoCapture(0)

def calculate_distance(features):
    features = torch.tensor(features, dtype=torch.float32).to(device)
    with torch.no_grad():
        distance = distance_model(features).item()
    return distance

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to a smaller size for faster processing
    frame = cv2.resize(frame, (1224, 375))
    
    height, width, channels = frame.shape


    # Perform detection
    results = yolo_model(frame)[0]
    detections = []

    for r in results.boxes:
        x1, y1, x2, y2 = map(int, r.xyxy[0].tolist())
        conf = r.conf[0].item()
        cls = int(r.cls[0].item())
        label = yolo_model.names[cls]

        bbox_height = abs(y1 - y2)
        bbox_width = abs(x1 - x2)
        
        if label in object_dimensions:
            object_height, object_width = object_dimensions[label]
            features = [object_height, bbox_height, object_width, bbox_width, height, width]
            distance = calculate_distance(features)
        else:
            distance = -1  # If object dimensions are not available, set distance to -1

        detections.append({
            'label': f"{label} {conf:.2f}",
            'bbox': [x1, y1, x2 - x1, y2 - y1],
            'color': class_colors[cls],
            'distance': distance
        })

    for detection in detections:
        x, y, w, h = detection['bbox']
        image = detection['label']
        color = detection['color']
        distance = detection['distance']

        label = f"{image} : {distance/10000:.2f}m" if distance != -1 else image

        # Draw bounding box with rounded corners
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)

        # Draw label text with background
        text_size = cv2.getTextSize(image, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x = x
        text_y = y - 10 if y - 10 > 10 else y + 10
        cv2.rectangle(frame, (text_x, text_y - text_size[1] - 4), (text_x + text_size[0], text_y), color, -1)
        cv2.putText(frame, label, (text_x, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    # Resize the frame for display
    frame = cv2.resize(frame, (1224, 375))

    cv2.imshow("YOLOv8 Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
