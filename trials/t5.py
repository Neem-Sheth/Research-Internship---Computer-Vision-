import cv2
from ultralytics import YOLO
import numpy as np
import os

# Load YOLOv8 model
model = YOLO('models/yolov8n.pt')

# Load class names
with open("data/coco.names", "r") as f:
    classes = [line.strip().replace(" ", "_") for line in f.readlines()]

# Define different colors for different classes
class_colors = np.random.uniform(0, 255, size=(len(classes), 3)).astype(int)

# Load object dimensions from file
object_dimensions = {}
with open("data/object_dimensions.txt", "r") as f:
    for line in f:
        obj_class, width, height = line.strip().split()
        object_dimensions[obj_class] = (float(width), float(height))

def calculate_distance(object_height, bbox_height, object_width, bbox_width, frame_height, frame_width):
    # Adjusted focal length estimation
    sensor_height_mm = 99.2  # Typical height of a smartphone camera sensor in mm
    sensor_width_mm = 328.6  # Typical width of a smartphone camera sensor in mm
    focal_length_mm = 190.8  # Typical focal length of a smartphone camera in mm

    focal_length_px_h = (focal_length_mm / sensor_height_mm) * frame_height  # Focal length in pixels (height)
    focal_length_px_w = (focal_length_mm / sensor_width_mm) * frame_width  # Focal length in pixels (width)

    # Distance calculation using height and width
    distance_h = (object_height * focal_length_px_h) / bbox_height
    distance_w = (object_width * focal_length_px_w) / bbox_width

    # Average the distances from height and width calculations
    distance_m = (distance_h + distance_w) / 2


    # # if object_height-(bbox_height/100)>1:
    # if object_width-(bbox_width/100)>1:
    # # if object_width-(bbox_width/100)>1 and object_height-(bbox_height/100)>1:
    #     distance_f = distance_m + (bbox_height/100) - ((bbox_width/100)/2)
    # else:
    #     distance_f = distance_m + (bbox_heightc/100)

    distance_f = distance_m + (object_width-bbox_width/100)

    print(object_height, bbox_height, object_width, bbox_width, frame_height, frame_width, distance_m, distance_f)

    return distance_f # Return distance in meters

# Load KITTI dataset
kitti_image_dir = "D:/SVNIT/Semester-5/CISMR/kitti_dataset/data_object_image_2/training/image_2"
kitti_label_dir = "D:/SVNIT/Semester-5/CISMR/kitti_dataset/data_object_label_2/training/label_2"

counter = 0

for img_file in os.listdir(kitti_image_dir):
    img_path = os.path.join(kitti_image_dir, img_file)
    label_path = os.path.join(kitti_label_dir, img_file.replace(".png", ".txt"))

    counter +=1
    if counter>5:
        break

    # Load image
    frame = cv2.imread(img_path)
    frame_height, frame_width, _ = frame.shape

    # Perform detection
    results = model(frame)[0]
    detections = []

    for r in results.boxes:
        x1, y1, x2, y2 = map(int, r.xyxy[0].tolist())
        conf = r.conf[0].item()
        cls = int(r.cls[0].item())
        label = classes[cls]

        detections.append({
            'label': label,
            'bbox': [x1, y1, x2 - x1, y2 - y1],
            'color': tuple(class_colors[cls].tolist())
        })

    # Compare with ground truth from KITTI labels
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            obj_class = parts[0]
            ground_truth_distance = float(parts[-2])

            for detection in detections:
                # print(detection['label'], obj_class)
                if detection['label'].lower() == obj_class.lower():
                    x, y, w, h = detection['bbox']
                    if obj_class.lower() in object_dimensions:
                        object_height, object_width = object_dimensions[obj_class.lower()]
                        predicted_distance = calculate_distance(object_height, h, object_width, w, frame_height, frame_width)
                    else:
                        predicted_distance = 0
                    
                    # Calculate error
                    if predicted_distance is not None:
                        error = abs(predicted_distance - ground_truth_distance)
                        print(x, y, x+w, y+h, "||", float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7]))
                        print(f"Object: {obj_class}, Ground Truth Distance: {ground_truth_distance}m, Predicted Distance: {predicted_distance:.2f}m, Error: {error:.2f}m")

                    # Draw bounding box and label
                    color = detection['color']
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)
                    label = f"{detection['label']} : {predicted_distance:.2f}m"
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    text_x = x
                    text_y = y - 10 if y - 10 > 10 else y + 10
                    cv2.rectangle(frame, (text_x, text_y - text_size[1] - 4), (text_x + text_size[0], text_y), color, -1)
                    cv2.putText(frame, label, (text_x, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Save the processed image (Optional)
    output_image_path = os.path.join("D:/SVNIT/Semester-5/CISMR/kitti_dataset_output", img_file)
    cv2.imwrite(output_image_path, frame)

    # Optionally display the frame
    cv2.imshow("YOLOv8 Object Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
