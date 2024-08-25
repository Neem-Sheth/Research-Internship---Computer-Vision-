import os
import cv2
import numpy as np

# Load class names
with open("data/coco.names", "r") as f:
    classes = [line.strip().replace(" ", "_") for line in f.readlines()]

# Generate random colors for each class
class_colors = {cls: tuple(np.random.randint(0, 255, size=3).tolist()) for cls in classes}

# Load KITTI dataset directories
kitti_image_dir = "D:/SVNIT/Semester-5/CISMR/kitti_dataset/data_object_image_2/training/image_2"
kitti_label_dir = "D:/SVNIT/Semester-5/CISMR/kitti_dataset/data_object_label_2/training/label_2"

def calculate_distance(object_height, bbox_height, object_width, bbox_width, frame_height_px, frame_width_px):
    # Adjusted focal length estimation based on object size
    sensor_height_mm = 70  # Typical height of a KITTI camera sensor in mm
    sensor_width_mm = 350  # Typical width of a KITTI camera sensor in mm
    focal_length_mm = 180  # Typical focal length of a KITTI camera in mm

    # Convert object dimensions from meters to millimeters
    object_height_mm = object_height * 1000  # Convert height to mm
    object_width_mm = object_width * 1000  # Convert width to mm

    # Focal length in pixels
    focal_length_px_h = (focal_length_mm / sensor_height_mm) * frame_height_px  # Focal length in pixels (height)
    focal_length_px_w = (focal_length_mm / sensor_width_mm) * frame_width_px  # Focal length in pixels (width)

    # Distance calculation using height and width
    distance_h = (object_height_mm * focal_length_px_h ) / bbox_height
    distance_w = (object_width_mm * focal_length_px_w) / bbox_width

    # Average the distances from height and width calculations
    distance_m = (distance_h + distance_w) / 2  # Average distance in millimeters

    # Convert distance to meters
    distance_m /= 1000

    # Adjust distance based on object size and bounding box dimensions

    distance_f = distance_m + object_width # 5.59

    # if object_width > 0.55:  # 5.38
    #     distance_f = distance_m + (object_width - bbox_width / 1000)
    # else:
    #     distance_f = distance_m +  (object_height - bbox_height / 1000) 

    # if object_width > 0.5 and object_height > 0.5:  # 5.95 
    #     distance_f = distance_m + (object_width - bbox_width / 1000)
    # else:
    #     distance_f = distance_m +  (object_height - bbox_height / 1000) 

    # distance_f = distance_m + object_height

    # distance_f = distance_m + (object_width - bbox_width / 1000) # 6.18

    # distance_f = distance_m # 6.19
    # if object_width > 0.5:  
    #     distance_f = distance_m + (object_width - bbox_width / 1000) 
    # if object_height > 1.5:
    #     distance_f = distance_m +  (object_height - bbox_height / 1000) 

    # distance_f = distance_m +  (object_height - bbox_height / 1000) # 6.29

    # if object_width > 1.5:  # 6.42
    #     distance_f = distance_m + (object_width - bbox_width / 1000) 

    # if object_height > 1.5: # 6.46
    #     distance_f = distance_m +  (object_height - bbox_height / 1000) 

    # distance_f = distance_m + bbox_width/1000

    # distance_f = distance_m + bbox_height/1000

    # distance_f = distance_m # 7.2

    return distance_f


def error(predicted_distance, actual_distance):
    return abs(predicted_distance - actual_distance)

number_of_images = 10
mean_error = 0
counter = 0

for img_file in os.listdir(kitti_image_dir):
    img_path = os.path.join(kitti_image_dir, img_file)
    label_path = os.path.join(kitti_label_dir, img_file.replace(".png", ".txt"))

    counter += 1
    if counter > number_of_images:
        break

    # Load image
    frame = cv2.imread(img_path)
    frame_height, frame_width, _ = frame.shape

    # Read the label file
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            obj_class = parts[0]
            if obj_class == "DontCare":
                continue

            truncated = float(parts[1])
            occluded = int(parts[2])
            alpha = float(parts[3])
            x1, y1, x2, y2 = map(float, parts[4:8])
            height, width, length = map(float, parts[8:11])
            loc_x, loc_y, loc_z = map(float, parts[11:14])
            rotation_y = float(parts[14])

            # Draw bounding box
            color = class_colors.get(obj_class, (255, 255, 255))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            predicted_distance = calculate_distance(height, round(abs(y1 - y2), 2), width, round(abs(x1 - x2), 2), frame_height, frame_width)

            mean_error += error(predicted_distance, loc_z)

            # Put class name and distance
            label = f"{obj_class} : {predicted_distance:.2f}m|{loc_z:.2f}m"
            # print(height, round(abs(y1 - y2)*0.00026, 2), width, round(abs(x1 - x2)*0.00026, 2), frame_height, frame_width)
            # print(label)
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            text_x = int(x1)
            text_y = int(y1) - 10 if int(y1) - 10 > 10 else int(y1) + 10
            cv2.rectangle(frame, (text_x, text_y - text_size[1] - 4), 
                          (text_x + text_size[0], text_y), color, -1)
            cv2.putText(frame, label, (text_x, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 0, 0), 1, cv2.LINE_AA)

    
    # Save the processed image (Optional)
    output_image_path = os.path.join("D:/SVNIT/Semester-5/CISMR/kitti_dataset_output", img_file)
    cv2.imwrite(output_image_path, frame)

    # Optionally display the frame
    # cv2.imshow("KITTI Object Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print(mean_error/number_of_images)
