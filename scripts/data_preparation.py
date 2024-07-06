import os
import cv2
import pandas as pd

def load_kitti_data(kitti_image_dir, kitti_label_dir):
    data = []
    image_count = 0

    for img_file in os.listdir(kitti_image_dir):
        img_path = os.path.join(kitti_image_dir, img_file)
        label_path = os.path.join(kitti_label_dir, img_file.replace(".png", ".txt"))

        frame = cv2.imread(img_path)
        frame_height, frame_width, _ = frame.shape

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                obj_class = parts[0]
                if obj_class == "DontCare":
                    continue

                x1, y1, x2, y2 = map(float, parts[4:8])
                height, width, length = map(float, parts[8:11])
                loc_z = float(parts[13])

                bbox_height = abs(y1 - y2)
                bbox_width = abs(x1 - x2)

                data.append([height, bbox_height, width, bbox_width, frame_height, frame_width, loc_z])

        image_count += 1
        # print(f"Image Number Processed: {image_count}")

    print(f"Number of images processed: {image_count}")
    return pd.DataFrame(data, columns=['object_height', 'bbox_height', 'object_width', 'bbox_width', 'frame_height_px', 'frame_width_px', 'distance'])
