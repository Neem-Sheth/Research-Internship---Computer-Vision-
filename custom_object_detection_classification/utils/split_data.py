# utils/split_data.py

import os
import shutil
import random

def split_data(image_dir, label_dir, train_split=0.8):
    images = sorted(os.listdir(image_dir))
    labels = sorted(os.listdir(label_dir))

    data_pairs = list(zip(images, labels))
    random.shuffle(data_pairs)

    train_size = int(len(data_pairs) * train_split)
    train_data = data_pairs[:train_size]
    val_data = data_pairs[train_size:]

    image_dir_output = "D:/SVNIT/Semester-5/CISMR/kitti_dataset_output/image"
    label_dir_output = "D:/SVNIT/Semester-5/CISMR/kitti_dataset_output/label"

    os.makedirs(os.path.join(image_dir_output, 'train'), exist_ok=True)
    os.makedirs(os.path.join(image_dir_output, 'val'), exist_ok=True)
    os.makedirs(os.path.join(label_dir_output, 'train'), exist_ok=True)
    os.makedirs(os.path.join(label_dir_output, 'val'), exist_ok=True)

    for img, lbl in train_data:
        shutil.move(os.path.join(image_dir, img), os.path.join(image_dir_output, 'train', img))
        shutil.move(os.path.join(label_dir, lbl), os.path.join(label_dir_output, 'train', lbl))

    for img, lbl in val_data:
        shutil.move(os.path.join(image_dir, img), os.path.join(image_dir_output, 'val', img))
        shutil.move(os.path.join(label_dir, lbl), os.path.join(label_dir_output, 'val', lbl))

if __name__ == "__main__":
    split_data("D:/SVNIT/Semester-5/CISMR/kitti_dataset/data_object_image_2/training/image_2", "D:/SVNIT/Semester-5/CISMR/kitti_dataset/data_object_label_2/training/label_2")
