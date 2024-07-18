# utils/dataset.py

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class KITTIDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(images_dir))
        self.label_files = sorted(os.listdir(labels_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        label_path = os.path.join(self.labels_dir, self.label_files[idx])
        
        image = np.array(Image.open(img_path).convert("RGB"))
        boxes = self.parse_labels(label_path)
        
        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes, labels=np.ones(len(boxes)))
            image = transformed['image']
            boxes = transformed['bboxes']
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        return image, boxes

    def parse_labels(self, label_path):
        boxes = []
        with open(label_path, 'r') as f:
            for line in f:
                data = line.strip().split(' ')
                if data[0] in ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']:
                    x_min = float(data[4])
                    y_min = float(data[5])
                    x_max = float(data[6])
                    y_max = float(data[7])
                    boxes.append([x_min, y_min, x_max, y_max])
        return boxes

transform = A.Compose([
    A.Resize(640, 640),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
