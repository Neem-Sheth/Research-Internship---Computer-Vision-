# scripts/evaluate.py

import torch
from torch.utils.data import DataLoader
from models.yolov8_custom import YOLOv8Custom
from utils.dataset import KITTIDataset, transform
from torch import nn

# Load dataset
val_img_dir = "D:/SVNIT/Semester-5/CISMR/kitti_dataset_output/image/val"
val_lbl_dir = "D:/SVNIT/Semester-5/CISMR/kitti_dataset_output/label/val"
val_dataset = KITTIDataset(val_img_dir, val_lbl_dir, transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

# Load model
num_classes = 9  # Number of classes in KITTI dataset
model = YOLOv8Custom(num_classes).cuda()
model.load_state_dict(torch.load("yolov8_custom.pth"))
model.eval()

# Evaluation
val_loss = 0.0
criterion = nn.CrossEntropyLoss()
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.cuda(), labels.cuda()
        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item() * images.size(0)

val_loss /= len(val_loader.dataset)
print(f"Validation Loss: {val_loss:.4f}")
