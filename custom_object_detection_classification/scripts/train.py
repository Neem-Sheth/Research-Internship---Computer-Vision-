# scripts/train.py

import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from yolov8_custom import YOLOv8Custom
from utils.dataset import KITTIDataset, transform

# Dataset directories
train_img_dir = "D:/SVNIT/Semester-5/CISMR/kitti_dataset_output/image/train"
train_lbl_dir = "D:/SVNIT/Semester-5/CISMR/kitti_dataset_output/label/train"
val_img_dir = "D:/SVNIT/Semester-5/CISMR/kitti_dataset_output/image/val"
val_lbl_dir = "D:/SVNIT/Semester-5/CISMR/kitti_dataset_output/label/val"

# Hyperparameters
num_classes = 9  # Number of classes in KITTI dataset
batch_size = 16
learning_rate = 0.001
num_epochs = 50

# Create datasets and dataloaders
train_dataset = KITTIDataset(train_img_dir, train_lbl_dir, transform)
val_dataset = KITTIDataset(val_img_dir, val_lbl_dir, transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Initialize model, loss function, and optimizer
model = YOLOv8Custom(num_classes).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

    val_loss /= len(val_loader.dataset)
    print(f"Validation Loss: {val_loss:.4f}")

# Save the model
torch.save(model.state_dict(), "yolov8_custom.pth")
