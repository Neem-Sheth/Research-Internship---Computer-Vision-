import os
from sklearn.model_selection import train_test_split
import pandas as pd
from scripts.data_preparation import load_kitti_data
from scripts.dataset import KITTIDataset
from scripts.model import DistancePredictor
from scripts.train import train_model
from scripts.evaluate import evaluate_model, plot_results
import torch

# Set directories
kitti_image_dir = "D:/SVNIT/Semester-5/CISMR/kitti_dataset/data_object_image_2/training/image_2"
kitti_label_dir = "D:/SVNIT/Semester-5/CISMR/kitti_dataset/data_object_label_2/training/label_2"

# Load data
df = load_kitti_data(kitti_image_dir, kitti_label_dir)

# Normalize features
features = df[['object_height', 'bbox_height', 'object_width', 'bbox_width', 'frame_height_px', 'frame_width_px']].values
labels = df['distance'].values

# Normalize the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Prepare dataset
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
train_dataset = KITTIDataset(X_train, y_train)
test_dataset = KITTIDataset(X_test, y_test)

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
train_dataset = KITTIDataset(X_train, y_train)
val_dataset = KITTIDataset(X_val, y_val)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

# Initialize and train model
model = DistancePredictor().to(device)
train_model(model, train_dataset, val_dataset, device)

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))

# Evaluate model
y_pred, test_loss = evaluate_model(model, test_dataset, device)

# Plot results
plot_results(y_test, y_pred)
