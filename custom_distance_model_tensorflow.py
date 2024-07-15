import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load class names
with open("data/coco.names", "r") as f:
    classes = [line.strip().replace(" ", "_") for line in f.readlines()]

# Generate random colors for each class
class_colors = {cls: tuple(np.random.randint(0, 255, size=3).tolist()) for cls in classes}

# Load KITTI dataset directories
kitti_image_dir = "D:/SVNIT/Semester-5/CISMR/kitti_dataset/data_object_image_2/training/image_2"
kitti_label_dir = "D:/SVNIT/Semester-5/CISMR/kitti_dataset/data_object_label_2/training/label_2"

data = []

for img_file in os.listdir(kitti_image_dir):
    img_path = os.path.join(kitti_image_dir, img_file)
    label_path = os.path.join(kitti_label_dir, img_file.replace(".png", ".txt"))

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

            x1, y1, x2, y2 = map(float, parts[4:8])
            height, width, length = map(float, parts[8:11])
            loc_z = float(parts[13])  # Actual distance (label)

            bbox_height = abs(y1 - y2)
            bbox_width = abs(x1 - x2)

            # Collect the data
            data.append([height, bbox_height, width, bbox_width, frame_height, frame_width, loc_z])

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(data, columns=['object_height', 'bbox_height', 'object_width', 'bbox_width', 'frame_height_px', 'frame_width_px', 'distance'])

# Split the dataset
X = df[['object_height', 'bbox_height', 'object_width', 'bbox_width', 'frame_height_px', 'frame_width_px']]
y = df['distance']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network model
model = Sequential([
    Dense(64, input_dim=6, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

# Print the model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {test_mae:.2f} meters")

# Predict distances on the test set
y_pred = model.predict(X_test)

# Visualize predictions vs actual distances
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Distance (meters)')
plt.ylabel('Predicted Distance (meters)')
plt.title('Actual vs Predicted Distance')
plt.show()
