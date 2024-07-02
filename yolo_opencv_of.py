import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("weights/yolov3.weights", "cfg/yolov3.cfg")
layer_names = net.getLayerNames()

# Ensure compatibility with different versions of OpenCV
try:
    unconnected_layers = net.getUnconnectedOutLayers()
    if isinstance(unconnected_layers[0], list) or isinstance(unconnected_layers[0], np.ndarray):
        output_layers = [layer_names[i[0] - 1] for i in unconnected_layers]
    else:
        output_layers = [layer_names[i - 1] for i in unconnected_layers]
except Exception as e:
    print(f"An error occurred: {e}")
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class names
with open("data/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define colors for different classes
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Capture video from the webcam
cap = cv2.VideoCapture(0)


# Function to calculate distance based on object dimensions
def calculate_distance(w):
    W = 20 # Known real-world width of the object (example, adjust based on your object)
    F = 1000 # Focal length of the camera (example, adjust based on your camera)
    # Calculate distance using the formula D = (W * F) / w
    distance = (W * F) / w
    return distance

while True:
    ret, frame = cap.read()
    height, width, channels = frame.shape

    # Prepare the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Analyze the output
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label_image = str(classes[class_ids[i]])
            label_confidence = confidences[i]

            # Calculate distance based on object dimensions (w, h)
            distance = calculate_distance(w)

            # Assign color based on class_id
            color = colors[class_ids[i]]

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f'{label_image} ({label_confidence:.2f}) : {distance:.2f} inches', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Resize the frame for display
    frame = cv2.resize(frame, (width * 2, height * 2))  # Double the size of the frame

    cv2.imshow("Image", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
