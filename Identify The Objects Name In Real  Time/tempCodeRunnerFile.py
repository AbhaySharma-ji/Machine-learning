import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import subprocess
import sys

# Install opencv-python-headless if not installed
def install_opencv():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
        print("opencv-python-headless installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while installing opencv-python-headless: {e}")

# Call the installation function
install_opencv()

# Paths to YOLOv3 weights, config, and COCO names file
weights_path = r"C:\Users\Nav\Desktop\mini 3rd project\yolov3.weights"
config_path = r"C:\Users\Nav\Desktop\mini 3rd project\yolov3.cfg"
names_path = r"C:\Users\Nav\Desktop\mini 3rd project\coco.names"

# Load class labels
with open(names_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Load YOLO model
net = cv2.dnn.readNet(weights_path, config_path)

# Enable GPU for faster processing (CUDA)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Set random colors for classes
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Initialize webcam
cap = cv2.VideoCapture(0)

# Create Tkinter window
window = tk.Tk()
window.title("Real-Time Object Detection")
label = tk.Label(window)
label.pack()

# Define frame skip (to speed up video processing)
frame_skip = 2  # Skip every 2nd frame for faster processing
frame_count = 0

def detect_objects(frame):
    height, width, _ = frame.shape

    # Resize the frame for faster processing (reduce size)
    frame = cv2.resize(frame, (640, 480))  # Resize to 640x480
    # Create blob from frame
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    # Process detections
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Confidence threshold for better accuracy
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indices) > 0:  # Ensure indices is not empty
        for i in indices.flatten():  # Flatten the indices
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]} {confidences[i]:.2f}"
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame

# Function to update the video frame
def update_frame():
    global frame_count
    ret, frame = cap.read()
    if ret and frame_count % frame_skip == 0:  # Skip frames to speed up
        frame = detect_objects(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)
    frame_count += 1
    window.after(10, update_frame)

# Start video stream
update_frame()
window.mainloop()

# Release resources
cap.release()
cv2.destroyAllWindows()
