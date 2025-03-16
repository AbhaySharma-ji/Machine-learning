import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import pygame
from twilio.rest import Client

# Twilio setup - You need to replace these with your own Twilio credentials


# Initialize Twilio Client
client = Client(ACCOUNT_SID, AUTH_TOKEN)

# Paths to YOLOv3 weights, config, and COCO names file
weights_path = r"C:\Users\Nav\Desktop\FINAL MINI PROJECT OF 3RD YEAR\yolov3.weights"
config_path = r"C:\Users\Nav\Desktop\FINAL MINI PROJECT OF 3RD YEAR\yolov3.cfg"
names_path = r"C:\Users\Nav\Desktop\FINAL MINI PROJECT OF 3RD YEAR\coco.names"

# Load class labels
with open(names_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Load YOLO model
net = cv2.dnn.readNet(weights_path, config_path)

# Enable GPU for faster processing (CUDA)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Set random colors for classes
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize pygame mixer for sound playback
pygame.mixer.init()

# Function to send an SMS via Twilio
def send_sms(message):
    try:
        client.messages.create(
            body=message,
            from_=TWILIO_PHONE,
            to=TO_PHONE
        )
        print(f"Message sent: {message}")
    except Exception as e:
        print(f"Failed to send message: {e}")

# Define the object you want to track
TARGET_OBJECT = "cell phone"  # Example: Look for "person" in the frame

# Function to play sound when object is detected
def play_sound():
    pygame.mixer.music.load(r'C:\Users\Nav\Desktop\FINAL MINI PROJECT OF 3RD YEAR\intruder_alert.mp3')
    pygame.mixer.music.play()

# Function to detect objects in a frame
def detect_objects(frame):
    height, width, _ = frame.shape

    # Resize the frame for faster processing (reduce size)
    frame = cv2.resize(frame, (640, 480))  # Resize to 640x480
    # Create blob from frame
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    detected_objects = []  # Keep track of detected objects

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

                # Check if the target object is detected
                if classes[class_id] == TARGET_OBJECT:
                    detected_objects.append(classes[class_id])

    # Apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indices) > 0:  # Ensure indices is not empty
        for i in indices.flatten():  # Flatten the indices
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]} {confidences[i]:.2f}"
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # If the target object is detected, send an SMS and play sound
    if TARGET_OBJECT in detected_objects:
        send_sms(f"{TARGET_OBJECT} detected in the frame!")
        play_sound()
        print("Alert Sound play")

    return frame

# Create Tkinter window
window = tk.Tk()
window.title("Real-Time Object Detection")
label = tk.Label(window)
label.pack()

# Define frame skip (to speed up video processing)
frame_skip = 5  # Skip every 2nd frame for faster processing
frame_count = 0

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
# Commenting out destroyAllWindows() to avoid error on Windows
# cv2.destroyAllWindows()
