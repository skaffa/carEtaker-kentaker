import cv2
import numpy as np
import time
import uuid
import os
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Path to Haar Cascade XML for vehicle detection
cascade_path = 'haarcascade_car.xml'  # Update this path if needed

# Check if the cascade file exists
if not os.path.exists(cascade_path):
    print(f"Error: The file {cascade_path} does not exist.")
    raise FileNotFoundError(f"The file {cascade_path} was not found.")
else:
    print(f"Found cascade file: {cascade_path}")

# Load Haar Cascade for vehicle detection
vehicle_cascade = cv2.CascadeClassifier(cascade_path)

# Verify if the cascade file is loaded
if vehicle_cascade.empty():
    raise Exception("Failed to load Haar Cascade file. Please check the file path.")

# Create a directory to save images
save_dir = 'saved_images'
os.makedirs(save_dir, exist_ok=True)

# Function to detect and draw cars on the frame
def detect_and_draw_cars(image):
    color_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    cars = vehicle_cascade.detectMultiScale(
        color_img,
        scaleFactor=1.1,
        minNeighbors=2,  # Loosened from 3
        minSize=(30, 30),  # Loosened from (50, 50)
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Merge close bounding boxes
    merged_boxes = merge_bounding_boxes(cars)

    print(f"Detected {len(merged_boxes)} cars in the image.")
    for (x, y, w, h) in merged_boxes:
        # Draw rectangle around detected car
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return image

# Function to merge overlapping bounding boxes
def merge_bounding_boxes(boxes, overlap_thresh=0.3):
    if len(boxes) == 0:
        return []

    # Convert bounding boxes to list of (x, y, x+w, y+h)
    boxes = np.array([(x, y, x+w, y+h) for (x, y, w, h) in boxes])
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    indices = np.argsort(y2)

    merged_boxes = []
    while len(indices) > 0:
        i = indices[-1]
        xx1 = np.maximum(x1[i], x1[indices[:-1]])
        yy1 = np.maximum(y1[i], y1[indices[:-1]])
        xx2 = np.minimum(x2[i], x2[indices[:-1]])
        yy2 = np.minimum(y2[i], y2[indices[:-1]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / (areas[indices[:-1]] + areas[i] - (w * h))

        to_merge = np.where(overlap > overlap_thresh)[0]

        if len(to_merge) > 0:
            x1[i] = np.min(x1[indices[to_merge]])
            y1[i] = np.min(y1[indices[to_merge]])
            x2[i] = np.max(x2[indices[to_merge]])
            y2[i] = np.max(y2[indices[to_merge]])

        merged_boxes.append((x1[i], y1[i], x2[i] - x1[i], y2[i] - y1[i]))
        indices = np.delete(indices, np.concatenate(([len(indices) - 1], to_merge)))

    return merged_boxes

# Function to detect white spots in the frame
def detect_white_spots(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply threshold to detect white areas
    _, thresholded = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    # Find contours of white spots
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    white_spots = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area > 500 and area < 3000 and (x + w / 2) > (image.shape[1] / 4) and (x + w / 2) < (3 * image.shape[1] / 4):
            white_spots.append((x, y, w, h))

    return white_spots

# Function to filter and save detected objects
def filter_and_save_objects(frame, contours):
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if cv2.contourArea(contour) > 500:
            moving_object_image = frame[y:y+h, x:x+w]

            # Save the detected object image with a random name
            file_name = f"{uuid.uuid4().hex}.jpg"
            file_path = os.path.join(save_dir, file_name)
            cv2.imwrite(file_path, moving_object_image)
            print(f"Saved detected object as {file_path}")

            # Draw rectangle around detected moving object
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return frame

# Function to capture snapshots and process them
def capture_and_process():
    camera_url = 'http://192.168.1.82:8080/video'
    capture_interval = 0.01  # Time between snapshots in seconds

    # Initialize background subtractor with parameters adjusted for less sensitivity
    backSub = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=90,  # Adjusted from 100 to reduce sensitivity
        detectShadows=True
    )

    # Cropping margins (in pixels) for each side of the frame
    margin_left = 50
    margin_right = 100
    margin_top = 435
    margin_bottom = 150

    while True:
        cap = cv2.VideoCapture(camera_url)
        if not cap.isOpened():
            print("Error: Unable to open camera stream.")
            time.sleep(capture_interval)
            continue

        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("Error: Unable to read frame from camera.")
            time.sleep(capture_interval)
            continue

        # Crop the frame based on margins
        height, width = frame.shape[:2]
        frame = frame[
            margin_top:height-margin_bottom,
            margin_left:width-margin_right
        ]

        # Apply background subtraction to get the foreground mask
        fg_mask = backSub.apply(frame)

        # Use morphological operations to clean up the foreground mask
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # Find contours of moving objects
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter, save objects, and keep the visual overlay
        frame_with_detections = filter_and_save_objects(frame, contours)

        # Display the frame with detections
        cv2.imshow('Detected Moving Objects', frame_with_detections)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(capture_interval)

    cv2.destroyAllWindows()

# Start capturing when the script is run
if __name__ == '__main__':
    capture_and_process()
