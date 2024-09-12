import cv2
import torch
import pytesseract
from flask import Flask, Response, render_template, request, jsonify
from PIL import Image
import numpy as np
import time
from fuzzywuzzy import fuzz
import warnings
import re
import os
from datetime import datetime

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Configure pytesseract path (change this to your Tesseract installation path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the YOLOv5x model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

# Adjust detection parameters for better accuracy
model.conf = 0.4  # Confidence threshold
model.iou = 0.45  # IoU threshold for NMS

# Flask app
app = Flask(__name__)

# RTSP stream URL
RTSP_URL = "http://192.168.1.82:8080/video"

# Directory to save images
SAVE_DIR = 'saved_frames'
os.makedirs(SAVE_DIR, exist_ok=True)

# Global variable for toggle
cars_only_mode = False

def clean_text(text):
    # Remove all characters that are not letters or digits
    text = re.sub(r'[^a-zA-Z0-9]', '', text)
    # Remove duplicate characters while preserving order
    return ''.join(sorted(set(text), key=text.index))

def extract_text_from_image(image):
    # Convert image to PIL format
    pil_image = Image.fromarray(image)
    # Extract text using pytesseract
    text = pytesseract.image_to_string(pil_image, config='--psm 6')
    return clean_text(text.strip().lower())

def text_similar(text, target_text, threshold=70):
    # Use fuzzywuzzy to check if the text closely resembles the target text
    similarity = fuzz.partial_ratio(text, target_text)
    return similarity >= threshold

def is_text_reliable(text, target_text):
    # Calculate the proportion of detected text that matches the target text
    detected_letters = sum(1 for char in text if char in target_text)
    total_letters = len(target_text)
    return (detected_letters / total_letters) >= 0.5

def special_function(detected_text, frame):
    # Implement your custom logic here
    print(f"Special function called with detected text: {detected_text}")

    # Save the frame with a timestamped filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(SAVE_DIR, f'{timestamp}.jpg')
    cv2.imwrite(filename, frame)
    print(f"Saved frame as {filename}")

def get_frames():
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print("Error: Unable to open video stream")
        return None

    prev_time = 0
    interval = 0.06  # 0.06 seconds between each frame
    tracked_cars = {}

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to fetch frame from the stream")
                continue

            # Get current time and calculate if interval is respected
            current_time = time.time()
            if current_time - prev_time >= interval:
                prev_time = current_time

                # YOLOv5x inference with improved accuracy
                results = model(frame)

                # Filter detections based on class (optional)
                filtered_results = results.pred[0]  # Extract the predicted boxes
                valid_classes = [2]  # Only car class

                # Convert class IDs to integers and create a mask for filtering
                class_ids = filtered_results[:, 5].int()
                mask = torch.isin(class_ids, torch.tensor(valid_classes))

                # Apply mask to filter results
                filtered_results = filtered_results[mask]

                current_frame_cars = []

                for *box, conf, cls in filtered_results:
                    x1, y1, x2, y2 = map(int, box)
                    car_id = (x1, y1, x2, y2)

                    # Track still cars
                    if car_id not in tracked_cars:
                        tracked_cars[car_id] = time.time()
                        car_region = frame[y1:y2, x1:x2]
                        car_text = extract_text_from_image(car_region)

                        # Check for security or police text with at least 70% similarity
                        if (text_similar(car_text, 'handhaving') or text_similar(car_text, 'politie')) and \
                           is_text_reliable(car_text, 'handhaving') or is_text_reliable(car_text, 'politie'):
                            label = 'Security' if 'handhaving' in car_text else 'Police'
                            print(f"Detected {label} car with text: {car_text}")
                            special_function(car_text, frame)
                    else:
                        # If car is still detected
                        current_frame_cars.append(car_id)
                
                # Remove cars that are no longer detected
                tracked_cars = {car_id: timestamp for car_id, timestamp in tracked_cars.items() if car_id in current_frame_cars}

                # Render the filtered detections on the frame
                rendered_frame = results.render()[0]

                # Convert frame to JPEG
                ret, jpeg = cv2.imencode('.jpg', rendered_frame)
                if not ret:
                    continue

                # Yield the frame in byte format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

        except Exception as e:
            print(f"Exception occurred: {e}")
            continue

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(get_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_cars_only', methods=['POST'])
def toggle_cars_only():
    global cars_only_mode
    try:
        data = request.get_json()
        if 'cars_only' in data:
            cars_only_mode = data['cars_only']
            return jsonify({'cars_only_mode': cars_only_mode})
        else:
            return jsonify({'error': 'Invalid request'}), 400
    except Exception as e:
        print(f"Exception occurred: {e}")
        return jsonify({'error': 'Failed to process request'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
