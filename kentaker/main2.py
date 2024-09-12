from flask import Flask, request, jsonify
from PIL import Image
import io
import cv2
import numpy as np
import os
import uuid
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Path to Haar Cascade XML for vehicle detection
cascade_path = 'haarcascade_car.xml'  # Update this path

# Load Haar Cascade for vehicle detection
vehicle_cascade = cv2.CascadeClassifier(cascade_path)

# Verify if the cascade file is loaded
if vehicle_cascade.empty():
    raise Exception("Failed to load Haar Cascade file. Please check the file path.")

# Function to generate a random folder name
def generate_random_folder():
    return uuid.uuid4().hex

# Function to generate a random filename
def generate_random_filename(extension="png"):
    return f"{uuid.uuid4().hex}.{extension}"

# Function to detect vehicles and license plates, and cut them out
def extract_vehicles_and_license_plates(image, save_path):
    np_image = np.array(image)
    gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)

    # Detect vehicles with looser parameters
    vehicles = vehicle_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20), flags=cv2.CASCADE_SCALE_IMAGE)

    saved_images = {'vehicles': [], 'license_plates': []}

    # Create subfolders for cutouts
    vehicle_folder = os.path.join(save_path, 'vehicles')
    license_plate_folder = os.path.join(save_path, 'license_plates')
    os.makedirs(vehicle_folder, exist_ok=True)
    os.makedirs(license_plate_folder, exist_ok=True)

    # Create a copy of the image for license plate detection
    gray_image_for_plates = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
    _, thresh_for_plates = cv2.threshold(gray_image_for_plates, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh_for_plates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through detected vehicles
    for (x, y, w, h) in vehicles:
        # Extract the vehicle from the image
        vehicle_image = np_image[y:y+h, x:x+w]
        vehicle_image_pil = Image.fromarray(vehicle_image)

        # Save vehicle image
        vehicle_filename = generate_random_filename()
        vehicle_filepath = os.path.join(vehicle_folder, vehicle_filename)
        vehicle_image_pil.save(vehicle_filepath)
        saved_images['vehicles'].append(vehicle_filepath)

    # Loop through contours in the whole image to extract possible license plates
    for contour in contours:
        x_plate, y_plate, w_plate, h_plate = cv2.boundingRect(contour)
        aspect_ratio = float(w_plate) / h_plate

        # Check if the shape is rectangular with reasonable aspect ratio
        if 0.5 < aspect_ratio < 2.0 and w_plate > 50 and h_plate > 15:
            # Extract the license plate from the image
            license_plate_image = np_image[y_plate:y_plate+h_plate, x_plate:x_plate+w_plate]
            license_plate_image_pil = Image.fromarray(license_plate_image)

            # Save license plate image
            license_plate_filename = generate_random_filename()
            license_plate_filepath = os.path.join(license_plate_folder, license_plate_filename)
            license_plate_image_pil.save(license_plate_filepath)
            saved_images['license_plates'].append(license_plate_filepath)

    return saved_images

# Single endpoint to accept an image and return extracted vehicles and license plates as JSON
@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    
    # Create a unique folder for this request
    folder_name = generate_random_folder()
    save_path = os.path.join('cutouts', folder_name)
    os.makedirs(save_path)

    try:
        image = Image.open(io.BytesIO(image_file.read()))
        saved_images = extract_vehicles_and_license_plates(image, save_path)
        return jsonify({'saved_images': saved_images}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
