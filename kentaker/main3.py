import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# Load the classifier model
model = load_model('car_classifier.h5')

# Function to classify the car
def classify_car(car_image):
    img = cv2.resize(car_image, (224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    predictions = model.predict(img)
    label = np.argmax(predictions[0])
    labels = ['normal', 'security', 'police']
    
    return labels[label]

# Function to detect and classify cars in the frame
def detect_and_classify_cars(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cars = vehicle_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    for (x, y, w, h) in cars:
        car_image = image[y:y+h, x:x+w]
        car_type = classify_car(car_image)
        
        # Draw bounding box and label
        color = (0, 255, 0)
        label = f'Car: {car_type}'
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    return image

# Update the capture_and_process function to use the new detection and classification function
def capture_and_process():
    camera_url = 'http://192.168.1.82:8080/video'
    capture_interval = 0.01

    backSub = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=90, detectShadows=True
    )

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

        # Process each contour
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if cv2.contourArea(contour) > 500:
                moving_object_image = frame[y:y+h, x:x+w]
                if np.mean(moving_object_image) < 220:
                    # Detect and classify cars
                    frame_with_detections = detect_and_classify_cars(frame)

        # Display the frame with detections and classifications
        cv2.imshow('Detected Moving Objects', frame_with_detections)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(capture_interval)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    capture_and_process()
