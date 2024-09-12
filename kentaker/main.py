from flask import Flask, request, jsonify
from PIL import Image
import pytesseract
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Function to process the image using pytesseract (OCR)
def extract_text_from_image(image):
    string = pytesseract.image_to_string(image)
    return string

# Single endpoint to accept an image and return extracted text as JSON
@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    
    try:
        image = Image.open(io.BytesIO(image_file.read()))
        extracted_text = extract_text_from_image(image)
        return jsonify({'extracted_text': extracted_text}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
