from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from backend_hybrid import predict_image
import os

app = Flask(__name__)
CORS(app)

DIGITS_SAMPLE_FOLDER = 'digits_sample'
#make upload button in frontend to upload images for the predict() function

image = Image.open("digits_sample/drawing_0.png")

@app.route('/predict', methods=['POST'])
def predict():
    image = Image.open("digits_sample/drawing_0.png")
    label = predict_image(image)
    print(f"Predicted label: {label}")
    return jsonify({'label': label})

# can you add a route to upload images?
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filepath = os.path.join(DIGITS_SAMPLE_FOLDER, file.filename)
        file.save(filepath)
        image = Image.open(filepath)
        label = predict_image(image)
        return jsonify({'label': label})

    return jsonify({'error': 'File upload failed'}), 500

if __name__ == '__main__':
    app.run(debug=True)