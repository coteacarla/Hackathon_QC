from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from backend_cnn import predict_image_cnn
#from backend_hybrid import predict_image_hybrid
from backend_cnn import predict_image_cnn
from backend_qnn import predict_image_qnn
from backend_hybrid import predict_image_hybrid
import os

app = Flask(__name__)
CORS(app)

DIGITS_SAMPLE_FOLDER = 'digits_sample'
#make upload button in frontend to upload images for the predict() function

image = Image.open("digits_sample/drawing_0.png")

@app.route('/predict', methods=['POST'])
def predict():
    image = Image.open("digits_sample/drawing_0.png")
    #label = predict_image_hybrid(image)
    label = 0
    print(f"Predicted label: {label}")
    return jsonify({'label': label})

# can you add a route to upload images?
@app.route('/upload_and_predict_hybrid', methods=['POST'])
def upload_and_predict_hybrid():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filepath = os.path.join(DIGITS_SAMPLE_FOLDER, file.filename)
        file.save(filepath)
        image = Image.open(filepath)
        label = predict_image_hybrid(image)
        print(f"Predicted label hybrid : {label}")
        return jsonify({'label': label})

    return jsonify({'error': 'File upload failed'}), 500

# Add route to upload image and get prediction from backend_cnn.py
@app.route('/upload_and_predict_cnn', methods=['POST'])
def upload_and_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filepath = os.path.join(DIGITS_SAMPLE_FOLDER, file.filename)
        file.save(filepath)
        image = Image.open(filepath)
        label = predict_image_cnn(image)
        print(f"Predicted label classical: {label}")
        return jsonify({'label': label})

    return jsonify({'error': 'File upload failed'}), 500

@app.route('/upload_and_predict_qnn', methods=['POST'])
def upload_and_predict_qnn():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filepath = os.path.join(DIGITS_SAMPLE_FOLDER, file.filename)
        file.save(filepath)
        image = Image.open(filepath)
        label = predict_image_qnn(image)
        print(f"Predicted label quantum: {label}")
        return jsonify({'label': label})

    return jsonify({'error': 'File upload failed'}), 500

if __name__ == '__main__':
    app.run(debug=True)