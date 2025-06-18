from flask import Flask, request, jsonify
from backend_hybrid import predict_image
import os

app = Flask(__name__)

DIGITS_SAMPLE_FOLDER = 'digits_sample'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'filename' not in data:
        return jsonify({'error': 'No filename provided'}), 400

    filename = data['filename']
    image_path = os.path.join(DIGITS_SAMPLE_FOLDER, filename)

    if not os.path.exists(image_path):
        return jsonify({'error': 'File not found'}), 404

    with open(image_path, 'rb') as image_file:
        label = predict_image(image_file)
    return jsonify({'label': label})

if __name__ == '__main__':
    app.run(debug=True)