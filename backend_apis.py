from PIL import Image
from flask import Flask, request, jsonify
from backend_hybrid import predict_image
import os

app = Flask(__name__)

DIGITS_SAMPLE_FOLDER = 'digits_sample'
#make upload button in frontend to upload images for the predict() function

image = Image.open("digits_sample/drawing_0.png")

@app.route('/predict', methods=['POST'])
def predict():
    image = Image.open("digits_sample/drawing_0.png")
    label = predict_image(image)
    print(f"Predicted label: {label}")
    return jsonify({'label': label})

if __name__ == '__main__':
    app.run(debug=True)