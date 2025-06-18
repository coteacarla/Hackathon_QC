




from flask import Flask, request, jsonify
import backend_hybrid 

from backend_hybrid import model5
from PIL import Image
from torchvision import transforms
from torch import no_grad

app = Flask(__name__)
image = Image.open("drawing_1.png")

@app.route('/predict_image', methods=['POST'])
def predict_image(image):
    # Check if an image file was sent
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read image from request and preprocess for model
    img = image.convert('L')
    img = img.resize((28, 28))
    img_tensor = transforms.ToTensor()(img).unsqueeze(0)

    model5.eval()
    with no_grad():
        output = model5(img_tensor)
        if len(output.shape) == 1:
            output = output.reshape(1, *output.shape)
        pred = output.argmax(dim=1, keepdim=True).item()
    print(f"Predicted label: {pred}")
    return jsonify({'predicted_label': int(pred)})
# Add more endpoints for other functions as needed

if __name__ == '__main__':
    app.run(debug=True)