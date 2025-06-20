from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
import numpy as np
from skimage.transform import resize
from skimage.filters import threshold_otsu
from skimage.exposure import rescale_intensity

classifier = NeuralNetworkClassifier.load("real_qcnn_classifier.model")

def preprocess_image(image):
    image = np.array(image).astype(np.float32)
    image = rescale_intensity(image, in_range=(0, 255), out_range=(0, 1))
    resized = resize(image, (8, 8), order=3, anti_aliasing=True, mode='reflect')
    thresh = threshold_otsu(resized)
    binary = (resized > thresh).astype(int)
    return binary * (np.pi / 2)

def predict_image_qnn(image):
    image = image.convert("L").resize((28, 28))

    image = preprocess_image(image)
    
    x_test = np.asarray(image).reshape((-1, 8, 8))
    x_test = x_test.mean(axis=2)  # shape: (n_samples, 8)

    y_predict = classifier.predict(x_test)

    if int(y_predict[0][0]) == 0:
        return 0
    elif int(y_predict[0][0]) == 1:
        return 1
    else:
        return "Unknown"