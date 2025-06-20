from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
import numpy as np

classifier = NeuralNetworkClassifier.load("real_qcnn_classifier.model")

def predict_image_qnn(image):
    x_test = np.asarray(image).reshape((-1, 8, 8))
    x_test = x_test.mean(axis=2)  # shape: (n_samples, 8)

    y_predict = classifier.predict(x_test)

    if int(y_predict[0][0]) == 0:
        return 0
    elif int(y_predict[0][0]) == 1:
        return 1
    else:
        return "Unknown"