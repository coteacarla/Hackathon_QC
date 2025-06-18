import numpy as np
from matplotlib import pyplot as plt

import torch
from torch.autograd import Function
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, CrossEntropyLoss, MSELoss
from torch.optim import LBFGS
import torchvision
from torch import cat, no_grad, manual_seed
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
from torch.nn import (Module, Conv2d, Linear, Dropout2d, NLLLoss, MaxPool2d, Flatten, Sequential, ReLU,)

from qiskit import QuantumCircuit
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.circuit import Parameter
from flask import Flask, jsonify, send_file, request
import io
from PIL import Image


def hybrid_model_train():
    # Train Dataset
    # -------------

    # Set train shuffle seed (for reproducibility)
    manual_seed(42)

    batch_size = 1
    n_samples = 100  # We will concentrate on the first 100 samples

    # Use pre-defined torchvision function to load MNIST train data
    X_train = datasets.MNIST(
        root="./data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])
    )

    # Filter out labels (originally 0-9), leaving only labels 0 and 1
    idx = np.append(
        np.where(X_train.targets == 0)[0][:n_samples], np.where(X_train.targets == 1)[0][:n_samples]
    )
    X_train.data = X_train.data[idx]
    X_train.targets = X_train.targets[idx]

    # Define torch dataloader with filtered data
    train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True)
    return (X_train, train_loader)

def hybrid_model_show_training(train_loader):
    n_samples_show = 6

    data_iter = iter(train_loader)
    fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))

    while n_samples_show > 0:
        images, targets = data_iter.__next__()

        axes[n_samples_show - 1].imshow(images[0, 0].numpy().squeeze(), cmap="gray")
        axes[n_samples_show - 1].set_xticks([])
        axes[n_samples_show - 1].set_yticks([])
        axes[n_samples_show - 1].set_title("Labeled: {}".format(targets[0].item()))

        n_samples_show -= 1

def hybrid_model_test(batch_size=1, n_samples=50):
    # Test Dataset
    # -------------

    # Set test shuffle seed (for reproducibility)
    # manual_seed(5)

    # Use pre-defined torchvision function to load MNIST test data
    X_test = datasets.MNIST(
        root="./data", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()])
    )

    # Filter out labels (originally 0-9), leaving only labels 0 and 1
    idx = np.append(
        np.where(X_test.targets == 0)[0][:n_samples], np.where(X_test.targets == 1)[0][:n_samples]
    )
    X_test.data = X_test.data[idx]
    X_test.targets = X_test.targets[idx]

    # Define torch dataloader with filtered data
    test_loader = DataLoader(X_test, batch_size=batch_size, shuffle=True)
    return (X_test, test_loader)

# Define and create QNN
def create_qnn():
    feature_map = ZZFeatureMap(2)
    ansatz = RealAmplitudes(2, reps=1)
    qc = QuantumCircuit(2)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)

    # REMEMBER TO SET input_gradients=True FOR ENABLING HYBRID GRADIENT BACKPROP
    qnn = EstimatorQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        input_gradients=True,
    )
    return qnn

# Define torch NN module
class Net(Module):
    def __init__(self, qnn):
        super().__init__()
        self.conv1 = Conv2d(1, 2, kernel_size=5)
        self.conv2 = Conv2d(2, 16, kernel_size=5)
        self.dropout = Dropout2d()
        self.fc1 = Linear(256, 64)
        self.fc2 = Linear(64, 2)  # 2-dimensional input to QNN
        self.qnn = TorchConnector(qnn)  # Apply torch connector, weights chosen
        # uniformly at random from interval [-1,1].
        self.fc3 = Linear(1, 1)  # 1-dimensional output from QNN

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.qnn(x)  # apply QNN
        x = self.fc3(x)
        return cat((x, 1 - x), -1)

def hybrid_model_actually_hybrid_train(model, train_loader):
    # Define model, optimizer, and loss function
    optimizer = optim.Adam(model4.parameters(), lr=0.001)
    loss_func = NLLLoss()

    # Start training
    epochs = 20  # Set number of epochs
    loss_list = []  # Store loss history
    model4.train()  # Set model to training mode

    for epoch in range(epochs):
        total_loss = []
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)  # Initialize gradient
            output = model4(data)  # Forward pass
            loss = loss_func(output, target)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize weights
            total_loss.append(loss.item())  # Store loss
        loss_list.append(sum(total_loss) / len(total_loss))
        print("Training [{:.0f}%]\tLoss: {:.4f}".format(100.0 * (epoch + 1) / epochs, loss_list[-1]))

def hybrid_model_evaluate_performance(model,batch_size=1):
    
    loss_func = NLLLoss()
    total_loss = []
    model5.eval()  # set model to evaluation mode
    with no_grad():

        correct = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            output = model5(data)
            if len(output.shape) == 1:
                output = output.reshape(1, *output.shape)

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            loss = loss_func(output, target)
            total_loss.append(loss.item())

        print(
            "Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%".format(
                sum(total_loss) / len(total_loss), correct / len(test_loader) / batch_size * 100
            )
        )


(X_train,train_loader) = hybrid_model_train()
hybrid_model_show_training(train_loader)
(X_test, test_loader) = hybrid_model_test()

qnn4 = create_qnn()
model4 = Net(qnn4)
hybrid_model_actually_hybrid_train(model4, train_loader)
qnn5 = create_qnn()
model5 = Net(qnn5)
torch.save(model4.state_dict(), "model4.pt")
model5.load_state_dict(torch.load("model4.pt"))
hybrid_model_evaluate_performance(model5, batch_size=1)
#hybrid_model_plot_predicted_labels(model5)

# To run the Flask app, add:
# if __name__ == '__main__':
#     app.run(debug=True)
image = Image.open("drawing_1.png")
def predict_image(image):
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

predict_image(image)