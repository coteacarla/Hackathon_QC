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
import time # Import time to measure epoch duration


def hybrid_model_train():
    # Train Dataset
    # -------------

    # Set train shuffle seed (for reproducibility)
    manual_seed(42)

    batch_size = 64
    n_samples_train_per_digit = 1500  # Increased samples per digit

    # Use pre-defined torchvision function to load MNIST train data
    X_train = datasets.MNIST(
        root="./data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])
    )

    # Filter out labels (originally 0-9), leaving only labels 0 and 1
    idx_0 = np.where(X_train.targets == 0)[0][:n_samples_train_per_digit]
    idx_1 = np.where(X_train.targets == 1)[0][:n_samples_train_per_digit]
    idx = np.concatenate((idx_0, idx_1))

    X_train.data = X_train.data[idx]
    X_train.targets = X_train.targets[idx]

    # Define torch dataloader with filtered data
    train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True)
    return (X_train, train_loader)

def hybrid_model_show_training(train_loader):
    n_samples_show = 6

    data_iter = iter(train_loader)
    fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))

    # Calculate the actual number of available batches in the DataLoader
    num_batches_available = len(train_loader)

    # Limit n_samples_show to the actual number of batches available
    n_samples_to_plot = min(n_samples_show, num_batches_available)

    for i in range(n_samples_to_plot):
        images, targets = data_iter.__next__()

        axes[i].imshow(images[0, 0].numpy().squeeze(), cmap="gray")
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_title("Labeled: {}".format(targets[0].item()))
    plt.show()


def hybrid_model_test(batch_size=64, n_samples_test_per_digit=500):
    # Test Dataset
    # -------------

    # Use pre-defined torchvision function to load MNIST test data
    X_test = datasets.MNIST(
        root="./data", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()])
    )

    # Filter out labels (originally 0-9), leaving only labels 0 and 1
    idx_0 = np.where(X_test.targets == 0)[0][:n_samples_test_per_digit]
    idx_1 = np.where(X_test.targets == 1)[0][:n_samples_test_per_digit]
    idx = np.concatenate((idx_0, idx_1))
    
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
        self.fc3 = Linear(1, 2)  # 2-dimensional output from QNN (for 2 classes)

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
        return F.log_softmax(x, dim=1) # Use log_softmax for NLLLoss

def hybrid_model_actually_hybrid_train(model4, train_loader):
    # Define model, optimizer, and loss function
    optimizer = optim.Adam(model4.parameters(), lr=0.001)
    loss_func = NLLLoss()

    # Start training
    epochs = 20  # Set number of epochs
    loss_list = []  # Store loss history
    model4.train()  # Set model to training mode

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = []
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)  # Initialize gradient
            output = model4(data)  # Forward pass
            loss = loss_func(output, target)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize weights
            total_loss.append(loss.item())  # Store loss
        loss_list.append(sum(total_loss) / len(total_loss))
        end_time = time.time()
        epoch_duration = end_time - start_time
        print("Training [{:.0f}%]\tLoss: {:.4f}\tEpoch Time: {:.2f}s".format(
            100.0 * (epoch + 1) / epochs, loss_list[-1], epoch_duration))

    # Plot loss convergence
    plt.plot(loss_list)
    plt.title("Hybrid NN Training Convergence")
    plt.xlabel("Training Iterations")
    plt.ylabel("Neg. Log Likelihood Loss")
    plt.show()

def hybrid_model_evaluate_performance(model, test_loader, batch_size=64):
    loss_func = NLLLoss()
    total_loss = []
    model.eval()  # set model to evaluation mode
    with no_grad():
        correct = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            output = model(data)
            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            loss = loss_func(output, target)
            total_loss.append(loss.item())

        print(
            "Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%".format(
                sum(total_loss) / len(total_loss), correct / len(test_loader.dataset) * 100
            )
        )

def hybrid_model_plot_predicted_labels(model, test_loader, n_samples_show=6):
    count = 0
    fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))

    model.eval()
    with no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if count == n_samples_show:
                break
            output = model(data[0:1])
            
            pred = output.argmax(dim=1, keepdim=True)

            axes[count].imshow(data[0].numpy().squeeze(), cmap="gray")
            axes[count].set_xticks([])
            axes[count].set_yticks([])
            axes[count].set_title("Predicted {}".format(pred.item()))

            count += 1
    plt.show() # Display the plot


def predict_image_hybrid(image):
    image = image.convert('L')
    image = image.resize((28, 28))
    img_tensor = transforms.ToTensor()(image).unsqueeze(0)

    model_loaded.eval()
    with no_grad():
        output = model_loaded(img_tensor)
        pred = output.argmax(dim=1, keepdim=True).item()
    return pred


# Main execution flow
# (X_train, train_loader) = hybrid_model_train()
# hybrid_model_show_training(train_loader)
# (X_test, test_loader) = hybrid_model_test()

# qnn4 = create_qnn()
# model4 = Net(qnn4)
# hybrid_model_actually_hybrid_train(model4, train_loader)
# torch.save(model4.state_dict(), "model4.pt")

# qnn5 = create_qnn()
# model5 = Net(qnn5)
# model5.load_state_dict(torch.load("model4.pt"))
# hybrid_model_evaluate_performance(model5, test_loader)
# hybrid_model_plot_predicted_labels(model5, test_loader)

# Load model once at startup
device = torch.device("cpu")
qnn_loaded = create_qnn()
model_loaded = Net(qnn_loaded)
model_loaded.load_state_dict(torch.load("model4.pt", map_location=device))
model_loaded.eval()