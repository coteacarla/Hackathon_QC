import torch as T
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

device = T.device("cpu")
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts to [0, 1]
])

full_train_ds = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
full_test_ds = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)


train_targets = np.array(full_train_ds.targets.tolist())
selected_train_indices = []
for digit in range(10):
    indices = np.where(train_targets == digit)[0]
    np.random.shuffle(indices)
    selected_train_indices.extend(indices[:500])

train_ds = T.utils.data.Subset(full_train_ds, selected_train_indices)

test_targets = np.array(full_test_ds.targets.tolist())
selected_test_indices = []
for digit in range(10):
    indices = np.where(test_targets == digit)[0]
    np.random.shuffle(indices)
    selected_test_indices.extend(indices[:20])

test_ds = T.utils.data.Subset(full_test_ds, selected_test_indices)

train_ldr = T.utils.data.DataLoader(train_ds, batch_size=10, shuffle=True)
class Net(T.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = T.nn.Conv2d(1, 32, 5)
        self.conv2 = T.nn.Conv2d(32, 64, 5)
        self.pool1 = T.nn.MaxPool2d(2, stride=2)
        self.pool2 = T.nn.MaxPool2d(2, stride=2)
        self.drop1 = T.nn.Dropout(0.25)
        self.drop2 = T.nn.Dropout(0.50)
        self.fc1 = T.nn.Linear(1024, 512)
        self.fc2 = T.nn.Linear(512, 256)
        self.fc3 = T.nn.Linear(256, 10)  
    def forward(self, x):
        z = T.relu(self.conv1(x))
        z = self.pool1(z)
        z = self.drop1(z)
        z = T.relu(self.conv2(z))
        z = self.pool2(z)
        z = z.view(-1, 1024)
        z = T.relu(self.fc1(z))
        z = self.drop2(z)
        z = T.relu(self.fc2(z))
        z = self.fc3(z)
        return z
def accuracy(model, ds):
    ldr = T.utils.data.DataLoader(ds, batch_size=len(ds), shuffle=False)
    n_correct = 0
    for data in ldr:
        (pixels, labels) = data
        with T.no_grad():
            oupts = model(pixels)
        (_, predicteds) = T.max(oupts, 1)
        n_correct += (predicteds == labels).sum().item()
    return n_correct / len(ds)

loss_history = []
net = Net().to(device)
loss_func = T.nn.CrossEntropyLoss()
optimizer = T.optim.SGD(net.parameters(), lr=0.02)
max_epochs = 80

#net.train()
# for epoch in range(max_epochs):
#     ep_loss = 0
#     for (X, y) in train_ldr:
#         optimizer.zero_grad()
#         output = net(X)
#         loss_val = loss_func(output, y)
#         ep_loss += loss_val.item()
#         loss_val.backward()
#         optimizer.step()
#     loss_history.append(ep_loss)
#     if epoch % 5 == 0:
#         print(f"Epoch {epoch} | Loss: {ep_loss:.4f}")

#import matplotlib.pyplot as plt
#plt.figure(figsize=(10, 5))
#plt.plot(range(max_epochs), loss_history, marker='o')
#plt.title('Training Loss Convergence')
#plt.xlabel('Epoch')
#plt.ylabel('Loss')
#plt.grid(True)
#plt.show()

#net.eval()
#train_acc = accuracy(net, train_ds)
#print(f"Accuracy on training set: {train_acc:.4f}")

#test_acc = accuracy(net, test_ds)
#print(f"Accuracy on test set: {test_acc:.4f}")

#T.save(net.state_dict(), "./mnist_model.pt")
#print("Model saved.")

#Load model once at startup
net = Net().to(device)
net.load_state_dict(T.load("./mnist_model.pt", map_location=device))
#net.eval()

def predict_image_cnn(image):
    image = image.convert("L").resize((28, 28))
    x = np.array(image, dtype=np.float32) / 255.0
    x = T.tensor(x, dtype=T.float32).to(device)
    x = x.view(1, 1, 28, 28)
    with T.no_grad():
        outputs = net(x)
        _, predicted = T.max(outputs, 1)
    return int(predicted.item())
