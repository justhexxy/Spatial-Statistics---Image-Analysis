import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import time

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Hyperparameters
input_size = 28 * 28  # Size of input images
hidden_size = 128  # Number of units in the hidden layer
num_classes = 10  # Number of output classes (digits 0-9)
learning_rate = 0.001
batch_size = 64
num_epochs = 5  # Maximum number of epochs to train


start_time = time.time()
# Load the MNIST dataset using torchvision
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Prepare data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

# Initialize the model, loss function, and optimizer
model = MLP(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Reshape images to (batch_size, input_size)
        images = images.reshape(-1, 28*28)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Compute accuracy on the training set
    with torch.no_grad():
        total = 0
        correct = 0
        for images, labels in train_loader:
            images = images.reshape(-1, 28*28)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Accuracy: {accuracy:.4f}")


    # Evaluation loop with printing indices of wrongly classified images
wrong_indices = []
with torch.no_grad():
    correct = 0
    total = 0
    for idx, (images, labels) in enumerate(test_loader):
        images = images.reshape(-1, 28*28)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Find indices of wrongly classified images
        wrong_indices.extend((predicted != labels).nonzero().squeeze().tolist())

    test_accuracy = correct / total
    print(f"Test Accuracy: {test_accuracy:.4f}")

# Record end time
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time
print("Elapsed time:", elapsed_time, "seconds")

# Print indices of wrongly classified images
#print("Indices of wrongly classified images:", wrong_indices)
