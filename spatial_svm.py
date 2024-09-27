import torch
import torchvision
from sklearn import svm
from sklearn.metrics import accuracy_score
import time

start_time = time.time()
# Load the MNIST dataset using torchvision
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Prepare data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

# Extract features and labels
train_data, train_labels = next(iter(train_loader))
test_data, test_labels = next(iter(test_loader))

# Flatten the images
train_data = train_data.view(train_data.size(0), -1)
test_data = test_data.view(test_data.size(0), -1)

# Initialize the SVM classifier
svm_classifier = svm.SVC()

# Train the classifier
svm_classifier.fit(train_data.numpy(), train_labels.numpy())

# Predict on the test set
test_predictions = svm_classifier.predict(test_data.numpy())

# Calculate accuracy on the test set
test_accuracy = accuracy_score(test_labels.numpy(), test_predictions)
print("Test Accuracy:", test_accuracy)

end_time = time.time()  # Record the end time
iteration_time = end_time - start_time  # Calculate the time for this iteration
print("Epoch", epoch, iteration_time, "seconds")
