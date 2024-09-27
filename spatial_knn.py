import matplotlib.pyplot as plt
import torch
import torchvision
from collections import Counter
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

def knn(train_data, train_labels, test_data, k=3):
    distances = torch.cdist(test_data, train_data)  # Calculate pairwise distances between test and train data
    _, indices = torch.topk(distances, k, largest=False)  # Find k nearest neighbors for each test sample
    knn_labels = train_labels[indices]  # Get labels of k nearest neighbors
    knn_predictions = torch.mode(knn_labels, dim=1).values  # Majority voting to predict labels
    return knn_predictions



train_predictions = knn(train_data, train_labels, train_data)

# Calculate accuracy on the training set
train_accuracy = torch.sum(train_predictions == train_labels).item() / len(train_labels)
print("Training Accuracy:", train_accuracy)


# Predict on the test set
test_predictions = knn(train_data, train_labels, test_data)

# Calculate accuracy on the test set
test_accuracy = torch.sum(test_predictions == test_labels).item() / len(test_labels)
print("Test Accuracy:", test_accuracy)

# Record end time
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time
print("Elapsed time:", elapsed_time, "seconds")

# Find indices of incorrectly classified digits
#incorrect_indices = (test_predictions != test_labels).nonzero().squeeze()

# Select three random incorrect indices
#sample_indices = torch.randint(0, len(incorrect_indices), (3,))


# Reverse the normalization process
# reverse_normalize = torchvision.transforms.Compose([
#     torchvision.transforms.Normalize(mean=(-0.5 / 0.5), std=(1.0 / 0.5)),
#     torchvision.transforms.ToPILImage()
# ])

# Plot the images of incorrectly classified digits
# for i, idx in enumerate(sample_indices):
#     plt.subplot(1, 3, i + 1)
#     # Reverse normalization and convert to PIL image
#     img = reverse_normalize(test_data[incorrect_indices[idx]].view(1, 28, 28))
#     plt.imshow(img, cmap='gray')
#     plt.title(f'Predicted: {test_predictions[incorrect_indices[idx]]}, Actual: {test_labels[incorrect_indices[idx]]}')
#     plt.axis('off')
#
# plt.show()
