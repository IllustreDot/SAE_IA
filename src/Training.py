import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import json
from model import ConvMLP

# Dataset Preparation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((360, 354)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])
# Dataset directory
dataset_dir = "./Dataset"

# Load dataset
dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)

# Mapping of class indices to class names
label_to_class = {idx: cls_name for idx, cls_name in enumerate(dataset.classes)}

# Find indices for each class
class_indices = {cls_name: [] for cls_name in dataset.classes}
for idx, (_, label) in enumerate(dataset.samples):
    class_name = label_to_class[label]
    class_indices[class_name].append(idx)

print(class_indices.keys())

# Determine the minimum class size
min_class_size = min(len(indices) for indices in class_indices.values())

# Limit each class to the minimum size
balanced_indices = []
for indices in class_indices.values():
    balanced_indices.extend(indices[:min_class_size])

# Create a subset dataset with balanced indices
balanced_dataset = Subset(dataset, balanced_indices)

# Shuffle the balanced dataset
np.random.shuffle(balanced_indices)  # Ensure randomness in splits

# Calculate train/test split sizes (80/20 split)
train_size = int(0.6 * len(balanced_dataset))  # 60% for training
test_size = len(balanced_dataset) - train_size  # 40% for testing

# Perform the split
train_dataset, test_dataset = random_split(balanced_dataset, [train_size, test_size])

# Print dataset statistics
print(f"Total dataset size: {len(dataset)}")
print(f"Minimum size for each class: {min_class_size}")
print(f"Total dataset size (balanced): {len(balanced_dataset)}")
print(f"Training dataset size: {len(train_dataset)}")
print(f"Testing dataset size: {len(test_dataset)}")

# Create DataLoader for training and testing with num_workers
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)


params = {
    'input_channels': 1,  # Grayscaled images
    'conv1_out_channels': 32,  # First convolution output channels
    'conv2_out_channels': 64,  # Second convolution output channels
    'conv_kernel_size': 3,  # 3x3 kernel
    'pool_kernel_size': 2,  # 2x2 pooling
    'l_i_size': 489984,  # Flattened size after conv and pooling
    'h_l1_size': 1024,  # Hidden layer 1 size
    'h_l2_size': 512,  # Hidden layer 2 size
    'h_l3_size': 256,  # Hidden layer 3 size
    'h_l4_size': 128,   # Hidden layer 4 size
    'l_o_size': 8  # Output classes
}

# Use Cuda if available
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    print("CUDA not available. Using CPU.")
    device = torch.device('cpu')

# Initialize the MLP model
model = ConvMLP(params).to(device)

# Set up loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Early Stopping Variables
best_loss = float('inf')
patience = 5  # Number of epochs to wait for improvement
counter = 0

# Scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Training Loop with Early Stopping
epochs = 15
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    log_interval = 100

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # Print progress
        if batch_idx % log_interval == 0:
            print(f'Epoch {epoch + 1}, Batch {batch_idx}: \n\t'
                  f'Loss: {running_loss / (batch_idx + 1):.4f}, \n\t'
                  f'Accuracy: {100.0 * correct / total:.2f}%')
    scheduler.step()

    # Validation Loss Check for Early Stopping
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    # Early stopping
    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0  # Reset the counter if loss improves
    else:
        counter += 1

    if counter >= patience:
        print("Early stopping due to no improvement in validation loss.")
        break

    # Print statistics
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.2f}%")

# Evaluation on Test Data
model.eval()
correct = 0
total = 0
with torch.no_grad():  # No gradients needed for evaluation
    for batch_idx, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Print final test accuracy
print(f"Test Accuracy: {100 * correct / total:.2f}%")

print("Do you want to save the model? (y/n)")
response = input()
if response == 'y':
    torch.save(model, 'model.pth')
    print("Model saved.")
else:
    print("Model not saved.")

with open("label_to_class.json", "w") as f:
    json.dump(label_to_class, f)