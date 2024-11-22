import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class ConvMLP(nn.Module):
    def __init__(self, params):
        '''
        Args:
            params: dictionary with the following
                input_channels: number of input channels for convolutional layer (int)
                conv_out_channels: number of output channels for convolutional layer (int)
                conv_kernel_size: kernel size for convolutional layer (int or tuple)
                pool_kernel_size: kernel size for pooling layer (int or tuple)
                l_i_size: input size for first fully connected layer after flattening (int)
                h_lx_size: hidden layer x size (int)
                l_o_size: output layer size (int)
        '''
        super(ConvMLP, self).__init__()
        
        # Convolutional Layers with Batch Normalization and LeakyReLU
        self.conv1 = nn.Conv2d(in_channels=params['input_channels'], 
                               out_channels=params['conv1_out_channels'], 
                               kernel_size=params['conv_kernel_size'])
        self.bn1 = nn.BatchNorm2d(params['conv1_out_channels'])
        
        self.conv2 = nn.Conv2d(in_channels=params['conv1_out_channels'], 
                               out_channels=params['conv2_out_channels'], 
                               kernel_size=params['conv_kernel_size'])
        self.bn2 = nn.BatchNorm2d(params['conv2_out_channels'])
        
        self.pool = nn.MaxPool2d(kernel_size=params['pool_kernel_size'])
        
        # Fully Connected Layers (MLP Part)
        self.fc1 = nn.Linear(params['l_i_size'], params['h_l1_size'])
        self.fc2 = nn.Linear(params['h_l1_size'], params['h_l2_size'])
        self.fc3 = nn.Linear(params['h_l2_size'], params['h_l3_size'])
        self.fc4 = nn.Linear(params['h_l3_size'], params['h_l4_size'])
        self.fc5 = nn.Linear(params['h_l4_size'], params['l_o_size'])
        
        # Output activation
        self.softmax = nn.Softmax(dim=1)
        
        # Initialize weights
        self.apply(initialization)

    def forward(self, x):
        '''
        Args:
            x: input tensor (e.g., a batch of images)
        '''
        # Pass through convolutional layers with LeakyReLU and BatchNorm
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.1)(x)  # LeakyReLU with negative slope 0.1
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.1)(x)
        x = self.pool(x)
        
        # Flatten the tensor for the fully connected layers
        x = torch.flatten(x, start_dim=1)
        
        # Pass through fully connected layers
        x = self.fc1(x)
        x = nn.LeakyReLU(0.1)(x)
        x = self.fc2(x)
        x = nn.LeakyReLU(0.1)(x)
        x = self.fc3(x)
        x = nn.LeakyReLU(0.1)(x)
        x = self.fc4(x)
        x = nn.LeakyReLU(0.1)(x)
        x = self.fc5(x)
        
        # Apply softmax to get probabilities
        x = self.softmax(x)
        return x

def initialization(model):
    '''
    Args:
        model: model to be initialized
    '''
    if isinstance(model, nn.Linear):
        nn.init.kaiming_normal_(model.weight)
        nn.init.zeros_(model.bias)
    if isinstance(model, nn.Conv2d):
        nn.init.kaiming_normal_(model.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(model.bias)

# Dataset Preparation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.Resize((360, 354)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

dataset_dir = "./Dataset/data_2classes_Mice"

dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)

train_size = int(0.8 * len(dataset))  # 80% for training
test_size = len(dataset) - train_size  # 20% for testing

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoader for training and testing with num_workers
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, pin_memory=True)

params = {
    'input_channels': 3,  # RGB images
    'conv1_out_channels': 32,  # First convolution output channels
    'conv2_out_channels': 64,  # Second convolution output channels
    'conv_kernel_size': 3,  # 3x3 kernel
    'pool_kernel_size': 2,  # 2x2 pooling
    'l_i_size': 489984,  # Flattened size after conv and pooling
    'h_l1_size': 512,  # Hidden layer 1 size
    'h_l2_size': 256,  # Hidden layer 2 size
    'h_l3_size': 128,  # Hidden layer 3 size
    'h_l4_size': 64,   # Hidden layer 4 size
    'l_o_size': 2  # Output classes
}

# Initialize the MLP model
model = ConvMLP(params)

# Set up loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Early Stopping Variables
best_loss = float('inf')
patience = 5  # Number of epochs to wait for improvement
counter = 0

# Training Loop with Early Stopping
epochs = 2
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx + 1}/{len(train_loader)}")  # Debug batch count
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

    # Validation Loss Check for Early Stopping
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

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
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
