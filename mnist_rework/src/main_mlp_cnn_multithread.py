# building of a MLP model for the MNIST dataset
# author: 37b7
# created: 9 Nov 2024

# Import =========================================================

# to use for next import
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
from itertools import product
import seaborn as sns
import pandas as pd
import numpy as np
import os

#to extract data and visualy see the most fitting parmeters
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, ConfusionMatrixDisplay

# model used
from sklearn.neural_network import MLPClassifier

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ================================================================

# Collect data and Standardization ===============================

# import the py which extract the data of the rsc/data folder and return the data and the label
from mnist_reader import load_mnist
path_to_data = "../rsc/data"

X_train, y_train = load_mnist(path_to_data, kind='train')
X_test, y_test = load_mnist(path_to_data, kind='t10k')
X_train, X_test = X_train / 255.0, X_test / 255.0

# Convert data for PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).view(-1, 1, 28, 28)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).view(-1, 1, 28, 28)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=64)

# ================================================================

# Variable init for MLPClassifier parameter finder for MNIST ======

choosen_layers = (784, 284, 84, 10)
learning_rate_init_number = 0.001
alpha_number = 1e-4
random_state_number = 1
activation_function = 'relu'
max_iter_number = 100
solvertype = 'adam'

# ================================================================

# init of collected data file ====================================

output_file = "../rsc/output/collected_mlp_cnn.csv"
if not os.path.exists(output_file):
    with open(output_file, "w") as f:
        f.write("type,accuracy,mse,conf_matrix,accuracies,losses\n")
    print("Output file:", output_file, " created")
else:
    print("Output file:", output_file, " exists")

# ================================================================

# Model Creation and Data Collection =============================

def mlp():
    model = MLPClassifier(
        hidden_layer_sizes=choosen_layers,
        max_iter=max_iter_number,
        activation=activation_function,
        alpha=alpha_number,
        solver=solvertype,
        random_state=random_state_number,
        learning_rate_init=learning_rate_init_number
    )
    
    mlp_accuracies = []
    mlp_losses = []
    
    for epoch in range(max_iter_number):
        # for i in range(X_train.shape[0]):
        #     model.partial_fit(X_train[i].reshape(1,-1), np.array([y_train[i]]), classes=np.unique(y_train))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        probas = model.predict_proba(X_test)
        mse = log_loss(y_test, probas)
        mlp_accuracies.append(accuracy * 100)
        mlp_losses.append(mse)
        
        print(f'Epoch {epoch+1}/{max_iter_number}, MLP Test Accuracy: {accuracy * 100:.2f}%, MSE: {mse:.4f}')
        
    conf_matrix_mlp = confusion_matrix(y_test, y_pred)

    result = {
        "accuracy": accuracy,
        "mse": mse,
        "conf_matrix": conf_matrix_mlp,
        "accuracies": mlp_accuracies,
        "losses": mlp_losses
    }
    
    with open(output_file, "a") as f:
        f.write(f"MLP,{accuracy},{mse},\"{conf_matrix_mlp}\",\"{mlp_accuracies}\",\"{mlp_losses}\"\n")
    print(result)
    return result


def visualize_feature_maps(model, image, layer_names=["conv1", "conv2"]):
    """ Visualize the output feature maps of the specified layers for a given input image."""
    model.eval()
    
    with torch.no_grad():
        x = image
        for name, layer in model.named_children():
            x = layer(x)
            if name in layer_names:
                num_feature_maps = x.shape[1]
                grid_size = math.ceil(math.sqrt(num_feature_maps))
                fig, axs = plt.subplots(grid_size, grid_size, figsize=(15, 15))
                fig.suptitle(f"Feature Maps after {name}", fontsize=16)
                for i in range(grid_size * grid_size):
                    ax = axs[i // grid_size, i % grid_size]
                    if i < num_feature_maps:
                        ax.imshow(x[0, i].cpu().numpy(), cmap='viridis')
                    ax.axis('off')
                    
                plt.show()
                
    model.train()

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 284)
        self.fc2 = nn.Linear(284, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def cnn():
    cnn_model = CNNModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn_model.parameters(), lr=learning_rate_init_number)
    
    cnn_accuracies = []
    cnn_losses = []

    for epoch in range(max_iter_number):
        cnn_model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = cnn_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        cnn_losses.append(running_loss / len(train_loader))
        
        if epoch == 0:  # Visualize at the start of training
            sample_image, _ = next(iter(test_loader))
            visualize_feature_maps(cnn_model, sample_image[:1])  # Take one sample for visualization
        
        cnn_model.eval()
        correct, total = 0, 0
        mse_loss = 0.0
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = cnn_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                y_true.extend(labels.tolist())
                y_pred.extend(predicted.tolist())
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                probabilities = F.softmax(outputs, dim=1)
                labels_one_hot = F.one_hot(labels, num_classes=10).float()
                mse_loss += F.mse_loss(probabilities, labels_one_hot, reduction='sum').item()

        accuracy_cnn = 100 * correct / total
        cnn_accuracies.append(accuracy_cnn)
        mse_cnn = mse_loss / total
        
        print(f'Epoch {epoch+1}/{max_iter_number}, CNN Test Accuracy: {accuracy_cnn:.2f}%, MSE: {mse_cnn:.4f}')

    conf_matrix_cnn = confusion_matrix(y_true, y_pred)

    result = {
        "accuracy": accuracy_cnn,
        "mse": mse_cnn,
        "conf_matrix": conf_matrix_cnn,
        "accuracies": cnn_accuracies,
        "losses": cnn_losses
    }
    
    with open(output_file, "a") as f:
        f.write(f"CNN,{accuracy_cnn},{mse_cnn},\"{conf_matrix_cnn}\",\"{cnn_accuracies}\",\"{cnn_losses}\"\n")
    print(result)
    return result

# ================================================================

# use model mlp and cnn ==========================================

with ThreadPoolExecutor() as executor:
    future_mlp = executor.submit(mlp)
    future_cnn = executor.submit(cnn)

    result_mlp = future_mlp.result()
    result_cnn = future_cnn.result()

# ================================================================

# Display results ================================================

print(f"MLP Test Accuracy: {result_mlp['accuracy'] * 100:.2f}%")
print(f"CNN Test Accuracy: {result_cnn['accuracy']:.2f}%")

ConfusionMatrixDisplay(result_mlp["conf_matrix"]).plot()
plt.title("MLP Confusion Matrix")
plt.show()
ConfusionMatrixDisplay(result_cnn["conf_matrix"]).plot()
plt.title("CNN Confusion Matrix")
plt.show()

plt.figure(figsize=(14, 10))
# Accuracy Plot
plt.subplot(2, 2, 1)
plt.plot(result_mlp["accuracies"], label="MLP Accuracy", color='blue')
plt.plot(result_cnn["accuracies"], label="CNN Accuracy", color='orange')
plt.title("Accuracy Progression")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
# Loss Plot
plt.subplot(2, 2, 2)
plt.plot(result_mlp["losses"], label="MLP Loss", color='blue')
plt.plot(result_cnn["losses"], label="CNN Loss", color='orange')
plt.title("Loss Progression")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.bar(['MLP', 'CNN'], [result_mlp["accuracy"] * 100, result_cnn["accuracy"]], color=['blue', 'orange'])
plt.title("Comparison of MLP and CNN Accuracy on MNIST")
plt.ylabel("Accuracy (%)")
plt.show()

# ================================================================