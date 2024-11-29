# building AI from mouse translated to point coordinate dataset
# author: 37b7
# created: 25 Nov 2024

# Import =========================================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.cuda.amp import autocast, GradScaler

import sys
sys.path.append("../rsc/config")
from allvariable import *

# ===============================================================

def create_output_file(path_to_output, file_name_output):
    if not os.path.exists(path_to_output + file_name_output):
        with open(path_to_output + file_name_output, "w") as f:
            f.write("behavior,layer,accuracy,mse,conf_matrix,accuracies,losses,mses\n")

class NNModel(nn.Module):
    def __init__(self, layer_config):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(layer_config[i], layer_config[i + 1]) for i in range(len(layer_config) - 1)])
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) if self.layers[-1] == layer else F.relu(layer(x))
            if self.layers[-1] != layer:
                x = self.dropout(x)
        return x

def nn_run(layer_config, behavior, train_loader, test_loader, device, learning_rate_init_number, alpha_number, max_iter_number, path_to_output, file_name_best_model):
    model = NNModel(layer_config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate_init_number, weight_decay=alpha_number)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=4, verbose=True)
    accuracies, losses, mses = [], [], []

    best_accuracy = 0
    epochs_without_improvement = 0
    early_stop_patience = 10

    for epoch in range(max_iter_number):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        losses.append(running_loss / len(train_loader))
        model.eval()
        correct, total, mse_loss, y_true, y_pred = 0, 0, 0.0, [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                y_true.extend(labels.tolist())
                y_pred.extend(predicted.tolist())
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                probabilities = F.softmax(outputs, dim=1)
                mse_loss += F.mse_loss(probabilities, F.one_hot(labels, num_classes=len(behavior)).float(), reduction='sum').item()
                
        epoch_accuracy = 100 * correct / total
        accuracies.append(epoch_accuracy)
        mses.append(mse_loss / total)

        scheduler.step(epoch_accuracy)

        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(f"Behavior: {behavior}, Layer: {layer_config}, Epoch: {epoch + 1}, Accuracy: {accuracies[-1]:.2f}%, Loss: {losses[-1]:.4f}, MSE: {mses[-1]:.4f}")
        if epochs_without_improvement >= early_stop_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs. Best accuracy: {best_accuracy:.2f}%")
            break
        

    create_output_file(path_to_output, file_name_data_output)
    with open(path_to_output + file_name_data_output, "a") as f:
        f.write(f"\"{behavior}\",\"{layer_config}\",{accuracies[-1]},{mses[-1]},\"{confusion_matrix(y_true, y_pred).tolist()}\",\"{accuracies}\",\"{losses}\",\"{mses}\"\n")
    return model