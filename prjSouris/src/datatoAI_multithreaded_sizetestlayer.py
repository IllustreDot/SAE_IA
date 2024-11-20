# building AI from mouse translated to point coordinate dataset
# author: 37b7
# created: 20 Nov 2024

# Import =========================================================

# to use for next import
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
import os

#to extract data and visualy see the most fitting parmeters
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, ConfusionMatrixDisplay

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ================================================================

# Collect data and Standardization ===============================

# TODO! : collect data in ../rsc/(data folder name) 

# ================================================================

# function to use in the program =================================

def generate_layer_configurations(hl_nb_dict_of_dict):
    to_return_layers = []
    for num_layers in range(1, len(hl_nb_dict_of_dict) + 1):
        current_layer_values = [hl_nb_dict_of_dict[str(i+1)].values() for i in range(num_layers)]
        for config in product(*current_layer_values):
            if all(config[i] > config[i+1] for i in range(len(config) - 1)):
                to_return_layers.append(config)
    return to_return_layers

# ================================================================

# Variable init ==================================================

hl_nb_dict_of_dict = {
    "1": {"1": 784},
    "2": {"1": 784, "2": 584, "3": 284, "4": 84, "5": 10},
    "3": {"1": 784, "2": 584, "3": 284, "4": 84, "5": 10},
    "4": {"1": 784, "2": 584, "3": 284, "4": 84, "5": 10}
}

layer_configs = generate_layer_configurations(hl_nb_dict_of_dict)
# TODO : need more reflexion for the following code part with pytorch to get a correct tester

learning_rate_init_number = 0.001
alpha_number = 1e-4
random_state_number = 1
activation_function = 'relu'
max_iter_number = 100
solvertype = 'adam'

# ================================================================

# init of collected data file ====================================

output_file = "../rsc/output/analitics.csv"
if not os.path.exists(output_file):
    with open(output_file, "w") as f:
        f.write("accuracy,mse,conf_matrix,accuracies,losses,mses\n")
    print("Output file:", output_file, " created")
else:
    print("Output file:", output_file, " exists")

# ================================================================

# Model Creation and Data Collection =============================

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # TODO : need to be done in a way that adapt to the selected layer at the beginning of the cnn function call
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
    cnn_mses = []

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
        cnn_mses.append(mse_cnn)
        
        print(f'Epoch {epoch+1}/{max_iter_number}, CNN Test Accuracy: {accuracy_cnn:.2f}%, MSE: {mse_cnn:.4f}')

    conf_matrix_cnn = confusion_matrix(y_true, y_pred)

    result = {
        "accuracy": accuracy_cnn,
        "mse": mse_cnn,
        "conf_matrix": conf_matrix_cnn,
        "accuracies": cnn_accuracies,
        "losses": cnn_losses,
        "mses": cnn_mses
    }
    
    with open(output_file, "a") as f:
        f.write(f"CNN,{accuracy_cnn},{mse_cnn},\"{conf_matrix_cnn}\",\"{cnn_accuracies}\",\"{cnn_losses}\",\"{cnn_mses}\"\n")
    print(result)
    return result

# ================================================================

# use model mlp and cnn ==========================================

with ThreadPoolExecutor() as executor:
    # TODO : write the for loop to test with each layer configuration
    future_cnn = executor.submit(cnn)
    result_cnn = future_cnn.result()

# ================================================================

# load the data from the file ====================================

data = pd.read_csv(output_file)

# ================================================================

# Display results ================================================

# TODO : display the results from each result_cnn of each layer configuration in one graph per parameter (accuracy, mse, conf_matrix, accuracies, losses, mses)

# ================================================================

# display best configuration =====================================

# ================================================================