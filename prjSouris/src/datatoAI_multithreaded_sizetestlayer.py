# building AI from mouse translated to point coordinate dataset
# author: 37b7
# created: 20 Nov 2024

# Import =========================================================

# to use for next import
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
from itertools import product
import pandas as pd
import numpy as np
import os

#to extract data and visualy see the most fitting parmeters
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ================================================================

# Variable config ================================================

basedirpath = "../rsc/"
path_to_data = basedirpath + "data_cleaned/"
data_files_name = "data.csv"
data_classification_files_name = "data_classification.csv"
output_path = basedirpath + "output/"
output_file = output_path + "analitics.csv"

# kind is the list of comportement of the mouse we will train the AI on
behavior = ["scratching","body grooming"]

# ================================================================

# Collect data and Standardization ===============================

def LoadDataAll():
    data = {"data": None, "classification": None}
    for cl in behavior:
        data_file = pd.read_csv(path_to_data + cl + "/" + data_files_name)
        data_classification_file = pd.read_csv(path_to_data + cl + "/" + data_classification_files_name)
        
        columns_to_keep = [col for col in data_classification_file.columns if col in behavior]
        data_classification_file = data_classification_file[columns_to_keep]
        
        if data["data"] is None:
            data["data"] = data_file
            data["classification"] = data_classification_file
        else:
            data["data"] = pd.concat([data["data"], data_file], ignore_index=True)
            data["classification"] = pd.concat([data["classification"], data_classification_file], ignore_index=True)
    
    return data

def PrepareData(data):
    features = torch.tensor(data["data"].values, dtype=torch.float32)
    one_hot_labels = data["classification"].values
    labels = np.argmax(one_hot_labels, axis=1)
    labels = torch.tensor(labels, dtype=torch.long)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader

def LoadData():
    raw_data = LoadDataAll()
    train_loader, test_loader = PrepareData(raw_data)
    return train_loader, test_loader

# ================================================================

# function to use in the program =================================

def generate_layer_configurations(hl_nb_dict_of_dict):
    to_return_layers = []
    for num_layers in range(1, len(hl_nb_dict_of_dict) + 1):
        current_layer_values = [hl_nb_dict_of_dict[str(i+1)].values() for i in range(num_layers)]
        for config in product(*current_layer_values):
            if all(config[i] > config[i+1] for i in range(len(config) - 1)):
                to_return_layers.append(config)
    to_return_layers = [layer for layer in to_return_layers if len(layer) > 1]
    return to_return_layers

# ================================================================

# Variable init programm =========================================

hl_nb_dict_of_dict = {
    "1": {"1": 19},
    "2": {"1": 15, "2": 12, "3": 9, "4": 6, "5": 3, "6": 2},
    "3": {"1": 15, "2": 12, "3": 9, "4": 6, "5": 3, "6": 2},
    "4": {"1": 15, "2": 12, "3": 9, "4": 6, "5": 3, "6": 2},
    "5": {"1": 15, "2": 12, "3": 9, "4": 6, "5": 3, "6": 2},
    "6": {"1": 15, "2": 12, "3": 9, "4": 6, "5": 3, "6": 2},
}

layer_configs = generate_layer_configurations(hl_nb_dict_of_dict)

learning_rate_init_number = 0.001
alpha_number = 1e-4
random_state_number = 1
max_iter_number = 100

# ================================================================

# init of collected data file ====================================

if not os.path.exists(output_file):
    with open(output_file, "w") as f:
        f.write("accuracy,mse,conf_matrix,accuracies,losses,mses\n")
    print("Output file:", output_file, " created")
else:
    print("Output file:", output_file, " exists")

# ================================================================

# Model Creation and Data Collection =============================

class NNModel(nn.Module):
    def __init__(self, layer_config):
        super(NNModel, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_config) - 1):
            self.layers.append(nn.Linear(layer_config[i], layer_config[i + 1]))
        if len(self.layers) == 0:
            print("No layers were added! Check the layer_config." , str(layer_config) , layer_config)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        for layer in self.layers:
            if self.layers[-1] == layer:
                x = layer(x)
            else:
                x = F.relu(layer(x))
                x = self.dropout(x)
        return x

def nnRun(layer_config):
    nn_model = NNModel(layer_config)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(nn_model.parameters(), lr=learning_rate_init_number)
    
    nn_accuracies, nn_losses, nn_mses = [], [], []

    for epoch in range(max_iter_number):
        nn_model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = nn_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        nn_losses.append(running_loss / len(train_loader))
        
        nn_model.eval()
        correct, total, mse_loss = 0, 0, 0.0
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = nn_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                y_true.extend(labels.tolist())
                y_pred.extend(predicted.tolist())
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                probabilities = F.softmax(outputs, dim=1)
                labels_one_hot = F.one_hot(labels, num_classes=len(behavior)).float()
                mse_loss += F.mse_loss(probabilities, labels_one_hot, reduction='sum').item()

        accuracy_cnn = 100 * correct / total
        nn_accuracies.append(accuracy_cnn)
        mse_cnn = mse_loss / total
        nn_mses.append(mse_cnn)
        
        print(f'Epoch {epoch+1}/{max_iter_number}, CNN Test Accuracy: {accuracy_cnn:.2f}%, MSE: {mse_cnn:.4f} of layer {layer_config}')

    conf_matrix_cnn = confusion_matrix(y_true, y_pred)

    result = {
        "accuracy": accuracy_cnn,
        "mse": mse_cnn,
        "conf_matrix": conf_matrix_cnn,
        "accuracies": nn_accuracies,
        "losses": nn_losses,
        "mses": nn_mses
    }
    
    with open(output_file, "a") as f:
        f.write(f"{accuracy_cnn},{mse_cnn},\"{conf_matrix_cnn}\",\"{nn_accuracies}\",\"{nn_losses}\",\"{nn_mses}\"\n")
    print(result)
    return result

# ================================================================

# use model mlp and cnn ==========================================

def RunNNMultithreaded():
    results = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(nnRun, layer_config) for layer_config in layer_configs]
        for future in as_completed(futures):
            results.append(future.result())
    return results

# ================================================================

# load the data from the file ====================================

def LoadDataResultat():
    data = pd.read_csv(output_file)
    return data

# ================================================================

# Display results ================================================

def DisplayResults(data = None):
    if data == None:
        data = LoadDataResultat()

    # Plot Accuracy, Loss, and MSE Progression for all configurations
    plt.figure(figsize=(18, 12))

    # Accuracy Plot
    plt.subplot(3, 1, 1)
    for index, row in data.iterrows():
        accuracies = eval(row["accuracies"])
        plt.plot(accuracies, label=f"Config {index + 1}: {row['accuracy']:.2f}%")
    plt.title("Accuracy Progression Across Configurations")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    # Loss Plot
    plt.subplot(3, 1, 2)
    for index, row in data.iterrows():
        losses = eval(row["losses"])
        plt.plot(losses, label=f"Config {index + 1}")
    plt.title("Loss Progression Across Configurations")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # MSE Plot
    plt.subplot(3, 1, 3)
    for index, row in data.iterrows():
        mses = eval(row["mses"])
        plt.plot(mses, label=f"Config {index + 1}")
    plt.title("MSE Progression Across Configurations")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Display Confusion Matrices
    for index, row in data.iterrows():
        conf_matrix = eval(row["conf_matrix"])
        ConfusionMatrixDisplay(conf_matrix).plot()
        plt.title(f"Confusion Matrix for Config {index + 1}")
        plt.show()

    # Bar plot comparison of final accuracies
    plt.figure(figsize=(10, 6))
    configurations = [f"Config {i + 1}" for i in range(len(data))]
    accuracies = data["accuracy"].tolist()
    plt.bar(configurations, accuracies, color='skyblue')
    plt.title("Comparison of Final Accuracies Across Configurations")
    plt.ylabel("Accuracy (%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# ================================================================

# display best configuration =====================================

    best_index = data["accuracy"].idxmax()
    best_row = data.iloc[best_index]
    print(f"Best Configuration: Config {best_index + 1}")
    print(f"  Accuracy: {best_row['accuracy']:.2f}%")
    print(f"  Final Loss: {eval(best_row['losses'])[-1]:.4f}")
    print(f"  Final MSE: {eval(best_row['mses'])[-1]:.4f}")

    ConfusionMatrixDisplay(eval(best_row["conf_matrix"])).plot()
    plt.title(f"Confusion Matrix for Best Config {best_index + 1}")
    plt.show()

# ================================================================

# Run the program ================================================

if __name__ == "__main__":
    print("Starting")
    train_loader, test_loader = LoadData()
    results = RunNNMultithreaded()
    DisplayResults(results)

# ================================================================