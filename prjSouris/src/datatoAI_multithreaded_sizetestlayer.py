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
import ast 
import os

#to extract data and visualy see the most fitting parmeters
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset



# ================================================================

# Variable config import =========================================

import sys
sys.path.append("../rsc/config")  # Relative path to the config directory
from allvariable import * 

# ================================================================

# validator behavior combinaison done ============================

def GetBehaviors(selector):
    data_classification_file = pd.read_csv(path_to_data_clean + selector + "/" + file_name_data_classification)
    data_classification = data_classification_file.columns.tolist()
    return data_classification

def checkBehavior(behaviorPairs):
    done_file = pd.read_csv(path_to_config + dile_name_config_done)
    done = []
    for item in done_file["behavior"]:
        if isinstance(item, str):
            parsed_item = ast.literal_eval(item)
            done.append(tuple(parsed_item))
        elif isinstance(item, list):
            done.append(tuple(item))
        else:
            done.append(item)
    for pair in behaviorPairs[:]:
        if pair in done:
            behaviorPairs.remove(pair)
            print(f"\"{pair[0]}\" and \"{pair[1]}\" already done")
    return behaviorPairs

def ValidateBehavior(selector):
    behaviors = GetBehaviors(selector)
    behaviorPairs = []
    for behavior in behaviors:
        if behavior == selector:
            continue
        behaviorPairs.append((selector, behavior))
    behaviorPairs = checkBehavior(behaviorPairs)
    return behaviorPairs

# ================================================================

# Collect data and Standardization ===============================

def LoadDataAll():
    data = {"data": None, "classification": None}
    file_sizes = []
    for cl in behavior:
        data_file = pd.read_csv(path_to_data_clean + cl + "/" + file_name_data)
        file_sizes.append(len(data_file))
    min_size = min(file_sizes)


    for cl in behavior:
        data_file = pd.read_csv(path_to_data_clean + cl + "/" + file_name_data)
        data_classification_file = pd.read_csv(path_to_data_clean + cl + "/" + file_name_data_classification)

        # TODO? : there is a huge flaw in the code below, the relation between the data and the classification is not kept when we sample the data (i think that why i put the random_state=13 but im not sure)
        data_file = data_file.sample(n=min_size, random_state=13)
        data_classification_file = data_classification_file.sample(n=min_size, random_state=13)

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
            if all(config[i] > config[i+1] for i in range(len(config) - 1)) and config[-1] == 2:
                to_return_layers.append(config)
    to_return_layers = [layer for layer in to_return_layers if len(layer) > 1]
    return to_return_layers

# ================================================================

# init of collected data file ====================================

if not os.path.exists(path_to_output + file_name_data_output):
    with open(path_to_output + file_name_data_output, "w") as f:
        f.write("behavior,layer,accuracy,mse,conf_matrix,accuracies,losses,mses\n")
    print("Output file:", path_to_output + file_name_data_output, " created")
else:
    print("Output file:", path_to_output + file_name_data_output, " exists")

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
    optimizer = optim.Adam(nn_model.parameters(), lr=learning_rate_init_number, weight_decay=alpha_number)

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

        accuracy_nn = 100 * correct / total
        nn_accuracies.append(accuracy_nn)
        mse_nn = mse_loss / total
        nn_mses.append(mse_nn)

        print(f'Epoch {epoch+1}/{max_iter_number}, NN Test Accuracy: {accuracy_nn:.2f}%, MSE: {mse_nn:.4f} of layer {layer_config}')

    conf_matrix_nn = confusion_matrix(y_true, y_pred)

    result = {
        "accuracy": accuracy_nn,
        "mse": mse_nn,
        "conf_matrix": conf_matrix_nn,
        "accuracies": nn_accuracies,
        "losses": nn_losses,
        "mses": nn_mses
    }

    with open(path_to_output + file_name_data_output, "a") as f:
        f.write(f"\"{behavior}\",\"{layer_config}\",{accuracy_nn},{mse_nn},\"{conf_matrix_nn.tolist()}\",\"{nn_accuracies}\",\"{nn_losses}\",\"{nn_mses}\"\n")
    print(result)
    return result

# ================================================================

# multithreading =================================================

def RunNNMultithreaded():
    results = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(nnRun, layer_config) for layer_config in layer_configs]
        for future in as_completed(futures):
            results.append(future.result())
    return results

# ================================================================

# Run the program ================================================

if __name__ == "__main__":
    print("Starting")
    layer_configs = generate_layer_configurations(hl_nb_dict_of_dict)
    behaviorPairs = ValidateBehavior(selector)
    for behavior in behaviorPairs:
        print(f"Starting {behavior[0]} and {behavior[1]}")
        train_loader, test_loader = LoadData()
        results = RunNNMultithreaded()
        with open(path_to_config + dile_name_config_done, "a") as f:
            f.write(f"\"{behavior}\"\n")


# ================================================================