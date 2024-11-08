# building of a MLP model for the MNIST dataset
# author: 37b7
# created: 6 Nov 2024

# train or display
mode = "train" 

# Import =========================================================

# to use for next import
import matplotlib.pyplot as plt
from itertools import product
import seaborn as sns
import pandas as pd
import numpy as np
import os

#to extract data and visualy see the most fitting parmeters
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error

# model used
from sklearn.neural_network import MLPClassifier

# ================================================================

# Collect data and Standardization ===============================

# import the py which extract the data of the rsc/data folder and return the data and the label
from mnist_reader import load_mnist
path_to_data = "../rsc/data"

X_train, y_train = load_mnist(path_to_data, kind='train')
X_test, y_test = load_mnist(path_to_data, kind='t10k')
X_train, X_test = X_train / 255.0, X_test / 255.0

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

# Variable init for MLPRegressor parameter finder for MNIST ======

hl_nb_dict_of_dict = {
    "1": {"1": 784},
    "2": {"1": 784, "2": 584, "3": 284, "4": 84, "5": 10},
    "3": {"1": 784, "2": 584, "3": 284, "4": 84, "5": 10},
    "4": {"1": 784, "2": 584, "3": 284, "4": 84, "5": 10}
}

layer_configs = generate_layer_configurations(hl_nb_dict_of_dict)
learning_rate_init_number = 0.001
alpha_number = 1e-4
random_state_number = 1
activation_function = 'relu'
max_iter_number = 100
solvertype = 'sgd'

best_mse= float('inf')
best_config = {}

# ================================================================

# init of collected data file ====================================

output_file = "../rsc/output/collected_data_classification.csv"
# si le fichier n'existe pas, on le crée
if mode == "train":
    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            f.write("layers,neurons,mse\n")
        print("Output file:", output_file, " created")
    else:
        print("Output file:", output_file, " exists")

# ================================================================

# Model Creation and Data Collection =============================
if mode == "train":
    for layers in layer_configs:
        if len(layers) == 1:
            continue
        print(f"Layers: {layers}, Alpha: {alpha_number}, Learning Rate: {learning_rate_init_number}, Random State: {random_state_number}")
        model = MLPClassifier(
            hidden_layer_sizes=layers,
            max_iter=max_iter_number,
            activation=activation_function,
            alpha=alpha_number,
            solver=solvertype,
            random_state=random_state_number,
            learning_rate_init=learning_rate_init_number,
            verbose=True
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mse = mean_squared_error(y_test, model.predict_proba(X_test))
        
        # Save best configuration
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_config = {
                "layers": layers,
                "alpha": alpha_number,
                "learning_rate_init": learning_rate_init_number,
                "random_state": random_state_number,
                "accuracy": accuracy,
                "mse": mse
            }
        
        print(f"Layers: {layers} Accuracy: {accuracy:.4f} MSE: {mse:.4f}")
        with open(output_file, "a") as f:
            f.write(f"{len(layers)},{layers},{accuracy},{mse}\n")

# ================================================================

# Model with the Best Parameters Found ===========================
    best_model = MLPClassifier(
        hidden_layer_sizes=best_config["layers"],
        activation=activation_function,
        solver=solvertype,
        alpha=best_config["alpha"],
        learning_rate_init=best_config["learning_rate_init"],
        max_iter=max_iter_number,
        random_state=best_config["random_state"]
    )
    best_model.fit(X_train, y_train)
    y_best_pred = best_model.predict(X_test)

# ================================================================

# Read and Process Data ==========================================
if mode == "display" or mode == "process_best":
    data = pd.read_csv(output_file)
    data.columns = ["layers", "neurons", "accuracy", "mse"]
    data["neurons"] = data["neurons"].apply(eval)
    
if mode == "process_best":
    # Find the best configuration
    best_row = data.loc[data["mse"].idxmin()]
    best_layers = best_row["neurons"]
    print("start best_layers_number : ", best_layers)
    best_model = MLPClassifier(
        hidden_layer_sizes=best_layers,
        activation=activation_function,
        solver=solvertype,
        alpha=alpha_number,
        learning_rate_init=learning_rate_init_number,
        max_iter=max_iter_number,
        random_state=random_state_number,
        verbose=True
    )
    best_model.fit(X_train, y_train)
    y_best_pred = best_model.predict(X_test)

    # Separate data by the number of layers
    data_1_layer = data[data["layers"] == 1]
    data_2_layers = data[data["layers"] == 2]
    data_3_layers = data[data["layers"] == 3]

# ================================================================

# Plotting Accuracy and MSE for Different Layer Configurations ===
if mode == "display":
    plt.figure(figsize=(15, 5))

    # Plot for 1-layer configurations
    plt.subplot(1, 3, 1)
    plt.plot(data_1_layer.index, data_1_layer["accuracy"], marker='o', color='b', label="Accuracy")
    plt.plot(data_1_layer.index, data_1_layer["mse"], marker='x', color='r', label="MSE")
    plt.title("1-Layer Configurations")
    plt.xlabel("Configuration Index")
    plt.ylabel("Metrics")
    plt.legend()
    plt.grid(True)

    # Plot for 2-layer configurations
    plt.subplot(1, 3, 2)
    plt.plot(data_2_layers.index, data_2_layers["accuracy"], marker='o', color='g', label="Accuracy")
    plt.plot(data_2_layers.index, data_2_layers["mse"], marker='x', color='r', label="MSE")
    plt.title("2-Layer Configurations")
    plt.xlabel("Configuration Index")
    plt.ylabel("Metrics")
    plt.legend()
    plt.grid(True)

    # Plot for 3-layer configurations
    plt.subplot(1, 3, 3)
    plt.plot(data_3_layers.index, data_3_layers["accuracy"], marker='o', color='c', label="Accuracy")
    plt.plot(data_3_layers.index, data_3_layers["mse"], marker='x', color='r', label="MSE")
    plt.title("3-Layer Configurations")
    plt.xlabel("Configuration Index")
    plt.ylabel("Metrics")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# ================================================================

    # Comparison Plot of Accuracy and MSE Across Layer Counts
    plt.figure(figsize=(10, 6))
    plt.plot(data_1_layer.index, data_1_layer["accuracy"], marker='o', label="1 Layer Accuracy", color='b')
    plt.plot(data_2_layers.index, data_2_layers["accuracy"], marker='o', label="2 Layers Accuracy", color='g')
    plt.plot(data_3_layers.index, data_3_layers["accuracy"], marker='o', label="3 Layers Accuracy", color='c')
    plt.plot(data_1_layer.index, data_1_layer["mse"], marker='x', label="1 Layer MSE", color='r')
    plt.plot(data_2_layers.index, data_2_layers["mse"], marker='x', label="2 Layers MSE", color='orange')
    plt.plot(data_3_layers.index, data_3_layers["mse"], marker='x', label="3 Layers MSE", color='purple')
    plt.title("Accuracy and MSE Comparison for Different Layer Counts")
    plt.xlabel("Configuration Index")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Confusion Matrix for Best Model
    conf_matrix = confusion_matrix(y_test, y_best_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# ================================================================
