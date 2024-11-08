# building of a MLP model for the MNIST dataset
# author: 37b7
# created: 6 Nov 2024

# train or display
mode = "display" 

# Import =================================

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
from sklearn.neural_network import MLPRegressor

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

# Variable init for MLPRegressor parameter finder for MNIST =====

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

best_mse= 0
best_config = {}

# ================================================================

# init of collected data file ====================================

output_file = "../rsc/output/collected_data.csv"
# si le fichier n'existe pas, on le crée
if mode == "train":
    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            f.write("layers,neurons,mse\n")
        print("Output file:", output_file, " created")
    else:
        print("Output file:", output_file, " exists")

# ================================================================

# Model creation and collect of data =============================
if mode == "train":
    for layers in layer_configs:
        if len(layers) == 1:
            continue
        print(f"Layers: {layers}, Alpha: {alpha_number}, Learning Rate: {learning_rate_init_number}, Random State: {random_state_number}")
        model = MLPRegressor(hidden_layer_sizes=layers,
                            max_iter=max_iter_number,
                            activation=activation_function,
                            alpha=alpha_number,
                            solver=solvertype,
                            random_state=random_state_number,
                            learning_rate_init=learning_rate_init_number,
                            verbose=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        # Save best configuration
        if mse > best_mse:
            best_mse = mse
            best_config = {
                "layers": layers,
                "alpha": alpha_number,
                "learning_rate_init": learning_rate_init_number,
                "random_state": random_state_number,
                "mse": mse
            }
        
        print(f"Layers: {layers} MSE: {mse:.4f}")
        with open(output_file, "a") as f:
            f.write(f"{len(layers)},\"{layers}\",{mse}\n")

# ================================================================

# Read and Process Data ==========================================
if mode == "display" or mode == "process_best":
    # Load data into a DataFrame
    data = pd.read_csv(output_file)
    data.columns = ["layers", "neurons", "mse"]
    data["neurons"] = data["neurons"].apply(eval)
    
if mode == "process_best":
    # Find the best configuration
    best_row = data.loc[data["mse"].idxmin()]
    best_layers = best_row["neurons"]
    print("start best_layers_number : ", best_layers)
    best_model = MLPRegressor(
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

# ================================================================

# Plotting MSE for Different Layer Configurations ================
if mode == "display" :
    # Separate data by the number of layers
    data_1_layer = data[data["layers"] == 2]
    data_2_layers = data[data["layers"] == 3]
    data_3_layers = data[data["layers"] == 4]

    plt.figure(figsize=(15, 5))
    
    # Plot for 1-layer configurations
    plt.subplot(1, 3, 1)
    plt.plot(data_1_layer.index, data_1_layer["mse"], marker='o', color='b')
    plt.title("MSE for 1-Layer Configurations")
    plt.xlabel("Configuration Index")
    plt.ylabel("MSE")
    plt.grid(True)

    # Plot for 2-layer configurations
    plt.subplot(1, 3, 2)
    plt.plot(data_2_layers.index, data_2_layers["mse"], marker='o', color='g')
    plt.title("MSE for 2-Layer Configurations")
    plt.xlabel("Configuration Index")
    plt.ylabel("MSE")
    plt.grid(True)

    # Plot for 3-layer configurations
    plt.subplot(1, 3, 3)
    plt.plot(data_3_layers.index, data_3_layers["mse"], marker='o', color='r')
    plt.title("MSE for 3-Layer Configurations")
    plt.xlabel("Configuration Index")
    plt.ylabel("MSE")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# ================================================================

# Data visualistion (with plt) ===================================

    plt.figure(figsize=(10, 6))
    plt.plot(data_1_layer.index, data_1_layer["mse"], marker='o', label="1 Layer", color='b')
    plt.plot(data_2_layers.index, data_2_layers["mse"], marker='o', label="2 Layers", color='g')
    plt.plot(data_3_layers.index, data_3_layers["mse"], marker='o', label="3 Layers", color='r')
    plt.title("MSE Comparison for Different Layer Counts")
    plt.xlabel("Configuration Index")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, np.argmax(y_best_pred, axis=1))
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# ================================================================