# building of a MLP model for the MNIST dataset
# author: 37b7
# created: 9 Nov 2024

# train or display or not
mode = "qsd"
# process_best or not
submode = "process_best"

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
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss

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

# Variable init for MLPClassifier parameter finder for MNIST ======

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
max_iter_number = 1000
solvertype = 'sgd'

best_accuracy = 0
best_config = {}

# ================================================================

# init of collected data file ====================================

output_file = "../rsc/output/collected_data_classification.csv"
best_output_file = "../rsc/output/best_collected_data_classification.csv"
# si le fichier n'existe pas, on le crée
if mode == "train":
    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            f.write("layers,neurons,accuracy,mse\n")
        print("Output file:", output_file, " created")
    else:
        print("Output file:", output_file, " exists")
if submode == "process_best":
    if not os.path.exists(best_output_file):
        with open(best_output_file, "w") as f:
            f.write("layers,neurons,alpha,learning_rate_init,random_state,accuracy,mse,y_best_pred\n")
            print("Best Output file:", best_output_file, " created")
    else:
        print("Best Output file:", best_output_file, " exists")

# ================================================================

# Model Creation and Data Collection =============================

def train_and_evaluate_model(layers):
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
    probas = model.predict_proba(X_test)
    mse = log_loss(y_test, probas)

    result = {
        "layers": layers,
        "accuracy": accuracy,
        "mse": mse
    }
    return result

if mode == "train":
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(train_and_evaluate_model, layers): layers for layers in layer_configs if len(layers) > 1}

        for future in as_completed(futures):
            result = future.result()
            layers = result["layers"]
            accuracy = result["accuracy"]
            mse = result["mse"]
            print(f"Layers: {layers} Accuracy: {accuracy:.4f} MSE: {mse:.4f}")

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

            with open(output_file, "a") as f:
                f.write(f"{len(layers)},\"{layers}\",{accuracy},{mse}\n")

    print("Best Configuration:", best_config)

# ================================================================

# Model with the Best Parameters Found ===========================
if submode == "process_best":
    if mode != "train":
        best_config_data = pd.read_csv(output_file).sort_values("accuracy", ascending=False).iloc[0].to_dict()
        best_config_data["neurons"] = eval(best_config_data["neurons"])

        best_config = {
            "layers": best_config_data["neurons"],
            "alpha": alpha_number,
            "learning_rate_init": learning_rate_init_number,
            "random_state": random_state_number,
            "accuracy": best_config_data["accuracy"],
            "mse": best_config_data["mse"]
        }


    best_model = MLPClassifier(
        hidden_layer_sizes=best_config["layers"],
        activation=activation_function,
        solver=solvertype,
        alpha=best_config["alpha"],
        learning_rate_init=best_config["learning_rate_init"],
        max_iter=max_iter_number,
        random_state=best_config["random_state"],
        verbose=True
    )
    best_model.fit(X_train, y_train)
    y_best_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_best_pred)
    probas = best_model.predict_proba(X_test)
    mse = log_loss(y_test, probas)

    with open(best_output_file, "a") as f:
        f.write(f"{len(best_config['layers'])},\"{best_config['layers']}\",{best_config['alpha']},{best_config['learning_rate_init']},{best_config['random_state']},{accuracy},{mse},{y_best_pred}\n")

# ================================================================

# Read and Process Data ==========================================
if mode == "display":
    data = pd.read_csv(output_file)
    data.columns = ["layers", "neurons", "accuracy", "mse"]
    data["neurons"] = data["neurons"].apply(eval)

# ================================================================

# Plotting Accuracy and MSE for Different Layer Configurations ===
if mode == "display":
    data_1_layer = data[data["layers"] == 2]
    data_2_layers = data[data["layers"] == 3]
    data_3_layers = data[data["layers"] == 4]

    # Comparison Plot of Accuracy Across Layer Counts
    plt.figure(figsize=(10, 6))
    plt.plot(data_1_layer.index, data_1_layer["accuracy"], marker='o', label="1 Layer Accuracy", color='b')
    plt.plot(data_2_layers.index, data_2_layers["accuracy"], marker='o', label="2 Layers Accuracy", color='g')
    plt.plot(data_3_layers.index, data_3_layers["accuracy"], marker='o', label="3 Layers Accuracy", color='c')
    plt.title("Accuracy Comparison for Different Layer Counts")
    plt.xlabel("Configuration Index")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Comparison Plot of MSE Across Layer Counts
    plt.figure(figsize=(10, 6))
    plt.plot(data_1_layer.index, data_1_layer["mse"], marker='x', label="1 Layer MSE", color='r')
    plt.plot(data_2_layers.index, data_2_layers["mse"], marker='x', label="2 Layers MSE", color='orange')
    plt.plot(data_3_layers.index, data_3_layers["mse"], marker='x', label="3 Layers MSE", color='purple')
    plt.title("MSE Comparison for Different Layer Counts")
    plt.xlabel("Configuration Index")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True)
    plt.show()

if submode == "process_best":
    # Confusion Matrix for Best Model
    conf_matrix = confusion_matrix(y_test, y_best_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# ================================================================
