# building of a MLP model for the MNIST dataset
# author: 37b7
# created: 6 Nov 2024

# Import =================================

# to use for next import
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
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

# y = label and x = data
X_train, y_train = load_mnist(path_to_data, kind='train')
X_test, y_test = load_mnist(path_to_data, kind='t10k')
#standardization
X_train = X_train / 255.0
X_test = X_test / 255.0

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

output_file = "../rsc/output/collected_data.txt"
# si le fichier n'existe pas, on le crée
if not os.path.exists(output_file):
    with open(output_file, "w") as f:
        f.write("layers,neurons,mse\n")
    print("Output file:", output_file, " created")
else:
    print("Output file:", output_file, " exists")

# ================================================================

# Model creation and collect of data =============================

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
    
    print(f"Layers: {layers}, Alpha: {alpha_number}, Learning Rate: {learning_rate_init_number}, Random State: {random_state_number}, MSE: {mse:.4f}")
    with open(output_file, "a") as f:
        f.write(f"{len(layers)},{layers}{mse}\n")

# ================================================================

# Model with the best parameter found ============================

best_model = MLPRegressor(
    hidden_layer_sizes=best_config["layers"],
    activation=activation_function,
    solver=solvertype,
    alpha=best_config["alpha"],
    learning_rate_init=best_config["learning_rate_init"],
    max_iter=max_iter_number,
    random_state=best_config["random_state"]
)
best_model.fit(X_train, y_train)
y_best_pred = np.round(best_model.predict(X_test))

# ================================================================

# Data visualistion (with plt) ===================================


# ================================================================

# code used previously

# # variable for MLPClassifier =====================================
# hl_nb = 16 -- dictionnary of dictionnay to get as many layer and theur number of neuron needed to find the best matching parameter
# activation_function = 'relu' mse
# max_iter_number = 1000 
# alpha_number = 1e-4 table of data to test the best alpha
# solvertype = 'sgd'
# random_state_number = 1 table of data to test the best random_state
# learning_rate_init_number = .2 table of data to test the best learning_rate_init
# # model ==========================================================
# modelMLP = MLPClassifier(hidden_layer_sizes=(hl_nb), 
#                         max_iter=max_iter_number,
#                         activation=activation_function,
#                         alpha=alpha_number,
#                         solver=solvertype,
#                         random_state=random_state_number,
#                         learning_rate_init=learning_rate_init_number,
#                         verbose=True)
# modelMLP.fit(X_train, y_train)
# print(f"Training set score: {modelMLP.score(X_train, y_train):.3f}")
# print(f"Test set score: {modelMLP.score(X_test, y_test):.3f}")
# # ================================================================


# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import mean_squared_error
# import numpy as np

# # Initialize and train MLPClassifier
# model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=1)
# model.fit(X_train, y_train)

# # Get probabilities and compute MSE
# probabilities = model.predict_proba(X_test)
# y_test_one_hot = np.eye(len(np.unique(y_test)))[y_test]  # Convert y_test to one-hot if needed
# mse = mean_squared_error(y_test_one_hot, probabilities)
# print("Mean Squared Error:", mse)