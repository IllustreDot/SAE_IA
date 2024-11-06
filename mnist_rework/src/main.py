# building of a MLP model for the MNIST dataset
# author: 37b7
# created: 6 Nov 2024

# Import =================================

# to use for next import
import numpy as np
import matplotlib.pyplot as plt
#to extract data and visualy see the most fitting parmeters
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# model used
from sklearn.neural_network import MLPClassifier

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

# Variable init for MLPClassifier parameter finder for MNIST =====




# ================================================================

# Model creation and collect of data =============================




# ================================================================

# Data visualistion (with plt) ===================================




# ================================================================

# Model with the best parameter found ============================



# ================================================================

# Construction of graph to show the performance of the model ======



# ================================================================











# code used previously

# # variable for MLPClassifier =====================================
# hl_nb = 16
# activation_function = 'relu'
# max_iter_number = 1000
# alpha_number = 1e-4
# solvertype = 'sgd'
# random_state_number = 1
# learning_rate_init_number = .2
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