# building of a MLP model for the MNIST dataset
# author: 37b7
# created: 23 Oct 2024

# explain how MLP works ==========================================
# MLP is a feedforward neural network that is trained using backpropagation
# it consists of an input layer, [1..inf] hidden layers, and an output layer
# each layer consists of neurons that are connected to the next layer
# the connection between neurons is represented by weights
# bais is added to the weighted sum of the inputs and passed through an activation function
# the weights are adjusted during the training process to minimize the error 

# W(0)*A(0) + B(0) = Z(0) -> activation(Z(0)) = A(1)
# W(1)*A(1) + B(1) = Z(1) -> activation(Z(1)) = A(2)
# and so on

# backpropagation is used to adjust the weights and bais
# the error is calculated and propagated back through the network
# the weights are adjusted based on the error and the learning rate

# TODO : finish the explaination and add the math behind it     


# import =================================
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
from sklearn.neural_network import MLPClassifier
# ================================================================

# collect data ===================================================
from mnist_reader import load_mnist

path_to_data = "../rsc/data"

# y = label and x = data
X_train, y_train = load_mnist(path_to_data, kind='train')
X_test, y_test = load_mnist(path_to_data, kind='t10k')
X_train = X_train / 255.0
X_test = X_test / 255.0

print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)

# ================================================================

# variable for MLPClassifier =====================================

hl_nb = 16
activation_function = 'relu'
max_iter_number = 1000
alpha_number = 1e-4
solvertype = 'sgd'
random_state_number = 1
learning_rate_init_number = .2

# ================================================================

# model ==========================================================

modelMLP = MLPClassifier(hidden_layer_sizes=(hl_nb), 
                        max_iter=max_iter_number,
                        activation=activation_function,
                        alpha=alpha_number,
                        solver=solvertype,
                        random_state=random_state_number,
                        learning_rate_init=learning_rate_init_number,
                        verbose=True)

modelMLP.fit(X_train, y_train)

print(f"Training set score: {modelMLP.score(X_train, y_train):.3f}")
print(f"Test set score: {modelMLP.score(X_test, y_test):.3f}")

# ================================================================