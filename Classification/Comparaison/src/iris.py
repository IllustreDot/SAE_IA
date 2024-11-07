import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from utils.plot_data import *
from sklearn import datasets


X, y= datasets.load_iris(return_X_y = True)

# for i in range (len(y)):
#     print(y[i] , X[i])
#     # print(X[i])
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# print("\n y_train ")
# print(y_train)
# print("\n y_test")
# print(y_test)

# for i in range (len(y_train)):
#     print(y_train[i] , X_train[i])

# print("\n test data \n")
# for i in range (len(y_test)):
#     print(y_test[i] , X_test[i])

# X = iris.data[:, :2] # Utiliser les deux premiers colonnes afin d'avoir un problème de classification binaire.&nbsp;
# y = (iris.target != 0) * 1 # re-étiquetage des fleurs

plt.ion()


mlp = MLPClassifier(hidden_layer_sizes=(4, ), max_iter=500, random_state=42)
mlp.fit(X_train,y_train)




plot_data(X_train, y_train, mlp)
plot_data(X_train, y_train, mlp, show_errors=True)
plt.ioff()
plt.show()

# plt.figure(figsize=(10, 6))
# plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='g', label='0')
# plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='y', label='1')
# plt.scatter(X[y == 2][:, 0], X[y == 2][:, 1], color='b', label='2')
# plt.legend()
# plt.show()