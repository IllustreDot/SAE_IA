import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from utils.plot_data import *

# Génération d'un jeu de données de classification

X, y = make_classification(n_samples=300, n_features=15, n_classes=2, n_informative=2, n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

plt.ion()

for i in range (len(y_train)):
    print(y_train[i] , X_train[i])

print("\n test data \n")
for i in range (len(y_test)):
    print(y_test[i] , X_test[i])


# Création du modèle MLP

mlp = MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=500, random_state=42)

# Afficher les données de base
#plot_data(X_train, y_train)

#fit the model for the data
mlp.fit(X_train, y_train)

# Afficher uniquement les erreurs de prédiction
print("Traitement final :")
plot_data(X_train, y_train, mlp)
plot_data(X_train, y_train, mlp, show_errors=True)
plt.ioff()
plt.show()


# for i in range (len(y)):
#     print(y[i], X[i])
    
