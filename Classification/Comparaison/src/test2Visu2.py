import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from utils.plot_data import *

# Génération d'un jeu de données de classification
X, y = make_classification(n_samples=500, n_features=25, n_classes=2, n_informative=2, n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

plt.ion()

# Création du modèle MLP

mlp = MLPClassifier(hidden_layer_sizes=(8, 4), max_iter=400, random_state=42)

# Afficher les données de base
#plot_data(X_train, y_train)

# Entraînement du modèle
mlp.fit(X_train, y_train)

# Afficher uniquement les erreurs de prédiction
print("Traitement final :")
plot_data(X_train, y_train, mlp)
plot_data(X_train, y_train, mlp, show_errors=True)
plt.ioff()
plt.show()