import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from utils.plot_data import *
import time

start_time = time.time()
# Génération d'un jeu de données de classification
X, y = make_classification(n_samples=1500, n_features=25, n_classes=2, n_informative=15, n_redundant=5, random_state=42)

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


mlp = MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=1700, random_state=42)

# Entraînement du modèle
# mlp.fit(X_train, y_train)

# Afficher les données de base
plot_data(X_train, y_train)

# Entraînement du modèle
mlp.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = mlp.predict(X_test)

end_time = time.time()
duration = end_time - start_time
# Afficher les données avec les prédictions finales
print("Traitement final :")
plot_data(X_train, y_train, mlp, True)

# Garder la fenêtre ouverte à la fin
plt.show()

print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred))
print("\nRapport de classification :\n", classification_report(y_test, y_pred))
print(f"La durée de l'action est de {duration:.6f} secondes.")