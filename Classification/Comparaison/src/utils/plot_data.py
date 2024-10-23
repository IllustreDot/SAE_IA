import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def plot_data(X, y, model=None, show_errors=False):
    plt.figure(figsize=(10, 6))
    
    if model is not None:
        if show_errors :
            # Prédictions du modèle
            y_pred = model.predict(X)
            # Identifier les indices où le modèle s'est trompé
            incorrect_indices = np.where(y_pred != y)[0]
            X_incorrect = X[incorrect_indices]
            y_incorrect = y[incorrect_indices]
            
            # Tracer uniquement les points incorrects
            for i in range(len(X_incorrect)):
                if y_incorrect[i] == 0:
                    plt.scatter(X_incorrect[i, 0], X_incorrect[i, 1], c='violet', marker='x', s=100, label='Erreur Classe 0' if i == 0 else "")
                else:
                    plt.scatter(X_incorrect[i, 0], X_incorrect[i, 1], c='orange', marker='x', s=100, label='Erreur Classe 1' if i == 0 else "")


            plt.title("Erreurs de prédiction")
        else:
            scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=100)
            y_pred = model.predict(X)
            plt.scatter(X[:, 0], X[:, 1], c=y_pred, marker='x', cmap='coolwarm', edgecolor='k', s=100, label='Prédictions')

            
            plt.legend(*scatter.legend_elements(), title="Classes")
            plt.title("Visualisation des données de classification")
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
    else:
        # Tracer les données réelles
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=100)
        plt.legend(*scatter.legend_elements(), title="Classes")
        plt.title("Visualisation des données de classification")
    
    plt.xlim(-4, 3.5)  # Ajustez ces valeurs selon vos données
    plt.ylim(-4, 3.5)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.draw()
    plt.pause(1) 
