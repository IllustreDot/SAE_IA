import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from iris2Plots import *

#X, y = load_iris(return_X_y = True)
data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)

print(len(X_train))
model = MLPClassifier(
    hidden_layer_sizes=(60, ),       
    solver='sgd',                    
    max_iter=1000,                   
    learning_rate_init=0.01,         
    momentum=0.9,                              
    tol=1e-4                         
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Précision du modèle : {accuracy * 100:.2f}%')

plot_learning_curve(model, X_train, y_train)

# 2. Validation Curve (ici on teste l'impact de la taille des couches cachées)
param_range = [5, 10, 20, 30, 40, 50, 60, 70, 80 ,90, 100, 110, 120, 130, 140, 150, 200, 300]
plot_validation_curve(model, X_train, y_train, param_name="hidden_layer_sizes", param_range=param_range)

# 3. Matrice de Confusion
plot_confusion_matrix(model, X_test, y_test)

# 4. Courbe ROC (si classification binaire ou multi-classes)
plot_roc_curve(model, X_test, y_test)

# 5. Projection PCA pour visualiser la séparation des classes

plot_pca(X_train, y_train)