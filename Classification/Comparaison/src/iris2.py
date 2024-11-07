import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y = True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(len(X_train))
model = MLPClassifier(
    hidden_layer_sizes=(120,),       
    solver='sgd',                    
    max_iter=350,                   
    learning_rate_init=0.01,         
    momentum=0.9,                              
    tol=1e-4                         
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Précision du modèle : {accuracy * 100:.2f}%')