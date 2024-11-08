import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve, train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelBinarizer

def plot_learning_curve(model, X_train, y_train, cv=None, scoring='accuracy'):
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5, n_jobs=-1)
    
    train_scores_mean = train_scores.mean(axis=1)
    train_scores_std = train_scores.std(axis=1)
    test_scores_mean = test_scores.mean(axis=1)
    test_scores_std = test_scores.std(axis=1)

    # Tracer les courbes
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, label='Training score', color='b')
    plt.plot(train_sizes, test_scores_mean, label='Test score', color='r')

    # Ecart type
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, color='b', alpha=0.2)
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, color='r', alpha=0.2)

    plt.xlabel('Training Size')
    plt.ylabel('Score')
    plt.title('Learning Curve')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
    

def plot_validation_curve(model, X, y, param_name, param_range, cv=None, scoring='accuracy'):
    train_scores, validation_scores = validation_curve(
        model, X, y, param_name=param_name, param_range=param_range, cv=cv,
        scoring=scoring, n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(validation_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, train_mean, label="Training score", color='blue')
    plt.plot(param_range, val_mean, label="Cross-validation score", color='red')
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.title(f"Validation Curve with {model.__class__.__name__}")
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix")
    plt.show()

def plot_roc_curve(model, X_test, y_test):
    # Binariser les classes si c'est un problème multi-classes
    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test)
    
    # Si c'est un problème multi-classes, on va utiliser "one-vs-rest"
    fpr, tpr, roc_auc = {}, {}, {}
    
    plt.figure(figsize=(10, 6))
    
    for i in range(y_test_bin.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], model.predict_proba(X_test)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

def plot_pca(X, y):
    # Applique PCA pour réduire à 2D pour une visualisation facile
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('PCA Projection')
    plt.colorbar()
    plt.show()