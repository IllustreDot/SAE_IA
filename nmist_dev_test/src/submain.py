# 1. Import required libraries
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score

# 2. Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)  # Fetches the MNIST dataset
X = mnist.data  # features (each 28x28 image is flattened into a 784 feature vector)
y = mnist.target.astype(int)  # labels (digits 0-9)

# 3. Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Preprocess the data (Standardization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Define the MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', max_iter=20, random_state=42)

# 6. Train the model
mlp.fit(X_train, y_train)

# 7. Make predictions on the test set
y_pred = mlp.predict(X_test)

# 8. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
