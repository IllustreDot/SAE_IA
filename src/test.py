from utils.generate import data_Two_Dimension
from utils.visualisation import plot_2D
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


data = []
labels = []
sample_size = 1000
data_size = 100

for _ in range(sample_size):
    sample_data = data_Two_Dimension(data_size)
    data.extend(sample_data) 
    for point in sample_data:
        if point[0] > 0 and point[1] > 0:
            labels.append("1")
        elif point[0] > 0 and point[1] < 0:
            labels.append("2")
        elif point[0] < 0 and point[1] > 0:
            labels.append("3")
        else:
            labels.append("4")

print("Dataset classification")

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print("Data splitted")

hidden_layer_size = (5, 5)
clf = MLPClassifier(hidden_layer_sizes=hidden_layer_size, max_iter=1000, verbose=True)

print("Model up")
clf.fit(x_train, y_train)
print("Model trained")

y_pred = clf.predict(x_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))


# data=data_Two_Dimension()
# plot_2D(data)