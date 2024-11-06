import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

#load csv
with open('src/Dataset/loan_data.csv', 'r') as file:
    reader = csv.reader(file)
    data = list(reader)

header = data[0]
data = data[1:]

x=[]
y=[]

scaler = StandardScaler()

for row in data:
    x.append(row[:-1])
    y.append(row[-1])

x_encoded = []

for col in zip(*x):
    try:
        float(col[0])
        x_encoded.append(list(map(float, col)))
    except ValueError:
        le = LabelEncoder()
        x_encoded.append(le.fit_transform(col))

x = np.array(x_encoded).T
y = LabelEncoder().fit_transform(y)  

x = scaler.fit_transform(x)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=5)

MLPClassifier = MLPClassifier(batch_size=32, nesterovs_momentum=False,solver="sgd",hidden_layer_sizes=(124))

train_acc=[]
val_acc=[]

epochs = 5
for epoch in range(epochs):
    print(epoch)
    MLPClassifier.fit(x_train, y_train)
    test = accuracy_score(y_train, MLPClassifier.predict(x_train))
    train_acc.append(test)
    val = accuracy_score(y_val, MLPClassifier.predict(x_val))
    val_acc.append(val)
    print("Difference: ", test - val)
    print("Byase Error: ", 1 - val)

print(MLPClassifier.get_params())

plt.figure(figsize=(10, 6))
plt.ylim(0, 1)
plt.plot(range(1, epochs + 1), train_acc, label='Training Accuracy')
plt.plot(range(1, epochs + 1), val_acc, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs. Validation Accuracy')
plt.legend()

y_pred = MLPClassifier.predict(x)
conf_mat=confusion_matrix(y, y_pred)
plt.matshow(conf_mat, cmap='Blues')

plt.colorbar() 
class_labels = ['Refusé', 'Accepté'] 
plt.xticks(ticks=np.arange(len(class_labels)), labels=class_labels)
plt.yticks(ticks=np.arange(len(class_labels)), labels=class_labels)

for i in range(conf_mat.shape[0]): 
    for j in range(conf_mat.shape[1]):  
        plt.text(j, i, str(conf_mat[i, j]), ha='center', va='center', color='black')

plt.show()