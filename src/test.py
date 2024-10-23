from utils.generate import data_Two_Dimension
from utils.visualisation import plot_2D
from utils.visualisation import plot_2D_with_label
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import time
import csv

def test(sample_size,data_size,hidden_layer_sizes,max_iter,verbose=True,graph=True):
    data = []
    labels = []
    for _ in range(sample_size):
        sample_data = data_Two_Dimension(data_size)
        data.extend(sample_data) 
        #categorization of the data
        for point in sample_data:
            if point[0] >= 0 and point[1] >= 0:
                labels.append("1")
            elif point[0] >= 0 and point[1] <= 0:
                labels.append("2")
            elif point[0] <= 0 and point[1] >= 0:
                labels.append("3")
            else:
                labels.append("4")
    if (verbose):
        print("Dataset classification")

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
    if (verbose):
        print("Data splitted")

    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, verbose=verbose)
    if (verbose):
        print("Model up")

    clf.fit(x_train, y_train)

    if (verbose):
        print("Model trained")

    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    if (verbose):
        print("Accuracy: ", acc)
    if (graph):
        plot_2D_with_label(x_test, y_pred)
    return acc

current_iteration=0
total_combinations=5*5*3
results=[]
for data_size in range(20,101,20):
    for sample_size in range(20,101,20):
        for hidden_layer_sizes in range(2,7,2):
            print(int((current_iteration / total_combinations) * 100),"%")
            results.append([sample_size,data_size,hidden_layer_sizes])
            s=time.time()
            acc=test(sample_size,data_size,(hidden_layer_sizes,hidden_layer_sizes),100000,verbose=False,graph=False)
            e=time.time()
            results[-1].append(e-s)
            results[-1].append(acc)
            current_iteration+=1

with open('results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    
    writer.writerow(["sample_size","data_size","hidden_layer_sizes","time","accuracy"])
    writer.writerows(results)