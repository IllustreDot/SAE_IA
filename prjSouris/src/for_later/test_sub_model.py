import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix


# Mock data as a dictionary for demonstration
data = pd.read_csv('../rsc/output/analitics.csv')

# Find the best accuracy and corresponding confusion matrix

# isoler la bahavior : "('rearing', 'wall rearing', 'jump')"
data = data.loc[data['behavior'] == "('rearing', 'wall rearing', 'jump')"]
data['conf_matrix'] = data['conf_matrix'].apply(ast.literal_eval)
data['accuracy'] = pd.to_numeric(data['accuracy'])
best_row = data.loc[data['accuracy'].idxmax()]
best_accuracy = best_row['accuracy']
best_conf_matrix = best_row['conf_matrix']

print(f"Best Accuracy: {best_accuracy}")
print(f"Confusion Matrix: {best_conf_matrix}")

plt.figure(figsize=(10, 8))
sns.heatmap(best_conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()