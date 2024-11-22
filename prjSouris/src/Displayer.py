# Display the results of all the data collected and processed
# author: 37b7
# created: 22 Nov 2024

# TODO! work in progress



# Import =========================================================

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd

# ================================================================

# load the data from the file ====================================

def LoadDataResultat():
    data = pd.read_csv(output_file)
    return data

# ================================================================

# Display results ================================================

def DisplayResults(data = None):
    if data == None:
        data = LoadDataResultat()

    # Plot Accuracy, Loss, and MSE Progression for all configurations
    plt.figure(figsize=(18, 12))

    # Accuracy Plot
    plt.subplot(3, 1, 1)
    for index, row in data.iterrows():
        accuracies = eval(row["accuracies"])
        plt.plot(accuracies, label=f"Config {index + 1}: {row['accuracy']:.2f}%")
    plt.title("Accuracy Progression Across Configurations")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    # Loss Plot
    plt.subplot(3, 1, 2)
    for index, row in data.iterrows():
        losses = eval(row["losses"])
        plt.plot(losses, label=f"Config {index + 1}")
    plt.title("Loss Progression Across Configurations")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # MSE Plot
    plt.subplot(3, 1, 3)
    for index, row in data.iterrows():
        mses = eval(row["mses"])
        plt.plot(mses, label=f"Config {index + 1}")
    plt.title("MSE Progression Across Configurations")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Display Confusion Matrices
    for index, row in data.iterrows():
        conf_matrix = eval(row["conf_matrix"])
        ConfusionMatrixDisplay(conf_matrix).plot()
        plt.title(f"Confusion Matrix for Config {index + 1}")
        plt.show()

    # Bar plot comparison of final accuracies
    plt.figure(figsize=(10, 6))
    configurations = [f"Config {i + 1}" for i in range(len(data))]
    accuracies = data["accuracy"].tolist()
    plt.bar(configurations, accuracies, color='skyblue')
    plt.title("Comparison of Final Accuracies Across Configurations")
    plt.ylabel("Accuracy (%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# ================================================================

# display best configuration =====================================

    best_index = data["accuracy"].idxmax()
    best_row = data.iloc[best_index]
    print(f"Best Configuration: Config {best_index + 1}")
    print(f"  Accuracy: {best_row['accuracy']:.2f}%")
    print(f"  Final Loss: {eval(best_row['losses'])[-1]:.4f}")
    print(f"  Final MSE: {eval(best_row['mses'])[-1]:.4f}")

    ConfusionMatrixDisplay(eval(best_row["conf_matrix"])).plot()
    plt.title(f"Confusion Matrix for Best Config {best_index + 1}")
    plt.show()

# ================================================================