# Display the results of all the data collected and processed
# author: 37b7
# created: 22 Nov 2024

# Import =========================================================

import os
import matplotlib.pyplot as plt
import pandas as pd

# ================================================================

# import variable config ==========================================

from do_not_touch_again.data_loader import get_classification_header
import sys
sys.path.append("../rsc/config")
from allvariable import *

# ================================================================

# load the data from the file ====================================

def LoadDataResultat():
    data = []
    for cl in get_classification_header(path_to_data_classification):
        data.append(pd.read_csv(path_to_data_clean + cl + "/" + file_name_data))
    return data

def RawData():
    data = []
    name_file = os.listdir(path_to_data)[0]
    input_data_file = path_to_data + name_file
    data.append(pd.read_csv(input_data_file))
    return data

# ================================================================

# Data analitics ================================================

def DisplayData(data):
    for cl in data:
        bodypart = {
            "nose": [],
            "forepaw_R": [],
            "forepaw_L": [],
            "hindpaw_R": [],
            "hindpaw_L": [],
            "tailbase": []
        }
        for i in range(0, len(cl.columns), 2):
            bodypart[cl.columns[i][:-2]] = ([cl.iloc[:, i], cl.iloc[:, i+1]])

        fig, ax = plt.subplots(figsize=(8, 6))
        for key, (x, y) in bodypart.items():
            ax.scatter(x, y, label=key)
        plt.legend()
        plt.show()

def RawDisplayData(data):
    for cl in data:
        bodypart = {
            "nose": {"x": [], "y": []},
            "forepaw_R": {"x": [], "y": []},
            "forepaw_L": {"x": [], "y": []},
            "hindpaw_R": {"x": [], "y": []},
            "hindpaw_L": {"x": [], "y": []},
            "tailbase": {"x": [], "y": []}
        }
        for i in range(1, len(cl.columns), 3):
            part = cl.iloc[0,i]
            if part in bodypart:
                bodypart[part]["x"] = cl.iloc[2:, i].values
                bodypart[part]["y"] = cl.iloc[2:, i + 1].values

        # Visualization
        fig, ax = plt.subplots(figsize=(8, 6))
        for key, coords in bodypart.items():
            if len(coords["x"]) != 0 and len(coords["y"]) != 0:
                ax.scatter(coords["x"], coords["y"], label=key)
        plt.legend()
        plt.show()
# ================================================================

# main ==========================================================

def main():
    RawDisplayData(RawData())

if __name__ == "__main__":
    main()

# ================================================================