# Display the results of all the data collected and processed
# author: 37b7
# created: 22 Nov 2024

# TODO! work in progress



# Import =========================================================

import matplotlib.pyplot as plt
import pandas as pd

# ================================================================

# import variable config ==========================================

import sys
sys.path.append("../rsc/config")
from allvariable import *

# ================================================================

# load the data from the file ====================================

def LoadDataResultat():
    data = pd.read_csv(path_to_data_clean + "still/" + file_name_data)
    return data

# ================================================================

# Data analitics ================================================

# we need to isolate the x and y of each body part of the mouse
# we will display the same point cloud graphic all of the body part with each of them in a different color

def DisplayData(data):
    bodypart = {
        "nose": [],
        "forepaw_R": [],
        "forepaw_L": [],
        "hindpaw_R": [],
        "hindpaw_L": [],
        "tailbase": []
    }
    for i in range(1, len(data.columns), 2):
        bodypart[data.columns[i][:-2]] = ([data.iloc[:, i], data.iloc[:, i+1]])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    for key, (x, y) in bodypart.items():
        ax.scatter(x, y, label=key)
    plt.legend()
    plt.show()

DisplayData(LoadDataResultat())
# ================================================================
