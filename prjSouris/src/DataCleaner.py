# script to clean the data into separate files for each classification data into csv files per video
# author: 37b7
# created: 20 Nov 2024

# Import =========================================================

import pandas as pd
import numpy as np
import os

# ================================================================

# import variable config ==========================================

import sys
sys.path.append("../rsc/config")
from allvariable import *

# ================================================================

# variable init ==================================================

file_name_data = os.listdir(path_to_data)
file_name_data_classification = os.listdir(path_to_data_classification)

file_name_data_header = pd.read_csv(path_to_data + file_name_data[0] , header=None, nrows=3)
file_name_data_true_header = file_name_data_header.iloc[1].tolist()
file_name_data_sub_header = file_name_data_header.iloc[2].tolist()
data_classification_header = pd.read_csv(path_to_data_classification + file_name_data_classification[0]).columns
data_classification_header = data_classification_header[1:].tolist()

matches = {}
save_matches = False # if you want to rewrite the matches file or not

# ================================================================

# visualisation needed variables =================================

print ("file_name_data_true_header : ", file_name_data_true_header)
print ("file_name_data_sub_header : ", file_name_data_sub_header)
print ("data_classification_header : ", data_classification_header)

# ================================================================

# init file output structure =====================================

def create_folder():
    if not os.path.exists(path_to_data_clean):
        os.makedirs(path_to_data_clean)
    for cl in data_classification_header :
        if not os.path.exists(path_to_data_clean + cl):
            os.makedirs(path_to_data_clean + cl)
        if not os.path.exists(path_to_data_clean + cl + "/data.csv"):
            with open(path_to_data_clean + cl + "/data.csv", "w") as f:
                for i in range(len(file_name_data_true_header)-2):
                    if file_name_data_sub_header[i] != "likelihood":
                        print (file_name_data_sub_header[i])
                        f.write(file_name_data_true_header[i] + "_" + file_name_data_sub_header[i] + ",")
                f.write(file_name_data_true_header[-2] + "_" + file_name_data_sub_header[-2] + "\n")
        if not os.path.exists(path_to_data_clean + cl + "/data_classification.csv"):
            with open(path_to_data_clean + cl + "/data_classification.csv", "w") as f:
                for i in range(len(data_classification_header)-1):
                    f.write(data_classification_header[i] + ",")
                f.write(data_classification_header[-1] + "\n")

# ================================================================

# find matches ===================================================

def find_matches():
    for data_file in file_name_data :
        for classification_file in file_name_data_classification :
            if data_file[:6] == classification_file[:6] :
                matches[data_file] = classification_file
                print ("match found : ", data_file, " with ", classification_file)
    if save_matches :
        with open(path_to_output + "matches.csv", "w") as f:
            f.write("data_file,classification_file\n")
            for key in matches.keys():
                f.write(key + "," + matches[key] + "\n")

# ================================================================

# load file names ================================================

def sort_files():
    for data_file, classification_file in matches.items():
        print("loading data : ", data_file , " with classification : ", classification_file)
        load_data = pd.read_csv(path_to_data + data_file)
        
        index_to_drop = []
        for i in range(len(load_data.columns)):
            if load_data.iloc[1, i] == "likelihood":
                index_to_drop.append(i)
        load_data = load_data.drop(load_data.columns[index_to_drop], axis=1)
        
        load_data_classification = pd.read_csv(path_to_data_classification + classification_file)
        load_data_classification = load_data_classification.drop(load_data_classification.columns[0], axis=1)

        for index, row in load_data_classification.iterrows():
            selected_classification = load_data_classification.columns[row == 1].tolist()
            for cl in selected_classification :
                with open(path_to_data_clean + cl + "/data.csv", "a") as f:
                    f.write(','.join(map(str, load_data.iloc[index+2].tolist())) + '\n')
                with open(path_to_data_clean + cl + "/data_classification.csv", "a") as f:
                    f.write(','.join(map(str, row.tolist())) + '\n')
    print("done")

# ================================================================

# init ===========================================================

create_folder()
find_matches()
sort_files()

# ================================================================
