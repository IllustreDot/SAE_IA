# script to clean the data into separate files for each classification data into csv files per video
# author: 37b7
# created: 20 Nov 2024

# Import =========================================================

import pandas as pd
import numpy as np
import os

# ================================================================

# variable init ==================================================

basedirpath = "../rsc/"
path_to_data = basedirpath + "data/"
path_to_data_classification = basedirpath + "data_classification/"
output_clean_data_path = basedirpath + "data_cleaned/"
output_path = basedirpath + "output/"
data_files_name = os.listdir(path_to_data)
data_classification_files_name = os.listdir(path_to_data_classification)

data_files_name_header = pd.read_csv(path_to_data + data_files_name[0] , header=None, nrows=3)
data_files_name_true_header = data_files_name_header.iloc[1].tolist()
data_files_name_sub_header = data_files_name_header.iloc[2].tolist()
data_classification_header = pd.read_csv(path_to_data_classification + data_classification_files_name[0]).columns
data_classification_header = data_classification_header[1:].tolist()

matches = {}
save_matches = False # if you want to rewrite the matches file or not

# ================================================================

# visualisation needed variables =================================

print ("data_files_name_true_header : ", data_files_name_true_header)
print ("data_files_name_sub_header : ", data_files_name_sub_header)
print ("data_classification_header : ", data_classification_header)

# ================================================================

# init file output structure =====================================

def create_folder():
    if not os.path.exists(output_clean_data_path):
        os.makedirs(output_clean_data_path)
    for cl in data_classification_header :
        if not os.path.exists(output_clean_data_path + cl):
            os.makedirs(output_clean_data_path + cl)
        if not os.path.exists(output_clean_data_path + cl + "/data.csv"):
            with open(output_clean_data_path + cl + "/data.csv", "w") as f:
                for i in range(len(data_files_name_true_header)-1):
                    f.write(data_files_name_true_header[i] + "_" + data_files_name_sub_header[i] + ",")
                f.write(data_files_name_true_header[-1] + "_" + data_files_name_sub_header[-1] + "\n")
        if not os.path.exists(output_clean_data_path + cl + "/data_classification.csv"):
            with open(output_clean_data_path + cl + "/data_classification.csv", "w") as f:
                for i in range(len(data_classification_header)-1):
                    f.write(data_classification_header[i] + ",")
                f.write(data_classification_header[-1] + "\n")

# ================================================================

# find matches ===================================================

def find_matches():
    for data_file in data_files_name :
        for classification_file in data_classification_files_name :
            if data_file[:6] == classification_file[:6] :
                matches[data_file] = classification_file
                print ("match found : ", data_file, " with ", classification_file)
    if save_matches :
        with open(output_path + "matches.csv", "w") as f:
            f.write("data_file,classification_file\n")
            for key in matches.keys():
                f.write(key + "," + matches[key] + "\n")

# ================================================================

# load file names ================================================

def sort_files():
    for data_file, classification_file in matches.items():
        print("loading data : ", data_file , " with classification : ", classification_file)
        load_data = pd.read_csv(path_to_data + data_file)
        load_data_classification = pd.read_csv(path_to_data_classification + classification_file)
        load_data_classification = load_data_classification.drop(load_data_classification.columns[0], axis=1)

        for index, row in load_data_classification.iterrows():
            selected_classification = load_data_classification.columns[row == 1].tolist()
            for cl in selected_classification :
                with open(output_clean_data_path + cl + "/data.csv", "a") as f:
                    f.write(','.join(map(str, load_data.iloc[index+2].tolist())) + '\n')
                with open(output_clean_data_path + cl + "/data_classification.csv", "a") as f:
                    f.write(','.join(map(str, row.tolist())) + '\n')
    print("done")

# ================================================================

# init ===========================================================

create_folder()
find_matches()
sort_files()

# ================================================================
