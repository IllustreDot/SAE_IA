# building AI from mouse translated to point coordinate dataset
# author: 37b7
# created: 26 Nov 2024

# Import =========================================================

# to use for next import
import pandas as pd
import numpy as np
import ast
import os

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

from do_not_touch_again.generator import layer_finder

torch.backends.cudnn.benchmark = True

# ===============================================================

import sys
sys.path.append("../rsc/config")
from allvariable import *

# ===============================================================

def get_classification_header(path_to_data_classification):
    file_name_data_classification = os.listdir(path_to_data_classification)
    return pd.read_csv(path_to_data_classification + file_name_data_classification[0]).columns[1:].tolist()

def get_data_header(path_to_data, file_name_data):
    behaviors_dir = os.listdir(path_to_data)
    file_name_data_header = pd.read_csv(path_to_data + behaviors_dir[0] + "/" + file_name_data).columns.tolist()
    return file_name_data_header

def get_best_layer(behaviors):
    if not isinstance(behaviors, tuple):
        behaviors = tuple(behaviors)
    try:
        output_data = pd.read_csv(path_to_output + file_name_data_output, on_bad_lines='skip')
    except pd.errors.ParserError as e:
        print(f"Error reading CSV file: {e}")
        return None
    output_data["behavior"] = output_data["behavior"].apply(lambda x: tuple(ast.literal_eval(x)) if isinstance(x, str) else tuple(x) if isinstance(x, list) else x)
    output_data["layer"] = output_data["layer"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    filtered_data = output_data[output_data["behavior"] == behaviors]
    if filtered_data.empty:
        print("No matching data found.")
        return None
    best_layer_row = filtered_data.loc[filtered_data["accuracy"].idxmax()]
    return best_layer_row["layer"]

def verify_logic(output_behavior, model_behaviors_to_merge=None, model_behaviors_disabled=None, silent=False):
    if model_behaviors_disabled:
        if silent != True:
            print(f"Warning: The following behaviors are disabled: {model_behaviors_disabled}")
        output_behavior = [behavior for behavior in output_behavior if behavior not in model_behaviors_disabled]
        for merged_behavior, behaviors_to_merge in model_behaviors_to_merge.items():
            if model_behaviors_disabled:
                disabled_in_merge = [behavior for behavior in behaviors_to_merge if behavior in model_behaviors_disabled]
                if disabled_in_merge:
                    if silent != True:
                        print(f"Warning: The following disabled behaviors are in the merge request for '{merged_behavior}': {disabled_in_merge}")
                    behaviors_to_merge = [behavior for behavior in behaviors_to_merge if behavior not in model_behaviors_disabled]
                    model_behaviors_to_merge[merged_behavior] = behaviors_to_merge
            if not all(behavior in output_behavior for behavior in behaviors_to_merge):
                if silent != True:
                    print(f"Blockage: Not all behaviors to merge are present in the data: {behaviors_to_merge}")
                return None
            if merged_behavior in output_behavior:
                if silent != True:
                    print(f"Blockage: The merged behavior '{merged_behavior}' is already present in the data")
                return None
    return output_behavior

def get_merged_classifications(model_behaviors_to_merge=None, model_behaviors_disabled=None):
    columns = get_classification_header(path_to_data_classification)
    if verify_logic(columns, model_behaviors_to_merge, model_behaviors_disabled, True) is None:
        return None
    if model_behaviors_disabled:
        columns = [col for col in columns if col not in model_behaviors_disabled]
    if model_behaviors_to_merge:
        for merged_behavior, behaviors_to_merge in model_behaviors_to_merge.items():
            relevant_columns = [col for col in behaviors_to_merge if col in columns]
            if relevant_columns: 
                columns.append(merged_behavior)
                for col in relevant_columns:
                    columns.remove(col)
    return columns
    
def is_merged_already_done(behaviors):
    print("Checking if behavior is already done...")
    done_file = pd.read_csv(path_to_config + dile_name_config_done)
    done = []
    for item in done_file["behavior"]:
        try:
            parsed_item = ast.literal_eval(item)
            if isinstance(parsed_item, list):
                done.append(tuple(parsed_item))
            elif isinstance(parsed_item, tuple):
                done.append(parsed_item)
        except Exception as e:
            print(f"Failed to parse item: {item}, Error: {e}")
            continue
    if tuple(behaviors) in done:
        print(f'"{behaviors}" already done')
        return True
    return False

def merge_behavior_classes(classification_file, model_behaviors_to_merge):
    merged_data = classification_file.copy()
    for merged_behavior, behaviors_to_merge in model_behaviors_to_merge.items():
        relevant_columns = [col for col in behaviors_to_merge if col in classification_file.columns]
        if relevant_columns: 
            merged_data[merged_behavior] = merged_data[relevant_columns].apply(lambda row: 1 if row.any() else 0, axis=1)
            merged_data.drop(columns=relevant_columns, inplace=True)
    return merged_data

def load_data_all(model_behaviors_to_merge=None, model_behaviors_disabled=None):
    list_data = []
    no_central_classes = False
    data = {"data": None, "classification": None, "layers": None, "name": None}
    output_behavior = [col for col in get_classification_header(path_to_data_classification) if col not in model_behaviors_disabled]
    file_sizes = [len(pd.read_csv(path_to_data_clean + cl + "/" + file_name_data)) for cl in output_behavior if cl not in sum(model_behaviors_to_merge.values(), [])]
    if file_sizes == []:
        no_central_classes = True
        file_sizes = [len(pd.read_csv(path_to_data_clean + cl + "/" + file_name_data)) for cl in output_behavior]
    min_size = min(file_sizes)
    if verify_logic(output_behavior, model_behaviors_to_merge, model_behaviors_disabled) is None:
        return None
    data_file = None
    classification_file = None
    print("Minimum size for central classes:", min_size)
    for cl in output_behavior:
        for merged_behavior, behaviors_to_merge in model_behaviors_to_merge.items():
            if cl in behaviors_to_merge:
                sub_file_sizes = [len(pd.read_csv(path_to_data_clean + cl + "/" + file_name_data)) for cl in model_behaviors_to_merge[merged_behavior]if cl in output_behavior]
                sub_min_size = min(sub_file_sizes)
                if no_central_classes:
                    sub_min_size = min_size
                else:
                    new_size_nb_behaviors = int(min_size/len(behaviors_to_merge))
                    if new_size_nb_behaviors < sub_min_size:
                        sub_min_size = new_size_nb_behaviors
                print("Minimum size for central " , merged_behavior , " -> ", cl, ":", sub_min_size)
                data_file = pd.read_csv(path_to_data_clean + cl + "/" + file_name_data).sample(n=sub_min_size, random_state=13)
                classification_file = pd.read_csv(path_to_data_clean + cl + "/" + file_name_data_classification).sample(n=sub_min_size, random_state=13)
                break
        if cl not in sum(model_behaviors_to_merge.values(), []):
            data_file = pd.read_csv(path_to_data_clean + cl + "/" + file_name_data).sample(n=min_size, random_state=13)
            classification_file = pd.read_csv(path_to_data_clean + cl + "/" + file_name_data_classification).sample(n=min_size, random_state=13)
        if model_behaviors_disabled:
            classification_file = classification_file.drop(columns=[col for col in model_behaviors_disabled if col in classification_file.columns])
        if model_behaviors_to_merge:
            classification_file = merge_behavior_classes(classification_file, model_behaviors_to_merge)
        data["data"] = pd.concat([data["data"], data_file], ignore_index=True) if data["data"] is not None else data_file
        data["classification"] = pd.concat([data["classification"], classification_file], ignore_index=True) if data["classification"] is not None else classification_file
    merged_classification = get_merged_classifications(model_behaviors_to_merge, model_behaviors_disabled)
    if is_merged_already_done(merged_classification):
        data["layers"] = get_best_layer(merged_classification)
        if data["layers"][0] != len(data["data"].columns):
                data["layers"] = layer_finder(len(data["data"].columns), len(classification_file.columns), selected_nb_hlayers)
        else:
            print(f"Using the best layer configuration for '{merged_classification}': {data['layers']}")
    else:
        data["layers"] = layer_finder(len(data["data"].columns), len(classification_file.columns), selected_nb_hlayers)
    data["name"] = "central"
    list_data.append(prepare_data(data))
    data = {"data": None, "classification": None}
    if model_behaviors_to_merge:
        for merged_behavior, behaviors_to_merge in model_behaviors_to_merge.items():
            file_sizes = [len(pd.read_csv(path_to_data_clean + cl + "/" + file_name_data)) for cl in behaviors_to_merge]
            min_size = min(file_sizes)
            print("Minimum size for", merged_behavior, ":", min_size)
            for cl in behaviors_to_merge:
                data_file = pd.read_csv(path_to_data_clean + cl + "/" + file_name_data).sample(n=min_size, random_state=13)
                classification_file = pd.read_csv(path_to_data_clean + cl + "/" + file_name_data_classification).sample(n=min_size, random_state=13)
                if model_behaviors_disabled:
                    classification_file = classification_file.drop(columns=[col for col in model_behaviors_disabled if col in classification_file.columns])
                columns_to_keep = [col for col in classification_file.columns if col in behaviors_to_merge]
                classification_file = classification_file[columns_to_keep]
                data["data"] = pd.concat([data["data"], data_file], ignore_index=True) if data["data"] is not None else data_file
                data["classification"] = pd.concat([data["classification"], classification_file], ignore_index=True) if data["classification"] is not None else classification_file
            if is_merged_already_done(behaviors_to_merge):
                data["layers"] = get_best_layer(behaviors_to_merge)
                if data["layers"][0] != len(data["data"].columns):
                    data["layers"] = layer_finder(len(data["data"].columns), len(classification_file.columns), selected_nb_hlayers)
                else:
                    print(f"Using the best layer configuration for '{behaviors_to_merge}': {data['layers']}")
            else:
                data["layers"] = layer_finder(len(data["data"].columns), len(classification_file.columns), selected_nb_hlayers)
            data["name"] = merged_behavior
            list_data.append(prepare_data(data))
            data = {"data": None, "classification": None}
    train_loaders, test_loaders, layer_configs , names = zip(*list_data)
    return list(train_loaders), list(test_loaders), list(layer_configs), names

def prepare_data(data):
    features = torch.tensor(data["data"].values, dtype=torch.float32)
    labels = torch.tensor(np.argmax(data["classification"].values, axis=1), dtype=torch.long)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    return (DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True), DataLoader(TensorDataset(X_test, y_test), batch_size=128, shuffle=False), data["layers"], data["name"])

def load_data(model_behaviors_to_merge=None, model_bahaviors_disabled=None):
    return load_data_all(model_behaviors_to_merge, model_bahaviors_disabled)



