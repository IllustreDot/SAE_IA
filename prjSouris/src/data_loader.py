# building AI from mouse translated to point coordinate dataset
# author: 37b7
# created: 20 Nov 2024

# Import =========================================================

# to use for next import
import pandas as pd
import numpy as np
import ast
import os

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

torch.backends.cudnn.benchmark = True

# ===============================================================

import sys
sys.path.append("../rsc/config")
from allvariable import *

# ===============================================================

def remove_behavior_done(selector):
    behaviors = pd.read_csv(path_to_data_clean + selector + "/" + file_name_data_classification).tolist()
    behavior_pairs = [(selector, behavior) for behavior in behaviors if behavior != selector]
    done_file = pd.read_csv(path_to_config + dile_name_config_done)
    done = [tuple(ast.literal_eval(item)) if isinstance(item, str) else tuple(item) if isinstance(item, list) else item for item in done_file["behavior"]]
    remaining_pairs = []
    for pair in behavior_pairs:
        if pair not in done:
            remaining_pairs.append(pair)
        else:
            print(f'"{pair[0]}" and "{pair[1]}" already done')
    return remaining_pairs

def get_classification_header(path_to_data_classification):
    file_name_data_classification = os.listdir(path_to_data_classification)
    return pd.read_csv(path_to_data_classification + file_name_data_classification[0]).columns[1:].tolist()

def merge_behavior_classes(classification_file, model_behaviors_to_merge, model_behaviors_disabled):
    merged_data = classification_file.copy()
    for merged_behavior, behaviors_to_merge in model_behaviors_to_merge.items():
        if model_behaviors_disabled:
            disabled_in_merge = [behavior for behavior in behaviors_to_merge if behavior in model_behaviors_disabled]
            if disabled_in_merge:
                print(f"Warning: The following disabled behaviors are in the merge request for '{merged_behavior}': {disabled_in_merge}")
                behaviors_to_merge = [behavior for behavior in behaviors_to_merge if behavior not in model_behaviors_disabled]
        relevant_columns = [col for col in behaviors_to_merge if col in classification_file.columns]
        if relevant_columns:
            merged_data[merged_behavior] = classification_file[relevant_columns].apply(lambda row: 1 if row.any() else 0, axis=1)
            merged_data.drop(columns=relevant_columns, inplace=True)
    
    return merged_data

def load_data_all(behavior, model_behaviors_to_merge=None, model_behaviors_disabled=None):
    data = {"data": None, "classification": None}
    file_sizes = [len(pd.read_csv(path_to_data_clean + cl + "/" + file_name_data)) for cl in behavior]
    min_size = min(file_sizes)
    for cl in behavior:
        data_file = pd.read_csv(path_to_data_clean + cl + "/" + file_name_data).sample(n=min_size, random_state=13)
        classification_file = pd.read_csv(path_to_data_clean + cl + "/" + file_name_data_classification).sample(n=min_size, random_state=13)
        if model_behaviors_disabled:
            classification_file = classification_file.drop(columns=[col for col in model_behaviors_disabled if col in classification_file.columns])
        if model_behaviors_to_merge:
            classification_file = merge_behavior_classes(classification_file, model_behaviors_to_merge, model_behaviors_disabled)
        columns_to_keep = [col for col in classification_file.columns if col in behavior]
        classification_file = classification_file[columns_to_keep]
        data["data"] = pd.concat([data["data"], data_file], ignore_index=True) if data["data"] is not None else data_file
        data["classification"] = pd.concat([data["classification"], classification_file], ignore_index=True) if data["classification"] is not None else classification_file
    return data

def prepare_data(data):
    features = torch.tensor(data["data"].values, dtype=torch.float32)
    labels = torch.tensor(np.argmax(data["classification"].values, axis=1), dtype=torch.long)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    return DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True), DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)

def load_data(behavior, model_behaviors_to_merge=None, model_bahaviors_disabled=None):
    return prepare_data(load_data_all(behavior, model_behaviors_to_merge, model_bahaviors_disabled))
