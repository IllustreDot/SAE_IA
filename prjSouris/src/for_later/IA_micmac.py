# building AI from mouse translated to point coordinate dataset
# author: 37b7
# created: 25 Nov 2024

# Import =========================================================

# to use for next import
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
from itertools import product
import multiprocessing
import pandas as pd
import numpy as np
import ast
import os

#to extract data and visualy see the most fitting parmeters
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset

torch.backends.cudnn.benchmark = True

# ===============================================================

import sys
sys.path.append("../rsc/config")
from allvariable import *
from nn_model import NNModel, nn_run

# data related ==================================================

def get_behaviors(selector):
    data_classification_file = pd.read_csv(path_to_data_clean + selector + "/" + file_name_data_classification)
    return data_classification_file.columns.tolist()

def check_behavior(behavior_pairs):
    done_file = pd.read_csv(path_to_config + dile_name_config_done)
    done = [tuple(ast.literal_eval(item)) if isinstance(item, str) else tuple(item) if isinstance(item, list) else item for item in done_file["behavior"]]
    return [pair for pair in behavior_pairs if pair not in done]

def validate_behavior(selector):
    behaviors = get_behaviors(selector)
    behavior_pairs = [(selector, behavior) for behavior in behaviors if behavior != selector]
    return check_behavior(behavior_pairs)

def get_complete_behavior(path_to_data_classification):
    file_name_data_classification = os.listdir(path_to_data_classification)
    return pd.read_csv(path_to_data_classification + file_name_data_classification[0]).columns[1:].tolist()

def merge_behavior_classes(classification_file, model_behaviors_to_merge):
    merged_data = classification_file.copy()
    for merged_behavior, behaviors_to_merge in model_behaviors_to_merge.items():
        relevant_columns = [col for col in behaviors_to_merge if col in classification_file.columns]
        merged_data[merged_behavior] = classification_file[relevant_columns].apply(lambda row: 1 if row.any() else 0, axis=1)
        merged_data.drop(columns=relevant_columns, inplace=True)

def load_data_all(behavior, model_behaviors_to_merge=None):
    data = {"data": None, "classification": None}
    file_sizes = [len(pd.read_csv(path_to_data_clean + cl + "/" + file_name_data)) for cl in behavior]
    min_size = min(file_sizes)

    for cl in behavior:
        data_file = pd.read_csv(path_to_data_clean + cl + "/" + file_name_data).sample(n=min_size, random_state=13)
        classification_file = pd.read_csv(path_to_data_clean + cl + "/" + file_name_data_classification).sample(n=min_size, random_state=13)
        
        if model_behaviors_to_merge:
            classification_file = merge_behavior_classes(classification_file, model_behaviors_to_merge)

        
        columns_to_keep = [col for col in classification_file.columns if col in behavior]
        classification_file = classification_file[columns_to_keep]

        data["data"] = pd.concat([data["data"], data_file], ignore_index=True) if data["data"] is not None else data_file
        data["classification"] = pd.concat([data["classification"], classification_file], ignore_index=True) if data["classification"] is not None else classification_file
    return data

def prepare_data(data):
    features = torch.tensor(data["data"].values, dtype=torch.float32)
    labels = torch.tensor(np.argmax(data["classification"].values, axis=1), dtype=torch.long)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    return DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True), DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)

def load_data(behavior, model_behaviors_to_merge=None):
    return prepare_data(load_data_all(behavior, model_behaviors_to_merge))

# ===============================================================

def generate_layer_configurations(hl_nb_dict_of_dict, behavior):
    layer_configs = []
    for num_layers in range(1, len(hl_nb_dict_of_dict) + 1):
        current_layer_values = [hl_nb_dict_of_dict[str(i+1)].values() for i in range(num_layers)]
        for config in product(*current_layer_values):
            if all(config[i] >= config[i+1] for i in range(len(config) - 1)) and config[-1] == len(behavior):
                layer_configs.append(config)
    return [layer for layer in layer_configs if len(layer) > 1]

def create_output_file(path_to_output, file_name_best_model):
    if not os.path.exists(path_to_output + file_name_best_model):
        with open(path_to_output + file_name_best_model, "w") as f:
            f.write("behavior,layer,accuracy,mse,conf_matrix,accuracies,losses,mses\n")

def train_behavior(behavior, layer_configs, train_loader, test_loader, device, learning_rate_init_number, alpha_number, max_iter_number, path_to_output, file_name_best_model):
    if device == "cpu":
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            for config in layer_configs:
                executor.submit(nn_run, config, behavior, train_loader, test_loader, device, learning_rate_init_number, alpha_number, max_iter_number, path_to_output, file_name_best_model)
    else:
        for config in layer_configs:
            nn_run(config, behavior, train_loader, test_loader, device, learning_rate_init_number, alpha_number, max_iter_number, path_to_output, file_name_best_model)


def main():
    create_output_file(path_to_output, file_name_best_model)
    behavior = get_complete_behavior(path_to_data_classification)
    layer_configs = generate_layer_configurations(hl_nb_dict_of_dict, behavior)
    train_loader, test_loader = load_data(behavior, model_behaviors_to_merge)
    train_behavior(behavior, layer_configs, train_loader, test_loader, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

if __name__ == "__main__":
    main()
