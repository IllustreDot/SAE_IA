# building AI from mouse translated to point coordinate dataset
# author: 37b7
# created: 20 Nov 2024

# Import =========================================================

# to use for next import
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import multiprocessing
import pandas as pd
import numpy as np
import ast
import os

#to extract data and visualy see the most fitting parmeters
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset

from prjSouris.src.data_loader import remove_behavior_done
from prjSouris.src.datatoAI import LoadData
from prjSouris.src.nn_model import nn_run

torch.backends.cudnn.benchmark = True

# ================================================================

# Variable config import =========================================

import sys
sys.path.append("../rsc/config")  # Relative path to the config directory
from allvariable import *

# ================================================================

# validator behavior combinaison done ============================

def GetBehaviors(selector):
    data_classification_file = pd.read_csv(path_to_data_clean + selector + "/" + file_name_data_classification)
    data_classification = data_classification_file.columns.tolist()
    return data_classification

# ================================================================

# function to use in the program =================================

def generate_layer_configurations(hl_nb_dict_of_dict):
    to_return_layers = []
    for num_layers in range(1, len(hl_nb_dict_of_dict) + 1):
        current_layer_values = [hl_nb_dict_of_dict[str(i+1)].values() for i in range(num_layers)]
        for config in product(*current_layer_values):
            if all(config[i] > config[i+1] for i in range(len(config) - 1)) and config[-1] == 2:
                to_return_layers.append(config)
    to_return_layers = [layer for layer in to_return_layers if len(layer) > 1]
    return to_return_layers

# ================================================================

# init of collected data file ====================================

def create_output_file(path_to_output, file_name_data_output):
    if not os.path.exists(path_to_output + file_name_data_output):
        with open(path_to_output + file_name_data_output, "w") as f:
            f.write("behavior,layer,accuracy,mse,conf_matrix,accuracies,losses,mses\n")
        print("Output file:", path_to_output + file_name_data_output, " created")
    else:
        print("Output file:", path_to_output + file_name_data_output, " exists")

# ================================================================

# multithreading =================================================

def cpu_train_single_behavior(behavior, train_loader, test_loader, layer_configs):
    print("Using CPU")
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        for config in layer_configs:
            executor.submit(nn_run, config, behavior, train_loader, test_loader, torch.device('cpu'))

def gpu_train_with_multiprocessing(behavior, layer_configs, train_loader, test_loader, devices):
    print("Using GPU")
    for config in layer_configs:
        nn_run(config, behavior, train_loader, test_loader, devices[0])
    # with mp.Pool(processes=len(devices)) as pool:
    #     pool.starmap(nnRun, [(config, behavior, train_loader, test_loader, device) for config, device in zip(layer_configs, devices)])

def run_parallel_behavior_training(behavior_pairs, layer_configs, choose_gpu=False):
    for behavior in behavior_pairs:
        train_loader, test_loader = LoadData(behavior)
        print(f"Training for {behavior}")
        if choose_gpu:
            num_gpus = torch.cuda.device_count()
            print(f"Using {num_gpus} GPUs.")
            devices = [torch.device(f'cuda:{i}') for i in range(num_gpus)]
            print(devices)
            gpu_train_with_multiprocessing(behavior, layer_configs, train_loader, test_loader, devices)
        else:
            cpu_train_single_behavior(behavior, train_loader, test_loader, layer_configs)
        with open(path_to_config + dile_name_config_done, "a") as f:
            f.write(f"\"{behavior}\"\n")

# ================================================================

# Run the program ================================================

if __name__ == "__main__":
    print("Starting")
    create_output_file(path_to_output, file_name_data_output)
    behaviorPairs = remove_behavior_done(selector)
    layer_configs = generate_layer_configurations(hl_nb_dict_of_dict)
    run_parallel_behavior_training(behaviorPairs, layer_configs, choose_gpu)

# ================================================================"