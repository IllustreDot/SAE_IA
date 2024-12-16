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

from do_not_touch_again.data_loader import get_classification_header, load_pair_data
from do_not_touch_again.generator import behavior_pairs, layer_finder
from do_not_touch_again.nn_model import nn_run

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

def train_behavior(layer_configs, behavior, train_loader, test_loader, device, learning_rate_init_number, alpha_number, max_iter_number, path_to_output, file_name_data_output):
    print(f"Training behavior {behavior} using device: {device}")
    if device == torch.device("cpu"):
        print("Starting multiprocessing on CPU")
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            futures = []
            for config in layer_configs:
                futures.append(executor.submit( nn_run, config, behavior, train_loader, test_loader, device, learning_rate_init_number, alpha_number, max_iter_number, path_to_output, file_name_data_output))
            for future in futures:
                future.result() 
    else:
        print("Starting sequential processing on GPU")
        for config in layer_configs:
            nn_run( config, behavior, train_loader, test_loader, device, learning_rate_init_number, alpha_number, max_iter_number, path_to_output, file_name_data_output)
    with open(path_to_config + dile_name_config_done, "a") as f:
        f.write(f"\"{behavior}\"\n")

def run_parallel_behavior_training(behavior_pairs, layer_configs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for behavior_pair in behavior_pairs:
        train_loader, test_loader = load_pair_data(behavior_pair)
        print(f"Training for {behavior_pair}")
        train_behavior( layer_configs, behavior_pair, train_loader, test_loader, device, learning_rate_init_number, alpha_number, max_iter_number, path_to_output, file_name_data_output)

def remove_behavior_done(gen_behavior_pairs):
    print("Loading done behaviors...")
    try:
        done_file = pd.read_csv(path_to_config + dile_name_config_done)
    except FileNotFoundError:
        print("Done file not found. Proceeding without exclusions.")
        done = set()
    except Exception as e:
        print(f"Error reading the done file: {e}")
        return gen_behavior_pairs
    else:
        done = set()
        for item in done_file["behavior"]:
            try:
                parsed_item = ast.literal_eval(item)
                if isinstance(parsed_item, (list, tuple)) and tuple(sorted(parsed_item)) not in done:
                    done.add(tuple(sorted(parsed_item)))
            except Exception as e:
                print(f"Failed to parse item: {item}, Error: {e}")
                continue
    unique_gen_pairs = {tuple(sorted(behavior)) for behavior in gen_behavior_pairs}
    remaining_behaviors = unique_gen_pairs - done
    return list(remaining_behaviors)

# ================================================================

# Run the program ================================================

if __name__ == "__main__":
    print("Starting")
    create_output_file(path_to_output, file_name_data_output)
    gen_behavior_pairs = behavior_pairs(get_classification_header(path_to_data_classification))
    behaviorPairs = remove_behavior_done(gen_behavior_pairs)
    layer_configs = layer_finder(12, len(behaviorPairs[0]), selected_nb_hlayers)
    run_parallel_behavior_training(behaviorPairs, layer_configs)

# ================================================================"