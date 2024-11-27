# building AI from mouse translated to point coordinate dataset
# author: 37b7
# created: 25 Nov 2024

# Import =========================================================

# to use for next import
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import multiprocessing
import seaborn as sns
import pandas as pd
import os

# ===============================================================

import sys

import torch
sys.path.append("../rsc/config")
from allvariable import *
from do_not_touch_again.nn_model import nn_run
from do_not_touch_again.data_loader import get_best_layer, get_classification_header, get_merged_classifications, load_data

# data related ==================================================

# ===============================================================

def train_behavior(layer_configs, behavior, train_loader, test_loader, device, learning_rate_init_number, alpha_number, max_iter_number, path_to_output, file_name_data_output):
    if isinstance(layer_configs, list):
        print("Using :", device)
        if device == "cpu":
            with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                for config in layer_configs:
                    executor.submit(nn_run,config,behavior,train_loader,test_loader,device,learning_rate_init_number,alpha_number,max_iter_number,path_to_output,file_name_data_output)
        else:
            for config in layer_configs:
                model = nn_run(config, behavior, train_loader, test_loader, device, learning_rate_init_number, alpha_number, max_iter_number, path_to_output, file_name_data_output)
        with open(path_to_config + dile_name_config_done, "a") as f:
            f.write(str(behavior) + "\n")
        return nn_run(get_best_layer(behavior), behavior, train_loader, test_loader, device, learning_rate_init_number, alpha_number, max_iter_number, path_to_output, file_name_data_output)
    else:
        model = nn_run(layer_configs, train_loader, test_loader, device, learning_rate_init_number, alpha_number, max_iter_number, path_to_output, file_name_data_output)
    return model

def dispatcher(central_model, list_model, names, file_path, model_behaviors_to_merge, model_bahaviors_disabled):
    data = pd.read_csv(file_path)
    classifications = get_merged_classifications(model_behaviors_to_merge, model_bahaviors_disabled)
    central_output = central_model(data)
    central_output = tuple(central_output)
    central_index = central_output.index(1)
    central_behavior = classifications[central_index]
    if central_behavior in model_behaviors_to_merge:
        sub_model_index = names.index(central_behavior)
        sub_model = list_model[sub_model_index]
        sub_model_output = sub_model(data)
        sub_model_output = tuple(sub_model_output)
        sub_behaviors = model_behaviors_to_merge[central_behavior]
        sub_behavior_index = sub_model_output.index(1)
        sub_behavior = sub_behaviors[sub_behavior_index]
        decompressed_output = [0] * len(classifications + sum(model_behaviors_to_merge.values(), []))
        decompressed_index = classifications.index(central_behavior) + sub_behaviors.index(sub_behavior)
        decompressed_output[decompressed_index] = 1
        return tuple(decompressed_output)
    decompressed_output = [0] * len(classifications + sum(model_behaviors_to_merge.values(), []))
    decompressed_output[central_index] = 1
    return tuple(decompressed_output)


def dispatcher(central_model, list_model, names, file_path, model_behaviors_to_merge, model_behaviors_disabled):
    data = pd.read_csv(file_path)
    classifications = get_merged_classifications(model_behaviors_to_merge, model_behaviors_disabled)
    central_output = tuple(central_model(data))
    central_index = central_output.index(1)
    central_behavior = classifications[central_index]
    decompressed_output = [0] * (len(classifications) + len(sum(model_behaviors_to_merge.values(), [])))
    if model_behaviors_to_merge:
        if central_behavior in model_behaviors_to_merge:
            sub_model_index = names.index(central_behavior)
            sub_model = list_model[sub_model_index]
            sub_model_output = tuple(sub_model(data))
            sub_behavior_index = sub_model_output.index(1)
            decompressed_index = classifications.index(central_behavior) + sub_behavior_index
            decompressed_output[decompressed_index] = 1
        else:
            decompressed_output[central_index] = 1
    else:
        decompressed_output[central_index] = 1
    return tuple(decompressed_output), classifications

def reorder_and_include_disabled(current_output, current_order, desired_order, model_behaviors_disabled):
    full_order = current_order + model_behaviors_disabled
    reordered_output = [0] * len(desired_order)
    for i, behavior in enumerate(desired_order):
        if behavior in full_order:
            index = full_order.index(behavior)
            reordered_output[i] = current_output[index]
    return reordered_output

def create_ia_output_file(path_to_output, file_name):
    if not os.path.exists(path_to_output + file_name):
        with open(path_to_output + file_name, "w") as f:
            f.write(get_classification_header(path_to_data_classification).split(",") + "\n")

def compare_and_plot(input_file, output_file, column_names):
    input_data = pd.read_csv(input_file, names=column_names)
    output_data = pd.read_csv(output_file, names=column_names)
    if input_data.shape != output_data.shape:
        raise ValueError("Input and output files have mismatched shapes.")
    comparison_data = pd.DataFrame({
        "Value": input_data.values.flatten().tolist() + output_data.values.flatten().tolist(),
        "Source": ["Input"] * input_data.size + ["Output"] * output_data.size,
        "Index": list(range(input_data.size)) * 2
    })
    plt.figure(figsize=(14, 7))
    sns.stripplot(data=comparison_data, x="Index", y="Value", hue="Source", jitter=True, dodge=True, alpha=0.6)
    plt.title("Comparison of Input and Output Files")
    plt.xlabel("Index (Flattened)")
    plt.ylabel("Classification Value")
    plt.legend(title="Source")
    plt.tight_layout()
    plt.show()

def main():
    list_train_loader, list_test_loader , list_layer_config, names = load_data(model_behaviors_to_merge, model_bahaviors_disabled)
    dico_model = {}
    for name in names:
        dico_model[name] = [None]
    if isinstance(list_train_loader, list) and isinstance(list_test_loader, list) and isinstance(list_layer_config, list):
        for train_loader, test_loader, layer_configs, name in zip(list_train_loader, list_test_loader, list_layer_config, names):
            if name == 'central':
                sub_behaviors = get_merged_classifications(model_behaviors_to_merge, model_bahaviors_disabled)
            else:
                sub_behaviors = model_behaviors_to_merge.get(name, [])
            print("Training model for:", sub_behaviors)
            dico_model[name] = train_behavior(layer_configs, sub_behaviors, train_loader, test_loader, torch.device('cuda' if torch.cuda.is_available() else 'cpu'), learning_rate_init_number, alpha_number, max_iter_number, path_to_output, file_name_data_output)
    create_ia_output_file(path_to_output, file_name_data_ia_output)
    
    input_data_file = path_to_data + os.listdir(path_to_data)[0]
    print("Processing data from:", input_data_file)
    data = pd.read_csv(input_data_file)
    with open(path_to_output + file_name_data_ia_output, "a") as f:
        for index, row in data.iterrows():
            dispatcher_output, dispatcher_classification_order = dispatcher(dico_model["central"],[dico_model[name] for name in names if name != "central"],names,input_data_file,model_behaviors_to_merge, model_bahaviors_disabled)
            reordered_output = reorder_and_include_disabled(dispatcher_output, dispatcher_classification_order,get_classification_header(path_to_data_classification).split(","),model_bahaviors_disabled)
            f.write(",".join(map(str, reordered_output)) + "\n")
    print("Processing complete. Results saved to:", path_to_output + file_name_data_ia_output)
    
    compare_and_plot(input_data_file, path_to_output + file_name_data_ia_output, get_classification_header(path_to_data_classification).split(","))

if __name__ == "__main__":
    main()
