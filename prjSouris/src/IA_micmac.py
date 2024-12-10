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
import numpy as np
import os

# ===============================================================

import sys

import torch
from sklearn.metrics import confusion_matrix

sys.path.append("../rsc/config")
from allvariable import *
from do_not_touch_again.nn_model import nn_run
from do_not_touch_again.data_cleaner import get_single_data, rotate_coordinates, translate_coordinates
from do_not_touch_again.data_loader import get_best_layer, get_classification_header, get_merged_classifications, load_data

# data related ==================================================

# ===============================================================

def train_behavior(layer_configs, behavior, train_loader, test_loader, device, learning_rate_init_number, alpha_number, max_iter_number, path_to_output, file_name_data_output):
    print("Using:", device)
    if isinstance(layer_configs, list):
        if device == torch.device("cpu"):
            print("Starting multiprocessing on CPU")
            with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                for config in layer_configs:
                    executor.submit(nn_run, config, behavior, train_loader, test_loader, device, learning_rate_init_number, alpha_number, max_iter_number, path_to_output, file_name_data_output)
        else:
            print("Starting sequential processing on GPU")
            results = [nn_run(config, behavior, train_loader, test_loader, device, learning_rate_init_number, alpha_number, max_iter_number, path_to_output, file_name_data_output) for config in layer_configs]
        with open(path_to_config + dile_name_config_done, "a") as f:
            f.write("\"" + str(behavior) + "\"\n")
        return nn_run(get_best_layer(behavior), behavior, train_loader, test_loader, device, learning_rate_init_number, alpha_number, max_iter_number, path_to_output, file_name_data_output)
    else:
        return nn_run(layer_configs, behavior, train_loader, test_loader, device, learning_rate_init_number, alpha_number, max_iter_number, path_to_output, file_name_data_output)

def dispatcher(data_row, central_model, list_model, names, model_behaviors_to_merge, model_bahaviors_disabled, classifications, full_header, merged_behaviors):
    data_tensor = torch.tensor(data_row.values, dtype=torch.float32).unsqueeze(0)
    central_output = central_model(data_tensor).argmax(dim=1).item()
    central_behavior = classifications[central_output]
    decompressed_output = [0] * len(full_header)
    if model_behaviors_to_merge:
        if central_behavior in model_behaviors_to_merge:
            sub_model_index = names.index(central_behavior) - 1
            sub_model = list_model[sub_model_index]
            sub_model_output = sub_model(data_tensor).argmax(dim=1).item()
            offset = 0
            for b in full_header:
                if b in model_bahaviors_disabled:
                    continue
                if b not in merged_behaviors:
                    offset += 1
            for behavior, merged_classes in model_behaviors_to_merge.items():
                if behavior == central_behavior:
                    break
                offset += len(merged_classes)
            decompressed_index = offset + sub_model_output
            decompressed_output[decompressed_index] = 1
        else:
            decompressed_output[central_output] = 1
    else:
        decompressed_output[central_output] = 1
    return tuple(decompressed_output)

def validator(data_row, central_model, list_model, names, model_behaviors_to_merge, model_bahaviors_disabled, classifications, full_header, merged_behaviors):
    run_validator_cycle = 3
    num_classes = len(full_header)
    output_sums = [0] * num_classes
    for _ in range(run_validator_cycle):
        output = dispatcher(data_row, central_model, list_model, names, model_behaviors_to_merge, model_bahaviors_disabled, classifications, full_header, merged_behaviors)
        output_sums = [sum(x) for x in zip(output_sums, output)]
    while output_sums.count(max(output_sums)) > 1:
        output = dispatcher(data_row, central_model, list_model, names, model_behaviors_to_merge, model_bahaviors_disabled, classifications, full_header, merged_behaviors)
        output_sums = [sum(x) for x in zip(output_sums, output)]
    final_output = [0] * num_classes
    max_index = output_sums.index(max(output_sums))
    final_output[max_index] = 1
    print("Final Output : ", final_output)
    return final_output

def reorder_and_include_disabled(current_output, current_order, desired_order):
    behavior_to_index = {behavior: i for i, behavior in enumerate(current_order)}
    reordered_output = [[row[behavior_to_index[behavior]] if behavior in behavior_to_index else 0 for behavior in desired_order]for row in current_output]
    return reordered_output

def create_ia_output_file(path_to_output, file_name, classification_headers):
    if not os.path.exists(path_to_output + file_name):
        with open(path_to_output + file_name, "w") as f:
            f.write(",".join(classification_headers) + "\n")

def process_data(input_data_file, classification_headers, path_to_output, file_name_data_ia_output, path_to_data_classification,model_behaviors_to_merge, model_bahaviors_disabled, dico_model, names):
    create_ia_output_file(path_to_output, file_name_data_ia_output, classification_headers)
    print("Processing data from:", input_data_file)
    data = get_single_data(input_data_file)
    list_dispatcher_output = []
    classifications = get_merged_classifications(model_behaviors_to_merge, model_bahaviors_disabled)
    full_header = get_classification_header(path_to_data_classification)
    if model_behaviors_to_merge:
        merged_behaviors = [behavior for behaviors in model_behaviors_to_merge.values() for behavior in behaviors]
    else:
        print("No behaviors to merge.")
        merged_behaviors = []
    dispatcher_classification_order = []
    for b in full_header:
        if b in model_bahaviors_disabled:
            continue
        if b not in merged_behaviors:
            dispatcher_classification_order.append(b)
    for behavior in merged_behaviors + model_bahaviors_disabled:
        dispatcher_classification_order.append(behavior)
    data = translate_coordinates(data)
    for index, row in data.iterrows():
        row = rotate_coordinates(row)
        dispatcher_output = validator(row, dico_model["central"],[dico_model[name] for name in names if name != "central"],names,model_behaviors_to_merge, model_bahaviors_disabled,classifications,full_header, merged_behaviors)
        list_dispatcher_output.append(dispatcher_output)
    reordered_output = reorder_and_include_disabled(list_dispatcher_output, dispatcher_classification_order,classification_headers)
    with open(path_to_output + file_name_data_ia_output, "a") as f:
        for row in reordered_output:
            f.write(",".join(map(str, row)) + "\n")

def compare_and_plot(input_data_file, ia_output_file, classification_headers, path_to_file_name_matches, path_to_data_classification):
    matches_df = pd.read_csv(path_to_file_name_matches)
    classification_file = None
    for _, row in matches_df.iterrows():
        if row['data_file'] == input_data_file:
            classification_file = row['classification_file']
            break
    if classification_file is None:
        print(f"Error: No matching classification file found for {input_data_file}")
        return
    real_data = pd.read_csv(path_to_data_classification + classification_file, header=None)
    ia_output = pd.read_csv(ia_output_file, header=None)
    real_data = real_data.drop(real_data.columns[0], axis=1)
    if real_data.shape != ia_output.shape:
        print("Error: The shape of the real classification data and IA output are not the same.")
        return
    correct_predictions = []
    for real_row, ia_row in zip(real_data.iterrows(), ia_output.iterrows()):
        real_class = np.array(real_row[1])
        ia_class = np.array(ia_row[1])
        if np.array_equal(real_class, ia_class):
            correct_predictions.append(1)
        else:
            correct_predictions.append(0)
    accuracy = np.mean(correct_predictions)
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    
    if "scratching" in classification_headers:
        scratching_index = classification_headers.index("scratching")
        real_scratching = real_data.iloc[:, scratching_index].values
        ia_scratching = ia_output.iloc[:, scratching_index].values
        cm = confusion_matrix(real_scratching, ia_scratching)
        true_positives = cm[1][1]
        false_positives = cm[0][1]
        false_negatives = cm[1][0]
        scratching_accuracy = true_positives / false_positives
        overall_scratching_accuracy = true_positives / (true_positives + false_negatives + false_positives)
        print(f"Scratching Accuracy: {scratching_accuracy * 100:.2f}%")
        print(f"Overall Scratching Accuracy: {overall_scratching_accuracy * 100:.2f}%")
    else:
        print("Error: 'scratching' not found in classification headers.")
    cm = confusion_matrix(real_data.values.argmax(axis=1), ia_output.values.argmax(axis=1))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classification_headers, yticklabels=classification_headers)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(correct_predictions)), correct_predictions, label='Accuracy per row', color='blue')
    plt.axhline(y=accuracy, color='r', linestyle='--', label=f'Overall Accuracy: {accuracy:.2f}')
    plt.title('Prediction Accuracy')
    plt.xlabel('Row Index')
    plt.ylabel('Accuracy (1 for correct, 0 for incorrect)')
    plt.legend()
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

    classification_headers = get_classification_header(path_to_data_classification)
    name_file = os.listdir(path_to_data)[0]
    input_data_file = path_to_data + name_file
    process_data(input_data_file, classification_headers, path_to_output, name_file[:6] + file_name_data_ia_output, path_to_data_classification, model_behaviors_to_merge, model_bahaviors_disabled, dico_model, names)
    print("Processing complete. Results saved to:", path_to_output + file_name_data_ia_output)
    compare_and_plot(name_file, path_to_output + name_file[:6] + file_name_data_ia_output, classification_headers, path_to_output + file_name_matches, path_to_data_classification)

if __name__ == "__main__":
    main()
