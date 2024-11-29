# command charged of generating (for now)
# author: 37b7
# created: 26 Nov 2024

# Import =========================================================

from itertools import combinations

# ===============================================================

def generate_layer_configurations(layer_config, output_size):
    layer_configs = []
    for num_layers in range(2, len(layer_config) + 1):
        for config in combinations(layer_config, num_layers):
            if all(config[i] >= config[i+1] for i in range(len(config) - 1)) and config[-1] == output_size and config[0] == layer_config[0]:
                layer_configs.append(config)
    return layer_configs

def layer_finder(input_size, output_size, hidden_layer_size):
    hidden_layer_size = abs(hidden_layer_size)
    layer_config = [input_size]
    diff = input_size - output_size
    if hidden_layer_size >= diff:
        for i in range(1, diff):
            layer_config.append(input_size - i * diff // diff)
        layer_config.append(output_size)
    else:
        for i in range(1, hidden_layer_size + 1):
            next_layer = input_size + i * (output_size - input_size) // hidden_layer_size
            layer_config.append(next_layer)
        layer_config[-1] = output_size
    print("Generating layer configurations")
    return generate_layer_configurations(layer_config, output_size)