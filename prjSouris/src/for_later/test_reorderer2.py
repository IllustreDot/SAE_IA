def reorder_and_include_disabled(current_output, current_order, desired_order):
    # Create a mapping from behaviors in current_order to their indices in current_output
    behavior_to_index = {behavior: i for i, behavior in enumerate(current_order)}
    
    # Reorder each row in current_output
    reordered_output = [
        [
            row[behavior_to_index[behavior]] if behavior in behavior_to_index else 0
            for behavior in desired_order
        ]
        for row in current_output
    ]
    
    return reordered_output

current_output = [[0, 0, 0, 0, 1, 0, 0, 0, 0],[0, 0, 0, 0, 1, 0, 0, 0, 0]]
current_order = ['scratching', 'hind paw licking', 'locomotion', 'still', 'body grooming', 'face grooming', 'rearing', 'wall rearing', 'jump']
desired_order = ['body grooming', 'face grooming', 'hind paw licking', 'jump', 'locomotion', 'rearing', 'scratching', 'still', 'wall rearing']
reordered_output = reorder_and_include_disabled(current_output, current_order, desired_order)
print(reordered_output)