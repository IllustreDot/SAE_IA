import pandas as pd

def merge_behavior_classes(classification_file, model_behaviors_to_merge):
    merged_data = classification_file.copy()
    for merged_behavior, behaviors_to_merge in model_behaviors_to_merge.items():
        relevant_columns = [col for col in behaviors_to_merge if col in classification_file.columns]
        merged_data[merged_behavior] = classification_file[relevant_columns].apply(lambda row: 1 if row.any() else 0, axis=1)
        merged_data.drop(columns=relevant_columns, inplace=True)

    return merged_data

df = pd.DataFrame({
    'body grooming': [1],
    'face grooming': [0],
    'hind paw licking': [0],
    'jump': [0],
    'locomotion': [0],
    'rearing': [0],
    'scratching': [0],
    'still': [0],
    'wall rearing': [0]
})

model_behaviors_to_merge = {
    'grooming': ['body grooming', 'face grooming'],
    'rearing': ['rearing', 'wall rearing', 'jump']
}

merged_df = merge_behavior_classes(df, model_behaviors_to_merge)
print(merged_df)

# working
