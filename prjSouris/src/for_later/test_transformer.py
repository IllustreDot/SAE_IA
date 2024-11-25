import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np

# Path for the data
sys.path.append("../rsc/config")
from allvariable import *

# Load data from the file
def LoadDataResultat():
    data = pd.read_csv(path_to_data_clean + "scratching/" + file_name_data)
    data = data.sample(n=1000, random_state=6)
    return data

def collect_bodypart_coordinates(data):
    bodypart = {
        "nose": [],
        "forepaw_R": [],
        "forepaw_L": [],
        "hindpaw_R": [],
        "hindpaw_L": [],
        "tailbase": []
    }
    for i in range(0, len(data.columns), 2):
        bodypart_name = data.columns[i][:-2]
        bodypart[bodypart_name] = [data.iloc[:, i], data.iloc[:, i+1]]
    return bodypart

def translate_coordinates(bodypart):
    translated_bodypart = {}
    for part in bodypart:
        nose_x = bodypart['nose'][0]
        nose_y = bodypart['nose'][1]
        translated_x = bodypart[part][0] - nose_x
        translated_y = bodypart[part][1] - nose_y
        translated_bodypart[part] = [translated_x, translated_y]
    return translated_bodypart

def rotate_single_row(row_data):
    rotated_row = {}
    nose_x, nose_y = row_data["nose"]
    tail_x, tail_y = row_data["tailbase"]
    delta_x = tail_x - nose_x
    delta_y = tail_y - nose_y
    angle_to_vertical_axis = np.arctan2(delta_y, delta_x)
    angle = np.pi / 2 + angle_to_vertical_axis
    for part, (x, y) in row_data.items():
        if part == "nose":
            rotated_row[part] = (x, y)
        else:
            rotated_x = (x - nose_x) * np.cos(-angle) - (y - nose_y) * np.sin(-angle) + nose_x
            rotated_y = (x - nose_x) * np.sin(-angle) + (y - nose_y) * np.cos(-angle) + nose_y
            rotated_row[part] = (rotated_x, rotated_y)
    return rotated_row

def rotate_body(bodypart):
    rotated_bodypart = {}
    for i in range(len(bodypart['nose'][0])):
        row_data = {part: (bodypart[part][0].iloc[i], bodypart[part][1].iloc[i]) for part in bodypart}
        rotated_row = rotate_single_row(row_data)
        for part, (x, y) in rotated_row.items():
            if part not in rotated_bodypart:
                rotated_bodypart[part] = [[], []]
            rotated_bodypart[part][0].append(x)
            rotated_bodypart[part][1].append(y)
    return rotated_bodypart

# Example usage
data = LoadDataResultat()  # Load the sample data
bodypart = collect_bodypart_coordinates(data)  # Collect body part coordinates
translated_bodypart = translate_coordinates(bodypart)  # Apply translation
rotated_bodypart = rotate_body(translated_bodypart)  # Apply rotation

# Visualization of the translated data
plt.scatter(rotated_bodypart['nose'][0], rotated_bodypart['nose'][1], label='Nose', color='red')
plt.scatter(rotated_bodypart['forepaw_R'][0], rotated_bodypart['forepaw_R'][1], label='Forepaw Right', color='blue')
plt.scatter(rotated_bodypart['forepaw_L'][0], rotated_bodypart['forepaw_L'][1], label='Forepaw Left', color='green')
plt.scatter(rotated_bodypart['hindpaw_R'][0], rotated_bodypart['hindpaw_R'][1], label='Hindpaw Right', color='orange')
plt.scatter(rotated_bodypart['hindpaw_L'][0], rotated_bodypart['hindpaw_L'][1], label='Hindpaw Left', color='purple')
plt.scatter(rotated_bodypart['tailbase'][0], rotated_bodypart['tailbase'][1], label='Tailbase', color='brown')

plt.legend()
plt.xlabel('X Coordinates')
plt.ylabel('Y Coordinates')
plt.title('Sample Rotated Body Part Coordinates')
plt.show()


plt.scatter(translated_bodypart['nose'][0], translated_bodypart['nose'][1], label='Nose', color='red')
plt.scatter(translated_bodypart['forepaw_R'][0], translated_bodypart['forepaw_R'][1], label='Forepaw Right', color='blue')
plt.scatter(translated_bodypart['forepaw_L'][0], translated_bodypart['forepaw_L'][1], label='Forepaw Left', color='green')
plt.scatter(translated_bodypart['hindpaw_R'][0], translated_bodypart['hindpaw_R'][1], label='Hindpaw Right', color='orange')
plt.scatter(translated_bodypart['hindpaw_L'][0], translated_bodypart['hindpaw_L'][1], label='Hindpaw Left', color='purple')
plt.scatter(translated_bodypart['tailbase'][0], translated_bodypart['tailbase'][1], label='Tailbase', color='brown')

plt.legend()
plt.xlabel('X Coordinates')
plt.ylabel('Y Coordinates')
plt.title('Sample Translated Body Part Coordinates')
plt.show()

# working