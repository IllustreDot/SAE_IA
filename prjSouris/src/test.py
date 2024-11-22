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
    data = data.sample(n=1, random_state=42)
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
    for i in range(1, len(data.columns), 2):
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

def rotate_body(bodypart):
    rotated_bodypart = {}

    # Iterate over the range of nose data points
    for i in range(len(bodypart["nose"][0])):  # Assuming bodypart["nose"][0] is a list or series of X coordinates
        
        # Get the x, y coordinates for tailbase and nose at index i
        tail_x, tail_y = bodypart["tailbase"][0].iloc[i], bodypart["tailbase"][1].iloc[i]
        nose_x, nose_y = bodypart["nose"][0].iloc[i], bodypart["nose"][1].iloc[i]
        
        # Calculate the differences in position
        delta_x = tail_x - nose_x
        delta_y = tail_y - nose_y
        
        # Calculate the angle from the nose to the tailbase
        angle_to_vertical_axis = np.arctan2(delta_y, delta_x)
        print(f"Angle to vertical axis for point {i}: {angle_to_vertical_axis}")

        # Determine the angle of rotation based on the quadrant
        if delta_x > 0 and delta_y > 0:  # 1st quadrant
            angle = np.pi / 2 - angle_to_vertical_axis
        elif delta_x < 0 and delta_y > 0:  # 2nd quadrant
            angle = np.pi/2 + angle_to_vertical_axis
        elif delta_x < 0 and delta_y < 0:  # 3rd quadrant
            angle = -3 * np.pi / 2 - angle_to_vertical_axis
        elif delta_x > 0 and delta_y < 0:  # 4th quadrant
            angle = 3 * np.pi / 2 - angle_to_vertical_axis
        
        print(f"Calculated rotation angle: {angle}")

        # Rotate all body parts for the current index i
        for part in bodypart.keys():
            if part == 'nose':
                rotated_bodypart[part] = bodypart[part]  # Nose remains unchanged
                continue

            # Get the x, y coordinates for the current body part
            x_coords = bodypart[part][0].iloc[i]
            y_coords = bodypart[part][1].iloc[i]

            # Apply the rotation matrix to the current body part
            rotated_x = (x_coords - nose_x) * np.cos(-angle) - (y_coords - nose_y) * np.sin(-angle) + nose_x
            rotated_y = (x_coords - nose_x) * np.sin(-angle) + (y_coords - nose_y) * np.cos(-angle) + nose_y
            
            # Store the rotated coordinates for the current part
            if part not in rotated_bodypart:
                rotated_bodypart[part] = [[], []]
            rotated_bodypart[part][0].append(rotated_x)
            rotated_bodypart[part][1].append(rotated_y)
    
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


# plt.scatter(translated_bodypart['nose'][0], translated_bodypart['nose'][1], label='Nose', color='red')
# plt.scatter(translated_bodypart['forepaw_R'][0], translated_bodypart['forepaw_R'][1], label='Forepaw Right', color='blue')
# plt.scatter(translated_bodypart['forepaw_L'][0], translated_bodypart['forepaw_L'][1], label='Forepaw Left', color='green')
# plt.scatter(translated_bodypart['hindpaw_R'][0], translated_bodypart['hindpaw_R'][1], label='Hindpaw Right', color='orange')
# plt.scatter(translated_bodypart['hindpaw_L'][0], translated_bodypart['hindpaw_L'][1], label='Hindpaw Left', color='purple')
# plt.scatter(translated_bodypart['tailbase'][0], translated_bodypart['tailbase'][1], label='Tailbase', color='brown')

# plt.legend()
# plt.xlabel('X Coordinates')
# plt.ylabel('Y Coordinates')
# plt.title('Sample Translated Body Part Coordinates')
# plt.show()
