# config file for all variables
# author: 37b7
# created: 22 Nov 2024

basedirpath = "../rsc/"

path_to_data_classification = basedirpath + "data_classification/"
path_to_data_clean = basedirpath + "data_cleaned/"
path_to_output = basedirpath + "output/"
path_to_config = basedirpath + "config/"
path_to_data = basedirpath + "data/"

file_name_data_classification = "data_classification.csv"
file_name_data_output = "analitics.csv"
dile_name_config_done = "done.csv" 
file_name_data = "data.csv"

# kind is the list of comportement of the mouse we will train the AI on
selector = "scratching"

hl_nb_dict_of_dict = {
    "1": {"1": 19},
    "2": {"1": 15, "2": 12, "3": 9, "4": 6, "5": 3, "6": 2},
    "3": {"1": 15, "2": 12, "3": 9, "4": 6, "5": 3, "6": 2},
    "4": {"1": 15, "2": 12, "3": 9, "4": 6, "5": 3, "6": 2},
    "5": {"1": 15, "2": 12, "3": 9, "4": 6, "5": 3, "6": 2},
    "6": {"1": 15, "2": 12, "3": 9, "4": 6, "5": 3, "6": 2},
}

learning_rate_init_number = 0.001
alpha_number = 1e-4
max_iter_number = 100