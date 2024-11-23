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
    "1": {"1": 12},
    "2": {"1": 10, "2": 8, "3": 6, "4": 4, "5": 2},
    "3": {"1": 8, "2": 6, "3": 4, "4": 2},
    "4": {"1": 6, "2": 4, "3": 2},
    "5": {"1": 4, "2": 2},
    "6": {"1": 2}
}

learning_rate_init_number = 0.001
alpha_number = 1e-4
max_iter_number = 100

choose_gpu = True