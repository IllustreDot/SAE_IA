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
file_name_data_ia_output = "ia_output.csv"
file_name_data_output = "analitics.csv"
dile_name_config_done = "done.csv" 
file_name_data = "data.csv"

path_to_config + dile_name_config_done

# kind is the list of comportement of the mouse we will train the AI on
selector = "scratching"

model_behaviors_to_merge = {'M_NoScratching':['hind paw licking','locomotion','body grooming','still','face grooming','rearing','wall rearing']}

model_bahaviors_disabled = ["jump"]

selected_nb_hlayers = 5

learning_rate_init_number = 0.001
alpha_number = 1e-4
max_iter_number = 100

# if you want to choose the gpu
choose_gpu = False

# if you want to rewrite the matches file or not
save_matches = False