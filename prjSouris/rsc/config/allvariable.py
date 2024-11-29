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
file_name_matches = "matches.csv"

# kind is the list of comportement of the mouse we will train the AI on
selector = "scratching"

# model_behaviors_to_merge = {
#     'M_scratching':['scratching', 'hind paw licking'],
#     'M_mouvement':['locomotion','still'],
#     'M_grooming':['body grooming','face grooming'],
#     'M_rearing':['rearing','wall rearing']} 
# 56% overall accuracy

# model_behaviors_to_merge = {
#     'M_scratching':['scratching', 'hind paw licking','body grooming','face grooming'],
#     'M_mouvement':['locomotion','still'],
#     'M_rearing':['rearing','wall rearing']}
# 41% overall accuracy

model_behaviors_to_merge = {
    'M_scratching':['scratching','body grooming'],
    'M_noscratching':['hind paw licking','face grooming','rearing'],
    'M_mouvement':['locomotion','still']}

# because wall rearing is desactivated
# model_behaviors_to_merge = {
#     'M_scratching':['scratching','body grooming'],
#     'M_noscratching':['hind paw licking','face grooming'],
#     'M_mouvement':['locomotion','still']}

# model_behaviors_to_merge = {
#     'M_scratching':['scratching', 'hind paw licking','body grooming','face grooming'],
#     'M_noscratching':['locomotion','still','rearing','wall rearing']}

# model_behaviors_to_merge = {
#     'M_noscratching':['hind paw licking','body grooming','face grooming','locomotion','still','rearing','wall rearing']}

model_bahaviors_disabled = ["jump","wall rearing"]

selected_nb_hlayers = 5

learning_rate_init_number = 0.001
alpha_number = 1e-4
max_iter_number = 100



process_data = False
# if you want to rewrite the matches file or not
save_matches = False
