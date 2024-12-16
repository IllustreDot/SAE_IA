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

model_behaviors_to_merge = False
# 61% overall accuracy and best in scratching accuracy with 93% (wall rearing enabled) overall scratching accuracy 42%
# Overall Accuracy: 58.36% Scratching Accuracy: 61.90% Overall Scratching Accuracy: 26.14% with wall rearing disabled

# model_behaviors_to_merge = {
#   'M_mouvement':['locomotion','still']} 
#Overall Accuracy: 58.93% Scratching Accuracy: 87.64% Overall Scratching Accuracy: 39.48%
# with validator (3) : Overall Accuracy: 55.77% Scratching Accuracy: 95.06% Overall Scratching Accuracy: 43.18%
# with validator (6) : Overall Accuracy: 58.33% Scratching Accuracy: 77.78% Overall Scratching Accuracy: 36.47%

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

# model_behaviors_to_merge = {
#     'M_noscratching':['hind paw licking','face grooming','body grooming','rearing'],
#     'M_mouvement':['locomotion','still']}
# 53% overall accuracy

# model_behaviors_to_merge = {
#     'M_toilettestand':['face grooming','rearing'],
#     'M_toilettage':['hind paw licking','body grooming'],
#     'M_mouvement':['locomotion','still']}
#59% overall accuracy Scratching Accuracy: 16.19% Overall Scratching Accuracy: 12.53%
# with validator (3) :
# with validator (6) : Overall Accuracy: 57.67% Scratching Accuracy: 83.08% Overall Scratching Accuracy: 37.50%

# model_behaviors_to_merge = {
#     'M_toilettestand':['scratching','face grooming','rearing'],
#     'M_toilettage':['hind paw licking','body grooming'],
#     'M_mouvement':['locomotion','still']}
#Overall Accuracy: 45.52% Scratching Accuracy: 0.00% Overall Scratching Accuracy: 0.00% , really promising but need more data and neet the validator structure 

# model_behaviors_to_merge = {
#     'M_bodyrelated':['scratching', 'hind paw licking','body grooming','face grooming'],
#     'M_behavior':['locomotion','still','rearing','wall rearing']}
# 15% overall accuracy 0% scratching accuracy 0% overall scratching accuracy

# model_behaviors_to_merge = {
#     'M_noscratching':['hind paw licking','body grooming','face grooming','locomotion','still','rearing','wall rearing']}
# Overall Accuracy: 14.48% Scratching Accuracy: 11.37% Overall Scratching Accuracy: 10.06%

model_bahaviors_disabled = ["jump"]#,"wall rearing"]

selected_nb_hlayers = 5

learning_rate_init_number = 0.001
alpha_number = 1e-4
max_iter_number = 100
solvertype = 'sgd'
activation_function = 'relu'



process_data = False
# if you want to rewrite the matches file or not
save_matches = False


# for NN resultat Overall Accuracy: 52.33% Scratching Accuracy: 35.44% Overall Scratching Accuracy: 34.89%