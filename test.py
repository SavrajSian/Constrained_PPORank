import numpy as np
import os
import pandas as pd
import pickle
import torch
##### print npz to see whats in it #######################################################################
'''
currentdir = os.path.dirname(os.path.realpath(__file__))
npz_path = os.path.join(currentdir, 'preprocess', 'GDSC_ALL')

#read npz file
data = np.load(npz_path + '/GDSC_GEX.npz')

names = {'X': 'GEX', 'Y': 'IC50', 'cell_ids': 'GEX_cell_ids', 'cell_names': 'GEX_cell_names', 'drug_ids': 'IC50_drug_ids', 'drug_names': 'IC50_drug_names', 'GEX_gene_symbols': 'GEX_gene_symbols'}

#read data
for key in data.keys():
    array = data[key]
    df = pd.DataFrame(array)
    print(names[key])
    print(df)
'''
############################################################################################################
'''
ppo_npz_path = os.path.join(os.getcwd(), 'results', 'GDSC_ALL', 'FULL', '100Dim', 'ppo')

#read npz file
data = np.load(ppo_npz_path + '/ppo_0.npz')
Y_true = data['Y_true']
Y_pred = data['Y_pred']

true_df = pd.DataFrame(Y_true)
pred_df = pd.DataFrame(Y_pred)

#print(true_df)
#print(pred_df)

true_path = os.path.join(ppo_npz_path, 'true_df.csv')
pred_path = os.path.join(ppo_npz_path, 'pred_df.csv')
#write to csvs
true_df.to_csv(true_path)
pred_df.to_csv(pred_path)

gdsc_gex = np.load('preprocess/GDSC_ALL/GDSC_GEX.npz')
gdsc_x = gdsc_gex['X']
gdsc_y = gdsc_gex['Y']
gdsc_cell_ids = gdsc_gex['cell_ids']
gdsc_cell_names = gdsc_gex['cell_names']
gdsc_drug_ids = gdsc_gex['drug_ids']
gdsc_drug_names = gdsc_gex['drug_names']
gdsc_gene_symbols = gdsc_gex['GEX_gene_symbols']

gdsc_x_df = pd.DataFrame(gdsc_x)
gdsc_y_df = pd.DataFrame(gdsc_y)
gdsc_cell_ids_df = pd.DataFrame(gdsc_cell_ids)
gdsc_cell_names_df = pd.DataFrame(gdsc_cell_names)
gdsc_drug_ids_df = pd.DataFrame(gdsc_drug_ids)
gdsc_drug_names_df = pd.DataFrame(gdsc_drug_names)
gdsc_gene_symbols_df = pd.DataFrame(gdsc_gene_symbols)

#create temp folder
if not os.path.exists('temp'):
    os.makedirs('temp')
temppath = os.path.join(os.getcwd(), 'temp')
gdsc_x_df.to_csv(os.path.join(temppath, 'gdsc_x_df.csv')) #gene expression, row = cell, cols = gene expression data
gdsc_y_df.to_csv(os.path.join(temppath, 'gdsc_y_df.csv')) #drug response, row = cell, cols = drug response data (IC50)
gdsc_cell_ids_df.to_csv(os.path.join(temppath, 'gdsc_cell_ids_df.csv'))
gdsc_cell_names_df.to_csv(os.path.join(temppath, 'gdsc_cell_names_df.csv'))
gdsc_drug_ids_df.to_csv(os.path.join(temppath, 'gdsc_drug_ids_df.csv'))
gdsc_drug_names_df.to_csv(os.path.join(temppath, 'gdsc_drug_names_df.csv'))
gdsc_gene_symbols_df.to_csv(os.path.join(temppath, 'gdsc_gene_symbols_df.csv'))
'''




'''
check = np.load('/Users/savrajsian/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Documents/GitHub/PPORank_FYP/results/GDSC_ALL/FULL/100Dim/CaDRRes/CaDRRes_3.npz')
y_true = check['Y_true']
y_pred = check['Y_pred']

y_true_df = pd.DataFrame(y_true)
y_pred_df = pd.DataFrame(y_pred)

print('yoyooy')
'''

with open('/Users/savrajsian/Desktop/pickle.pkl', 'rb') as f:
    data = pickle.load(f)

action_log_probs = data['action log probs']
actions_batch = data['actions batch']
filter_masks = data['filter masks batch']
old_action_log_probs = data['old action log probs batch']
old_action_log_probs = old_action_log_probs.flatten()
# Get the rankings of the drugs
action_log_probs = action_log_probs.flatten()
actions_batch = actions_batch.flatten()

zeros = np.isclose(action_log_probs, 0)
zeroindices = np.where(zeros)
zeroactions = actions_batch[zeroindices]

#remove zero prob actions from probs and batch
action_log_probs = np.delete(action_log_probs, zeroindices)
actions_batch = np.delete(actions_batch, zeroindices)

rankings = np.argsort(action_log_probs)[::-1]
ordered_actions = actions_batch[rankings]

# get count of each action
unique = np.unique(ordered_actions)
action_counts = dict(zip(unique, range(len(unique)))) #combine into dict for formatting for avg position calc

# avg position of each drug in ordered_actions
action_positions = {}
for action in action_counts:
    action_positions[action] = np.where(ordered_actions == action)[0].mean()

sorted_action_positions = {k: v for k, v in sorted(action_positions.items(), key=lambda item: item[1])} #this is ordered but doesnt show that way in debug
print(sorted_action_positions)

action_prob_dict = {}
for action_id, prob in zip(actions_batch, action_log_probs):
    if action_id in action_prob_dict:
        action_prob_dict[action_id].append(prob)
    else:
        action_prob_dict[action_id] = [prob]

average_probs = {action_id: np.mean(probs) for action_id, probs in action_prob_dict.items()}
sorted_avg_probs = {k: v for k, v in sorted(average_probs.items(), key=lambda item: item[1], reverse=True)}
print(sorted_avg_probs)
arr = np.array(list(sorted_avg_probs.keys()))

severity_scores = pd.read_csv('GDSC_ALL/severity_scores.csv')

map = {0: '1032', 1: '281', 2: '1021', 3: '186', 4: '150', 5: '190', 6: '1019', 7: '249', 8: '1005', 9: '37', 10: '1006', 11: '1373', 12: '51', 13: '1', 14: '134', 15: '1010', 16: '238', 17: '34', 18: '119', 19: '1020', 20: '1008', 21: '1013', 22: '1017', 23: '1054', 24: '199', 25: '155', 26: '71', 27: '1175', 28: '206', 29: '30', 30: '5', 31: '1259', 32: '1199', 33: '1375', 34: '1372', 35: '1009', 36: '1033', 37: '1012'}

original_idx_ranking = [map[drug] for drug in arr]

severity_dict = {drug: severity_scores[severity_scores['drug_id'] == int(drug)]['normalised'].values[0] for drug in original_idx_ranking}
print(severity_dict)

penalties = [(severity / (rank + 1)) for rank, (drug, severity) in enumerate(severity_dict.items())] #pen for each drug, gets smaller as rank of drug lowers
total_penalty = sum(penalties)
t2 = total_penalty/len(arr)
# Print the rankings
print('Drug rankings:', ordered_actions)
print('breakpoint')


