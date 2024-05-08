### old severity stuff:
import numpy as np

severity_scores = rollouts.severities

#action log probs and batch shapes are [-1,1]

action_log_probs_list = action_log_probs.cpu().detach().numpy().flatten()
actions_batch_list = actions_batch.cpu().detach().numpy().flatten()
zeros = np.isclose(action_log_probs_list, 0)
zeroindices = np.where(zeros)
zeroactions = actions_batch[zeroindices]

# remove zero prob actions from probs and batch
action_log_probs_list = np.delete(action_log_probs_list, zeroindices)
actions_batch_list = np.delete(actions_batch_list, zeroindices)

# get rankings of drugs
rankings = np.argsort(action_log_probs_list)[::-1]
ordered_actions = actions_batch_list[rankings]

# get average prob of each drug
action_prob_dict = {}
for drug, prob in zip(actions_batch_list, action_log_probs_list):
    if drug in action_prob_dict:
        action_prob_dict[drug].append(prob)
    else:
        action_prob_dict[drug] = [prob]

average_probs = {action: np.mean(probs) for action, probs in action_prob_dict.items()}
sorted_avg_probs = {k: v for k, v in
                    sorted(average_probs.items(), key=lambda item: item[1], reverse=True)}

ranking_arr = np.array(list(sorted_avg_probs.keys()))

#convert to original drug ids
original_idx_ranking = [drug_idx_to_original_map[drug] for drug in ranking_arr]
severity_scores = pd.read_csv('GDSC_ALL/severity_scores.csv')

ranked_severity_dict = {drug: severity_scores[severity_scores['drug_id'] == int(drug)]['normalised'].values[0]
                 for drug in original_idx_ranking}

penalties = [(severity / (rank + 1)) for rank, (drug, severity) in
             enumerate(ranked_severity_dict.items())]  # pen for each drug, gets smaller as rank of drug lowers
total_penalty = sum(penalties)
total_pen2 = total_penalty / len(original_idx_ranking)