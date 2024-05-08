import numpy as np
import os
import pandas as pd
import pickle
from torch import nn
import torch

with open('/Users/savrajsian/Desktop/evalpickle.pkl', 'rb') as f:
    data = pickle.load(f)

best_paths_rewards = data['best_paths_rewards']
best_rewards = data['best_rewards']
ndcg_all = data['ndcg_all']
paths_rewards = data['paths_rewards']
preds_test = data['preds_test']
test_rewards = data['test_rewards']
trues_test = data['trues_test']

#take mean of each col in preds_test
preds_test_mean = pd.DataFrame(preds_test)
preds_test_mean = preds_test_mean.mean().reset_index()
preds_test_mean.columns = ['drug', 'mean_pred']
preds_test_mean = preds_test_mean.sort_values(by='mean_pred', ascending=False)

dimension = len(preds_test.shape)-1

preds_test = torch.from_numpy(preds_test)

probs = nn.functional.log_softmax(preds_test, dim=dimension).exp()

#take mean of each col in probs
probs_mean = pd.DataFrame(probs.numpy())
probs_mean = probs_mean.mean().reset_index()
probs_mean.columns = ['drug', 'mean_prob']
probs_mean = probs_mean.sort_values(by='mean_prob', ascending=False)

trues_test = torch.from_numpy(trues_test)
trues_test_mean = pd.DataFrame(trues_test)
trues_test_mean = (trues_test_mean.mean().reset_index())
trues_test_mean.columns = ['drug', 'mean_true']


#can use preds_test_mean to get order of drugs
preds_order = preds_test_mean['drug'].values

print('end')