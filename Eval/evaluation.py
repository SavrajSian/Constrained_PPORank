from __future__ import print_function
import torch
import argparse
import torch.nn as nn
import numpy as np
import utils
from utils import AverageMeter, TqdmLoggingHandler
import Reward_utils
import os
import torch.optim as optim
import pickle
import pandas as pd

def validate(agent, test_loader, args, device, drug_idx_to_original_map=None, rollouts=None, update_lambda=False):
    # losses=AverageMeter()
    # model.eval()
    ndcg_all = []
    paths_rewards = []
    best_paths_rewards = []
    preds = []
    trues = []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            input, true_scores = batch
            input_var = input.clone().detach().requires_grad_(False).float().to(device) #added .float()
            if args.algo == 'ppo':
                output, _ = agent.actor_critic(input_var)  # .squeeze().type(torch.DoubleTensor)
                output = output.squeeze()
                del input_var
            elif args.algo == 'pg':
                output = agent.policy_net(input_var).squeeze().type(torch.DoubleTensor)

            preds.append(output.cpu().squeeze().detach().numpy())
            trues.append(true_scores.numpy())
            ndcg_val = Reward_utils.ndcg(
                true_scores.numpy(),
                output.cpu().squeeze().detach().numpy(),
                args.k, full_rank=args.full)
            ndcg_all.append(ndcg_val)
            rewards = Reward_utils.dcg_general(true_scores.numpy(), output.cpu().squeeze(
            ).detach().numpy().reshape(true_scores.size()[0], -1), args.k, full_rank=args.full)
            paths_rewards.append(np.nanmean(rewards))
            best_rewards = Reward_utils.dcg_general(
                true_scores.numpy(),
                true_scores.numpy(),
                args.k, full_rank=args.full)
            best_paths_rewards.append(np.nanmean(best_rewards))
        ndcg_all = np.asarray(ndcg_all, dtype=np.float32)
        preds_test = np.concatenate(preds)
        trues_test = np.concatenate(trues)
        test_rewards = np.asarray(paths_rewards, dtype=np.float32).mean()
        best_rewards = np.asarray(best_paths_rewards, dtype=np.float32).mean()

        if update_lambda: #

            # get order of drugs
            preds_test_mean = pd.DataFrame(preds_test)
            preds_test_mean = preds_test_mean.mean().reset_index()
            preds_test_mean.columns = ['drug', 'mean_pred']
            preds_test_mean = preds_test_mean.sort_values(by='mean_pred', ascending=False)
            preds_order = preds_test_mean['drug'].values

            # calculate severity scores/penalties
            severity_scores = rollouts.severities
            indexed_severity_scores = severity_scores[preds_order]
            ranks = torch.arange(1, len(preds_order) + 1, dtype=torch.float)
            penalties2 = indexed_severity_scores / ranks
            total_penalty = penalties2.sum()
            total_pen_norm = total_penalty / len(preds_order)

            # calculate lagrange penalty
            lagrange_pen = rollouts.lagrange_lambda * total_pen_norm
            # update lambda
            #rollouts.lagrange_lambda = max(0, rollouts.lagrange_lambda + rollouts.lambda_lr * lagrange_pen)
            rollouts.lagrange_lambda = rollouts.lagrange_lambda + rollouts.lambda_lr * lagrange_pen # allows negative lambda
            print('Updated lambda:', rollouts.lagrange_lambda)

            '''
            pickledata = {
                'ndcg_all': ndcg_all,
                'paths_rewards': paths_rewards,
                'best_paths_rewards': best_paths_rewards,
                'test_rewards': test_rewards,
                'best_rewards': best_rewards,
                'preds_test': preds_test,
                'trues_test': trues_test
            }
            with open('/Users/savrajsian/Desktop/evalpickle.pkl', 'wb') as f:
                pickle.dump(pickledata, f)
            '''
    del preds
    del trues

    return np.nanmean(ndcg_all), test_rewards, best_rewards, preds_test


def evalDNN(model, data_loader, device, args, algo="dnn", resume_file=None):
    if resume_file and os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(resume_file))
        checkpoint = torch.load(args.resume)
        best_ndcg = checkpoint['best_ndcg']
        model.load_state_dict(checkpoint['state_dict'])
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        #optimizer = optim.Adagrad(parameters, lr=0.001)
        optimizer = optim.Adam(parameters, lr=4e-5)
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume_file, checkpoint['epoch']))
    else:
        ndcg_all = []
        paths_rewards = []
        best_paths_rewards = []
        preds = []
        approx_ndcgs = []
        MSE = []
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                input, true_scores = batch
                input_var = input.clone().detach().float().to(device) #added .float()
                output = model(input_var).squeeze()
                pred_scores = output.cpu().detach().numpy()
                preds.append(pred_scores)

                MSE.append(Reward_utils.MSEloss(output, true_scores.float().to(device)).item()) #added .float()

                ndcg_val = Reward_utils.ndcg(true_scores.numpy(), pred_scores, args.k, full_rank=args.full)
                # print(ndcg_val)
                ndcg_all.append(ndcg_val)

                approx_ndcg = Reward_utils.approxNDCGLoss(output, true_scores.float().to(device), device) #added .float()
                approx_ndcgs.append(-approx_ndcg.item())
                # rewards is array
                rewards = Reward_utils.dcg_general(true_scores.numpy(), pred_scores.reshape(
                    true_scores.size()[0], -1), args.k, full_rank=args.full)
                paths_rewards.append(rewards)
                best_rewards = Reward_utils.dcg_general(
                    true_scores.numpy(),
                    true_scores.numpy(),
                    args.k, full_rank=args.full)
                best_paths_rewards.append(best_rewards)

            ndcg_all = np.asarray(ndcg_all, dtype=np.float32)
            preds_test = np.concatenate(preds)

            test_rewards = np.asarray(paths_rewards, dtype=np.float32).mean()
            best_rewards = np.asarray(best_paths_rewards, dtype=np.float32).mean()
            test_approx_ndcg = np.asarray(approx_ndcgs, dtype=np.float32).mean()
            test_mse = np.asarray(MSE, dtype=np.float32).mean()
    return ndcg_all, test_rewards, test_approx_ndcg, best_rewards, preds_test, test_mse


def evaluation(agent, data_loader, device, resume_file, algo, args):
    checkpoint = torch.load(resume_file)
    best_ndcg = checkpoint['best_ndcg']
    agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
    agent.value_net.load_state_dict(checkpoint['value_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer'])
    preds = []
    ndcg_all = []
    paths_rewards = []
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            input, true_scores = batch
            input_var = input.clone().detach().requires_grad_(False).float().to(device) #added .float()
            if algo == 'ppo':
                output = agent.actor_critic(input_var).squeeze().type(torch.DoubleTensor)
            elif algo == 'pg':
                output = agent.policy_net(input_var).squeeze().type(torch.DoubleTensor)
            preds.append(output.cpu().squeeze().detach().numpy())
            ndcg_val = Reward_utils.ndcg(
                true_scores.numpy(),
                output.cpu().squeeze().detach().numpy(),
                args.k, full_rank=args.full)
            ndcg_all.append(ndcg_val)
            rewards = Reward_utils.dcg_general(
                true_scores.numpy(),
                output.cpu().squeeze().detach().numpy(),
                args.k, full_rank=args.full)
            paths_rewards.append(rewards.sum())
        ndcg_all = np.asarray(ndcg_all, dtype=np.float32)
        preds_test = np.concatenate(preds)
        test_rewards = np.asarray(paths_rewards, dtype=np.float32).sum()
    return np.mean(ndcg_all), test_rewards, preds_test
