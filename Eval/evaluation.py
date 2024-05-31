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
import torch.distributed as distributed
from torch.distributed import ReduceOp
import torchsort


def validate(agent, test_loader, args, device, drug_idx_to_original_map=None, rollouts=None, update_lambda=False, train=False):
    # losses=AverageMeter()
    # agent.actor_critic.eval()
    ndcg_all = []
    paths_rewards = []
    best_paths_rewards = []
    preds = []
    trues = []
    dist = args.distributed
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            input, true_scores = batch
            input_var = input.clone().detach().requires_grad_(False).to(device)  # added .float()
            if args.algo == 'ppo':
                output, _ = agent.actor_critic(input_var)  # .squeeze().type(torch.DoubleTensor)
                output = output.squeeze()
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

    mean_preds = preds_test.mean(axis=0)
    preds_order = np.argsort(mean_preds)[::-1].copy()

    # calculate severity scores/penalties
    severity_scores = rollouts.severities
    indexed_severity_scores = severity_scores[preds_order]

    indices = torch.arange(1, len(preds_order) + 1, dtype=torch.double, device=device)
    log_pens = indexed_severity_scores * (-1 * torch.log(indices) + 4)
    total_log_pen = log_pens.sum()
    total_log_pen_norm = total_log_pen / len(preds_order)  # number of drugs isnt always the same
    total_log_pen_norm = total_log_pen_norm
    target_pen = rollouts.target_pen
    pen_violation = total_log_pen_norm - target_pen
    if train:
        print('train pen violation:', pen_violation)
    else:
        print('test pen violation:', pen_violation)

    return np.nanmean(ndcg_all), test_rewards, best_rewards, preds_test


def validate_train(agent, test_loader, args, device, drug_idx_to_original_map=None, rollouts=None, update_lambda=False):
    # losses=AverageMeter()
    # agent.actor_critic.eval()
    ndcg_all = []
    paths_rewards = []
    best_paths_rewards = []
    preds = []
    trues = []
    dist = args.distributed
    for i, batch in enumerate(test_loader):
        input, true_scores = batch
        input_var = input.clone().detach().to(device)  # added .float()
        if args.algo == 'ppo':
            output, _ = agent.actor_critic(input_var)  # .squeeze().type(torch.DoubleTensor)
            output = output.squeeze()
        elif args.algo == 'pg':
            output = agent.policy_net(input_var).squeeze().type(torch.DoubleTensor)

        preds.append(output.squeeze())
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
    preds_test = torch.cat(preds)
    trues_test = np.concatenate(trues)
    test_rewards = np.asarray(paths_rewards, dtype=np.float32).mean()
    best_rewards = np.asarray(best_paths_rewards, dtype=np.float32).mean()

    # get order of drugs

    mean_preds = preds_test.mean(dim=0)
    #preds_order = torch.argsort(mean_preds, descending=True)
    mean_preds = mean_preds.cpu()
    preds_order = torchsort.soft_rank(mean_preds, regularization_strength=0.01)
    preds_order = torch.abs(preds_order - 39) # reverse the order - soft_rank does lowest to highest
    preds_order = preds_order.to(device)


    # calculate severity scores/penalties
    severity_scores = rollouts.severities
    severity_scores = severity_scores.double()
    severity_scores_expanded = severity_scores.expand(preds_order.shape[0], -1)
    severities_all = severity_scores_expanded * (-1 * torch.log(preds_order) + 4)

    severities_mean = severities_all.mean(dim=1)
    total_severity = severities_mean.sum()
    total_log_pen_norm = total_severity / len(severities_mean)
    total_log_pen_norm = total_log_pen_norm
    target_pen = rollouts.target_pen
    pen_violation = total_log_pen_norm - target_pen

    print('Total log pen norm:', total_log_pen_norm.item())
    print('Penalty violation:', pen_violation)

    lagrange_multiplier = agent.lagrange_lambda
    #lagrange_multiplier = nn.functional.softplus(lagrange_multiplier)
    print('Lagrange multiplier:', lagrange_multiplier.item())

    loss_pen = -lagrange_multiplier * pen_violation  # penalty_loss = -penalty_param * (cur_cost_ph - cost_lim), loss_penalty = -penalty_param*cost_dev
    # want it with a negative sign when in PPO_Agent its + pen. When -ve here lambda goes up, when +ve here lambda goes down. want lambda to increase as not meeting pen.
    # pytorch aims to decrease the loss, so if loss_pen is +ve, decreasing lambda achieves this, if loss_pen is -ve, increasing lambda achieves this.
    # therefore want grad ascent on lambda. total_loss in PPO_agent needs to be  total_loss = objective + lambda*constraint_violation so when doing backprop w.r.t input,
    # it will aim to decrease the constraint function as it wants to decrease total loss, so changing the inputs to minimise constraint_violation does this
    agent.lagrange_optimizer.zero_grad()
    loss_pen.backward()
    torch.nn.utils.clip_grad_norm_([agent.lagrange_lambda], max_norm=0.01)
    agent.lagrange_optimizer.step()
    print('Updated lambda:', agent.lagrange_lambda.item())
    '''
    If using quadratic/augmented
    if pen_violation > 100:
        agent.rho = agent.rho * 0.8
        print('Updated rho:', agent.rho.item())
    elif pen_violation < 30:
        agent.rho = agent.rho * 1.2
        print('Updated rho:', agent.rho.item())
    '''

    '''
    Using pen vals to update lambda:
    # calculate lagrange penalty
    lagrange_pen = total_log_pen_norm  # had this timesed by lambda before, but if lambda is increasing then this will increase too. Want to see if the penalty itself is increasing.
    recent_lagrange_pens = rollouts.recent_lagrange_pens
    recent_lagrange_pens.append(lagrange_pen)

    if len(recent_lagrange_pens) > 3:
        recent_lagrange_pens.pop(0)
        print('Recent lagrange penalties:', recent_lagrange_pens)

    if len(recent_lagrange_pens) == 3:
        if recent_lagrange_pens[0] >= recent_lagrange_pens[1] >= recent_lagrange_pens[2]:  # if pens are getting smaller
            rollouts.lagrange_lambda = rollouts.lagrange_lambda * 0.8  # decrease lambda
            print('Updated lambda:', rollouts.lagrange_lambda)

        elif recent_lagrange_pens[0] <= recent_lagrange_pens[1] <= recent_lagrange_pens[2]:  # if pens are getting bigger
            rollouts.lagrange_lambda = rollouts.lagrange_lambda * 1.2  # increase lambda
            print('Updated lambda:', rollouts.lagrange_lambda)

        else:
            print('No change in lambda, lambda:', rollouts.lagrange_lambda)

    '''

    if dist:
        distributed.all_reduce(rollouts.lagrange_lambda, op=ReduceOp.SUM)  # sum across all processes
        rollouts.lagrange_lambda /= distributed.get_world_size()  # get average lambda - dont think summing is best way to do this (grads are summed though)

    return np.nanmean(ndcg_all), test_rewards, best_rewards, preds_test


def evalDNN(model, data_loader, device, args, algo="dnn", resume_file=None):
    if resume_file and os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(resume_file))
        checkpoint = torch.load(args.resume)
        best_ndcg = checkpoint['best_ndcg']
        model.load_state_dict(checkpoint['state_dict'])
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        # optimizer = optim.Adagrad(parameters, lr=0.001)
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
                input_var = input.clone().detach().float().to(device)  # added .float()
                output = model(input_var).squeeze()
                pred_scores = output.cpu().detach().numpy()
                preds.append(pred_scores)

                MSE.append(Reward_utils.MSEloss(output, true_scores.float().to(device)).item())  # added .float()

                ndcg_val = Reward_utils.ndcg(true_scores.numpy(), pred_scores, args.k, full_rank=args.full)
                # print(ndcg_val)
                ndcg_all.append(ndcg_val)

                approx_ndcg = Reward_utils.approxNDCGLoss(output, true_scores.float().to(device),
                                                          device)  # added .float()
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
    dist = args.distributed

    agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
    agent.value_net.load_state_dict(checkpoint['value_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer'])
    preds = []
    ndcg_all = []
    paths_rewards = []
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            input, true_scores = batch
            input_var = input.clone().detach().requires_grad_(False).to(device)  # added .float()
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
