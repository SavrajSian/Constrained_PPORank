import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import pandas as pd
from models.Policy import *


# python main.py --env-name "Reacher-v2" --algo ppo --use_gae(true) --log-interval 1 --num-steps 2048 --num-processes 1
# --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32
#  --gamma 0.99 --gae-lambda 0.95 --num-env-steps 1000000 --use_linear_lr_decay(true)
# --use-proper-time-limits(true)


class PPO(nn.Module):
    def __init__(self, args, N, M, P, f, WP, drug_embs, device, drug_mean=None, overall_mean=None):
        super(PPO, self).__init__()  # Initialize the superclass (torch.nn.Module)
        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.value_loss_coef = args.value_loss_coef
        self.lr = args.lr
        self.eps = args.eps
        self.max_grad_norm = args.max_grad_norm
        self.use_clipped_value_loss = True
        self.N = N  # no of cell lines
        self.M = M  # no of drugs
        self.gene_dim = P
        self.cell_dim = f
        self.drug_dim = f
        self.total_dim = self.drug_dim + self.cell_dim + 1  # 1 is for cosine similarity
        self.entropy_coef = args.entropy_coef
        self.value_loss_coef = args.value_loss_coef
        self.shared_params = args.shared_params
        if not self.shared_params:
            # PPO_Policy comes from nn.module
            self.actor_critic = PPO_Policy(
                N, M, self.gene_dim, self.drug_dim, drug_embs, self.cell_dim,
                WP, args.nlayers_cross, args.nlayers_deep, args.deep_hidden_sizes, args.deep_out_size,
                args.nlayers_value, args.value_hidden_sizes, device,
                train_cell=args.train_cell, train_drug=args.train_drug,
                drug_mean=drug_mean, overall_mean=overall_mean)

        else:
            self.actor_critic = PPO_Shared_Policy(
                N, M, self.gene_dim, self.drug_dim, drug_embs, self.cell_dim,
                WP, args.nlayers_cross, args.nlayers_deep, args.deep_hidden_sizes, args.deep_out_size,
                args.nlayers_value, args.value_hidden_sizes,
                train_cell=args.train_cell, train_drug=args.train_drug)
        # self.critic_size = self.actor_critic.critic_size

        # self.actor_critic = nn.DataParallel(self.actor_critic)

        # self.parameters = self.actor_critic.parameters
        self.optimizer = optim.AdamW(self.actor_critic.parameters, lr=args.lr, eps=args.eps)
        self.actor_critic = self.actor_critic.float()  # added .float()
        self.actor_critic.to(device)
        print(type(self.actor_critic), " - this should show something to do with DDP")

    def update(self, rollouts, drug_idx_to_original_map, do_constrained, device=None, distributed=False):
        advantages = rollouts.returns - rollouts.value_preds
        advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        losses_epoch = 0
        ppo_update_ind = 0
        if hasattr(rollouts.steps, 'item'):  # i added this, probs safe to remove and just used .item() version
            tot_ppo_update = self.ppo_epoch * (rollouts.steps.item() // self.num_mini_batch)
        else:
            tot_ppo_update = self.ppo_epoch * (rollouts.steps // self.num_mini_batch)
        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(  # gives all the data for the batch needed
                advantages, self.num_mini_batch)

            for sample in data_generator:
                # old_action_log_probs_batch requr_grad false
                obs_actor_batch, filter_masks_batch, actions_batch, value_preds_batch, \
                    return_batch, masks_batch, old_action_log_probs_batch, adv_targ = sample
                # print('actions_batch', actions_batch.cpu().detach().numpy().shape)
                # print('old_action_log_probs_batch', old_action_log_probs_batch.cpu().detach().numpy().shape)
                # print('filter_masks_batch', filter_masks_batch.cpu().detach().numpy().shape)
                # print('obs_actor_batch', obs_actor_batch.cpu().detach().numpy().shape)
                # Reshape to do in a single forward pass for all steps, all requr_grad true
                values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(  # evaluate for this batch
                    obs_actor_batch, filter_masks_batch, actions_batch)
                # print('old_action_log_probs_batch', old_action_log_probs_batch.cpu().detach().numpy().shape)
                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)  # [64,1] #ratio of new to old policy
                surr1 = ratio * adv_targ  # [64,1] . first term in LtCLIP

                cur_lrmult = 1 - ppo_update_ind / tot_ppo_update  # linearly decaying learning rate
                clip_param = self.clip_param * cur_lrmult  # epsilon in LtCLIP
                surr2 = torch.clamp(ratio, 1.0 - clip_param,
                                    1.0 + clip_param) * adv_targ  # 2nd term in LtCLIP - the clip(rt, 1-epsilon, 1+epsilon)
                action_loss = -torch.min(surr1, surr2).mean()  # full LtCLIP

                ppo_update_ind += 1  # increment the ppo update index

                if self.use_clipped_value_loss:  # if using LtVF (clipped value function)
                    value_pred_clipped = value_preds_batch + \
                                         (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                            value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()


                ########## Drug severity stuff ##########

                # Get unique drugs and their inverse indices
                severity_scores = rollouts.severities
                non_zero_mask = ~torch.isclose(action_log_probs.squeeze(), torch.tensor(0.0))
                actions_batch = actions_batch[non_zero_mask]
                action_log_probs = action_log_probs[non_zero_mask]
                unique_actions, inverse_indices = torch.unique(actions_batch, sorted=True, return_inverse=True)

                # Get average log probabilities for each drug
                counts = torch.zeros_like(unique_actions, dtype=torch.float)
                inverse_indices = inverse_indices.squeeze()

                prob_sums = torch.zeros(unique_actions.shape, dtype=torch.float, device=device)

                prob_sums.index_add_(0, inverse_indices, action_log_probs.squeeze())
                counts.index_add_(0, inverse_indices, torch.ones_like(action_log_probs.squeeze()))

                average_log_probs = prob_sums / counts

                # Sort unique drugs by average log probabilities
                sorted_average_probs_indices = torch.argsort(average_log_probs, descending=True)
                sorted_actions = unique_actions[sorted_average_probs_indices]
                sorted_average_probs = average_log_probs[sorted_average_probs_indices]

                indexed_severity_scores = severity_scores[sorted_actions]
                ranks = torch.arange(1, len(sorted_actions) + 1, dtype=torch.float, device=device)
                penalties2 = indexed_severity_scores / ranks
                total_penalty = penalties2.sum()
                total_pen_norm = total_penalty / len(sorted_actions)

                #log version:
                '''
                indices = torch.arange(1, len(sorted_actions) + 1, dtype=torch.float)
                log_pens = indexed_severity_scores * (-torch.log(indices)+4)
                total_log_pen = log_pens.sum()
                total_log_pen_norm = total_log_pen / len(sorted_actions)
                '''

                ########## end of drug severity stuff ##########

                self.optimizer.zero_grad()  # zero the gradients

                lagrange_multiplier = rollouts.lagrange_lambda
                PPO_loss = (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef)  # full surrogate loss func
                if do_constrained: # true by default
                    PPO_lagrange_loss = PPO_loss + lagrange_multiplier * total_pen_norm  # add the lagrange penalty if doing constrained
                else:
                    PPO_lagrange_loss = PPO_loss # if not doing constrained, just use the normal loss

                PPO_lagrange_loss.backward()  # backprop the loss
                # instead of MSE or something like that, this surrogate loss is used to update NN params
                nn.utils.clip_grad_norm_(self.actor_critic.parameters,  # stop gradients from exploding
                                         self.max_grad_norm)
                avg_grad = 0
                count = 0
                for param in self.actor_critic.parameters:
                    if param.grad is not None:
                        avg_grad += param.grad.abs().mean().item()
                        count += 1
                avg_grad /= count
                print(f'Average gradient value: {avg_grad}')

                self.optimizer.step()  # update the weights

                #del PPO_lagrange_loss, PPO_loss, value_losses, value_losses_clipped, value_pred_clipped, ratio, surr1, surr2
                #del values, action_log_probs
                #del obs_actor_batch, filter_masks_batch, actions_batch, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ
                '''
                print('ppo loss', PPO_loss.item())
                max_grad = None

                # Iterate over all parameters and find the maximum gradient
                for param in self.actor_critic.parameters:
                    if param.grad is not None:  # Gradients for some parameters might be None if they are not involved in the loss
                        # torch.max returns a tuple (max_value, max_indices) for multidimensional tensors, we are interested in max_value
                        param_max_grad = torch.max(
                            param.grad.abs()).item()  # Use .abs() to consider the absolute values of the gradients
                        if max_grad is None or param_max_grad > max_grad:
                            max_grad = param_max_grad

                print(f'Maximum gradient value: {max_grad}')
                '''

                ''' Lagrange multiplier update '''
                ''' not optimising a convex function though, so not sure if can do ADMM type thing'''
                #lagrange_lr = 0.5
                #lagrange_multiplier = lagrange_multiplier + lagrange_lr * total_pen2 # or something like that, eg update based on degree which constraint is being satisfied




                '''
                pickledata = {'actions batch': actions_batch.cpu().detach().numpy(),
                              'action log probs': action_log_probs.cpu().detach().numpy(),
                              'obs actor batch': obs_actor_batch.cpu().detach().numpy(),
                              'filter masks batch': filter_masks_batch.cpu().detach().numpy(),
                              'return batch': return_batch.cpu().detach().numpy(),
                              'value preds batch': value_preds_batch.cpu().detach().numpy(),
                              'values': values.cpu().detach().numpy(),
                              'masks batch': masks_batch.cpu().detach().numpy(),
                              'old action log probs batch': old_action_log_probs_batch.cpu().detach().numpy(),
                              }

                with open('/Users/savrajsian/Desktop/pickle.pkl', 'wb') as f:
                    pickle.dump(pickledata, f)
                '''

                '''
                Use actions_batch and action_log_probs to get what policy thinks is best?
                From here, can see what policy thinks is the best drugs - do some sort of ordering from this?
                
                Or can run an evaluation on the policy to see what it thinks is the best drugs?
                
                Not sure which is best.
                '''


                value_loss_epoch += value_loss.item()  # add to the value loss
                action_loss_epoch += action_loss.item()  # add to the action loss
                dist_entropy_epoch += dist_entropy.item()  # add to the entropy loss
                losses_epoch += value_loss.item() * self.value_loss_coef + action_loss.item() - dist_entropy.item() * self.entropy_coef  # add to the total loss

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        losses_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, losses_epoch
