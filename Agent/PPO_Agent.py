import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import pandas as pd
import torch.distributed
from torch.distributed import ReduceOp
from models.Policy import *
import os


# python main.py --env-name "Reacher-v2" --algo ppo --use_gae(true) --log-interval 1 --num-steps 2048 --num-processes 1
# --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32
#  --gamma 0.99 --gae-lambda 0.95 --num-env-steps 1000000 --use_linear_lr_decay(true)
# --use-proper-time-limits(true)


class PPO():
    def __init__(self, args, N, M, P, f, WP, drug_embs, device, drug_mean=None, overall_mean=None, dist=False):
        #super(PPO, self).__init__()  # Initialize the superclass (torch.nn.Module)
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
        self.lagrange_lambda = nn.Parameter(torch.tensor(args.lagrange_lambda, dtype=torch.float32, device=device, requires_grad=True))
        self.lagrange_lr = args.lambda_lr
        self.rho = args.rho
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

        if dist:
            local_rank = int(os.environ["LOCAL_RANK"])
            self.actor_critic = self.actor_critic.to(device)
            self.actor_critic = torch.nn.parallel.DistributedDataParallel(self.actor_critic, device_ids=[local_rank], output_device=local_rank)
            self.optimizer = optim.AdamW(self.actor_critic.parameters, lr=args.lr, eps=args.eps)
            print(type(self.actor_critic), " - this should show something to do with DDP")
            params = list(self.actor_critic.module.parameters)
            avg = 0
            for param in params:
                avg += param.abs().mean()
            avg /= len(params)  # Use the list length for division
            print('initial param avg is: ', avg)
        else:
        # self.parameters = self.actor_critic.parameters
            self.optimizer = optim.AdamW(self.actor_critic.parameters, lr=args.lr, eps=args.eps)
            self.lagrange_optimizer = optim.AdamW([self.lagrange_lambda], lr=self.lagrange_lr)
            #self.actor_critic = self.actor_critic.float()  # added .float()
            self.actor_critic.to(device)
            print(type(self.actor_critic), " - this should show just normal nn.module stuff")



    def update(self, rollouts, drug_idx_to_original_map, do_constrained, device=None, distributed=False, args=None):
        advantages = rollouts.returns - rollouts.value_preds
        advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-5) #normalise the advantages

        if args.do_cost_advantages:
            cost_advantages = rollouts.cost_returns - rollouts.value_cost_preds
            cost_advantages = (cost_advantages - cost_advantages.mean()) / (cost_advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        losses_epoch = 0
        ppo_update_ind = 0
        tot_ppo_update = self.ppo_epoch * (rollouts.steps.item() // self.num_mini_batch)
        avg_update_constraint = torch.Tensor([0.0]).to(device)
        curr_updates = 0
        for e in range(self.ppo_epoch): # 4 epochs
            if args.do_cost_advantages:
                data_generator = rollouts.feed_forward_generator(  # gives all the data for the batch needed
                    advantages, cost_advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(  # gives all the data for the batch needed
                    advantages, num_mini_batch=self.num_mini_batch)
            for sample in data_generator: # 4 iterations
                # old_action_log_probs_batch requr_grad false
                obs_actor_batch, filter_masks_batch, actions_batch, value_preds_batch, \
                    return_batch, masks_batch, old_action_log_probs_batch, adv_targ, cadv_targ, pens_batch, cost_returns_batch = sample
                # print('actions_batch', actions_batch.cpu().detach().numpy().shape)
                # print('old_action_log_probs_batch', old_action_log_probs_batch.cpu().detach().numpy().shape)
                # print('filter_masks_batch', filter_masks_batch.cpu().detach().numpy().shape)
                # print('obs_actor_batch', obs_actor_batch.cpu().detach().numpy().shape)
                #print(actions_batch.squeeze())
                #print(pens_batch.squeeze())
                # Reshape to do in a single forward pass for all steps, all requr_grad true
                if distributed:
                    values, action_log_probs, dist_entropy, cost_values = self.actor_critic.module.evaluate_actions(  # evaluate for this batch
                        obs_actor_batch, filter_masks_batch, actions_batch)
                else:
                    values, action_log_probs, dist_entropy, cost_values, actions_with_grad, action_log_probs_extra_grad, sorted_indices = self.actor_critic.evaluate_actions( #all outputs require_grad true
                        obs_actor_batch, filter_masks_batch, actions_batch, pens_batch)
                # print('old_action_log_probs_batch', old_action_log_probs_batch.cpu().detach().numpy().shape)
                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)  # [64,1] #ratio of new to old policy
                surr1 = ratio * adv_targ  # [64,1] . first term in LtCLIP

                cur_lrmult = 1 - ppo_update_ind / tot_ppo_update  # linearly decaying learning rate
                clip_param = self.clip_param * cur_lrmult  # epsilon in LtCLIP

                kl_div_check = args.kl_div_check
                if kl_div_check:
                    kl_divergence = (old_action_log_probs_batch - action_log_probs).mean()
                    kl_div_triggered = False
                    if kl_divergence > 0.08:
                        print('KL divergence triggered', kl_divergence.item())
                        kl_div_triggered = True
                        clip_param *= 0.5
                        curr_lr = self.optimizer.param_groups[0]['lr']
                        temp_lr = curr_lr * 0.5
                        self.optimizer.param_groups[0]['lr'] = temp_lr

                surr2 = torch.clamp(ratio, 1.0 - clip_param,
                                    1.0 + clip_param) * adv_targ  # 2nd term in LtCLIP - the clip(rt, 1-epsilon, 1+epsilon)
                action_loss = -torch.min(surr1, surr2).mean()  # full LtCLIP

                ppo_update_ind += 1  # increment the ppo update index

                if self.use_clipped_value_loss:  # if using LtVF (clipped value function)
                    value_pred_clipped = value_preds_batch + \
                                         (values - value_preds_batch).clamp(-self.clip_param, self.clip_param) #seems to be between 0 and 1
                    value_losses = (values - return_batch).pow(2) #on order of e08
                    if args.do_cost_advantages:
                        cost_value_losses = (cost_values - cost_returns_batch).pow(2).mean()
                    value_losses_clipped = (
                            value_pred_clipped - return_batch).pow(2) #on order of e08
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()
                    if args.do_cost_advantages:
                        cost_value_losses = (cost_values - cost_returns_batch).pow(2).mean()

                if args.do_cost_advantages:
                    constrained_action_loss = (ratio * cadv_targ).mean()
                    constrained_value_loss = cost_value_losses
                    total_constrained_loss = constrained_action_loss + 0.5* constrained_value_loss
                ########## Drug severity stuff ##########

                # Get unique drugs and their inverse indices

                #if do_constrained:
                # computational graph should link back to when drugs were selected as actions in the policy
                # cant use actions batch - need action_log_probs to get the log probs of the actions for proper order according to policy

                '''
                non_zero_mask = ~torch.isclose(action_log_probs.squeeze(), torch.tensor(0.0, dtype=torch.double, device=device))
                actions_batch = actions_batch[non_zero_mask]
                action_log_probs = action_log_probs[non_zero_mask] #requires grad
                unique_actions, inverse_indices = torch.unique(actions_batch, sorted=True, return_inverse=True)
                # Get average log probabilities for each drug
                counts = torch.zeros_like(unique_actions, dtype=torch.double)
                inverse_indices = inverse_indices.squeeze()


                prob_sums = torch.zeros(unique_actions.shape, dtype=torch.double, device=device)

                prob_sums.index_add_(0, inverse_indices, action_log_probs.squeeze())
                counts.index_add_(0, inverse_indices, torch.ones_like(action_log_probs.squeeze()))
                average_log_probs = prob_sums / counts

                # Sort unique drugs by average log probabilities
                sorted_average_probs_indices = torch.argsort(average_log_probs, descending=True) #argsort not differentiable
                sorted_actions = unique_actions[sorted_average_probs_indices]
                sorted_average_probs = average_log_probs[sorted_average_probs_indices]

                indexed_severity_scores = severity_scores[sorted_actions]

                ranks = torch.arange(1, len(sorted_actions) + 1, dtype=torch.double, device=device)
                penalties2 = indexed_severity_scores / ranks
                total_penalty = penalties2.sum()
                total_pen_norm = total_penalty / len(sorted_actions)

                #log version:

                indices = torch.arange(1, len(sorted_actions) + 1, dtype=torch.double, device=device)
                log_pens = indexed_severity_scores * (-10*torch.log(indices)+40)
                total_log_pen = log_pens.sum()
                total_log_pen_norm = total_log_pen / len(sorted_actions) # number of drugs isnt always the same
                total_log_pen_norm = total_log_pen_norm * 100 # scale up to make it more significant. works for reward sf=100
                target_pen = rollouts.target_pen # scale target penalty to match number of drugs
                pen_violation = total_log_pen_norm - target_pen
                print(total_log_pen_norm, pen_violation)
                #avg_update_constraint += pen_violation
                curr_updates += 1
                # sequential augmented penalty/lagrangian:
                #L_a = L + 1/eps ||pen function||^2

                #The update steps should happen in same place current lambda update happens (validation func)
                #step 1: find local min of L_a
                #step 2: update lambda based on local min of L_a: lambda = lambda + 2/eps * pen function
                #step 3: eps = beta * eps. beta=1 if g(x_k+1) < 1/4 g(x_k), beta<1 otherwise

                #sequential_loss = rollouts.lagrange_lambda * total_pen_norm + 1/sequential_eps * torch.pow(total_pen_norm, 2)
                # (sequential_eps will need to be added to rollouts)
                '''

                '''
                lagrange_multiplier = self.lagrange_lambda
                lagrange_multiplier = nn.functional.softplus(lagrange_multiplier)  # softplus to ensure positivity
                print('lagrange multiplier:', lagrange_multiplier.item())
                pen_loss = lagrange_multiplier * pen_violation
                print('pen violation:', pen_violation.item())
                print('pen loss:', pen_loss.item())

                self.lagrange_optimizer.zero_grad()
                pen_loss.backward()
                torch.nn.utils.clip_grad_norm_([self.lagrange_lambda], max_norm=1.0)
                self.lagrange_optimizer.step()

                print('new lambda:', self.lagrange_lambda.item())
                '''
                ########## end of drug severity stuff ##########
                # not using pensbatchnorm
                pens_batch_norm = pens_batch.sum()
                pens_batch_norm = pens_batch_norm / len(pens_batch)
                pens_batch_norm = pens_batch_norm * 100
                pens_batch_violation = pens_batch_norm - rollouts.target_pen  # set as 4000

                #There are 768 updates per epoch.
                #print(pens_batch.requires_grad)
                sorted_indices = torch.abs(sorted_indices - 39) # was in wrong order - was sorted st lowest score was 1, need other way around
                severity_scores = rollouts.severities
                severity_scores = severity_scores.double()
                severity_scores_expanded = severity_scores.expand(sorted_indices.shape[0], -1)
                severities_all = severity_scores_expanded * (-1 * torch.log(sorted_indices) + 4)

                severities_mean = severities_all.mean(dim=0)
                total_severity = severities_mean.sum()
                total_log_pen_norm = total_severity / len(severities_mean)
                total_log_pen_norm = total_log_pen_norm
                target_pen = rollouts.target_pen
                pen_violation = total_log_pen_norm - target_pen
                curr_updates = curr_updates + 1
                if curr_updates != 16:
                    avg_update_constraint = avg_update_constraint + pen_violation.item()
                else:
                    avg_update_constraint = avg_update_constraint + pen_violation
                    avg_update_constraint = avg_update_constraint / curr_updates

                #print(severity_scores_expanded)
                #print(sorted_indices)
                #print(severities_all)

                #print(total_log_pen_norm, pen_violation)

                self.optimizer.zero_grad()  # zero the gradients
                self.lagrange_optimizer.zero_grad()



                # GOTO: storage.py, line 164. scale factor to scale rewards
                PPO_loss = (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef)  # full surrogate loss func
                lagrange_multiplier = self.lagrange_lambda
                #lagrange_multiplier = nn.functional.softplus(lagrange_multiplier)  # softplus to ensure positivity
                lagrange_multiplier = nn.functional.celu(lagrange_multiplier, alpha=0.001)  # celu to ensure positivity
                if do_constrained:

                    #print('PPOLoss:', PPO_loss.item())
                    #print('penalty violation:', pen_violation.item())
                    #PPO_lagrange_loss = PPO_loss + lagrange_multiplier * pen_violation #was - lagrange_multiplier * total_pen_norm in
                    PPO_lagrange_loss = PPO_loss + lagrange_multiplier * pen_violation
                    #PPO_lagrange_loss = PPO_loss + lagrange_multiplier * pens_batch_violation
                    #PPO_lagrange_loss = PPO_loss + lagrange_multiplier * total_constrained_loss


                    rho = self.rho  # specified in args
                    PPO_quadratic_loss = PPO_loss + 1/rho * torch.pow(pen_violation, 2)
                    PPO_augmented_lagrange_loss = PPO_loss + lagrange_multiplier * pen_violation + 1/rho * torch.pow(pen_violation, 2)

                    if args.lagrange_loss:
                        PPO_loss_fn = PPO_lagrange_loss
                    elif args.quadratic_loss:
                        PPO_loss_fn = PPO_quadratic_loss
                        #print('penalty violation:', pen_violation.item())
                        #print('penalty violation squared:', torch.pow(pen_violation, 2).item())
                        #print('0.05 * pen vio', 0.05 * pen_violation.item())
                        #print('mult to equate', 0.05 * pen_violation.item() / torch.pow(pen_violation, 2).item())
                    elif args.augmented_lagrange_loss:
                        PPO_loss_fn = PPO_augmented_lagrange_loss
                else:
                    PPO_loss_fn = PPO_loss  # if not doing constrained, just use the normal loss

                if curr_updates == 16 and not args.update_in_val and (args.lagrange_loss or args.augmented_lagrange_loss): #only need to do this when updating lambda
                #if False:
                    PPO_loss_fn.backward(retain_graph=True)  # backprop the loss
                else:
                    PPO_loss_fn.backward()


                # instead of MSE or something like that, this surrogate loss is used to update NN params

                if distributed:
                    nn.utils.clip_grad_norm_(self.actor_critic.module.parameters,  # stop gradients from exploding
                                             self.max_grad_norm)
                else:
                    nn.utils.clip_grad_norm_(self.actor_critic.parameters,  # stop gradients from exploding
                                             self.max_grad_norm)

                    #nn.utils.clip_grad_norm_([self.lagrange_lambda], max_norm=1.0)


                if curr_updates == 16 and not args.update_in_val and (args.lagrange_loss or args.augmented_lagrange_loss): #only need to do this when updating lambda
                    #self.lagrange_lambda.grad = -self.lagrange_lambda.grad
                    self.lagrange_optimizer.zero_grad()
                    loss_pen = -lagrange_multiplier * avg_update_constraint
                    loss_pen.backward()
                    torch.nn.utils.clip_grad_norm_([self.lagrange_lambda], max_norm=0.001)
                    self.lagrange_optimizer.step()
                    print('new raw lambda:', self.lagrange_lambda.item())
                    #print('new celu lambda:', nn.functional.celu(self.lagrange_lambda, alpha=0.01).item())
                    print('average penalty violation:', avg_update_constraint.item())

                self.optimizer.step()  # update the weights after loss pen backprop and step otherwise params will updated before and cause inplace error


                if curr_updates == 16 and args.quadratic_loss:
                    print('average penalty violation:', avg_update_constraint.item())

                '''
                # if doing lagrange update in here every time:
                
                with torch.no_grad():
                    self.lagrange_lambda.grad = -self.lagrange_lambda.grad
                self.lagrange_optimizer.step()
                '''
                '''
                if not args.update_in_val and curr_updates == 16 and (args.lagrange_loss or args.augmented_lagrange_loss):  # if not updating in validation function
                # if doing lagrange update every 16 updates (and rho stuff in here too):
                
                    print('Average penalty violation:', avg_update_constraint.item() / curr_updates) # ignore .item() warning - becomes tensor
                    print(avg_update_constraint.requires_grad)
                    self.lagrange_optimizer.zero_grad()
                    avg_update_constraint = avg_update_constraint / curr_updates
                    loss_pen = -lagrange_multiplier * avg_update_constraint
                    loss_pen.backward()
                    torch.nn.utils.clip_grad_norm_([self.lagrange_lambda], max_norm=1.0)
                    self.lagrange_optimizer.step()
                    print('new lambda:', self.lagrange_lambda.item())

                    if args.augmented_lagrange_loss:
                        if avg_update_constraint > 180:
                            self.rho = self.rho * 0.99
                            print('Updated rho:', self.rho)
                        elif avg_update_constraint < 100:
                            self.rho = self.rho * 1.01
                            print('Updated rho:', self.rho)

                if not args.update_in_val and curr_updates == 16 and args.quadratic_loss:
                    if avg_update_constraint > 180:
                        self.rho = self.rho * 0.99
                        print('Updated rho:', self.rho)
                    elif avg_update_constraint < 100:
                        self.rho = self.rho * 1.01
                        print('Updated rho:', self.rho)

                    '''


                if kl_div_check:
                    if kl_div_triggered:
                        self.optimizer.param_groups[0]['lr'] = curr_lr

                def synchronize_model_parameters():
                    for param in self.actor_critic.module.parameters:
                        # Sum all parameters across all processes
                        torch.distributed.all_reduce(param.data, op=ReduceOp.SUM)
                        # Average by dividing by the number of processes
                        param.data /= torch.distributed.get_world_size()

                if distributed:
                    local_rank = int(os.environ["LOCAL_RANK"])

                if distributed and local_rank != 0:
                    torch.distributed.barrier()
                if distributed and local_rank == 0:
                    torch.distributed.barrier()
                if distributed:
                    synchronize_model_parameters()
                    torch.distributed.barrier()

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
                losses_epoch += PPO_loss_fn.item() # add to the total loss

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        losses_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, losses_epoch
