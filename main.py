import copy
import glob
import os
import logging
import gc

import time
import utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import datetime

from arguments import get_args
from models.Policy import *
from prepare_loader import *

from Eval.evaluation import validate, evaluation

from Agent.PPO_Agent import PPO
#from Agent.PGV_HybridF_Agent import PGV_forward_Agent, trainPG
from Agent.storage import RolloutStorage
#from torch.utils.tensorboard import SummaryWriter
from Eval import evaluation
from Eval.evaluation import *
from set_log import set_logging
from Reward_utils import *
from results import get_result_filename
from models.DNN_models import DeepCrossModel, LinearModel
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'


import gc
import psutil
import tracemalloc
from tqdm import tqdm


def main(Debug=False):
    #tracemalloc.start()
    args = get_args()

    # torch setup
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.mps.manual_seed(args.seed)

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            'nccl')
        device = torch.device(f'cuda:{args.local_rank}')
    else:
        if args.use_mps:
            device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    np.random.seed(args.seed)

    # if torch.cuda.is_available() and torch.cuda.current_device()==0:
    #    torch.cuda.set_device(5)

    # print(device)
    NAME = initialize_logger_name(args)
    logger = set_logging(NAME)
    #logger.info("current cuda device is {}".format(torch.cuda.current_device()))
    logger.info("current device is {}".format(device))

    utils.create_save_dir(args.saved_dir)
    data_dir = os.path.join(os.getcwd(), args.Data) #/GDSC_ALL

    dtype = torch.DoubleTensor
    # dtype = torch.FloatTensor

    early_stopping_iter = 100000
    early_stopping_counter = 0

    data_dir = utils.create_train_data_dir(data_dir, args)

    k, kr, keepk = utils.generate_k(args.analysis, args.keepk_ratio, args.keepk, args.scenario, args.k) #based on args.analysis
    args.k = k #if analysis=KEEPK, k=args.keepk else k=args.k
    args.kr = kr #if analysis=KEEPK, kr=args.keepk_ratio else kr=1
    args.keepk = keepk #if analysis=KEEPK, keepk=args.keepk elif analysis=FULL, keepk="" else keepk=args.scenario
    rank_str = "All" if args.full == True else args.k
    miss_rate = args.miss_rate if args.analysis == "sparse" else ""
    scale = args.scale if args.normalize_y else "raw"

    model_name = NAME

    model_save_dir = os.path.join(os.getcwd(), args.saved_dir, model_name)  # checkpoint saving directory
    utils.create_save_dir(model_save_dir)
    #writer = SummaryWriter('runs/'+model_name)
    logger.info('model saved directory {}'.format(model_save_dir))

    if args.analysis == "noise" or args.analysis == 'sparse':
        N, M, P, WP, drug_embs, train_dataset, test_dataset, train_input, test_input, Ytest, Ynoise_test, original_drug_ids = prepare_loader_simu(
            data_dir, args)
    else:
        N, M, P, WP, drug_embs, train_dataset, test_dataset, train_input, test_input, Ytest, \
            cell_mean, drug_mean, overall_mean, original_drug_ids = prepare_loader(
                data_dir, args)

    #drug_idx_to_original_map = {i: original_drug_ids[i] for i in range(M)}
    drug_idx_to_original_map = {original_drug_ids[i]: i for i in range(M)}

    if WP is not None:
        logger.info("Load WP with dimension {},{}".format(*(list(WP.size())))) #WP is matrix to project cell-line features onto latent space
        logger.info("Load drug embedding with dimension {},{}".format(*(list(drug_embs.size()))))
    else:
        logger.info("No pretrained WP with dimension {},{}".format(*([P, args.f])))
        logger.info("No pretrained drug embedding  with dimension {},{}".format(*([args.f, M])))

    cell_dim = WP.shape[1] if WP is not None else args.f
    drug_dim = drug_embs.shape[1] if drug_embs is not None else args.f

    logger.info(
        "Sample size N  is {}, drug size M is {}, cellline features P is {} with projection dimension is {}".format(
            N, M, P, cell_dim))

    if args.distributed:
        tr_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        tr_sampler = RandomSampler(train_dataset)

    train_loader = DataLoader(train_dataset, sampler=tr_sampler, num_workers=4, pin_memory=True,
                              batch_size=args.num_processes, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.num_processes, drop_last=True)

    best_ndcg = 0

    if args.algo == 'ppo':

        fp = open("./results/{}/{}_PPOresult.txt".format(args.Data, model_name), "w")
        fp.write("epoch,train_ndcg,train_rewards,test_ndcg,test_rewards\n")
        logger.info("results saved file name is ./results/{}/{}_PPOresult.txt".format(args.Data, model_name))
        agent = PPO(args, N, M, P, cell_dim, WP, drug_embs, device, drug_mean=drug_mean, overall_mean=overall_mean)
        total_params = [p.numel() for p in agent.actor_critic.parameters]  # this does not take gpu memory
        logger.info("total params in PPO is {}".format(sum(total_params)))

        # if args.distributed:
        #     agent = torch.nn.parallel.DistributedDataParallel(agent, device_ids=[args.local_rank])

        num_steps_cell = min(args.num_steps, M)  # number of forward steps in a single episode, only used when cut off the episode

        obs_actor_shape = train_input.shape[2] - 1  # origin cell-line shape
        obs_critic_shape = [M, agent.actor_critic.critic_size]

        #drug severities
        drug_severities = pd.read_csv('GDSC_ALL/severity_scores.csv')
        lagrange_lambda = 0.1
        lambda_lr = 0.01
        drug_ids = drug_severities['drug_id'].values
        severity_scores = drug_severities['normalised'].values
        drug_ids = [drug_idx_to_original_map[str(drug)] for drug in drug_ids]
        drug_ids = torch.Tensor(drug_ids)
        severity_scores = torch.FloatTensor(severity_scores)
        drug_idx_to_original_map = {i: original_drug_ids[i] for i in range(M)}

        # rollout still on cpu
        rollouts = RolloutStorage(num_steps_cell, args.num_processes, obs_actor_shape, obs_critic_shape, M, severity_scores, lagrange_lambda, lambda_lr) #store info needed for PPO rollout (generating episodes)
        rollouts.to(device)
        # episode_rewards = deque(maxlen=10)

        ###################################################
        # (1) every update, num_processes actors, each with num_steps_cell, then each update (for a batch) has obs of
        # at most Batch_Size = num_steps_cell*num_process, so for a single epoch, it has updates N/num_process
        # (2) when comes to update, for a ppo epoch, it has total sample size as  Batch_Size = num_steps_cell*num_process,
        #  it has ppo_epochs(3~30),and for each ppo epoch, it has num_mini_batch batches, each mini batch has mini_batch_size of
        # Batch_Size/num_mini_batch, so in each ppo epoch, the updates are num_mini_batch
        # (3) when loop over all the cell-lines in one epoch,
        # the updates are N/num_processes*ppo_epochs*num_mini_batch
        # (4) the def of num_updates are different from the original paper, where we don't have the total_timestamps
        # we can think the num_updates are epochs*N/num_processes
        ###################################################
        num_updates = int(args.num_env_steps) // num_steps_cell // args.num_processes
        num_updates_ppo = args.epochs * N//args.num_processes
        num_optimizer_steps = args.epochs * N//args.num_processes*args.ppo_epoch*args.num_mini_batch
        # the num_updates here are not considering all the cell-lines, only num_process of cells,
        # Batch_size= num_steps_cell * args.num_processes, which takes one update,
        # so num_updates is just as epoch
        print('num of updates is {}, num of updates in ppo is {}, total optimizer steps are {} and num of epochs is {}'.format(
            num_updates, num_updates_ppo, num_optimizer_steps, args.epochs))

        updates_per_epoch = N//args.num_processes*args.ppo_epoch*args.num_mini_batch
        print("for a single epoch, it updates {} times ".format(updates_per_epoch))
        epochs = args.epochs

        if args.resume:
            checkpoint = torch.load(args.resume)
            best_ndcg = checkpoint['best_ndcg']
            print("previous bset ndcg is {}".format(best_ndcg))
            agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            agent.value_net.load_state_dict(checkpoint['value_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer'])

        do_constrained = args.constrained
        print('constrained is {}'.format(do_constrained))

        train_ndcgs_all = np.array([], dtype=np.float32)
        test_ndcgs_all = np.array([], dtype=np.float32)
        train_rewards_all = np.array([], dtype=np.float32)
        test_rewards_all = np.array([], dtype=np.float32)
        ppo_epoch_losses_all = np.array([], dtype=np.float32)

        for epoch in range(epochs):
            # train_loader = DataLoader(train_dataset, sampler=tr_sampler, num_workers=4, pin_memory=True,
            #                           batch_size=args.num_processes, drop_last=True)
            if args.use_linear_lr_decay:
                utils.update_linear_schedule(agent.optimizer, epoch, num_updates, args.lr) #decrease lr linearly
            ppo_epoch_loss = []
            ppo_epoch_value_loss = []
            ppo_epoch_action_loss = []
            ppo_epoch_entropy_loss = []

            epoch_start = time.time()

            loop = tqdm(enumerate(train_loader), total=len(train_loader), colour='#3C8F3D')
            for i, batch in loop: # train_loader length = 48

                start = time.time()
                input, true_scores = batch  # input[B,M,P+1],1 is drug index
                input_var = input.float().clone().detach().to(device)  # input.detach().clone().requires_grad_(False).to(device). 16x38x664
                true_scores = true_scores.float().clone().detach().to(device) # true scores is the true scores of the drugs, shape = 16x38
                # Data collections, in paper as runner to collect  Batch_size= num_steps_cell * args.num_processes
                # corresponding to line2-line 3 in algorithm 1
                with torch.no_grad(): # disable gradient calculation, save memory, more efficient
                    # scores come from actor network,[B,M,1]
                    scores, critic_inputs = agent.actor_critic(input_var)  # B=16, 250MB. input into PPO_Policy. goes into forward of self.actor which is the deepcross network. outs are scores and state representation for critic
                    # scores shape = 16x38x1, critic_inputs shape = 16x38x266
                    scores = scores.squeeze()
                    start1 = time.time()
                    paths = rollouts.sample_num_steps(agent, input_var, scores, true_scores, critic_inputs) #sample a specific number of (time) steps. returns batch for training, containing all the things needed from paths
                    #paths contains rewards, log_probs of policy, dist_entropy, value_preds, obs_actor for all the time steps
                    #for p in paths:
                        #print('actions', p['actions'])

                    # TODO: multi process for sample collection
                    #############################################################################################################################
                    #rollouts.sample_episodes(agent, input_var, scores, true_scores, critic_inputs)
                    # reward_paths = [path['rewards'] for path in paths]
                    # value_paths = [torch.stack(path['value_pred'] )for path in paths]
                    # rollouts.return_of_rewards(reward_paths, value_paths,args.use_gae,args.gamma,args.gae_lambda)
                    rollouts.sample_concatenate(paths) #concatenate the paths, store info like rewards, value prediction etc in rollout self. variables, only keeping time steps
                rollouts.compute_returns(args.use_gae, args.gamma,
                                         args.gae_lambda, args.use_proper_time_limits) #uses gae unless specified otherwise. computes returns (using info just stored in prev line). stored in self.returns in rollout
                # for the update step Line 6,  Optimize surrogate L wrt θ, with K epochs and minibatch size M ≤ NT (in original PPO paper)
                # for e in range(self.ppo_epoch):
                # has num_mini_batchs,
                value_loss, action_loss, dist_entropy, losses = agent.update(rollouts, drug_idx_to_original_map, do_constrained=do_constrained) #calc standardised advantages, does PPO updates, returns losses
                # to track the  loss from each part

                ppo_epoch_value_loss.append(value_loss)
                ppo_epoch_action_loss.append(action_loss)
                ppo_epoch_entropy_loss.append(dist_entropy)
                ppo_epoch_loss.append(losses)
                rollouts.after_update()
                rollouts.to(device)

                #logger.info("on index {} out of {}, in epoch {}. Time for this update was {}".format(i, len(train_loader), epoch, time.time()-start))
                loop.set_description('Epoch [{}/{}]'.format(epoch, epochs-1))

            #gone through all data in train_loader

            # every update call the ndcg train

            ndcg_train, train_rewards, best_train_rewards, pred_train = validate(
                agent, train_loader, args, device, drug_idx_to_original_map, rollouts, update_lambda=do_constrained) #validate the model on the training data, returns ndcg, rewards, best rewards, and predictions. same process as what was just done basically
            epoch_loss = sum(ppo_epoch_loss)/len(ppo_epoch_loss) #average loss over the epoch
            ppo_epoch_loss.append(epoch_loss)

            #writer.add_scalar('ndcg_train', ndcg_train, epoch) #theres a writer = SummaryWriter commented out in this main, ive uncommented it now and it works fine

            # train_value_loss = np.array(ppo_epoch_value_loss).mean()
            # writer.add_scalar("train_value_loss", train_value_loss, epoch)

            # train_action_loss = np.array(ppo_epoch_action_loss).mean()
            # writer.add_scalar("train_action_loss", train_action_loss, epoch)

            # train_entropy_loss = np.array(ppo_epoch_entropy_loss).mean()
            # writer.add_scalar("train_entropy_loss", train_value_loss, epoch)
            with torch.no_grad():
                ndcg_test, test_rewards, best_test_rewards, pred_test = validate(
                    agent, test_loader, args, device, update_lambda=False) #validate the model on the test data, returns ndcg, rewards, best rewards, and predictions. dont update lambda§
            #writer.add_scalar('ndcg_test', ndcg_test, epoch)

            fp.write(f"{epoch} {ndcg_train:.4f} {train_rewards:.6f},{ndcg_test:.4f},{test_rewards:.6f}\n") #write to result file
            logger.info("DEV_and_Test@{}:train_ndcg {:.4f},test_ndcg {:.4f},train_rewards {:.6f},test_rewards {:.6f},ppo_epoch_loss {:6f}".format(
                epoch, ndcg_train, ndcg_test, train_rewards, test_rewards, epoch_loss)) #log the results
            logger.info("Time for this epoch was {:.4f}".format(time.time()-epoch_start))

            train_ndcgs_all = np.append(train_ndcgs_all, ndcg_train)
            test_ndcgs_all = np.append(test_ndcgs_all, ndcg_test)
            train_rewards_all = np.append(train_rewards_all, train_rewards)
            test_rewards_all = np.append(test_rewards_all, test_rewards)
            ppo_epoch_losses_all = np.append(ppo_epoch_losses_all, epoch_loss)

            # save for every interval-th episode

            is_best = ndcg_test > best_ndcg #if the ndcg on the test data is better than the best ndcg so far
            best_ndcg = max(ndcg_test, best_ndcg) #update the best ndcg

            if is_best and not args.Debug: #if the model is the best and not in debug mode
                # print("epoch {} best_rewards for {} test data is train {} and test {}, and the current best test ndcg is {}".format(
                #     epoch, model_name, best_train_rewards, best_test_rewards, best_ndcg))
                with torch.no_grad(): #disable gradient calculation, save memory, more efficient
                    test_input_var = test_input.clone().detach().float().to(device) #added .float()
                    Y_pred, _ = agent.actor_critic(test_input_var)  # .squeeze().cpu().detach().numpy(). #get the predictions for test data
                    Y_pred = Y_pred.squeeze().cpu().detach().numpy()
                    result_fn = get_result_filename(args.algo, args.analysis, args.Data, int(args.fold[-1]), args.f,
                                                    debug=args.Debug, keepk=args.keepk, ratio=args.keepk_ratio,
                                                    scenario=args.scenario)
                    np.savez(result_fn, Y_true=Ytest, Y_pred=Y_pred) #save the true and predicted values. results/gdsc_all/full/...
                    del test_input_var, Y_pred, _
            if (epoch % args.save_interval == 0 or epoch == num_updates - 1) and args.saved_dir != "": #save the model every save_interval epochs or at the end of training
                utils.save_checkpoint({ #save the model
                    'epoch': epoch,
                    'Policy_state_dict': agent.actor_critic.state_dict(),
                    'best_ndcg': best_ndcg,
                    'optimizer': agent.optimizer.state_dict(),
                }, is_best, model_save_dir, filename='/checkpoint.{0:03d}.tar'.format(epoch))

            elif is_best and args.Debug: #if the model is the best and in debug mode
                print("epoch {} best_rewards for {} test data is train {} and test {}, and the current best test ndcg is {}".format(
                    epoch, model_name, best_train_rewards, best_test_rewards, best_ndcg))
                with torch.no_grad():
                    test_input_var = test_input.clone().detach().float().to(device) #added .float()
                    Y_pred, _ = agent.actor_critic(test_input_var) #get the predictions for test data
                    Y_pred = Y_pred.squeeze().cpu().detach().numpy()
                    result_dir_name = get_result_filename(
                        args.algo, args.analysis, args.Data, int(args.fold[-1]), args.f, debug=True)
                    result_dir_name = result_dir_name[:-4]  # ppo_0
                    utils.create_save_dir(result_dir_name)
                    data_dims_name = "N{}_P{}_M{}".format(args.simu_N, P, M)

                    result_fn = get_result_filename(
                        args.algo, args.analysis, args.Data, int(args.fold[-1]),
                        args.f, debug=args.Debug, data_dims_name=data_dims_name, miss_ratio=args.miss_rate,
                        scenario=args.scenario)

                    np.savez(result_fn, Y_true=Ytest, Y_pred=Y_pred) #save the true and predicted values

            '''
            print('collecting garbage, current memory usage is {}'.format(torch.mps.current_allocated_memory()))
            print(torch.mps.driver_allocated_memory())
            gc.collect() #try this to help with memory use increasing over epochs. python rose to 7.5GB after 4 epochs. didnt help on its own
            torch.mps.empty_cache() #try this too. also didnt help... (even with gc.collect)
            print('garbage collected, current memory usage is {}'.format(torch.mps.current_allocated_memory()))
            print(torch.mps.driver_allocated_memory())
            '''




            '''
            for obj in gc.get_objects():  # Iterate through all objects
                print('printing gc objects:')
                try:
                    if torch.is_tensor(obj) and obj.requires_grad:  # Check if it is a tensor and requires grad. none of these objects increase in size over time
                        tag = getattr(obj, 'tag', 'No Tag')
                        print(obj.shape, obj.device, obj.dtype, obj.requires_grad, obj.grad_fn, obj.is_leaf, obj.numel(), obj.element_size()*obj.numel(), tag)
                except AttributeError as e:  # In case the object doesn't have the expected attributes
                    pass
                    
            '''

        fp.close()
        #writer.close()

    now = datetime.datetime.now()
    now = now.strftime('%y%m%d_%H%M')
    np.savez("./results/{}/{}_PPOresults{}.npz".format(args.Data, args.analysis, now), train_ndcgs_all=train_ndcgs_all, test_ndcgs_all=test_ndcgs_all, train_rewards_all=train_rewards_all, test_rewards_all=test_rewards_all, ppo_epoch_losses_all=ppo_epoch_losses_all)

    return best_ndcg #return the best ndcg, more detailed results are saved in the result file


if __name__ == "__main__":
    main(Debug=True)
