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
import random

from arguments import get_args
from models.Policy import *
from prepare_loader import *

from Eval.evaluation import validate, evaluation

from Agent.PPO_Agent import PPO
# from Agent.PGV_HybridF_Agent import PGV_forward_Agent, trainPG
from Agent.storage import RolloutStorage
# from torch.utils.tensorboard import SummaryWriter
from Eval import evaluation
from Eval.evaluation import *
from set_log import set_logging
from Reward_utils import *
from results import get_result_filename
from models.DNN_models import DeepCrossModel, LinearModel
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'


import gc
import tracemalloc
from tqdm import tqdm


def main(Debug=False):
    args = get_args()

    # torch setup
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.mps.manual_seed(args.seed)

    # num_threads = args.num_threads
    # torch.set_num_threads(num_threads)
    # torch.set_num_interop_threads(num_threads)

    if args.distributed:
        local_rank = int(os.environ["LOCAL_RANK"])  # added this according to pytorch docs for torchrun
        print('cuda device count', torch.cuda.device_count())
        print('local rank', local_rank)
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            'nccl')
        device = torch.device(f'cuda:{local_rank}')
        print('device is', device)
        try:
            print(f"Environment Rank: {os.getenv('RANK')}")
        except:
            print("Environment Rank: None")
        print(f"[Rank {dist.get_rank()}] Message here...")


    else:
        if args.use_mps:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    print('Debug is ', args.Debug)

    np.random.seed(args.seed)

    # if torch.cuda.is_available() and torch.cuda.current_device()==0:
    #    torch.cuda.set_device(5)

    # print(device)
    NAME = initialize_logger_name(args)
    logger = set_logging(NAME)
    # logger.info("current cuda device is {}".format(torch.cuda.current_device()))
    logger.info("current device is {}".format(device))

    utils.create_save_dir(args.saved_dir)
    data_dir = os.path.join(os.getcwd(), args.Data)  # /GDSC_ALL

    dtype = torch.DoubleTensor
    # dtype = torch.FloatTensor

    early_stopping_iter = 100000
    early_stopping_counter = 0

    data_dir = utils.create_train_data_dir(data_dir, args)

    k, kr, keepk = utils.generate_k(args.analysis, args.keepk_ratio, args.keepk, args.scenario,
                                    args.k)  # based on args.analysis
    args.k = k  # if analysis=KEEPK, k=args.keepk else k=args.k
    args.kr = kr  # if analysis=KEEPK, kr=args.keepk_ratio else kr=1
    args.keepk = keepk  # if analysis=KEEPK, keepk=args.keepk elif analysis=FULL, keepk="" else keepk=args.scenario
    rank_str = "All" if args.full == True else args.k
    miss_rate = args.miss_rate if args.analysis == "sparse" else ""
    scale = args.scale if args.normalize_y else "raw"
    model_name = NAME

    model_save_dir = os.path.join(os.getcwd(), args.saved_dir, model_name)  # checkpoint saving directory
    utils.create_save_dir(model_save_dir)
    # writer = SummaryWriter('runs/'+model_name)
    logger.info('model saved directory {}'.format(model_save_dir))

    if args.analysis == "noise" or args.analysis == 'sparse':
        N, M, P, WP, drug_embs, train_dataset, test_dataset, train_input, test_input, Ytest, Ynoise_test, original_drug_ids = prepare_loader_simu(
            data_dir, args)
    else:
        N, M, P, WP, drug_embs, train_dataset, test_dataset, train_input, test_input, Ytest, \
            cell_mean, drug_mean, overall_mean, original_drug_ids = prepare_loader(
            data_dir, args)

    # drug_idx_to_original_map = {i: original_drug_ids[i] for i in range(M)}
    drug_idx_to_original_map = {original_drug_ids[i]: i for i in range(M)}

    if WP is not None:
        logger.info("Load WP with dimension {},{}".format(
            *(list(WP.size()))))  # WP is matrix to project cell-line features onto latent space
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
        tr_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    else:
        tr_sampler = RandomSampler(train_dataset)

    train_loader = DataLoader(train_dataset, sampler=tr_sampler, num_workers=4, pin_memory=True,
                              batch_size=args.num_processes, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.num_processes, drop_last=True)

    best_ndcg = 0

    if args.algo == 'ppo':

        print('args.Data', args.Data)
        print('model_name', model_name)
        fp = open("./results/{}/{}_PPOresult.txt".format(args.Data, model_name), "w")
        fp.write("epoch,train_ndcg,train_rewards,test_ndcg,test_rewards\n")
        logger.info("results saved file name is ./results/{}/{}_PPOresult.txt".format(args.Data, model_name))
        agent = PPO(args, N, M, P, cell_dim, WP, drug_embs, device, drug_mean=drug_mean, overall_mean=overall_mean,
                    dist=args.distributed)
        if args.distributed:
            total_params = [p.numel() for p in agent.actor_critic.module.parameters]  # this does not take gpu memory
        else:
            total_params = [p.numel() for p in agent.actor_critic.parameters]
        logger.info("total params in PPO is {}".format(sum(total_params)))
        logger.info('nlayers_cross is {}, nlayers_deep is {}, deep_hidden_sizes is {}'.format(
            args.nlayers_cross, args.nlayers_deep, args.deep_hidden_sizes))

        # if args.distributed:
        #    agent = torch.nn.parallel.DistributedDataParallel(agent, device_ids=[local_rank], output_device=local_rank)

        num_steps_cell = min(args.num_steps,
                             M)  # number of forward steps in a single episode, only used when cut off the episode

        obs_actor_shape = train_input.shape[2] - 1  # origin cell-line shape

        if args.distributed:
            obs_critic_shape = [M, agent.actor_critic.module.critic_size]
        else:
            obs_critic_shape = [M, agent.actor_critic.critic_size]

        # drug severities
        drug_severities = pd.read_csv('GDSC_ALL/severity_scores.csv')
        lagrange_lambda = args.lagrange_lambda
        lambda_lr = args.lambda_lr
        drug_ids = drug_severities['drug_id'].values
        severity_scores = drug_severities['normalised'].values
        drug_ids = [drug_idx_to_original_map[str(drug)] for drug in drug_ids]
        drug_ids = torch.Tensor(drug_ids)
        severity_scores = torch.FloatTensor(severity_scores)
        drug_idx_to_original_map = {i: original_drug_ids[i] for i in range(M)}

        # rollout still on cpu

        rollouts = RolloutStorage(num_steps_cell, args.num_processes, obs_actor_shape, obs_critic_shape, M,
                                  severity_scores, lagrange_lambda,
                                  lambda_lr)  # store info needed for PPO rollout (generating episodes)
        rollouts.to(device)
        # episode_rewards = deque(maxlen=10)

        ###################################################
        # (1) every update, num_processes actors, each with num_steps_cell, then each update (for a batch) has obs of
        # at most Batch_Size = num_steps_cell*num_process = 38*16 so for a single epoch, it has updates N/num_process
        # (2) when comes to update, for a ppo epoch, it has total sample size as  Batch_Size = num_steps_cell*num_process,
        #  it has ppo_epochs(3~30),and for each ppo epoch, it has num_mini_batch batches, each mini batch has mini_batch_size of
        # Batch_Size/num_mini_batch, so in each ppo epoch, the updates are num_mini_batch
        # (3) when loop over all the cell-lines in one epoch,
        # the updates are N/num_processes*ppo_epochs*num_mini_batch
        # (4) the def of num_updates are different from the original paper, where we don't have the total_timestamps
        # we can think the num_updates are epochs*N/num_processes
        ###################################################
        num_updates = int(args.num_env_steps) // num_steps_cell // args.num_processes
        num_updates_ppo = args.epochs * N // args.num_processes
        num_optimizer_steps = args.epochs * N // args.num_processes * args.ppo_epoch * args.num_mini_batch
        # the num_updates here are not considering all the cell-lines, only num_process of cells,
        # Batch_size= num_steps_cell * args.num_processes, which takes one update,
        # so num_updates is just as epoch
        print(
            'num of updates is {}, num of updates in ppo is {}, total optimizer steps are {} and num of epochs is {}'.format(
                num_updates, num_updates_ppo, num_optimizer_steps, args.epochs))

        updates_per_epoch = N // args.num_processes * args.ppo_epoch * args.num_mini_batch
        print("for a single epoch, it updates {} times ".format(updates_per_epoch))
        epochs = args.epochs

        if args.resume:
            checkpoint = torch.load(args.resume, map_location=device)
            best_ndcg = checkpoint['best_ndcg']
            print("previous bset ndcg is {}".format(best_ndcg))
            print(checkpoint.keys())
            agent.actor_critic.load_state_dict(checkpoint['Policy_state_dict'])
            #agent.actor_critic.load_state_dict(checkpoint['value_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer'])

        do_constrained = args.constrained
        do_constrained = bool(do_constrained)
        print('constrained is {}, lagrange lambda is {}'.format(do_constrained, lagrange_lambda))

        train_ndcgs_all = np.array([], dtype=np.float32)
        test_ndcgs_all = np.array([], dtype=np.float32)
        train_rewards_all = np.array([], dtype=np.float32)
        test_rewards_all = np.array([], dtype=np.float32)
        ppo_epoch_losses_all = np.array([], dtype=np.float32)

        start_time = datetime.datetime.now()
        overall_start_time = start_time.strftime('%y%m%d_%H%M')
        PBS_job_name = os.getenv('PBS_JOBNAME', overall_start_time)

        initial_gae_lambda = args.gae_lambda


        ndcg_test, test_rewards, best_test_rewards, pred_test = validate(
            agent, test_loader, args, device,
            update_lambda=False)  # validate the model on the test data, returns ndcg, rewards, best rewards, and predictions. dont update lambda§
        # writer.add_scalar('ndcg_test', ndcg_test, epoch)

        with torch.no_grad():  # disable gradient calculation, save memory, more efficient
            test_input_var = test_input.clone().detach().to(device)  # added .float()
            Y_pred, _ = agent.actor_critic(
                test_input_var)  # .squeeze().cpu().detach().numpy(). #get the predictions for test data
            Y_pred = Y_pred.squeeze().cpu().detach().numpy()

        np.savez('PPORank_FYP_lr1.5e-4_15xlrsched_epochs700_rwdscl100.76029', Y_pred=Y_pred)





if __name__ == "__main__":
    main(Debug=True)
