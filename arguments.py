import argparse

import torch
import math

def get_args():
    parser = argparse.ArgumentParser(description='RL ranking algorithm')
    parser.add_argument(
        '--algo',
        default='ppo')
    parser.add_argument(
        '--lr',
        type=float,
        default=1.5e-4,
        help='learning rate (default: 3e-4)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer alpha (default: 0.99)')
    parser.add_argument(
        '--cuda_id',
        type=int,
        default=2,
        help='cuda device to be used ')
    parser.add_argument(
        '--use_gae',
        default=True,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae_lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy_coef',
        type=float,
        default=0, # in paper this is 0.001, originally 0. Try 0.01
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value_loss_coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max_grad_norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='random seed (default: 1)')
    parser.add_argument(
        '--cuda_deterministic',
        action='store_true',
        default=True,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num_processes',
        type=int,
        default=16,
        help='how many training actors to do sample(default: 16) - basically batch size')
    parser.add_argument(
        '--num_steps',
        type=int,
        default=300, #try 2048
        help='number of forward steps in a single episode, only used when cut off the episode (default: 300)')
    parser.add_argument(
        '--ppo_epoch',
        type=int,
        default=4, #8 in paper, 4 in original. Try 8
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--epochs',
        type=int,
        default=700,
        help='number of epochs for training (default:1000)')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='number of batches which is the sampled cell-lines')
    parser.add_argument(
        '--num_mini_batch',
        type=int,
        default=4,#try 8? was 4 in original. Try 16
        help='number of batches for ppo (default: 8)')
    # parser.add_argument(
    #     '--mini_batch_size',
    #     type=int,
    #     default=32,
    #     help='mini batch size for ppo (default: 32)')
    parser.add_argument(
        '--clip_param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 100)')
    parser.add_argument(
        '--save_interval',
        type=int,
        default=10,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--eval_interval',
        type=int,
        default=None,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--num_env_steps',
        type=int,
        default=30000000, # original was 30000000
        help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--Data',
        default="GDSC_ALL",
        help='data environment to train on (default: GDSC_ALL)')
    parser.add_argument(
        '--simu_N',
        default=1000,
        type=int,
        help='simulation data X size N')
    parser.add_argument(
        '--simu_P',
        default=1000,
        type=int,
        help='simulation data X features dim P')
    parser.add_argument(
        '--simu_M',
        default=120,
        type=int,
        help='simulation data drug dim M')
    parser.add_argument(
        '--miss_rate',
        default=0.5,
        type=float,
        help='simulation data missing rate')
    parser.add_argument(
        '--log_dir',
        default='/logs',
        help='directory to save agent logs ')
    parser.add_argument(
        '--CV_dir',
        default='CV',
        help='directory to do cross validation ')
    parser.add_argument(
        '--saved_dir',
        default='Saved',
        help='directory to save models')
    parser.add_argument(
        '--use_proper_time_limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--recurrent_policy',
        action='store_true',
        default=False,
        help='use a recurrent policy')
    parser.add_argument(
        '--use_linear_lr_decay',
        action="store_false",
        default=True,
        help='use a linear schedule on the learning rate')
    parser.add_argument(
        "--fold",
        type=str,
        default="Fold0",
        help="folder of CV to train")
    parser.add_argument(
        '--k',
        type=int,
        default=5,
        help="top k drugs recommended used in combination with rank all")
    parser.add_argument(
        "--f",
        type=int,
        default=100,
        help="dimension in the projected space")  # Number of dimensions
    parser.add_argument(
        '--full',
        default=True,
        action="store_false",
        help="whether to use full ranking")
    parser.add_argument(
        '--nn_baseline',
        default=False,
        help="whether to use nn baseline")
    parser.add_argument(
        '--avg_baseline',
        default=True,
        help="whether to use avg baseline")
    parser.add_argument(
        '--normalize_y',
        action="store_false",
        default=True,
        help="whether to normalize the response")
    parser.add_argument(
        '--augment',
        action="store_true",
        default=False,
        help="whether to augment data with noise for data augumentation")
    parser.add_argument(
        '--weight_decay',
        action="store_false",
        default=True,
        help="whether weight decay seperately in dnn")
    parser.add_argument(
        '--weight_decay_para',
        default=3e-4,
        type=float,
        help="weight decay params for all dnn params")
    parser.add_argument(
        '--scale',
        default="Snorm",
        type=str,
        help="when normalize y, scale it to make it invariant for ndcg, currently support minmax,normalzie,maxabs")
    parser.add_argument(
        '--nlayers_cross',
        type=int,
        default=1, #was 1 in original, 2 in paper i think
        help="layers of cross network")
    parser.add_argument(
        '--nlayers_deep',
        type=int,
        default=2, #2 in paper, 2 in original
        help="layers of deep network")
    parser.add_argument(
        '--nlayers_value',
        type=int,
        default=2,
        help="layers of value function network")
    parser.add_argument(
        '--train_drug',
        action="store_false",
        default=True,
        help="whether to train drug embedding")
    parser.add_argument(
        '--train_cell',
        default=True,
        action="store_false",
        help="whether to train cell embedding")
    parser.add_argument(
        '--normalize_advantages',
        default=True,
        help="whether to tnormalize the advange")
    parser.add_argument(
        '--hybrid',
        default=True,
        help="whether to hybrid the loss")
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99, #0.95 in paper?, 0.99 in original. Try 0.95
        help="gae discount factor")
    parser.add_argument(
        '--lambda1',
        type=float,
        default=1e-5,
        help="L1 regulization coeff")
    parser.add_argument(
        '--value_hidden_sizes',
        type=int, nargs="+",
        default=[128, 64],
        help="value NN baseline hidden units")
    parser.add_argument(
        '--deep_hidden_sizes',
        type=int,
        nargs="+",
        default=[256,  128], #worth trying [128, 64] & 32 for deep_out_size
        help="deep part NN hidden units")
    parser.add_argument(
        '--deep_out_size',
        type=int, default=64,
        help="deep part NN output size")
    parser.add_argument(
        '--truncated',
        default=False,
        help='when ranking k whether to truncted trajectory')
    parser.add_argument(
        '--analysis',
        type=str,
        default="FULL", # FULL, KEEPK, noise, sparse
        help='analysis can be FULL or KEEPK or noise or sparse')
    parser.add_argument(
        '--keepk',
        type=int,
        default=20,
        help='when analysis is KEEPK, the keepk value')
    parser.add_argument(
        '--keepk_ratio',
        type=float,
        default=0.1,
        help='when analysis is KEEPK, the keepk_ratio value')
    parser.add_argument(
        '--Debug',
        default=False,
        help='whether it is debug style to decide if saving pre_mat')
    parser.add_argument(
        '--scenario',
        default='linear',
        help='linear or quad or exp')
    parser.add_argument(
        '--resume',
        default='',
        type=str,
        help='resume file')
    parser.add_argument(
        '--ess_genes_fn',
        default='',
        type=str,
        help='essential genes names file')
    parser.add_argument(
        '--pretrain',
        default=True,
        action="store_false",
        help='WP pretrained or not')
    parser.add_argument(
        '--MSE',
        default=True,
        action="store_false",
        help='dnn use MSE or not')
    parser.add_argument(
        '--local_rank',
        type=int, default=0,
        help='node rank for distributed training')
    parser.add_argument(
        '--distributed',
        default=False,
        action="store_true",
        help='if in debugging, it will be set as false, in terminal,with ' +
        '--distributed it will be true, without it, will be false')
    parser.add_argument(
        '--nproc_per_node',
        type=int, default=2,
        help='num of gpus  for distributed training')
    parser.add_argument(
        '--prop',
        type=float,
        default=1.0,
        help='proportion to be used')
    parser.add_argument(
        '--sigma',
        type=float,
        default=1.0,
        help='noise std if using noise data')
    parser.add_argument(
        '--mu',
        type=float,
        default=0.0,
        help='noise mean if using noise data')
    parser.add_argument(
        '--lr_sched_mult',
        type=float,
        default=15,
        help='multiplier for lr scheduler')
    parser.add_argument(
        '--constrained',
        default=False,
        type=bool,
        help='whether to use constrained optimization')
    parser.add_argument(
        '--lagrange_lambda',
        default=1,
        type=float,
        help='lagrange lambda for constrained optimization')
    parser.add_argument(
        '--lambda_lr',
        default=0.1,
        type=float,
        help='lagrange lr for constrained optimization')
    parser.add_argument(
        '--use_mps',
        default=False,
        help='whether to use mps (Mac)')
    parser.add_argument(
        '--num_threads',
        default='13',
        type=int,
        help='number of threads for cpu')
    parser.add_argument(
        '--reward_scale_factor',
        default='100',
        type=int,
        help='scale factor for reward')
    parser.add_argument(
        '--kl_div_check',
        default=False,
        type=bool,
        help='whether to check kl div and compensate')
    parser.add_argument(
        '--increase_gae_lambda', #doesnt work - leaving it in just in case
        default=False,
        type=str,
        help='whether to increase gae lambda over time')
    parser.add_argument(
        '--rho',
        default=0.4,
        type=float,
        help='rho value for quadratic/augmented penalty')
    parser.add_argument(
        '--update_in_val',
        default=False,
        type=bool,
        help='update lambda/rho in validate function')
    parser.add_argument(
        '--lagrange_loss',
        default=False,
        type=bool,
        help='use lagrange loss in loss function')
    parser.add_argument(
        '--quadratic_loss',
        default=False,
        type=bool,
        help='use quadratic loss in loss function')
    parser.add_argument(
        '--augmented_lagrange_loss',
        default=False,
        type=bool,
        help='use augmented lagrange loss in loss function')
    parser.add_argument(
        '--target_pen',
        default=36,
        type=float,
        help='target penalty value for constrained')
    parser.add_argument(
        '--do_cost_advantages',
        default=False,
        type=bool,
        help='target penalty value for constrained')
    parser.add_argument(
        '--sequential_data',
        default=False,
        type=bool,
        help='sequential data for RL simulation')






    args = parser.parse_args()

    if args.use_mps:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.shared_params = False

    if args.distributed:
        args.lr = args.lr * math.sqrt(args.nproc_per_node)
        print('changed lr to', args.lr)

    #args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.cuda = torch.cuda.is_available()

    assert args.algo in ['pg', 'ppo', 'dnn']

    return args


if __name__ == "__main__":
    args = get_args()
