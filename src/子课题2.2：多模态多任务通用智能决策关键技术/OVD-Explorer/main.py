import os
import os.path as osp

import argparse
import torch

import utils.pytorch_util as ptu
from replay_buffer import ReplayBuffer
from utils.env_utils import NormalizedBoxEnv, domain_to_epoch, env_producer
from utils.rng import set_global_pkg_rng_state
from launcher_util import run_experiment_here
from path_collector import MdpPathCollector, RemoteMdpPathCollector
from trainer.policies import TanhGaussianPolicy, MakeDeterministic
from trainer.trainer import SACTrainer
from networks import FlattenMlp, QuantileMlp
from rl_algorithm import BatchRLAlgorithm
# torch.set_num_threads(1)
import ray
import logging
os.environ["OMP_NUM_THREADS"] = "2"

N_Q = 20
# import tensorflow as tf
# tf_sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1))
ray.init(
    # If true, then output from all of the worker processes on all nodes will be directed to the driver.
    num_cpus=4,
    log_to_driver=True,
    logging_level=logging.WARNING,
    # The amount of memory (in bytes)
    object_store_memory=1073741824, # 1g
    redis_max_memory=1073741824 # 1g
)

variant = None

def get_current_branch(dir):

    from git import Repo

    repo = Repo(dir)
    return repo.active_branch.name


def get_policy_producer(obs_dim, action_dim, hidden_sizes):

    def policy_producer(deterministic=False):

        policy = TanhGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
        )

        if deterministic:
            policy = MakeDeterministic(policy)

        return policy

    return policy_producer


def get_q_producer(obs_dim, action_dim, hidden_sizes, use_aleatoric=False, n_quantiles=N_Q):
    def q_producer():
        if not use_aleatoric:
            return FlattenMlp(input_size=obs_dim + action_dim,
                          output_size=1,
                          hidden_sizes=hidden_sizes, )
        else:
            return QuantileMlp(input_size=obs_dim + action_dim,
                    output_size=1,
                    num_quantiles=n_quantiles,
                    hidden_sizes=hidden_sizes, )

    return q_producer


def experiment(variant, prev_exp_state=None):
    domain = variant['domain']
    seed = variant['seed']
    annealing = variant['annealing']
    redq = variant['redq']
    use_aleatoric = variant["use_aleatoric"]
    expl_env = env_producer(domain, seed)

    obs_dim = expl_env.observation_space.low.size
    action_dim = expl_env.action_space.low.size

    # Get producer function for policy and value functions
    M = variant['layer_size']

    q_producer = get_q_producer(obs_dim, action_dim, hidden_sizes=[M, M], use_aleatoric=use_aleatoric)
    policy_producer = get_policy_producer(
        obs_dim, action_dim, hidden_sizes=[M, M])
    # Finished getting producer

    remote_eval_path_collector = RemoteMdpPathCollector.remote(
        domain, seed * 10 + 1,
        policy_producer
    )

    expl_path_collector = MdpPathCollector(
        expl_env,
    )
    replay_buffer = ReplayBuffer(
        variant['replay_buffer_size'],
        ob_space=expl_env.observation_space,
        action_space=expl_env.action_space
    )
    trainer = SACTrainer(
        policy_producer,
        q_producer,
        action_space=expl_env.action_space,
        **variant['trainer_kwargs']
    )

    algorithm = BatchRLAlgorithm(
        trainer=trainer,

        exploration_data_collector=expl_path_collector,
        remote_eval_data_collector=remote_eval_path_collector,
        annealing=annealing,
        redq=redq,
        replay_buffer=replay_buffer,
        optimistic_exp_hp=variant['optimistic_exp'],
        entropy_based_exp_hp=variant['entropy_based_exp'],
        **variant['algorithm_kwargs']
    )

    algorithm.to(ptu.device)

    if prev_exp_state is not None:

        expl_path_collector.restore_from_snapshot(
            prev_exp_state['exploration'])

        ray.get([remote_eval_path_collector.restore_from_snapshot.remote(
            prev_exp_state['evaluation_remote'])])
        ray.get([remote_eval_path_collector.set_global_pkg_rng_state.remote(
            prev_exp_state['evaluation_remote_rng_state']
        )])

        replay_buffer.restore_from_snapshot(prev_exp_state['replay_buffer'])

        trainer.restore_from_snapshot(prev_exp_state['trainer'])

        set_global_pkg_rng_state(prev_exp_state['global_pkg_rng_state'])

    start_epoch = prev_exp_state['epoch'] + \
        1 if prev_exp_state is not None else 0

    algorithm.train(start_epoch)


def get_cmd_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--domain', type=str, default='invertedpendulum')
    parser.add_argument('--no_gpu', default=False, action='store_true')
    parser.add_argument('--bo', default=False, action='store_true')
    parser.add_argument('--base_log_dir', type=str, default='./data')

    # optimistic_exp_hyper_param
    parser.add_argument('--beta_UB', type=float, default=0.0)
    parser.add_argument('--delta', type=float, default=0.0)

    # entropy-based exploration
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--alpha_2', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--sigma', type=float, default=0.0)
    parser.add_argument('--nor', type=float, default=3.0)
    parser.add_argument('--z', type=float, default=0.0)
    parser.add_argument('--version', type=int, default=0)
    parser.add_argument('--ee', default=False, action='store_true')
    parser.add_argument('--use_quantile_cdf', default=False, action='store_true')
    parser.add_argument('--autoz', default=False, action='store_true')
    parser.add_argument('--annealing', default=False, action='store_true')
    parser.add_argument('--use_aleatoric', default=False, action='store_true')
    parser.add_argument('--redq', default=False, action='store_true')

    # Training param
    # 这两个地方原本是1000  改成了200
    # parser.add_argument('--num_expl_steps_per_train_loop', type=int, default=1000)
    parser.add_argument('--num_expl_steps_per_train_loop', type=int, default=100)
    # parser.add_argument('--num_trains_per_train_loop', type=int, default=1000)
    parser.add_argument('--num_trains_per_train_loop', type=int, default=100)

    args = parser.parse_args()

    return args


def get_log_dir(args, should_include_base_log_dir=True, should_include_seed=True, should_include_domain=True):

    log_dir = osp.join(
        get_current_branch('./'),

        # Algo kwargs portion
        f'num_expl_steps_per_train_loop_{args.num_expl_steps_per_train_loop}_num_trains_per_train_loop_{args.num_trains_per_train_loop}'

        # optimistic exploration dependent portion
        f'_beta_UB_{args.beta_UB}_delta_{args.delta}'
        f'_alpha_{args.alpha}_alpha_2_{args.alpha_2}_beta_{args.beta}_sigma_{args.sigma}_nor_{args.nor}_z_{args.z}_ee_{args.ee}_version_{args.version}'
        ,
    )

    if should_include_domain:
        log_dir = osp.join(log_dir, args.domain)

    if should_include_seed:
        log_dir = osp.join(log_dir, f'seed_{args.seed}')

    if should_include_base_log_dir:
        log_dir = osp.join(args.base_log_dir, log_dir)

    return log_dir


if __name__ == "__main__":
    # Parameters for the experiment are either listed in variant below
    # or can be set through cmdline args and will be added or overrided
    # the corresponding attributein variant
    # global variant

    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=256,
        # replay_buffer_size=int(1E6),
        # replay_buffer_size=int(1E5),
        replay_buffer_size=0,
        algorithm_kwargs=dict(
            # num_eval_steps_per_epoch=1000,
            # min_num_steps_before_training=1000,
            # max_path_length=100,
            num_eval_steps_per_epoch=0,
            min_num_steps_before_training=0,
            max_path_length=0,
            num_trains_per_train_loop=None,
            num_expl_steps_per_train_loop=None,
            # num_eval_steps_per_epoch=5000,
            # min_num_steps_before_training=10000,
            # max_path_length=1000,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            train_num=20,
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
        optimistic_exp={},
        entropy_based_exp={}
    )

    args = get_cmd_args()
    if args.num_trains_per_train_loop == 1000:
        variant['replay_buffer_size'] = int(1E6)
        variant['algorithm_kwargs']['num_eval_steps_per_epoch'] = 5000
        variant['algorithm_kwargs']['min_num_steps_before_training'] = 10000
        variant['algorithm_kwargs']['max_path_length'] = 1000
    elif args.num_trains_per_train_loop == 100:
        variant['replay_buffer_size'] = int(1E5)
        variant['algorithm_kwargs']['num_eval_steps_per_epoch'] = 1000
        variant['algorithm_kwargs']['min_num_steps_before_training'] = 1000
        variant['algorithm_kwargs']['max_path_length'] = 100
    elif args.num_trains_per_train_loop == 50:
        variant['replay_buffer_size'] = int(1E5)
        variant['algorithm_kwargs']['num_eval_steps_per_epoch'] = 250
        variant['algorithm_kwargs']['min_num_steps_before_training'] = 500
        variant['algorithm_kwargs']['max_path_length'] = 50
    elif args.num_trains_per_train_loop == 200:
        variant['replay_buffer_size'] = int(1E5)
        variant['algorithm_kwargs']['num_eval_steps_per_epoch'] = 1000
        variant['algorithm_kwargs']['min_num_steps_before_training'] = 2000
        variant['algorithm_kwargs']['max_path_length'] = 200
    elif args.num_trains_per_train_loop == 250:
        variant['replay_buffer_size'] = int(1E5)
        variant['algorithm_kwargs']['num_eval_steps_per_epoch'] = 1250
        variant['algorithm_kwargs']['min_num_steps_before_training'] = 2500
        variant['algorithm_kwargs']['max_path_length'] = 250
    elif args.num_trains_per_train_loop == 750:
        variant['replay_buffer_size'] = int(1E6)
        variant['algorithm_kwargs']['num_eval_steps_per_epoch'] = 3750
        variant['algorithm_kwargs']['min_num_steps_before_training'] = 7500
        variant['algorithm_kwargs']['max_path_length'] = 750
    elif args.num_trains_per_train_loop == 400:
        variant['replay_buffer_size'] = int(1E5)
        variant['algorithm_kwargs']['num_eval_steps_per_epoch'] = 2000
        variant['algorithm_kwargs']['min_num_steps_before_training'] = 4000
        variant['algorithm_kwargs']['max_path_length'] = 400
    elif args.num_trains_per_train_loop == 500:
        variant['replay_buffer_size'] = int(1E5)
        variant['algorithm_kwargs']['num_eval_steps_per_epoch'] = 2500
        variant['algorithm_kwargs']['min_num_steps_before_training'] = 5000
        variant['algorithm_kwargs']['max_path_length'] = 500
    else:
        assert True == False
    variant['log_dir'] = get_log_dir(args)

    variant['seed'] = args.seed
    variant['annealing'] = args.annealing
    variant['redq'] = args.redq
    variant['bo'] = args.bo
    variant['domain'] = args.domain
    variant['use_aleatoric'] = args.use_aleatoric
    variant['trainer_kwargs']['use_aleatoric'] = args.use_aleatoric
    variant['trainer_kwargs']['redq'] = args.redq

    variant['algorithm_kwargs']['num_epochs'] = domain_to_epoch(args.domain)
    variant['algorithm_kwargs']['num_trains_per_train_loop'] = args.num_trains_per_train_loop
    variant['algorithm_kwargs']['num_expl_steps_per_train_loop'] = args.num_expl_steps_per_train_loop

    variant['optimistic_exp']['should_use'] = args.beta_UB > 0 or args.delta > 0
    variant['optimistic_exp']['beta_UB'] = args.beta_UB
    variant['optimistic_exp']['delta'] = args.delta
    variant['optimistic_exp']['use_aleatoric'] = args.use_aleatoric 
    variant['optimistic_exp']['redq'] = args.redq 
    variant['optimistic_exp']['version'] = args.version 
    variant['optimistic_exp']['seed'] = args.seed 

    variant['entropy_based_exp']['should_use'] = args.ee
    variant['entropy_based_exp']['alpha'] = args.alpha 
    variant['entropy_based_exp']['use_quantile_cdf'] = args.use_quantile_cdf 
    variant['entropy_based_exp']['alpha_2'] = args.alpha_2 
    variant['entropy_based_exp']['beta'] = args.beta 
    variant['entropy_based_exp']['nor'] = args.nor
    variant['entropy_based_exp']['sigma'] = args.sigma 
    variant['entropy_based_exp']['z'] = args.z 
    variant['entropy_based_exp']['version'] = args.version 
    variant['entropy_based_exp']['seed'] = args.seed 
    variant['entropy_based_exp']['use_automatic_z_tuning'] = args.autoz 
    variant['entropy_based_exp']['use_aleatoric'] = args.use_aleatoric 
    variant['entropy_based_exp']['redq'] = args.redq 
      

    if torch.cuda.is_available():
        gpu_id = 0
        # gpu_id = int(args.seed % torch.cuda.device_count())
    else:
        gpu_id = None

    run_experiment_here(experiment, variant,
                        seed=args.seed,
                        use_gpu=not args.no_gpu and torch.cuda.is_available(),
                        gpu_id=gpu_id,

                        # Save the params every snap0shot_gap and override previously saved result
                        snapshot_gap=10,
                        #snapshot_gap=100,
                        snapshot_mode='last_every_gap',

                        log_dir=variant['log_dir']

                        )
