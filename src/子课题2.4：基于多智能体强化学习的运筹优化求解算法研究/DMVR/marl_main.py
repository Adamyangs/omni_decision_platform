# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy

import argparse
import os
import random
import shutil
import time
from distutils.util import strtobool


import gym
import numpy as np
import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from wrappers.syncVectorEnvPomo import SyncVectorEnv
from wrappers.recordWrapper import RecordEpisodeStatistics

from marl_models.Policy.Low_Policy import low_agent_calculate_rewards
from envs.high_cvrp_vector_env import Added_High_Envs
from marl_models.Policy.High_Policy import Multi_Agent
from rl_utils import ReplayBuffer, angle
from learners.ppo import PPO
from learners.mappo import MAPPO
from copy import deepcopy
import torch.optim.lr_scheduler as lr_scheduler

def make_env(env_id, seed, cfg={}):
    def thunk():
        env = gym.make(env_id, **cfg)
        env = RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--PS", type=str, default="remove_tbd_in_Q",
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=5,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--writer_track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--problem", type=str, default="cvrp",
        help="the OR problem we are trying to solve, it will be passed to the agent")
    parser.add_argument("--customer_nodes", type=int, default=50,
        help="the OR problem we are trying to solve, it will be passed to the agent")
    parser.add_argument("--env-id", type=str, default="cvrp-v0",
        help="the id of the environment")
    parser.add_argument("--env-entry-point", type=str, default="envs.cvrp_vector_env:CVRPVectorEnv",
        help="the path to the definition of the environment, for example `envs.cvrp_vector_env:CVRPVectorEnv` if the `CVRPVectorEnv` class is defined in ./envs/cvrp_vector_env.py")

    parser.add_argument("--total-timesteps", type=int, default=10_000_000, # 6_000_000_000
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2e-3,
        help="the learning rate of the optimizer")
    parser.add_argument("--weight-decay", type=float, default=0,
        help="the weight decay of the optimizer")
    parser.add_argument("--num-envs", type=int, default=512,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=100,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=8,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=2,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--n-traj", type=int, default=1,
        help="number of trajectories in a vectorized sub-environment")
    parser.add_argument("--n-test", type=int, default=1000,
        help="how many test instance")
    parser.add_argument("--multi-greedy-inference", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to use multiple trajectory greedy inference")
    
    parser.add_argument("--agent-num", type=int, default=10,
        help="the number of agents")
    parser.add_argument("--embedding-dim", type=int, default=128,
        help="the number of agents")
    
    parser.add_argument("--depot_central", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    
    parser.add_argument("--n_traj_node_order", type=int, default=2,
        help="number of trajectories in a vectorized sub-environment")
    
    
    parser.add_argument("--n_traj_low_agent", type=int, default=4,
        help="number of trajectories in a vectorized sub-environment")
    
    parser.add_argument("--n_traj_low_agent_test", type=int, default=2,
        help="number of trajectories in a vectorized sub-environment")
    
    parser.add_argument("--agent_seed", type=int, default=5,
        help="seed of the experiment")
     
    
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def dis_func(x1, x2):
    dis = np.sqrt((x1[0] - x2[0])*(x1[0] - x2[0]) + (x1[1] - x2[1])*(x1[1] - x2[1]))
    return dis


def Unqiue_sort(x, agent_num=10):
    # x : numpy.array
    # x.shape: [512, 50]
    # Example: x[0]:
    # array([9, 3, 9, 9, 8, 6, 8, 8, 8, 9, 9, 3, 6, 8, 6, 9, 8, 8, 8, 6, 6, 9,
    #    7, 8, 3, 7, 7, 7, 6, 7, 5, 7, 7, 7, 5, 3, 6, 3, 5, 7, 3, 5, 4, 4,
    #    5, 5, 4, 4, 4, 2])
    
    # return y: numpy.array; shape: [512, 10]
    # Example: y[0]:
    # array([9,3,8,6,7,5,4,2])
    y = np.zeros((x.shape[0], agent_num), dtype=int)
    
    for i in range(x.shape[0]):
        unqiue_values = np.unique(x[i], return_index=True)[0]
        sorted_indices = np.argsort(np.unique(x[i], return_index=True)[1])
        y[i,:unqiue_values.shape[0]] = unqiue_values[sorted_indices]
    
    return y  



def Resort_order(visited_nodes, visited_len,  agent_order, customer_nodes):
    # visited_nodes: numpy.array
    # visited_nodes.shape: [512, 10, 28]
    # Example: visited_nodes[0]:
    # array([ 0,  1, 13,  5, 42, 23, 16, 49, 50,  0,  0,  0,  0,  0,  0,  0,  0,
        # 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0])
    # 
    # visited_len: numpy.array
    # visited_len.shape: [512, 10]
    # Example: visited_nodes[0]:
    # array([ 1,  2, 3,  5, 8, 7, ...])
    
    # agent_order: numpy.array
    # agent_order.shape: [512, 10]
    # Example: agent_order[0]:
    # array([9,3,8,6,7,5,4,2])
        
    # return trajectories_tensor: torch.tensor; shape: [512, 50]
    # Example: trajectories_tensor[0]:
    # trajectories_tensor
    # tensor([ 1, 37, 15, 18, 21, 30, 48,  6, 27, 34, 39,  9, 26,  4, 19, 20,  7, 11,
        # 44, 32, 13,  8, 35, 41, 40,  5, 42, 17, 29, 33, 36,  3, 12, 23, 22, 43,
        # 28, 10,  2, 38, 45, 31, 16, 46, 49, 24, 14, 25, 47, 50])

    trajectories_tensor = np.zeros((visited_nodes.shape[0], customer_nodes), dtype=int)
    
    for i in range(visited_nodes.shape[0]):
        trajectories_order = np.concatenate([visited_nodes[i,j] for j in agent_order[i]], axis=0)
        trajectories_tensor[i] = trajectories_order[trajectories_order!=0][:customer_nodes]
        assert trajectories_tensor[i].sum() == (1 + customer_nodes) * customer_nodes / 2
        
    return t.tensor(trajectories_tensor, dtype=t.int64)


#! 待补充
#* 待改善
if __name__ == "__main__":
    ####!###################
    ####! Hyper Params #####
    ####!###################
    args = parse_args()
    run_name = f"marl_cvrp_{args.customer_nodes}_{args.agent_seed}__{args.PS}__{int(time.time())}"
    # if args.track:
    #     import wandb
    #     wandb.init(
    #         project=args.wanbd_project_name,
    #         entity=args.wandb_entity,
    #         sync_tensorboard=True,
    #         config=vars(args),
    #         name=run_name,
    #         monitor_gym=True,   
    #         save_code=True,
    #     )
    if args.writer_track:
        writer = SummaryWriter(f"marl_runs_422/{run_name}")
        os.makedirs(os.path.join(f"marl_runs_422/{run_name}", "ckpt"), exist_ok=True)
        shutil.copy("marl_main.py", os.path.join(f"marl_runs_422/{run_name}","marl_main.py"))
    
    # set seeding & devive & buffer
    random.seed(args.seed)
    np.random.seed(args.seed)
    # t.manual_seed(args.seed)
    t.manual_seed(args.agent_seed)
    t.backends.cudnn.deterministic = args.torch_deterministic
    device = t.device("cuda" if t.cuda.is_available() and args.cuda else "cpu")

    
    ####!###################
    ####! Env defintion ####
    ####!###################   
    # *************base_cvrp_envs*************
    # if args.depot_central:
    #     gym.envs.register(id=args.env_id, entry_point="envs.cvrp_vector_env2:CVRPVectorEnv")
    # else:
    
    if args.customer_nodes == 50:
        capacity_limit = 60
        # order_agent_ckpt_path =  '/root/git_zjk/RLOR/runs/tsp-50__ppo_or__1__1711790404/ckpt/18300.pt'
        order_agent_ckpt_path = "/root/git_zjk/RLOR/runs/tsp-20__ppo_or__1__1711677659_final/ckpt/700.pt"
        
    elif args.customer_nodes == 100:
        capacity_limit = 80
        # order_agent_ckpt_path =  '/root/git_zjk/RLOR/runs/tsp-v0__ppo_or__1__1722789776/ckpt/2200.pt'
        # order_agent_ckpt_path = "/root/git_zjk/RLOR/runs/tsp-20__ppo_or__1__1711677659_final/ckpt/700.pt"
        order_agent_ckpt_path =  '/root/git_zjk/RLOR/runs/tsp-50__ppo_or__1__1711790404/ckpt/18300.pt'
    
    elif args.customer_nodes == 150:
        capacity_limit = 100
        order_agent_ckpt_path =  '/root/git_zjk/RLOR/runs/tsp-50__ppo_or__1__1711790404/ckpt/18300.pt'
    
    elif args.customer_nodes == 200:
        capacity_limit = 120
        order_agent_ckpt_path =  '/root/git_zjk/RLOR/runs/tsp-50__ppo_or__1__1711790404/ckpt/18300.pt'
    
    elif args.customer_nodes == 250:
        capacity_limit = 80
        
    gym.envs.register(id=args.env_id, entry_point=args.env_entry_point)
    envs = SyncVectorEnv([make_env(args.env_id, args.seed + i,
                                   cfg={"n_traj": args.n_traj,
                                        "max_nodes": args.customer_nodes,
                                        "capacity_limit": capacity_limit}) 
                                    for i in range(args.num_envs)])
    
    test_envs = SyncVectorEnv(
        [make_env(
                args.env_id,
                args.seed + i,
                cfg={"eval_data": True, "eval_partition": "eval", "n_traj": args.n_traj, "max_nodes": args.customer_nodes,"capacity_limit": capacity_limit,
                     "eval_data_idx": i},)
            for i in range(args.n_test)])
    
    
    # *************high_envs*************
    added_high_envs = Added_High_Envs(num_envs=args.num_envs, 
                                      max_nodes=args.customer_nodes,
                                      agent_num=args.agent_num, 
                                      n_traj = args.n_traj)
    
    test_added_high_envs = Added_High_Envs(num_envs=1000, 
                                      max_nodes=args.customer_nodes,
                                      agent_num=args.agent_num, 
                                      n_traj = args.n_traj)
        
    next_base_cvrp_obs = envs.reset()
    node4tsp = np.concatenate((next_base_cvrp_obs['depot'][:, np.newaxis], next_base_cvrp_obs['observations'],), axis=1)
    
    test_next_base_cvrp_obs = test_envs.reset()
    test_node4tsp = np.concatenate((test_next_base_cvrp_obs['depot'][:, np.newaxis], test_next_base_cvrp_obs['observations'],), axis=1)
    
    
    # *************node_order_envs 0-50*************
    node_order_env_id = 'node_env-v0'
    node_order_env_entry_point = 'envs.tsp_vector_env2:TSPVectorEnv'
    node_order_seed = 0
    
    gym.envs.register(
        id=node_order_env_id,
        entry_point=node_order_env_entry_point)
    
    node_order_envs = SyncVectorEnv(
        [
            make_env(
                node_order_env_id,
                node_order_seed+1,
                cfg={"n_traj":args.n_traj_node_order, "eval_data": True, 
                     "max_nodes": args.customer_nodes+1,
                     "node4tsp": node4tsp[i]})
            for i in range(args.num_envs * args.n_traj )
        ])
    
    test_node_order_envs = SyncVectorEnv(
        [
            make_env(
                node_order_env_id,
                node_order_seed+1,
                cfg={"n_traj":args.n_traj_node_order, "eval_data": True,
                      "max_nodes": args.customer_nodes+1,
                      "node4tsp": test_node4tsp[i]})
            for i in range(1000 * args.n_traj )
        ])
    
    #*************low_envs*************
    low_env_id = 'low_tsp-v0'
    low_env_entry_point = 'envs.low_tsp_vector_env:TSPVectorEnv'
    low_seed = 0

    gym.envs.register(
        id=low_env_id,
        entry_point=low_env_entry_point)
    
    low_envs = SyncVectorEnv(
        [
            make_env(
                low_env_id,
                low_seed + i,
                cfg={"n_traj":args.n_traj_low_agent, "eval_data": True,
                     "max_nodes": 20,
                     "eval_agents_idx": j, "eval_data_idx": i,})
            for i in range(args.num_envs * args.n_traj) for j in range(args.agent_num)
        ])
    
    test_low_envs = SyncVectorEnv(
        [
            make_env(
                low_env_id,
                low_seed + i,
                cfg={"n_traj":args.n_traj_low_agent_test, "eval_data": True,  
                     "max_nodes": 20,
                     "eval_agents_idx": j, "eval_data_idx": i,})
            for i in range(1000 * args.n_traj) for j in range(args.agent_num)
        ])
    
    ###!####################
    ###! Agent defintion ###
    ###!####################
    #* Node Order Agents: Pretrained TSP Agents
    from models.attention_model_wrapper import Agent
    device = 'cuda'
    # order_agent_ckpt_path = '/root/git_zjk/RLOR/runs/tsp-v50__ppo_or__1__1709020151/ckpt/100.pt'
    # order_agent_ckpt_path = '/root/git_zjk/RLOR/runs/tsp-50_8-1__ppo_or_50_200__1__1722486678/ckpt/75.pt'
    # order_agent_ckpt_path = "/root/git_zjk/RLOR/runs/tsp-20__ppo_or__1__1711677659_final/ckpt/700.pt"
    
    # order_agent_ckpt_path =  '/root/git_zjk/RLOR/runs/tsp-50__ppo_or__1__1711790404/ckpt/18300.pt'
    # order_agent_ckpt_path = "/root/git_zjk/RLOR/runs/tsp-20__ppo_or__1__1722585716/ckpt/700.pt"
    node_order_agent = Agent(device=device, name='tsp').to(device)
    node_order_agent.load_state_dict(t.load(order_agent_ckpt_path))
    node_order_agent.backbone.init()
    
    
    #* High Agents
    high_agents = Multi_Agent(agent_num = args.agent_num, 
                              embedding_dim=args.embedding_dim,
                              envs_num=args.num_envs, 
                              device=device, 
                              name='marl_cvrp').to(device)
    
    
    # high_agent_ckpt_path = '/root/git_zjk/RLOR/marl_runs_422/marl_cvrp_50_8__510__1722761459/ckpt/180.pt'
    high_agents.backbone.decoder.init()
    # high_agents.load_state_dict(t.load(high_agent_ckpt_path))
    
    
    
    #* Low Agents: Pretrained TSP Agents 
    device = 'cuda'
    # low_agent_ckpt_path = "/root/git_zjk/RLOR/runs/tsp-50__ppo_or__1__1711790404/ckpt/18400.pt"
    # low_agent_ckpt_path = "/root/git_zjk/RLOR/runs/tsp-20__ppo_or__1__1711987501/ckpt/12600.pt"
    low_agent_ckpt_path = '/root/git_zjk/RLOR/runs/tsp-50__ppo_or__1__1711790404/ckpt/18300.pt'
    # low_agent_ckpt_path = "/root/git_zjk/RLOR/runs/tsp-50__ppo_or__1__1722585754/ckpt/8300.pt"
    low_agent = Agent(device=device, name='tsp').to(device)
    low_agent.load_state_dict(t.load(low_agent_ckpt_path))
    
    low_agent.backbone.init()
    
    #######!################
    ######! Algorithm ######
    ######!#################
    high_buffer = ReplayBuffer(capacity=1000000)
    learner = MAPPO(high_agents, gamma=args.gamma, 
                  lmbda=args.gae_lambda, 
                  lr=args.learning_rate, 
                  update_epochs=args.update_epochs, 
                  agent_num=args.agent_num,
                  steps=args.customer_nodes,
                  envs_num=args.num_envs,
                  device='cuda',
                  num_minibatches = args.num_minibatches,
                  max_grad_norm = args.max_grad_norm,
                  high_envs = added_high_envs,
                  clip_vloss = args.clip_vloss, 
                  clip_coef = args.clip_coef, 
                  norm_adv = args.norm_adv,
                  vf_coef = args.vf_coef)
    
    #####!##################
    #####! Piplelines ######
    #####!##################
    steps = args.customer_nodes
    obs = [None] * (steps) # 每个obs已经包含了agent_num信息，即所有的agent的观测值
    actions = t.zeros((steps, args.num_envs, args.agent_num)).to(device)
    logprobs = t.zeros((steps, args.num_envs, args.agent_num)).to(device)
    high_rewards = np.zeros((steps, args.num_envs, args.agent_num))
    values = t.zeros((steps, args.num_envs, args.agent_num)).to(device)
    tbd_node_idxs = t.zeros(steps,args.num_envs, 1).to(device)
    
    #* collect best trajectory in tbd_node_idx
    # node_order_env进行reset; order agent按照顺序选择第一个节点
    TBD_node_idx = t.zeros((steps+1, args.num_envs, args.n_traj_node_order), dtype=int).to('cpu')
    order_obs = node_order_envs.reset()
    node_order_state = node_order_agent.backbone.encode(order_obs)
    reward_episode = 0
    for step in range(0, steps+1): # args.numsteps
        with t.no_grad():
            #* node order policy
            tbd_node_idx, logits,_,_,_ = node_order_agent.get_action_and_value_cached(order_obs, state = node_order_state) # 0.1
        if step == 0:
            #* depot node, matters!
            tbd_node_idx = (t.arange(args.n_traj_node_order)+1).repeat(order_obs['action_mask'].shape[0], 1)
        order_obs, order_reward, done, info = node_order_envs.step(tbd_node_idx.cpu().numpy())
        reward_episode += order_reward
        TBD_node_idx[step]=deepcopy(tbd_node_idx.to('cpu'))
    
    max_id = np.argmax(reward_episode, axis=-1)[:, np.newaxis]
    max_id_tensor = t.tensor(max_id, device='cpu')
    best_trajectories_tensor = t.gather(TBD_node_idx, -1, max_id_tensor.unsqueeze(0).repeat(steps+1,1,1))
    nonzero_mask = (best_trajectories_tensor != 0)
    assert nonzero_mask.sum() == (nonzero_mask.shape[0]-1) * nonzero_mask.shape[1]
    # x = best_trajectories_tensor.squeeze(-1).T
    # y = nonzero_mask.squeeze(-1).T
    #* [1024, 50]
    Best_trajectories_tensor = best_trajectories_tensor.squeeze(-1).T[nonzero_mask.squeeze(-1).T].view(best_trajectories_tensor.shape[1], -1)
    
    
    
    #* start the game
    global_step = 0
    num_updates = args.total_timesteps // (args.num_envs * steps)
    indices = t.arange(0, args.customer_nodes).unsqueeze(0)
    step_idx = (indices + t.arange(num_updates).unsqueeze(1)) % args.customer_nodes
    random_permutation = t.randperm(args.customer_nodes)
    test_episode_rewards_list = []
    test_trajectories_list = []

    for update in range(0, num_updates):
        #! try
        t.cuda.empty_cache()
        #* reset some things
        high_buffer.reset()
        high_agents.train()
        tbd_node_idx_list = []
        action_conflict_list = []
        
        #* high_envs reset; high_agents进行encode
        next_obs_multi = added_high_envs.reset(next_base_cvrp_obs)
        encoder_state = high_agents.backbone.encode(next_obs_multi)

        step=0
        for step_id in step_idx[update]: # args.numsteps
        # for step_id in random_permutation: # args.numsteps
            global_step += 1 * args.num_envs
            obs[step] = deepcopy(next_obs_multi)
            # if update == 0:
            tbd_node_idx = deepcopy(Best_trajectories_tensor[:,step_id].unsqueeze(-1).to('cuda'))
            # else:
            #     tbd_node_idx = deepcopy(tbd_node_idx_agents[:,step_id].unsqueeze(-1).to("cuda"))
            with t.no_grad():
                #* high policy
                action, high_actions_u, log_probs, value, entropy, action_conflict = high_agents.get_action_and_value_cached(
                    next_obs_multi, state=encoder_state, tbd_node_idx=tbd_node_idx) 
            #* high_envs step.
            next_base_cvrp_obs, _, done, info = envs.step(tbd_node_idx.cpu().numpy())
            next_obs_multi, _, intr_reward = added_high_envs.step(next_base_cvrp_obs, high_actions_u.unsqueeze(-1).cpu().numpy(), tbd_node_idx.cpu().numpy())
            assert next_obs_multi['multi_current_load'].max()<1   
            #* intrinsic rewards
            high_rewards[step] = 0.1 * np.repeat(intr_reward[:,None], repeats=high_rewards.shape[-1], axis=-1)
            # for i in range(args.num_envs):
            #     high_rewards[step][i, high_actions_u[i][0]] = reward[i]  
            #     high_rewards[step][i, :] = reward[i]  
            actions[step]=action
            logprobs[step]=log_probs
            values[step] = value
            tbd_node_idxs[step] = deepcopy(tbd_node_idx)
            action_conflict_list.append(action_conflict.cpu())
            tbd_node_idx_list.append(high_actions_u.cpu().numpy())
            step+=1
        
        
        # if update >=30 :
        #     print("trajectories_tensor: changed.")
        #     agent_action_order = Unqiue_sort(np.concatenate(tbd_node_idx_list,-1), agent_num=args.agent_num) 
        #     Best_trajectories_tensor =Resort_order(next_obs_multi['multi_visited_nodes'][:,0,:,:],next_obs_multi['multi_visited_len_idx'][:,0,:], agent_action_order, args.customer_nodes)
        
        tbd_node_idx_agents = t.tensor(deepcopy(next_obs_multi['multi_visited_nodes'][:,0,:,:].reshape(-1,1)[next_obs_multi['multi_visited_nodes'][:,0,:,:].reshape(-1,1)!=0].reshape(args.num_envs,-1)))
        
        assert next_obs_multi['multi_visited_nodes'].sum() == args.num_envs * args.n_traj * (1+args.customer_nodes)*args.customer_nodes/2
        #* low agents take actions here.
        t7 = time.time()
        max_nodes_in_agents = added_high_envs.max_nodes_in_agents()+1
        added_high_envs._high_policy_out()
        episode_rewards, trajectories = low_agent_calculate_rewards(TSPDataset1 = added_high_envs.high_policy_out, 
                                                                    env_num = args.num_envs, 
                                                                    n_traj = args.n_traj, 
                                                                    low_envs = low_envs,
                                                                    low_agent = low_agent,
                                                                    agent_num=args.agent_num, 
                                                                    max_nodes=max_nodes_in_agents,
                                                                    multi_visited_len_idx = next_obs_multi['multi_visited_len_idx'],
                                                                    device=device)
        
        R_max = episode_rewards.sum(-1).max()
        R_min = episode_rewards.sum(-1).min()
        #* try: intrinsic rewards
        # high_rewards[-1] = np.repeat(np.expand_dims(episode_rewards.sum(-1), -1), args.agent_num, axis=-1) + 0.5*episode_rewards
        high_rewards[-1] = np.repeat(np.expand_dims(episode_rewards.sum(-1), -1), args.agent_num, axis=-1)
        print("[train] episode_return={}, max={}, min={}, high_rewards_sum={}, intr_rewards_sum={} global_step={}, action_conflict={} \n".format(episode_rewards.sum(-1).mean(), R_max, R_min, high_rewards[:,:,0].sum(0).mean(), high_rewards[:-1,:,0].sum(0).mean(), global_step, np.array(action_conflict_list).mean()))
        print("[train] valid_vehicle_num={}\n".format((next_obs_multi['multi_current_load'][:,0]>0.0001).sum()/(next_obs_multi['multi_current_load'].shape[0])))
        
        # add valid agent_num
        # valid_num = next_obs_multi['multi_current_load'].shape[0]
        # for i in range(valid_num):
        #     (next_obs_multi['multi_current_load'][:,0]>0.0001).sum()
        
        if args.writer_track:
            writer.add_scalar("charts/episode_return", episode_rewards.sum(-1).mean(), global_step)
            writer.add_scalar("charts/episode_return_max", R_max, global_step)
            writer.add_scalar("charts/episode_return_min", R_min, global_step)
            writer.add_scalar("charts/action_conflict", np.array(action_conflict_list).mean(), global_step)
            # (next_obs_multi['multi_current_load'][:,0]>0.0001).sum()/(next_obs_multi['multi_current_load'].shape[0])
            writer.add_scalar("charts/{}_agent_num".format(args.customer_nodes), (next_obs_multi['multi_current_load'][:,0]>0.000001).sum()/(next_obs_multi['multi_current_load'].shape[0]), global_step)
            
        # if update>=1:
        #* update high_agents
        high_buffer.add(obs, actions, logprobs, high_rewards, values, tbd_node_idxs)
        learner.update(high_buffer, update, num_updates)
    
        print("hhhhhh")
        
        t.cuda.empty_cache()
        #* save high_agent model
        if args.writer_track:
            if update % 20 == 0 or update == num_updates:
                # t.save(high_agents.state_dict(), f"marl_runs/{run_name}/ckpt/{update}.pt")
                t.save(high_agents.state_dict(), f"marl_runs_422/{run_name}/ckpt/{update}.pt")
                print("high_agents model saved,{}-epoch.".format(update))
            
        # #####!##################
        # #####! Test ############
        # #####!##################
        # #* record and test
        # if update % 10 == 0 or update == num_updates:
        #     t.cuda.empty_cache()
        #     #* collect best trajectory in tbd_node_idx
        #     test_TBD_node_idx = t.zeros((steps+1, 1000, args.n_traj_low_agent_test), dtype=int).to('cpu')
        #     test_order_obs = test_node_order_envs.reset()
        #     test_node_order_state = node_order_agent.backbone.encode(test_order_obs)
        #     test_reward_episode = 0
        #     for step in range(0, steps+1): # args.numsteps
        #         with t.no_grad():
        #             #* node order policy
        #             test_tbd_node_idx, logits,_,_,_ = node_order_agent.get_action_and_value_cached_test(test_order_obs, state = test_node_order_state) # 0.1
        #         if step == 0:
        #             #* depot node, matters!
        #             test_tbd_node_idx = (t.arange(args.n_traj_low_agent_test)+1).repeat(test_order_obs['action_mask'].shape[0], 1)
        #         test_order_obs, test_order_reward, done, info = test_node_order_envs.step(test_tbd_node_idx.cpu().numpy())
        #         test_reward_episode += test_order_reward
        #         test_TBD_node_idx[step]=deepcopy(test_tbd_node_idx.to('cpu'))
            
        #     test_max_id = np.argmax(test_reward_episode, axis=-1)[:, np.newaxis]
        #     test_max_id_tensor = t.tensor(test_max_id, device='cpu')
        #     test_best_trajectories_tensor = t.gather(test_TBD_node_idx, -1, test_max_id_tensor.unsqueeze(0).repeat(steps+1,1,1))
        #     test_nonzero_mask = (test_best_trajectories_tensor != 0)
        #     assert test_nonzero_mask.sum() == (test_nonzero_mask.shape[0]-1) * test_nonzero_mask.shape[1]
        #     # x = best_trajectories_tensor.squeeze(-1).T
        #     # y = nonzero_mask.squeeze(-1).T
        #     #* [1000, 50]
        #     test_Best_trajectories_tensor = test_best_trajectories_tensor.squeeze(-1).T[test_nonzero_mask.squeeze(-1).T].view(test_best_trajectories_tensor.shape[1], -1)
            
            
        #     n_traj_test_for_best_result = 3
        #     jian_ge = args.customer_nodes//n_traj_test_for_best_result
        #     indices = t.arange(0, args.customer_nodes).unsqueeze(0)
        #     step_idx_test = (indices + t.arange(0, args.customer_nodes, jian_ge).unsqueeze(1)) % args.customer_nodes
        #     test_episode_rewards_list = []
        #     test_trajectories_list = []
        #     print("step_idx_test.shape[0]:{}".format(step_idx_test.shape[0]))
            
        #     for i in range(step_idx_test.shape[0]):
        #         #* reset some things
        #         test_tbd_node_idx_list = []
        #         test_action_conflict_list = []
        
        #         #* high_envs reset; high_agents进行encode
        #         test_next_obs_multi = test_added_high_envs.reset(test_next_base_cvrp_obs)
        #         with t.no_grad():
        #             test_encoder_state = high_agents.backbone.encode(test_next_obs_multi)

        #         # for step in range(0, steps): # args.numsteps
        #         for step in step_idx_test[i]: # args.numsteps
        #             test_tbd_node_idx = deepcopy(test_Best_trajectories_tensor[:,step].unsqueeze(-1).to('cuda'))
        #             with t.no_grad():
        #                 #* high policy
        #                 action, high_actions_u, log_probs, value, entropy, action_conflict = high_agents.get_action_and_value_cached_test(
        #                     test_next_obs_multi, state=test_encoder_state, tbd_node_idx=test_tbd_node_idx) 
        #             # t4 = time.time()
        #             #* high_envs step.
        #             test_next_base_cvrp_obs, _, done, info = test_envs.step(test_tbd_node_idx.cpu().numpy())
        #             test_next_obs_multi, _, reward = test_added_high_envs.step(test_next_base_cvrp_obs, high_actions_u.unsqueeze(-1).cpu().numpy(), test_tbd_node_idx.cpu().numpy())
                
        #             assert next_obs_multi['multi_current_load'].max()<1   
        #             test_action_conflict_list.append(action_conflict.cpu())
                
            
        #         assert test_next_obs_multi['multi_visited_nodes'].sum() == 1000 * args.n_traj * (1+args.customer_nodes)*args.customer_nodes/2
        #         #* low agents take actions here.
        #         max_nodes_in_agents = test_added_high_envs.max_nodes_in_agents()+1
        #         test_added_high_envs._high_policy_out()
        #         test_episode_rewards, trajectories = low_agent_calculate_rewards(TSPDataset1 = test_added_high_envs.high_policy_out, 
        #                                                                     env_num = 1000, 
        #                                                                     n_traj = args.n_traj, 
        #                                                                     low_envs = test_low_envs,
        #                                                                     low_agent = low_agent,
        #                                                                     agent_num=args.agent_num, 
        #                                                                     max_nodes=max_nodes_in_agents,
        #                                                                     multi_visited_len_idx = test_next_obs_multi['multi_visited_len_idx'],
        #                                                                     device=device)
                
        #         test_episode_rewards_list.append(test_episode_rewards.sum(-1))
        #         test_trajectories_list.append(trajectories)
        #         t.cuda.empty_cache()
            
        #     test_episode_rewards_final = np.array(test_episode_rewards_list)
        #     tb_print_episode_rewards = test_episode_rewards_final.max(0)
        #     test_R_max = tb_print_episode_rewards.max()
        #     test_R_min = tb_print_episode_rewards.min()
        #     print("[test] episode_return={}, max={}, min={}, global_step={}, action_conflict={}".format(tb_print_episode_rewards.mean(),test_R_max, test_R_min, global_step, np.array(test_action_conflict_list).mean()))
            
        #     if args.writer_track:
        #         writer.add_scalar("test/episode_return", tb_print_episode_rewards.mean(), global_step)
        #         writer.add_scalar("test/episode_return_max", test_R_max, global_step)
        #         writer.add_scalar("test/episode_return_min", test_R_min, global_step)
        #         writer.add_scalar("test/action_conflict", np.array(test_action_conflict_list).mean(), global_step)
            
            t.cuda.empty_cache()
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
                
        

    
    
    
    