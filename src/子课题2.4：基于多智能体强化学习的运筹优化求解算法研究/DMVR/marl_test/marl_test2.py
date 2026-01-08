# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy

import sys
import os
sys.path.insert(0, os.getcwd())
sys.path.insert(1, "../")
sys.path.insert(2, "../../")

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
from rl_utils import ReplayBuffer
from learners.ppo import PPO
from copy import deepcopy


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
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=False,
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
    parser.add_argument("--learning-rate", type=float, default=3e-3,
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
    parser.add_argument("--n-test", type=int, default=500,
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
    
    parser.add_argument("--n_test_start", type=int, default=0,
        help="number of trajectories in a vectorized sub-environment")

    
    parser.add_argument("--n_traj_low_agent_test", type=int, default=12,
        help="number of trajectories in a vectorized sub-environment")
     
    
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def dis_func(x1, x2):
    dis = np.sqrt((x1[0] - x2[0])*(x1[0] - x2[0]) + (x1[1] - x2[1])*(x1[1] - x2[1]))
    return dis

def agents2agent(trajectories, nodes, real_len):
        # trajectories: agent_num, steps_num
        # nodes: agent_num, max_nodes+1
        agent_num = trajectories.shape[0]
        real_trajectory = []
        
        for i in range(agent_num):
            real_traj_id_per_agent = trajectories[i][:real_len[i]+1].astype(np.int32)
            real_traj_per_agent = nodes[i][real_traj_id_per_agent]
            if real_traj_per_agent[0] != 0:
                traj = np.zeros(len(real_traj_per_agent), dtype=np.int32)
                start_idx = int(np.where(real_traj_per_agent==0)[0])
                traj[:-start_idx]=real_traj_per_agent[start_idx:]
                traj[-start_idx:-1]=real_traj_per_agent[1: start_idx]
                traj[-1]=0
            else:
                traj = real_traj_per_agent
                
            real_trajectory.append(traj[:-1])
        
        return np.append(np.concatenate(real_trajectory),0)


import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    return segments

def colorline(x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), linewidth=2, alpha=1.0):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''
    
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.3, 1.0, len(x))
        
    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])
        
    z = np.asarray(z)
    
    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    
    ax = plt.gca()
    ax.add_collection(lc)
    
    return lc

def plot(env_id, coords, demand, traj_color):
    x,y = coords.T
    lc = colorline(x,y,z = np.array([traj_color]*len(x)), cmap='jet')
    plt.axis('square')
    x, y =test_next_base_cvrp_obs['observations'][env_id].T
    # h = demand/4
    h = test_next_base_cvrp_obs['demand'][env_id]/4
    h = np.vstack([h*0,h])
    plt.errorbar(x,y,h,fmt='None',elinewidth=2)

    return lc


import time

#! 待补充
#* 待改善
if __name__ == "__main__":
    # -1  --->  -1000  由好到坏
    tb_print_id = -100
    
    
    ####!###################
    ####! Hyper Params #####
    ####!###################
    args = parse_args()
    run_name = f"marl_cvrp_{args.customer_nodes}_{args.seed}__{args.PS}__{int(time.time())}"

    # set seeding & devive & buffer
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.customer_nodes == 50:
        t.manual_seed(3)
        capacity_limit = 60 
        high_agent_ckpt_path = './marl_runs_422/marl_cvrp_50_8__510__1723005886/ckpt/100.pt'
        low_agent_ckpt_path = './runs/tsp-50__ppo_or__1__1711790404/ckpt/18300.pt'
    
    t.backends.cudnn.deterministic = False
    device = t.device("cuda" if t.cuda.is_available() and args.cuda else "cpu")

    
    ####!###################
    ####! Env defintion ####
    ####!###################   
    
    t1 = time.time()
    # *************base_cvrp_envs*************
    gym.envs.register(id=args.env_id, entry_point=args.env_entry_point)
    
    test_envs = SyncVectorEnv(
        [make_env(
                args.env_id,
                args.seed + i,
                cfg={"eval_data": True, "eval_partition": "eval", "n_traj": args.n_traj, "max_nodes": args.customer_nodes,
                     "eval_data_idx": i},)
            for i in range(args.n_test_start, args.n_test_start + args.n_test)])
    
    
    t2 = time.time()
    # *************high_envs************
    test_added_high_envs = Added_High_Envs(num_envs=args.n_test, 
                                      max_nodes=args.customer_nodes,
                                      agent_num=args.agent_num, 
                                      n_traj = args.n_traj)
        
    test_next_base_cvrp_obs = test_envs.reset()
    test_node4tsp = np.concatenate((test_next_base_cvrp_obs['depot'][:, np.newaxis], test_next_base_cvrp_obs['observations'],), axis=1)
    
    
    t3 = time.time()
    # *************node_order_envs 0-50*************
    node_order_env_id = 'node_env-v0'
    node_order_env_entry_point = 'envs.tsp_vector_env2:TSPVectorEnv'
    node_order_seed = 0
    
    gym.envs.register(
        id=node_order_env_id,
        entry_point=node_order_env_entry_point)
    
    t4 = time.time()
    test_node_order_envs = SyncVectorEnv(
        [
            make_env(
                node_order_env_id,
                node_order_seed+1,
                cfg={"n_traj":args.n_traj_node_order, "eval_data": True,
                      "max_nodes": args.customer_nodes+1,
                      "node4tsp": test_node4tsp[i]})
            for i in range(args.n_test * args.n_traj )
        ])
    
    #*************low_envs*************
    low_env_id = 'low_tsp-v0'
    low_env_entry_point = 'envs.low_tsp_vector_env:TSPVectorEnv'
    low_seed = 0

    gym.envs.register(
        id=low_env_id,
        entry_point=low_env_entry_point)
    
    t5 = time.time()
    test_low_envs = SyncVectorEnv(
        [
            make_env(
                low_env_id,
                low_seed + i,
                cfg={"n_traj":args.n_traj_low_agent_test, "eval_data": True,  
                     "max_nodes": 60,
                     "eval_agents_idx": j, "eval_data_idx": i,})
            for i in range(args.n_test * args.n_traj) for j in range(args.agent_num)
        ])
    
    t55 = time.time()
    ###!####################
    ###! Agent defintion ###
    ###!####################
    #* Node Order Agents: Pretrained TSP Agents
    from models.attention_model_wrapper import Agent
    device = 'cuda'
    order_agent_ckpt_path = './runs/tsp-50__ppo_or__1__1711790404/ckpt/18300.pt'
    node_order_agent = Agent(device=device, name='tsp').to(device)
    node_order_agent.load_state_dict(t.load(order_agent_ckpt_path))
    node_order_agent.backbone.init()
    
    #* High Agents
    high_agents = Multi_Agent(agent_num = args.agent_num, 
                              embedding_dim=args.embedding_dim,
                              envs_num=args.num_envs, 
                              device=device, 
                              name='marl_cvrp').to(device)
       
    high_agents.backbone.decoder.init()
    high_agents.load_state_dict(t.load(high_agent_ckpt_path))
    
    #* Low Agents: Pretrained TSP Agents 
    device = 'cuda'
    low_agent = Agent(device=device, name='tsp').to(device)
    low_agent.load_state_dict(t.load(low_agent_ckpt_path))
    low_agent.backbone.init()
    

    t6 = time.time()
    #! start test
    steps = args.customer_nodes
    t.cuda.empty_cache()
    #* collect best trajectory in tbd_node_idx
    test_TBD_node_idx = t.zeros((steps+1, args.n_test, args.n_traj_node_order), dtype=int).to('cpu')
    test_order_obs = test_node_order_envs.reset()
    t661 = time.time()
    test_node_order_state = node_order_agent.backbone.encode(test_order_obs)
    t662 = time.time()
    test_reward_episode = 0
    t66 = time.time()
    for step in range(0, steps+1): # args.numsteps
        with t.no_grad():
            #* node order policy
            test_tbd_node_idx, logits,_,_,_ = node_order_agent.get_action_and_value_cached_test(test_order_obs, state = test_node_order_state) # 0.1
        if step == 0:
            #* depot node, matters!
            test_tbd_node_idx = (t.arange(args.n_traj_node_order)+1).repeat(test_order_obs['action_mask'].shape[0], 1)
        test_order_obs, test_order_reward, done, info = test_node_order_envs.step(test_tbd_node_idx.cpu().numpy())
        test_reward_episode += test_order_reward
        test_TBD_node_idx[step]=deepcopy(test_tbd_node_idx.to('cpu'))
    
    # test_max_id = np.argmax(test_reward_episode, axis=-1)[:, np.newaxis]
    # test_max_id_tensor = t.tensor(test_max_id, device='cpu')
    # test_best_trajectories_tensor = t.gather(test_TBD_node_idx, -1, test_max_id_tensor.unsqueeze(0).repeat(steps+1,1,1))
    # test_nonzero_mask = (test_best_trajectories_tensor != 0)
    # assert test_nonzero_mask.sum() == (test_nonzero_mask.shape[0]-1) * test_nonzero_mask.shape[1]
    # #* [1000, 50]
    # test_Best_trajectories_tensor = test_best_trajectories_tensor.squeeze(-1).T[test_nonzero_mask.squeeze(-1).T].view(test_best_trajectories_tensor.shape[1], -1)
    
    
    best_result = 0
    test_episode_rewards_list = []
    test_trajectories_list = []
    for z in range(test_TBD_node_idx.shape[-1]):
        test_Best_trajectories_tensor = test_TBD_node_idx[:,:,z].T[test_TBD_node_idx[:,:,z].T != 0].reshape(-1, args.customer_nodes)
        # test_Best_trajectories_tensor = test_TBD_node_idx[:,:,z]
        # test_Best_trajectories_tensor_different_order = test_Best_trajectories_tensor
        x = t.arange(25).unsqueeze(1)
        y = t.arange(args.customer_nodes-25, args.customer_nodes).unsqueeze(1)
        
        indices = t.arange(0, args.customer_nodes).unsqueeze(0)
        # t.arange(20).unsqueeze(1)
        step_idx = (indices + t.cat((x,y))) % args.customer_nodes

        
        t7 = time.time()
        #* high_envs reset; high_agents进行encode
        # test_next_obs_multi_copy = test_added_high_envs.reset(test_next_base_cvrp_obs)
        t.cuda.empty_cache()
        test_next_base_cvrp_obs_copy = test_next_base_cvrp_obs
        test_next_obs_multi_copy = test_added_high_envs.reset(test_next_base_cvrp_obs_copy)
        with t.no_grad():
            test_encoder_state_copy = high_agents.backbone.encode(test_next_obs_multi_copy)
        
        t.cuda.empty_cache()
        t8888 = time.time() 
        for i in range(30):
        #* reset some things
            t8 = time.time() 
            test_tbd_node_idx_list = []
            test_action_conflict_list = []
            
            test_next_obs_multi = deepcopy(test_next_obs_multi_copy)
            test_encoder_state = deepcopy(test_encoder_state_copy)
            #  = test_next_base_cvrp_obs
            test_next_base_cvrp_obs = deepcopy(test_next_base_cvrp_obs_copy)
            
            _ = test_added_high_envs.reset(test_next_base_cvrp_obs_copy)
            
            for step in step_idx[i]: # args.numsteps
                test_tbd_node_idx = deepcopy(test_Best_trajectories_tensor[:,step].unsqueeze(-1).to('cuda'))
                with t.no_grad():
                    #* high policy
                    action, high_actions_u, log_probs, value, entropy, action_conflict = high_agents.get_action_and_value_cached_test(
                        test_next_obs_multi, state=test_encoder_state, tbd_node_idx=test_tbd_node_idx) 
                
                t881 = time.time()
                #* high_envs step.
                test_next_base_cvrp_obs, _, done, info = test_envs.step(test_tbd_node_idx.cpu().numpy())
                t882 = time.time()
                test_next_obs_multi, _, reward = test_added_high_envs.step(test_next_base_cvrp_obs, high_actions_u.unsqueeze(-1).cpu().numpy(), test_tbd_node_idx.cpu().numpy())
                t883 = time.time()
                # print(" ")
            
                assert test_next_obs_multi['multi_current_load'].max()<1   
                test_action_conflict_list.append(action_conflict.cpu())
            
            t81 = time.time() 
            assert test_next_obs_multi['multi_visited_nodes'].sum() == args.n_test * args.n_traj * (1+args.customer_nodes)*args.customer_nodes/2
            #* low agents take actions here.
            max_nodes_in_agents = test_added_high_envs.max_nodes_in_agents()+1
            test_added_high_envs._high_policy_out()
            test_episode_rewards, trajectories = low_agent_calculate_rewards(TSPDataset1 = test_added_high_envs.high_policy_out, 
                                                                        env_num = args.n_test, 
                                                                        n_traj = args.n_traj, 
                                                                        low_envs = test_low_envs,
                                                                        low_agent = low_agent,
                                                                        agent_num=args.agent_num, 
                                                                        max_nodes=max_nodes_in_agents,
                                                                        multi_visited_len_idx = test_next_obs_multi['multi_visited_len_idx'],
                                                                        device=device)
            
            t82 = time.time() 
            test_episode_rewards_list.append(test_episode_rewards.sum(-1))
            test_trajectories_list.append(trajectories)
            
            # best_result = max(best_result, np.array(test_episode_rewards_list).max(0).mean())
            # print("第{}次测试完成,目前最好值是{}, 所需要的时间为{}s".format(i + (z*args.customer_nodes), np.array(test_episode_rewards_list).max(0).mean(), t82-t8))
            print("目前求解值是{}, 总耗时为{}s".format(np.array(test_episode_rewards_list).max(0).mean(), t82-t8888))
            t.cuda.empty_cache()
            t9 = time.time()
            print(" ")
        
        t10 = time.time()
        test_episode_rewards_final = np.array(test_episode_rewards_list)
        
        tb_print_episode_rewards = test_episode_rewards_final.max(0)
            
        test_R_max = tb_print_episode_rewards.max()
        test_R_min = tb_print_episode_rewards.min()
        print("[test] episode_return={}, max={}, min={}, action_conflict={}".format(tb_print_episode_rewards.mean(),test_R_max, test_R_min, np.array(test_action_conflict_list).mean()))


            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
                
        

    
    
    
    