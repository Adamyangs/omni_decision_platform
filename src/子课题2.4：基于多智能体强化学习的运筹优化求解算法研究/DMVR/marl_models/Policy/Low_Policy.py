import numpy as np
import torch
import gym
import sys
import os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, "/root/git_zjk/RLOR/models")
sys.path.insert(0, "../")
sys.path.insert(0, "../../")
from models.attention_model_wrapper import Agent
from wrappers.syncVectorEnvPomo import SyncVectorEnv
from wrappers.recordWrapper import RecordEpisodeStatistics

def make_env(env_id, seed, cfg={}):
    def thunk():
        env = gym.make(env_id, **cfg)
        env = RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

import time 

##############################################
############ Calcualte Rewards ###############
##############################################
def low_agent_calculate_rewards(TSPDataset1, 
                                low_envs, 
                                low_agent, 
                                env_num, 
                                n_traj, 
                                agent_num, 
                                max_nodes,
                                multi_visited_len_idx,
                                device):
    ############ Environment Setting #############
    t11 = time.time()
    TSPD = {'TSPDataset': TSPDataset1}
    obs = low_envs.reset(options = TSPD) # 9.09
    test_traj = obs['action_mask'].shape[1]
    t22 = time.time()
    num_steps = int(max_nodes*0.75)
    trajectories = np.zeros((num_steps, env_num * n_traj * agent_num, test_traj))
    low_agent.eval()
    reward_episode = 0
    
    ############ calculate rewards #############  
    encoder_state = low_agent.backbone.encode(obs)
    t221 = time.time()
    for step in range(0, num_steps):
        # ALGO LOGIC: action logic
        t111=time.time()
        with torch.no_grad():
            action, logprob, _, value, _ = low_agent.get_action_and_value_cached_test(
            obs, state=encoder_state)
        if step == 0:
            action_arange = torch.arange(num_steps).repeat(env_num * n_traj * agent_num, 1)
            mask  = np.arange(num_steps) < multi_visited_len_idx.reshape(-1,1)
            action = (action_arange * mask)[:,:test_traj]
            
        t222=time.time()
        obs, reward, done, info = low_envs.step(action.cpu().numpy()) # 0.58
        t333=time.time()
        reward_episode += reward
        trajectories[step] = action.cpu().numpy()

    t33 = time.time()
    max_id = np.argmax(reward_episode, axis=-1)[:, np.newaxis]
    
    reward_episode_tensor = torch.tensor(reward_episode, device='cpu')
    max_id_tensor = torch.tensor(max_id, device='cpu')
    trajectories_tensor = torch.tensor(trajectories, device='cpu')
    
    best_reward_episode_tensor = torch.gather(reward_episode_tensor, -1, max_id_tensor)
    best_trajectories_tensor = torch.gather(trajectories_tensor, -1, max_id_tensor.unsqueeze(0).repeat(num_steps,1,1))

    best_reward_episode = best_reward_episode_tensor.numpy()
    best_trajectories = best_trajectories_tensor.numpy()
    return best_reward_episode.reshape(env_num, agent_num), np.transpose(best_trajectories.reshape(num_steps,env_num, agent_num), axes=(1,2,0)) # (env_num, agent_num, num_steps)



if __name__ == "__main__":
    reward_episode = low_agent_calculate_rewards()
    print("low_rewards: ", reward_episode)