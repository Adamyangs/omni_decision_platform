
import numpy as np
from collections import OrderedDict
from copy import deepcopy
import torch as t



class Added_High_Envs():
    def __init__(self, num_envs, max_nodes, agent_num, n_traj):
        self.n_traj = n_traj
        self.num_envs = num_envs
        self.max_nodes = max_nodes # depot
        self.agent_num = agent_num
        self.action_mask = 0 #!
        self.next_obs_multi = OrderedDict()
        self.step_max = 50
        self.depot_cus_nodes_num = max_nodes+1
        self.max_nodes_len = 70

        # self.reset(next_obs)
        

        
    #* high_actions_u 暂定是one-hot编码的
    def reset(self, next_obs):
        # to be determined node
        # self.tbd_node_idx = tbd_node_idx
        
        # static info
        self.next_obs_multi['observations'] = next_obs['observations']
        self.next_obs_multi['depot'] = next_obs['depot']
        self.next_obs_multi['demand'] = next_obs['demand']
        # self.tbd_node_demand = next_obs['demand'][:,self.tbd_node_idx-1]
        
        # dynamic info
        self.next_obs_multi['multi_visited_len_idx'] = np.ones((self.num_envs, self.n_traj, self.agent_num), dtype = int)  #每个agent分配到的长度， 类似于指针 从1开始，默认是depot
        self.next_obs_multi['multi_visited_nodes'] = np.zeros((self.num_envs, self.n_traj, self.agent_num, self.max_nodes_len), dtype=int) #每个agent分配到的节点列表
        self.next_obs_multi['multi_current_load'] = np.zeros((self.num_envs, self.n_traj, self.agent_num)) #每个agent分配到的负载总和 
        self.next_obs_multi['cluster_nodes_loc'] = np.zeros((self.num_envs, self.n_traj, self.agent_num,2), dtype=float)
        # self.next_obs_multi['tbd_node_idx'] = self.tbd_node_idx
        self.intr_reward = np.zeros(self.num_envs)
        
        return self.next_obs_multi

        
    # high_actions_u 暂定是不one-hot编码的 (num_envs, n_traj, 1) 作为索引
    def step(self, next_obs, high_actions_u, tbd_node_idx):
        self.tbd_node_idx = tbd_node_idx
        
        #! dynamic info
        #* multi_visited_len_idx 
        self.next_obs_multi['multi_visited_len_idx'] += np.squeeze(np.eye(self.agent_num)[high_actions_u], axis=1).astype(int)
        
        #* multi_current_load
        args=np.arange(0,next_obs['demand'].shape[0])
        tbd_node_demand = next_obs['demand'][args,(self.tbd_node_idx-1).squeeze()][:, np.newaxis][:, np.newaxis]
        self.next_obs_multi['multi_current_load'] += tbd_node_demand.repeat(self.agent_num, -1) * np.squeeze(np.eye(self.agent_num)[high_actions_u], axis=1).astype(int)
        
        #* multi_visited_nodes
        _nodes = self.tbd_node_idx[:,:,np.newaxis].repeat(self.agent_num,1).repeat(self.max_nodes_len,-1) # 25 --> [25, 25, 25, ...]
        _onehot_idx = np.eye(self.max_nodes_len)[(self.next_obs_multi['multi_visited_len_idx']-1).squeeze()] # 指针转换为one-hot
        _mask = np.eye(self.agent_num)[high_actions_u].squeeze()[:,:,np.newaxis].repeat(self.max_nodes_len,-1) # 通过high_actions_u来mask掉其他agent
        self.next_obs_multi['multi_visited_nodes'] += (_nodes * _onehot_idx * _mask)[:,np.newaxis,:,:].astype(int)
        
        # for i in range(self.next_obs_multi['multi_visited_len_idx'].shape[0]):
        #     for j in range(self.next_obs_multi['multi_visited_len_idx'].shape[1]):
        #         self.next_obs_multi['multi_visited_nodes'][i, j, high_actions_u[i,j], self.next_obs_multi['multi_visited_len_idx'][i,j][high_actions_u[i,j]]] = self.tbd_node_idx[i,j]
        #         self.next_obs_muslti['multi_visited_len_idx'][i,j,high_actions_u[i,j]] += 1
        #         self.next_obs_multi['multi_current_load'][i,j, high_actions_u[i,j]] += next_obs['demand'][i,self.tbd_node_idx[i,j]-1]

        import copy
        #! intrinsic rewards
        # 假设每个agent都有一个聚类中心，而且这个聚类中心是动态的：
        # 1. 当每个agent没有获得点的时候，它没有聚类中心
        # 2. 当每个agent有点的时候，它的聚类中心点是customer_node的均值点
        # intrinsic reward = -dist(tbd_node, cluster_center)
        # 如果没有聚类中心点的话，intrinsic reward = 0
        for i in range(self.num_envs):
            now_node_idx = self.next_obs_multi['multi_visited_nodes'][i,0,high_actions_u[i][0]][0][self.next_obs_multi['multi_visited_len_idx'][i,0][high_actions_u[i][0]]-1]
            now_loc = self.next_obs_multi['observations'][i][now_node_idx-1]
            
            if self.next_obs_multi['multi_visited_len_idx'][i,0,high_actions_u[i][0][0]]-1 <= 1:
                self.intr_reward[i] = 0
                # 更新cluster_nodes
                self.next_obs_multi['cluster_nodes_loc'][i, 0, high_actions_u[i][0][0]] = copy.copy(now_loc[0])
            else:
                visited_nodes_len = copy.copy(self.next_obs_multi['multi_visited_len_idx'][i,0,high_actions_u[i][0]]-2)
                self.intr_reward[i] = -self.dist(now_loc[0], self.next_obs_multi['cluster_nodes_loc'][i, 0, high_actions_u[i][0][0]])
                # 更新cluster_nodes
                self.next_obs_multi['cluster_nodes_loc'][i, 0, high_actions_u[i][0][0]] = \
                    (self.next_obs_multi['cluster_nodes_loc'][i, 0, high_actions_u[i][0][0]] * visited_nodes_len + now_loc) / (visited_nodes_len+1)

        #     last_node_idx = self.next_obs_multi['multi_visited_nodes'][i,0,high_actions_u[i][0]][self.next_obs_multi['multi_visited_len_idx'][i,0][high_actions_u[i][0]]-2]
            
        #     now_loc = self.next_obs_multi['observations'][i][now_node_idx-1] if now_node_idx != 0 else self.next_obs_multi['depot'][i]
        #     last_loc = self.next_obs_multi['observations'][i][last_node_idx-1] if last_node_idx != 0 else self.next_obs_multi['depot'][i]
            
        #     self.reward[i] = -self.dist(now_loc, last_loc)
        # reward = 0
        self.next_obs_multi['tbd_node_idx'] = self.tbd_node_idx
        return self.next_obs_multi, self.tbd_node_idx, self.intr_reward
    
    
    def dist(self, loc1, loc2):
        return ((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2) ** 0.5
    
    
    def _high_policy_out(self):
        #* for low_agent&low env
        self.depot_and_observation = np.concatenate(
            (np.expand_dims(self.next_obs_multi['depot'], 1),
            self.next_obs_multi['observations']),
            1)
        self.idx = deepcopy(self.next_obs_multi['multi_visited_nodes'].reshape((self.num_envs * self.n_traj, self.agent_num, self.max_nodes_len)))
    
    # 返回obs的idx对应的数据
    def high_policy_out(self, env_num_idx, agents_num_idx):
        # import time
        # t777 = time.time()        
        # t888 = time.time()
        # max_step = np.max(np.count_nonzero(idx, axis=-1))
        return self.depot_and_observation[env_num_idx][self.idx[env_num_idx, agents_num_idx]]
        # return self.next_obs_multi['observations'][env_num_idx][idx[env_num_idx, agents_num_idx]]
    
    def max_nodes_in_agents(self):
        idx = self.next_obs_multi['multi_visited_nodes'].reshape((self.num_envs * self.n_traj, self.agent_num, self.max_nodes_len))
        max_step = np.max(np.count_nonzero(idx, axis=-1))
        
        return max_step
        