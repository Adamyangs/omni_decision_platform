import gym
import numpy as np
from gym import spaces
import torch as t
from torch import nn
import torch.nn.functional as F

from marl_models.Nets.embedding import AutoEmbedding
from marl_models.Nets.encoder import GraphAttentionEncoder
from marl_models.Nets.docoder import Decoder

from copy import deepcopy
from torch.distributions import Categorical



class Problem:
    def __init__(self, name):
        self.NAME = name
        

class Backbone(nn.Module):
    def __init__(
        self,
        embedding_dim=128,
        problem_name="marl_cvrp",
        n_encode_layers=3,
        tanh_clipping=10.0,
        n_heads=8,
        device="cuda",
        agent_num=10,
        envs_num=40
    ):
        super(Backbone, self).__init__()
        self.agent_num = agent_num
        self.envs_num = envs_num
        self.device = device
        self.problem = Problem(problem_name)
        self.embedding = AutoEmbedding(self.problem.NAME, {"embedding_dim": embedding_dim})
        
        self.encoder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim = embedding_dim,
            n_layers=n_encode_layers)
        
        if t.cuda.device_count() > 1:
            device_ids = []
            for i in range(t.cuda.device_count()):
                device_ids.append(i)
        # device_ids = [0,1]
            self.encoder = nn.DataParallel(self.encoder, device_ids=device_ids).to(device)
        
        self.decoder = Decoder(
            embedding_dim,
            agent_num=agent_num,
            envs_num=envs_num,)
    
    def forward(self, obs):
        state = stateWrapper(obs, device=self.device, problem=self.problem.NAME)
        input = state.states["observations"]
        embedding = self.embedding(input)
        encoded_inputs, _ = self.encoder(embedding)
        print("encoding done.")
        
    def encode(self, obs):
        state = stateWrapper(obs, device=self.device, problem=self.problem.NAME)
        input = state.states['observations']
        embedding = self.embedding(input)
        encoded_inputs, _ = self.encoder(embedding)
        # encoded_iputs 包含了depot的embedding
        cached_embeddings = self.decoder._precompute(encoded_inputs)
        
        return cached_embeddings  # cache = (embeddings, graph_context)


    
    def decode(self, obs_multi, cached_embeddings, tbd_node_idx):
        cached_multi_state = MA_StateWrapper(obs_multi, cached_embeddings, tbd_node_idx, device='cuda')
        x = self.decoder.advance(cached_multi_state)
        
        # x = self.decoder.advance(MA_StateWrapper(obs_multi, cached_embeddings, tbd_node_idx, device='cuda'))
        
        return x
    

class Critic(nn.Module):
    def __init__(self, embedding_dim):
        super(Critic, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim*2+1, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )
    
    def forward(self, xs):
        agent_num = xs.shape[1]
        env_num = xs.shape[0]
        agent_id = (t.arange(agent_num*env_num) % agent_num).reshape(-1,agent_num).unsqueeze(-1).to('cuda')
        input = t.cat([xs.mean(1).unsqueeze(1).expand(-1,xs.shape[1] ,-1),
                       xs,
                       agent_id],
                       dim=-1).reshape(agent_num*env_num, -1)
        out = self.mlp(input)
        return out.view(env_num, agent_num)





#* discrete action space 0 or 1
class Actor_forward(nn.Module):
    def __init__(self, embedding_dim):
        super(Actor_forward, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 2) # no agent_id; action space: (0, 1)
    
    def forward(self, xs):
        # xs:        [env_num, agent_num, embedding_dim]
        # agent_num: [env_num, agent_num]
        # return:    [env_num, agent_num]
        action_01 = F.softmax(self.fc1(xs), dim=-1)
        return action_01  

#* discrete action space 0 or 1
class Actor(nn.Module):
    def __init__(self, embedding_dim):
        super(Actor, self).__init__()
        self.fc1 = Actor_forward(embedding_dim) # no agent_id; action space: (0, 1)
        
        if t.cuda.device_count() > 1:
            device_ids = []
            for i in range(t.cuda.device_count()):
                device_ids.append(i)

        # device_ids = [0,1]
            self.fc1 = nn.DataParallel(self.fc1, device_ids=device_ids).to('cuda')
        
    def take_action(self, xs, next_obs_multi, tbd_node_idx, action):
        u_action = None
        action_01 = self.fc1(xs)                                    # [env_num, agent_num, 2_prob]
        mutli_dist = t.distributions.Categorical(probs=action_01)
        action_conflict=0
        if action == None:
            action = mutli_dist.sample()   # 0 or 1                              # [env_num, agent_num, 1]         
            #* 1. mask掉不合法的action: load超过1就不能再接受新的node了
            #* masked_logits： [-1, 0, 1]: [非法action, 不选, 选]
            masked_logits, agent_mask = self._agent_mask(action, next_obs_multi, tbd_node_idx) # [env_num, agent_num, 1]
            action_conflict = (masked_logits>0).sum()/action.shape[0]
            
            #* 2. 若全0, 则随机选择一个合法action
            if ((masked_logits>=1).sum(-1) == 0).sum() >= 1:
                zero_indices = ((masked_logits>=1).sum(-1)==0).nonzero().squeeze(-1)
                masked_logits_clone = deepcopy(masked_logits[zero_indices])
                masked_logits_clone_softmax = F.softmax(masked_logits_clone, dim=-1)
                masked_logits_clone_softmax[~agent_mask[zero_indices]]=1e-9
                masked_logits[zero_indices] = t.eye(action_01.shape[-2]).to('cuda')[(Categorical(masked_logits_clone_softmax).sample().long())]
                
            u_action = t.argmax(action_01[:,:,1] * masked_logits, -1).unsqueeze(-1)
            action = (t.eye(action_01.shape[-2]).to('cuda'))[u_action.squeeze()]
            
        log_probs = mutli_dist.log_prob(action)                          # [env_num, agent_num, 1]
        return action, u_action, log_probs, mutli_dist.entropy(), action_conflict
    
    def take_action_test(self, xs, next_obs_multi, tbd_node_idx, action):
        u_action = None
        action_01 = self.fc1(xs)                                    # [env_num, agent_num, 2_prob]
        action_conflict=0
        if action == None:
            action = t.argmax(action_01, dim=-1)   # 0 or 1                              # [env_num, agent_num, 1]         
            # action_conflict = action.sum()/action.shape[0]
            
            #* 1. mask掉不合法的action: load超过1就不能再接受新的node了
            #* masked_logits： [-1, 0, 1]: [非法action, 不选, 选]
            masked_logits, agent_mask = self._agent_mask(action, next_obs_multi, tbd_node_idx) # [env_num, agent_num, 1]
            action_conflict = (masked_logits>0).sum()/action.shape[0]
            
            #* 2. 若全0, 则随机选择一个合法action
            if ((masked_logits>=1).sum(-1) == 0).sum() >= 1:
                zero_indices = ((masked_logits>=1).sum(-1)==0).nonzero().squeeze(-1)
                masked_logits_clone = deepcopy(masked_logits[zero_indices])
                masked_logits_clone_softmax = F.softmax(masked_logits_clone, dim=-1)
                masked_logits_clone_softmax[~agent_mask[zero_indices]]=1e-9
                masked_logits[zero_indices] = t.eye(action_01.shape[-2]).to('cuda')[(Categorical(masked_logits_clone_softmax).sample().long())]
                
            
            u_action = t.argmax(action_01[:,:,1] * masked_logits, -1).unsqueeze(-1)
            action = (t.eye(action_01.shape[-2]).to('cuda'))[u_action.squeeze()]
            
        log_probs = 0                         # [env_num, agent_num, 1]
        return action, u_action, log_probs, 0, action_conflict

    def _onehot_mask(self, masked_logits):
        #* 考虑全0行情况
        all_0_idx = (masked_logits.sum(-1)<1).nonzero().squeeze(-1)
        masked_logits_clone = deepcopy(masked_logits)
        masked_logits[masked_logits == 0] = 1e-9
        if len(all_0_idx) != 0:    
            for i in all_0_idx:
                masked_logits[i,t.nonzero(masked_logits_clone[i]==0)] = 1
                # masked_logits[985,(masked_logits[985]!=1e-9).to(int)] = 1
        onehot_dist = t.distributions.Categorical(probs=masked_logits)
        u_action = onehot_dist.sample().unsqueeze(-1) 
        
        assert t.gather(masked_logits,-1, u_action).sum() == masked_logits.shape[0]
        agent_num = masked_logits.shape[-1]
        onehot_mask = t.eye(agent_num).to('cuda')[u_action.squeeze(-1)]
        return u_action, onehot_mask
    
    
    #* load超过1就不能再接受新的node了
    def _agent_mask(self, action, next_obs_multi, tbd_node_idx):
        # //agent_mask = t.from_numpy(next_obs_multi['multi_current_load'].squeeze(1) < 1).to('cuda')
        logits = deepcopy(action).to(t.float32)
        # // if type(tbd_node_idx) != t.Tensor:
        #//     tbd_node_idx = t.tensor(tbd_node_idx).long().to('cuda')
        agent_mask = (t.Tensor(next_obs_multi['multi_current_load'].squeeze(1)).to('cuda') + t.gather(t.Tensor(next_obs_multi['demand']).to('cuda'), 1, (tbd_node_idx-1))) < 1
        # //agent_mask = t.from_numpy(next_obs_multi['multi_current_load'].squeeze(1) + next_obs_multi['demand'][:,tbd_node_idx[0]-1] <= 1).to('cuda')
        
        
        logits[~agent_mask] = -1
        
        # //agent_mask = t.full(logits.shape, float("-inf")) * (t.from_numpy(next_obs_multi['multi_current_load'].squeeze(1) > 1).to('cuda'))
        # //agent_mask = t.ones_like(probs, dtype=t.bool, device='cuda')
        return logits, agent_mask    
    

class Multi_Agent(nn.Module):
    def __init__(self, envs_num, agent_num = 10, embedding_dim=128, device='cuda', name='marl_cvrp'):
        super(Multi_Agent, self).__init__()
        self.agent_num = agent_num
        self.backbone = Backbone(embedding_dim=embedding_dim, device=device, problem_name=name, agent_num=agent_num, envs_num=envs_num)
        self.actor = Actor(embedding_dim)
        self.critic = Critic(embedding_dim)
        
        if t.cuda.device_count() > 1:
            device_ids = []
            for i in range(t.cuda.device_count()):
                device_ids.append(i)
        # device_ids = [0,1]
            self.critic = nn.DataParallel(self.critic, device_ids=device_ids).to(device)
    
    def get_action_and_value_cached(self, next_obs_multi, tbd_node_idx,  u_actions=None, state=None, action=None):
        xs = self.backbone.decode(next_obs_multi, state, tbd_node_idx)  
        action, u_action, log_probs, entropy, action_conflict = self.actor.take_action(xs, next_obs_multi, tbd_node_idx, action)
        return (action, u_action, log_probs, self.critic(xs), entropy, action_conflict)
    
    def get_action_and_value_cached_test(self, next_obs_multi, tbd_node_idx,  u_actions=None, state=None, action=None):
        xs = self.backbone.decode(next_obs_multi, state, tbd_node_idx)  
        action, u_action, log_probs, entropy, action_conflict = self.actor.take_action_test(xs, next_obs_multi, tbd_node_idx, action)
        return (action, u_action, log_probs, self.critic(xs), entropy, action_conflict)

    
    def get_value_cached(self, next_obs_multi, state):
        tbd_node_idx = next_obs_multi['tbd_node_idx']
        xs = self.backbone.decode(next_obs_multi, state, tbd_node_idx)  
        return self.critic(xs)
        
    


class stateWrapper:
    def __init__(self, states, device, problem="marl_cvrp"):
        self.device = device
        self.states = {k: t.tensor(v, device=self.device) for k, v in states.items()}
        input = {
            "loc": self.states["observations"],
            "depot": self.states["depot"].squeeze(-1),
            "demand": self.states["demand"]
        }
        self.states["observations"] = input

# class MA_StateWrapper:
def MA_StateWrapper(multi_state, cached_embeddings, tbd_node_idx, device):
    multi_state = {k: t.tensor(v, device=device) for k, v in multi_state.items()}
    node_embeddings, graph_context = cached_embeddings
    
    
    multi_current_load = multi_state['multi_current_load'].squeeze(1)
    if type(tbd_node_idx) != t.Tensor:
        tbd_node_idx = t.tensor(tbd_node_idx).long().to(device)
    tbd_node_embedding = t.gather(node_embeddings, 1, tbd_node_idx.unsqueeze(-1).expand(-1,-1,node_embeddings.shape[-1])).squeeze(1) # env_num, embedding_dim
    depot_embedding = node_embeddings[:,0,:].clone()

    agent_num=multi_state['multi_visited_len_idx'].shape[-1]
    env_num = multi_state['multi_visited_len_idx'].shape[0]
    multi_visited_nodes_len = multi_state['multi_visited_len_idx'].squeeze(1)
    multi_node_idx = multi_state['multi_visited_nodes'][:,:,:,:multi_visited_nodes_len.max()+1].squeeze(1).reshape(multi_state['multi_visited_nodes'].shape[0],-1).unsqueeze(-1).expand(-1,-1,128)
    multi_visited_nodes_embeddings = t.gather(node_embeddings,1,multi_node_idx).reshape(env_num, agent_num, -1, 128)
    
    
    
    
    
    


    cached_multi_state = (
        #* static embedding
        # node_embeddings,
        graph_context.squeeze(1),
        depot_embedding,
        #* dynamic embedding
        # tbd_node_demand,
        multi_current_load,
        tbd_node_embedding,
        multi_visited_nodes_embeddings,
        multi_visited_nodes_len 
    )
    
    return cached_multi_state

