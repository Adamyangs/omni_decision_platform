import torch as t
from torch import nn
from marl_models.Nets.multi_head_attention import AttentionScore, MultiHeadAttention

class class_project_query(nn.Module):
    def __init__(self, embedding_dim):
        super(class_project_query, self).__init__()
        self.fc = nn.Linear(embedding_dim*3+2, embedding_dim, bias=False).to('cuda')
    
    def forward(self, x):
        return self.fc(x)


class class_project_keys(nn.Module):
    def __init__(self, embedding_dim):
        super(class_project_keys, self).__init__()
        self.fc = nn.Linear(embedding_dim+1, embedding_dim, bias=False).to('cuda')
    
    def forward(self, x):
        return self.fc(x)

class class_project_values(nn.Module):
    def __init__(self, embedding_dim):
        super(class_project_values, self).__init__()
        self.fc = nn.Linear(embedding_dim+1, embedding_dim, bias=False).to('cuda')
    
    def forward(self, x):
        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self, embedding_dim, agent_num, envs_num):
        super(Decoder, self).__init__()
        
        self.agent_num = agent_num
        self.envs_num = envs_num
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self._project_query = class_project_query(embedding_dim).to('cuda')
        self._project_keys = class_project_keys(embedding_dim).to('cuda')
        self._project_values = class_project_values(embedding_dim).to('cuda')
        
        
        # self._project_query = nn.Linear(embedding_dim*3+2, embedding_dim, bias=False).to('cuda')
        # self._project_keys = nn.Linear(embedding_dim+1, embedding_dim, bias=False).to('cuda')
        # self._project_keys = nn.GRU(embedding_dim+1, embedding_dim, num_layers=2).to('cuda')
        # self._project_keys = [nn.Linear(embedding_dim, embedding_dim, bias=False).to('cuda') for _ in range(self.agent_num)]
        
        # self._project_values = nn.Linear(embedding_dim+1, embedding_dim, bias=False).to('cuda')
        # self._project_values = nn.GRU(embedding_dim+1, embedding_dim, num_layers=2).to('cuda')
        # self._project_values = [nn.Linear(embedding_dim, embedding_dim, bias=False).to('cuda') for _ in range(self.agent_num)]
        self.attention = MultiHeadAttention(embedding_dim, n_heads=8)
        
        self.agent_hidden_hn = [0]*self.agent_num
        # self.agent_id = (t.arange(agent_num*self.envs_num) % agent_num).reshape(-1,agent_num).unsqueeze(-1).to('cuda')
    
    def init(self):
        if t.cuda.device_count() > 1:
            device_ids = []
            for i in range(t.cuda.device_count()):
                device_ids.append(i)
            # device_ids = [0,1]
            self._project_query_parallel = nn.DataParallel(self._project_query, device_ids=device_ids).to('cuda')
            self._project_keys_parallel = nn.DataParallel(self._project_keys, device_ids=device_ids).to("cuda")
            self._project_values_parallel = nn.DataParallel(self._project_values, device_ids=device_ids).to("cuda")
            # for i in range(self.agent_num):
        #     self._project_keys[i] = nn.DataParallel(self._project_keys[i], device_ids=device_ids).to("cuda")
        #     self._project_values[i] = nn.DataParallel(self._project_values[i], device_ids=device_ids).to("cuda")
    
    
    
    def _precompute(self, embeddings):
        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)
        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        graph_context = self.project_fixed_context(graph_embed).unsqueeze(-2)
        
        cache = (
            embeddings,
            graph_context
        )
        return cache
    
    def project_query(self, graph_context, depot_embedding, tbd_node_embedding, multi_current_load):
        # graph_context: (env_num, embedding_dim)
        # depot_embedding: (env_num, embedding_dim)
        # tbd_node_embedding: (env_num, embedding_dim)
        # multi_current_load: (env_num, agent_num)
        # agent_num: (env_num, agent_num)
        
        # input:(env_num, agent_num, 128+128+128+1(容量)+1(agent_id))
        # input_per_agent: graph_context(128) + depot_embedding(128) + tbd_node_embedding(128) + current_load(1)
        # return: (env_num, agent_num, embedding_dim)
        
        agent_num = multi_current_load.shape[-1]
        envs_num = multi_current_load.shape[0]
        self.agent_id = (t.arange(agent_num*envs_num) % agent_num).reshape(-1,agent_num).unsqueeze(-1).to('cuda')
        input = t.cat([
                       graph_context.unsqueeze(1).expand(-1, self.agent_num, -1), 
                       depot_embedding.unsqueeze(1).expand(-1, self.agent_num, -1), 
                       tbd_node_embedding.unsqueeze(1).expand(-1, self.agent_num, -1), 
                       multi_current_load.unsqueeze(-1),
                       self.agent_id
                       ],
                       dim=-1)
        q = self._project_query_parallel(input.to(t.float32))
        
        return q
    
    def project_key(self, multi_visited_nodes_embeddings, multi_visited_nodes_len):
        # multi_vsisited_nodes_embeddings: env_num, agent_num, node_num, embedding_dim
        # multi_visited_nodes_len : env_num, agent_num
        
        # key: (env_num, agent_num, node_num, embedding_dim)
        # return key_list: (key_1, key_2, ..., key_agent_num)
        agent_num = multi_visited_nodes_len.shape[-1]
        key_list = []
        # idx = multi_visited_nodes_len.unsqueeze(-1).expand(-1,-1,multi_visited_nodes_embeddings.shape[-1])
        for i in range(agent_num):
            input = t.cat([
                multi_visited_nodes_embeddings[:,i,:t.max(multi_visited_nodes_len[:,i]),:], 
                self.agent_id[:,i].unsqueeze(-1).repeat(1,t.max(multi_visited_nodes_len[:,i]),1),
                ],
                dim=-1)
            
            
            # input = multi_visited_nodes_embeddings[:,i,:t.max(multi_visited_nodes_len[:,i]),:].reshape(-1, multi_visited_nodes_embeddings.shape[-1])
            # input = multi_visited_nodes_embeddings[:,i,:t.max(multi_visited_nodes_len[:,i]),:]
            # //key_list.append(self._project_keys[i](input.to(t.float32)).reshape(-1,t.max(multi_visited_nodes_len[:,i]), multi_visited_nodes_embeddings.shape[-1]))
            # key_list.append(self._project_keys[i](input.to(t.float32)))
            key_list.append(self._project_keys_parallel(input.to(t.float32)))
        
        
        return key_list
    
    def project_value(self, multi_visited_nodes_embeddings, multi_visited_nodes_len):
        # multi_vsisited_nodes_embeddings: env_num, agent_num, node_num, embedding_dim
        # multi_visited_nodes_len : env_num, agent_num
        
        # key: (env_num, valid_node_num, embedding_dim)
        # return key_list: (key_1, key_2, ..., key_agent_num)
        agent_num = multi_visited_nodes_len.shape[-1]
        value_list = []
        # idx = multi_visited_nodes_len.unsqueeze(-1).expand(-1,-1,multi_visited_nodes_embeddings.shape[-1])
        for i in range(agent_num):
            input = t.cat([
                    multi_visited_nodes_embeddings[:,i,:t.max(multi_visited_nodes_len[:,i]),:], 
                    self.agent_id[:,i].unsqueeze(-1).repeat(1,t.max(multi_visited_nodes_len[:,i]),1),
                    ],
                    dim=-1)
            
            # input = multi_visited_nodes_embeddings[:,i,:t.max(multi_visited_nodes_len[:,i]),:]
            # value_list.append(self._project_values[i](input.to(t.float32)))
            value_list.append(self._project_values_parallel(input.to(t.float32)))
        
        return value_list
    
    def _mask(self, multi_visited_nodes_embeddings, multi_visited_nodes_len):
        # multi_visited_nodes_len : env_num, agent_num
        
        # mask: (env_num, valid_node_num, embedding_dim)
        # return mask_list: (mask_1, mask_2, ..., mask_agent_num)
        mask_list = []
        for i in range(self.agent_num):
            mask = t.arange(multi_visited_nodes_len[:,i].max()).unsqueeze(0).expand(multi_visited_nodes_len.shape[0], -1).to('cuda') < multi_visited_nodes_len[:,i].unsqueeze(-1)
            # //mask_list.append(mask.unsqueeze(-1).expand(-1,-1,multi_visited_nodes_embeddings.shape[-1]))
            mask_list.append(mask.unsqueeze(-1).transpose(-2, -1))        
        return mask_list
    
    def attention4q(self, query, key_list, value_list, mask_list):
        
        x_list = []
        for i in range(self.agent_num):
            # q: env_num, 1, embedding_dim
            # key: env_num, valid_node_num, embedding_dim
            # v: env_num, valid_node_num, embedding_dim
            # mask: env_num, valid_node_num, 1
            x_i = self.attention(query[:,i,:].unsqueeze(1), key_list[i], value_list[i], ~mask_list[i])
            x_list.append(x_i)
        
        return x_list
    
    def advance(self, cached_multi_state):
        # node_embeddings, graph_context, depot_embedding, tbd_node_demand, multi_current_load, tbd_node_embedding, multi_visited_nodes_embeddings, multi_visited_nodes_len = cached_multi_state
        graph_context, depot_embedding, multi_current_load, tbd_node_embedding, multi_visited_nodes_embeddings, multi_visited_nodes_len = cached_multi_state
        # query from: graph_context, depot_embedding, tbd_node_embedding, multi_current_load
        # key & value from: multi_visited_nodes_embeddings, multi_visited_nodes_len
        query = self.project_query(graph_context, depot_embedding, tbd_node_embedding, multi_current_load)
        key_list = self.project_key(multi_visited_nodes_embeddings, multi_visited_nodes_len)
        value_list = self.project_value(multi_visited_nodes_embeddings, multi_visited_nodes_len)
        
        mask_list = self._mask(multi_visited_nodes_embeddings, multi_visited_nodes_len)
        
        x_list = self.attention4q(query, key_list, value_list, mask_list)
        xs = t.stack(x_list, dim=1).squeeze(2)
        
        return xs # env_num, agent_num, embedding_dim
    