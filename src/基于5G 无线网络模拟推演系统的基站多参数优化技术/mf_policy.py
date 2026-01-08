import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import itertools
from ATSPModel import EncodingBlock, MultiGraphEncoderLayer
from ATSPModel_LIB import MultiHeadAttention, AddAndInstanceNormalization
from environment import BaseStationSleepingProblem
from util import to_onehot, init_weights
import setproctitle

class Buffer:
    def __init__(self, max_length=100, n_agent=1, n_input=15):
        self.max_length = max_length
        self._buffer = {
            'state': np.zeros([max_length, n_agent, n_input], dtype='float32'),
            'action': np.zeros([max_length, n_agent], dtype='int64'),
            'reward': np.zeros([max_length, n_agent], dtype='float32'),
            'next_state': np.zeros([max_length, n_agent, n_input], dtype='float32'),
            'done': np.zeros([max_length], dtype='float32'),
        }
        self.ptr = 0
        self.length = 0

    def is_full(self):
        return self.length == self.max_length

    def append(self, transition:dict):
        for k in transition:
            self._buffer[k][self.ptr] = transition[k]
        self.ptr = (self.ptr + 1) % self.max_length
        self.length = min(self.length + 1, self.max_length)

    def sample(self, batchsize:int):
        # if self.length <= 10:
        #     return None
        idx = np.random.randint(low=0, high=self.length, size=batchsize)

        state = self._buffer['state'][idx]              # [n_batch, n_agent, n_input]
        action = self._buffer['action'][idx]            # [n_batch, n_agent]
        reward = self._buffer['reward'][idx]            # [n_batch, n_agent]
        next_state = self._buffer['next_state'][idx]    # [n_batch, n_agent, n_input]
        done = self._buffer['done'][idx]                # [n_batch]

        return state, action, reward, next_state, done

class Encoder(nn.Module):
    def __init__(self, n_input, n_output, n_embed, n_graph, n_mask, device, use_ensemble=False):
        super(Encoder, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.n_embed = n_embed
        self.n_graph = n_graph
        self.n_mask = n_mask
        self.n_ensemble = 10
        self.use_ensemble = use_ensemble
        self.device = device

        self.embed = nn.Sequential(
            nn.Linear(self.n_input, self.n_embed),
            nn.ReLU(),
            nn.Linear(self.n_embed, self.n_embed),
            nn.ReLU(),
        )
        self.attention_layer = nn.Linear(self.n_embed, self.n_embed * 3 * n_graph)    # n_graph * 3 (3=k q v)
        self.output = nn.Sequential(
            nn.Linear(self.n_embed * (1 + self.n_graph) + self.n_mask, self.n_embed),
            nn.ReLU(),
            nn.Linear(self.n_embed, self.n_output * self.n_ensemble if self.use_ensemble else self.n_output),
        )

        self.apply(init_weights)
        self.to(self.device)

    def _attention_graph(self, x, graph):
        """

        :param x: [..., n_agent, n_embed]
        :param graph [..., n_graph, n_agent, n_agent]
        :return: [..., n_agent, n_output]
        """
        shape = x.shape[:-2]
        n_agent = x.shape[-2]
        n_embed = x.shape[-1]

        x1 = self.attention_layer(x)                            # [..., n_agent, n_graph * 3 * n_embed]
        x1 = torch.reshape(x1, [*shape, n_agent, self.n_graph, 3, self.n_embed])
                                                                # [..., n_agent, n_graph, 3, n_embed]

        q, k, v = x1[..., 0, :], x1[..., 1, :], x1[..., 2, :]   # [..., n_agent, n_graph, n_embed]
        q, k = torch.transpose(q, -2, -3), torch.transpose(k, -2, -3)
                                                                # [..., n_graph, n_agent, n_embed]
        x2_list = []
        for i in range(self.n_graph):
            g_i = graph[i].float().coalesce()                   # sparse[n_agent, n_agent]
            q_i = q[..., i, g_i.indices()[0], :]                # [..., ?, n_embed]
            k_i = k[..., i, g_i.indices()[1], :]                # [..., ?, n_embed]
            score_i = (q_i * k_i).sum(-1) / np.sqrt(self.n_embed)
                                                                # [..., ?]
            if score_i.dim() == 1:
                score_i = torch.sparse.FloatTensor(g_i.indices(), score_i, torch.Size([n_agent, n_agent]))
                                                                # sparse[n_agent, n_agent]
                score_i = torch.sparse.softmax(score_i, dim=1)  # sparse[n_agent, n_agent]
            elif score_i.dim() == 2:
                score_i = torch.cat([torch.sparse.FloatTensor(g_i.indices(), score_i[j], torch.Size([n_agent, n_agent])).unsqueeze(0)
                                     for j in range(score_i.shape[0])],
                                    dim=0)                      # sparse[n_batch, n_agent, n_agent]
                score_i = torch.sparse.softmax(score_i, dim=2)  # sparse[n_agent, n_agent]
            else:
                raise NotImplementedError

            if score_i.dim() == 2:
                x2 = torch.sparse.mm(score_i, v[:, i, :])       # [n_agent, n_embed]
            elif score_i.dim() == 3:
                x2 = torch.cat([torch.sparse.mm(score_i[j], v[j, :, i, :]).unsqueeze(0)
                                for j in range(score_i.shape[0])],
                               dim=0)                           # [..., n_agent, n_embed]
            else:
                raise NotImplementedError
            x2_list.append(x2)
        x2 = torch.cat(x2_list, dim=-1)                         # [..., n_agent, n_graph * n_embed]

        return x2

    def forward(self, x, graph, agent_id=None, mask=None):
        """

        :param x: [n_agent, n_input]
        :param graph [n_graph, n_agent, n_agent]
        :param agent_id [n_selected_agent]
        :param mask [n_selected_agent, n_mask]
        :return: [n_agent, n_output]
        """
        x = self.embed(x)                                               # [..., n_agent, n_embed]
        x1 = self._attention_graph(x, graph)                            # [..., n_agent, n_graph * n_embed]
        if agent_id is not None:
            x = x[..., agent_id, :]
            x1 = x1[..., agent_id, :]
            if mask is None:
                x = self.output(torch.cat([x, x1], dim=-1))             # [..., n_agent, n_embed]
            else:
                x = self.output(torch.cat([x, x1, mask], dim=-1))       # [..., n_agent, n_embed]
        else:
            x = self.output(torch.cat([x, x1], dim=-1))                 # [..., n_agent, n_embed]
        if self.use_ensemble:
            x = x.reshape([*x.shape[:-1], self.n_ensemble, self.n_output])    # [..., n_agent, n_ensemble, n_embed]

        return x

class PriorEncoder(nn.Module):
    def __init__(self, n_input, n_output, n_embed, n_graph, n_mask, device, use_ensemble=False):
        super(PriorEncoder, self).__init__()

        self.encoder = Encoder(n_input, n_output, n_embed, n_graph, n_mask, device, use_ensemble)
        self.random_encoder = Encoder(n_input, n_output, n_embed, n_graph, n_mask, device, use_ensemble)
        self.random_encoder.requires_grad_(False)

    def forward(self, x, graph, agent_id=None, mask=None):
        x1 = self.encoder(x, graph, agent_id, mask)
        x2 = self.random_encoder(x, graph, agent_id, mask)
        # with torch.no_grad():
        #     print('output', x1.mean().item(), x2.mean().item())

        return x1 + 5 * x2

def get_sparse_graph(node_groups, node_in_group_ids, n_node, device):
    """

    :param node_groups: list({group_id_0:[node0,node1,...], ...})
    :param node_in_group_ids: list({node0:group_id, ...})
    :param n_node: int
    :param device: str
    :return:
    """
    graph_list= []
    for node_group, node_in_group_id in zip(node_groups, node_in_group_ids):
        indices = torch.LongTensor(
            [[i1 for i1 in range(n_node) for _ in range(len(node_group[node_in_group_id[i1]]))],
             [i2 for i1 in range(n_node) for i2 in node_group[node_in_group_id[i1]]]])
        value = torch.ones_like(indices[0])
        graph = torch.sparse.FloatTensor(indices, value, torch.Size([n_node, n_node])).unsqueeze(0)
        graph_list.append(graph)
    graph = torch.cat(graph_list, dim=0).to(device)

    return graph

class GraphTool:
    def __init__(self, n_node, graph):
        """

        :param n_node: int
        :param graph:  SparseTensor[n_node, n_node] or SparseTensor[n_graph, n_node, n_node]
        """
        self.n_node = n_node
        if graph.dim() == 2:
            self.graph = graph.unsqueeze(0).coalesce()
        else:
            self.graph = graph.coalesce()
        self.n_graph = graph.shape[0]
        self.graph_sum = graph[0]
        for i in range(1, self.n_graph):
            self.graph_sum = self.graph_sum + graph[i]
        self.graph_sum = self.graph_sum.float()
        self._get_graph_group()

    def _get_graph_group(self):
        """

        :return:
        """
        # the node_ids connected to input node_id
        self.graph_group = [{j:[] for j in range(self.n_node)} for _ in range(self.n_graph)]

        for i in range(self.n_graph):
            graph = self.graph[i].coalesce()
            indices = graph.indices().cpu().numpy()
            for k in range(indices.shape[1]):
                self.graph_group[i][indices[0, k]].append(indices[1, k])

    def get_connected_nodes(self, node_ids, order=1):
        """

        :param node_ids: Tensor[n_batch]
        :return:
        """
        with torch.no_grad():
            assert node_ids.dim() == 1
            node_ids = node_ids.flatten()                                   # [n_batch]
            node_ids_onehot = F.one_hot(node_ids, self.n_node)              # [n_batch, n_cell]

            c_node_ids = node_ids_onehot.float().T                          # [n_cell, n_batch]
            for i in range(order):
                if i == order - 1:
                    latter_c_node_ids = c_node_ids.T.bool().to_sparse()     # [n_batch, n_cell]
                c_node_ids = torch.sparse.mm(self.graph_sum, c_node_ids)    # [n_cell, n_batch]

        return latter_c_node_ids, c_node_ids.T.bool().to_sparse()           # [n_batch, n_cell]

    def _get_feature(self, node_id_marks, fea):
        """

        :param node_id_marks: Sparse[n_batch, n_cell]
        :param fea:      Tensor[n_batch, n_cell, n_dim]
        :return:
        """
        with torch.no_grad():
            n_batch, _, n_dim = fea.shape
            node_id_marks = node_id_marks.indices()                                     # [2, ?]
            batch_node_ids = node_id_marks[0] * self.n_node + node_id_marks[1]          # [?]
            fea = fea.reshape((-1, n_dim))[batch_node_ids]                              # [?, n_dim]

        return fea

    def _get_subgraph(self, node_id_marks0, node_id_marks1):
        """

        :param node_id_marks0: Sparse[n_batch, n_cell]
        :param node_id_marks1: Sparse[n_batch, n_cell]
        :return:
        """
        with torch.no_grad():
            # _t=time.time()
            n_batch = node_id_marks0.shape[0]
            device = node_id_marks0.device
            node_id_marks1 = node_id_marks1.indices()
            batch_node_id_tensor = node_id_marks1[0] * self.n_node + node_id_marks1[1]
            node_id_marks0 = node_id_marks0.indices().cpu().numpy()
            node_id_marks1 = node_id_marks1.cpu().numpy()
            batch_node_id = node_id_marks1[0] * self.n_node + node_id_marks1[1]
            batch_node_id_to_compressed_batch_node_id = {
                id:i for i, id in enumerate(batch_node_id)
            }
            # print('0', time.time()-_t)

            _t = time.time()
            subgraph = []
            for gi in range(self.n_graph):
                i1, i2 = [], []
                for i in range(n_batch):
                    node_ids = node_id_marks0[1][node_id_marks0[0]==i]
                    l1 = list(itertools.chain.from_iterable(
                        [[batch_node_id_to_compressed_batch_node_id[node_id + self.n_node * i]]
                         * len(self.graph_group[gi][node_id])
                         for node_id in node_ids]))
                    l2 = [batch_node_id_to_compressed_batch_node_id[node_id1 + self.n_node * i]
                          for node_id in node_ids
                          for node_id1 in self.graph_group[gi][node_id]]
                    i1 += l1
                    i2 += l2

                graph = torch.sparse.FloatTensor(torch.LongTensor([i1, i2]),
                                                 torch.ones([len(i1)]),
                                                 torch.Size([batch_node_id.shape[0], batch_node_id.shape[0]]))
                subgraph.append(graph.unsqueeze(dim=0))
            subgraph = torch.cat(subgraph, dim=0).to(device)
            # print('1', time.time()-_t)
        return subgraph, batch_node_id_tensor

    def get_fea_graph(self, node_ids, fea, order=1):
        """

        :param node_ids:    Tensor[n_batch]
        :param fea:         Tensor[n_batch, n_cell, n_dim] or list(Tensor[n_batch, n_cell, n_dim])
        :param order:         int >0
        :return:
        """
        with torch.no_grad():
            node_id_marks0, node_id_marks1 = self.get_connected_nodes(node_ids, order)
            if isinstance(fea, list):
                compressed_fea = [self._get_feature(node_id_marks1, item) for item in fea]
            else:
                compressed_fea = self._get_feature(node_id_marks1, fea)
            subgraph, batch_node_id_tensor = self._get_subgraph(node_id_marks0, node_id_marks1)

        return compressed_fea, subgraph, batch_node_id_tensor


class CellControlMFAgent:
    def __init__(self, n_input, n_output, n_agent, gamma, w_entropy, learning_rate, buffer_size, graph, device):
        self.start_t = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        print(self.start_t)
        self.n_input = n_input
        self.n_output = n_output
        self.n_agent = n_agent
        self.n_graph = 2
        self.n_mf = 2
        self.n_embed = 32
        self.gamma = gamma
        self.w_entropy = w_entropy
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.soft_update_scale = 0.01
        self.epsilon = 0.01
        self.use_ensemble = False
        self.ensemble_train_ratio = 0.5
        self.graph = graph.long().detach()
        self.device = device

        self.control_network = Encoder(self.n_input, self.n_output, self.n_embed, self.n_graph, 0, device, self.use_ensemble)
        self.value_network = Encoder(self.n_input, self.n_output, self.n_embed, self.n_graph, self.n_mf, device)
        self.target_control_network = Encoder(self.n_input, self.n_output, self.n_embed, self.n_graph, 0, device, self.use_ensemble)
        self.target_value_network = Encoder(self.n_input, self.n_output, self.n_embed, self.n_graph, self.n_mf, device)

        for target_param, param in zip(self.target_control_network.parameters(), self.control_network.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_value_network.parameters(), self.value_network.parameters()):
            target_param.data.copy_(param.data)

        self.target_control_network.eval()
        self.target_value_network.eval()

        self.value_optim = torch.optim.Adam(list(self.value_network.parameters()), lr=self.learning_rate)
        self.policy_optim = torch.optim.Adam(list(self.control_network.parameters()), lr=self.learning_rate)
        self.buffer = Buffer(self.buffer_size, self.n_agent, self.n_input)

        self.n_update = 0

    def sample_action(self, state, graph=None, eval_mode=False, use_target_network=False):
        """

        :param state:           [..., n_in]
        :param graph:           Sparse[]
        :param use_target_network: bool
        :return:
        """
        with torch.no_grad():
            original_graph = graph is None
            graph = graph if graph is not None else self.graph
            if use_target_network:
                action_value = self.target_control_network(state, graph)        # [..., (n_ens,) n_act]
            else:
                action_value = self.control_network(state, graph)               # [..., (n_ens,) n_act]
            if self.use_ensemble:
                if eval_mode:
                    action_value = action_value.mean(dim=-2)                    # [..., n_act]
                else:
                    ens_id = torch.randint(action_value.shape[-2], [*action_value.shape[:-2], 1, self.n_output],
                                           device=self.device)                  # [..., 1, n_act]
                    action_value = torch.gather(action_value, dim=-2, index=ens_id).squeeze(-2)
                                                                                # [..., n_act]

            random_action = torch.randint(self.n_output, size=state.shape[:-1], device=self.device)
                                                                                # [...]
            greedy_action = torch.argmax(action_value, dim=-1)                  # [...]
            if eval_mode:
                selection = torch.ones(size=state.shape[:-1], device=self.device, dtype=torch.bool)
            else:
                selection = torch.rand(size=state.shape[:-1], device=self.device) > self.epsilon
                                                                                # [...]
            action = torch.where(selection, greedy_action, random_action)
            if original_graph and action_value.dim() == 2:
                print('action distribution = ', F.one_hot(greedy_action, self.n_output).sum(dim=0), action_value.std(dim=0))
                print('action value = ',action_value[torch.randint(action_value.shape[0], size=(2,))])

            return action

    def train(self, state, action, reward, next_state, done, subgraph, agent_id, mask,
              new_action, new_mask, next_action, next_mask):
        """
        :param state:           [n_batch, n_in]
        :param action:          [n_batch]
        :param reward:          [n_batch]
        :param next_state:      [n_batch, n_in] or None
        :param done             [n_batch]
        :param subgraph         []
        :param batch_agent_id   []
        :param mask             [n_batch, 2]
        :param new_action       [n_batch]
        :param new_mask         [n_batch, 2]
        :param next_action      [n_batch]
        :param next_mask        [n_batch, 2]
        :return:                []
        """
        self.n_update += 1

        # cal target value
        with torch.no_grad():
            next_qvalue = self.target_value_network(next_state, subgraph, agent_id, next_mask)
                                                                                    # [n_batch, n_act]
            next_value = torch.gather(next_qvalue, dim=-1, index=next_action.unsqueeze(-1)).squeeze(-1)
            # [n_batch]

            target_value = next_value * self.gamma * (1-done) + reward         # [n_batch] # TODO:DEBUG

        # cal value loss
        qvalue = self.value_network(state, subgraph, agent_id, mask)                # [n_batch, n_act]
        value = torch.gather(qvalue, dim=-1, index=action.unsqueeze(-1)).squeeze(-1)# [n_batch]
        if self.n_update % 50==0:
            print('qvalue', qvalue[torch.randint(qvalue.shape[0], size=(2,))].detach(),
                  'max q loss', (value - target_value).abs().max().item(), (value - target_value).abs().mean().item())
        value_loss = (0.5 * (value - target_value) ** 2).mean()                     # []

        self.value_optim.zero_grad()
        value_loss.backward()
        self.value_optim.step()
        # soft update
        for target_param, param in zip(self.target_value_network.parameters(), self.value_network.parameters()):
            target_param.data.copy_(target_param * (1 - self.soft_update_scale) + param * self.soft_update_scale)

        policy_loss, entropy_loss = torch.zeros([1], device=self.device), torch.zeros([1], device=self.device)

        # cal policy loss
        with torch.no_grad():
            target_qvalue = self.target_value_network(state, subgraph, agent_id, new_mask)
                                                                                    # [n_batch, n_act]
        action_value = self.control_network(state, subgraph, agent_id)              # [..., (n_ens,) n_act]
        if self.use_ensemble:
            ens_mask = (torch.rand(action_value.shape[:-1], device=self.device).unsqueeze(-1)
                        < self.ensemble_train_ratio)
                                                                                    # [..., n_ens, 1]
            policy_loss = (0.5 * (action_value - target_qvalue.unsqueeze(-2)) ** 2 * ens_mask).mean() \
                          / self.ensemble_train_ratio                               # []
        else:
            policy_loss = (0.5 * (action_value - target_qvalue) ** 2).mean()        # []

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        # soft update
        for target_param, param in zip(self.target_control_network.parameters(), self.control_network.parameters()):
            target_param.data.copy_(target_param * (1 - self.soft_update_scale) + param * self.soft_update_scale)

        return value_loss.item(), policy_loss.item(), entropy_loss.item()

    def save(self):
        if not os.path.exists('output/{}'.format(self.start_t)):
            os.makedirs('output/{}'.format(self.start_t))
        model_path = 'output/{}/sleep_controller.pt'.format(self.start_t)
        torch.save({
                'control_network_state': self.control_network.state_dict(),
                'value_network_state_dict': self.value_network.state_dict(),
                'target_control_network_state': self.target_control_network.state_dict(),
                'target_value_network_state': self.target_value_network.state_dict(),
                'value_optim_state_dict': self.value_optim.state_dict(),
                'policy_optim_state_dict': self.policy_optim.state_dict(),
        }, model_path)

    def load(self, model_path):
        checkpoint = torch.load(model_path)
        self.control_network.load_state_dict(checkpoint['control_network_state'])
        self.value_network.load_state_dict(checkpoint['value_network_state_dict'])
        self.target_control_network.load_state_dict(checkpoint['target_control_network_state'])
        self.target_value_network.load_state_dict(checkpoint['target_value_network_state'])
        self.value_optim.load_state_dict(checkpoint['value_optim_state_dict'])
        self.policy_optim.load_state_dict(checkpoint['policy_optim_state_dict'])

def get_mask(env, b_act_onehot, cell_idx, device):
    """

    :param env:
    :param b_act_onehot:    [n_batch, n_cell, 3]
    :param cell_idx         [n_batch]
    :param device:
    :return:
    """
    mask_bs = env.can_turn_off_bs(b_act_onehot, cell_idx)  # [n_batch]
    mask_cp = env.cell_grid_total_capacity(b_act_onehot, cell_idx)  # [n_batch]
    b_mask = torch.from_numpy(np.concatenate([mask_bs[:, None], mask_cp[:, None]], axis=-1)).to(device) # [n_batch, 2]

    return b_mask

def get_near_action_for_selected_agent(agent, gtool, b_state, cell_idx_tensor, order, batchsize, n_cell, device):
    compressed_states, subgraph, batch_node_id \
        = gtool.get_fea_graph(node_ids=cell_idx_tensor, fea=b_state, order=order)
    b_act = agent.sample_action(compressed_states, subgraph, use_target_network=True)   # [?]
    tmp = torch.zeros([batchsize * n_cell], device=device, dtype=torch.int64)
    tmp[batch_node_id] = b_act
    b_act = tmp.reshape([batchsize, n_cell])                                            # [batchsize, n_cell]

    return b_act

def get_approximate_optimal_solution_in_grid(cap, power_coef, traffic):
    """

    :param cap:                 [n_cell]
    :param power_coef:          [n_cell, 2]
    :param traffic:             float
    :return:
    """
    optimal_value, optimal_solution = np.inf, np.zeros(cap.size, dtype='int64')
    for n_part in range(1, cap.size+1):
        traffic_part = traffic / n_part
        power = power_coef[:, 0] * traffic_part + power_coef[:, 1] + (cap < traffic_part) * 1e12
        sort_id = np.argsort(power)
        power_total = np.sum(power[sort_id[:n_part]])
        if np.sum(cap[sort_id[:n_part]]) > traffic and power_total < optimal_value:
            optimal_value = power_total
            optimal_solution = np.ones(cap.size, dtype='int64')
            optimal_solution[sort_id[:n_part]] = 0
    return optimal_solution

def get_optimal_solution_in_grid(cap, power_coef, traffic):
    """

    :param cap:                 [n_cell]
    :param power_coef:          [n_cell, 2]
    :param traffic:             float
    :return:
    """
    sort_id = np.argsort(-cap)
    temp = traffic
    for i in range(cap.size):
        temp -= cap[sort_id[i]]
        if temp <= 0 :
            break
    n_sel1 = i + 1

    optimal_value, optimal_solution = np.inf, np.zeros(cap.size, dtype='int64')
    for n_sel in range(n_sel1, min(cap.size, n_sel1 * 3)):
        if n_sel >= 3:
            debug = 1
        sol = [i for i in range(n_sel)]
        while True:
            print(sol)
            sort_id = np.argsort(power_coef[sol, 0])
            tmp_traffic = traffic
            power = 0.
            for i in sort_id:
                id = sol[i]
                power += power_coef[id, 0] * min(cap[id], tmp_traffic) + power_coef[id, 1]
                tmp_traffic = max(0, tmp_traffic - cap[id])
            if tmp_traffic <= 0 and power < optimal_value:
                optimal_value = power
                optimal_solution = np.ones(cap.size, dtype='int64')
                optimal_solution[sol] = 0

            add_i = None
            for i in range(1, n_sel+1):
                if sol[-i] + 1 <= cap.size - 1 - (i-1):
                    add_i = n_sel - i
                    break
            if add_i is None:
                break
            sol[add_i] = sol[add_i] + 1

            for i in range(add_i+1, n_sel):
                sol[i] = sol[i-1] + 1

    return optimal_solution

def get_solution_for_all_grid(grid_groups, bs_groups, capacity, power_coef, grid_traffic, method='optimal'):
    """

    :param grid_groups:         {gid:[cid, ...], ...}
    :param bs_groups:           {bid:[cid, ...], ...}
    :param capacity:            [n_cell, 1]
    :param power_coef:          [n_cell, 2]
    :param grid_traffic:        [n_grid]
    :param method               {'approximate', 'optimal'}
    :return:
    """
    solution = np.ones(capacity.shape[0], dtype='int64')
    for gid in grid_groups:
        ids = grid_groups[gid]
        if method == 'approximate':
            temp = get_approximate_optimal_solution_in_grid(capacity[ids, 0], power_coef[ids], grid_traffic[gid])
        elif method == 'optimal':
            temp = get_optimal_solution_in_grid(capacity[ids, 0], power_coef[ids], grid_traffic[gid])
        else:
            raise NotImplementedError
        solution[ids] = temp

    bs_status = np.ones(len(bs_groups))
    cell_status = np.zeros(capacity.shape[0])
    bs_cell_activated = []
    for bid in range(len(bs_groups)):
        bs_status[bid] = (1-solution[bs_groups[bid]]).sum()==0
        if (solution[bs_groups[bid]]==0).sum() <= 2:
            cell_status[bs_groups[bid]] = 1

    solution1 = np.ones(capacity.shape[0], dtype='int64')
    for gid in grid_groups:
        ids = []
        for i in grid_groups[gid]:
            if cell_status[i] != 1:
                ids.append(i)
        if len(ids) == 0 or capacity[ids, 0].sum() < grid_traffic[gid]:
            solution1[grid_groups[gid]] = solution[grid_groups[gid]]
        else:
            if method == 'approximate':
                temp = get_approximate_optimal_solution_in_grid(capacity[ids, 0], power_coef[ids], grid_traffic[gid])
            elif method == 'optimal':
                temp = get_optimal_solution_in_grid(capacity[ids, 0], power_coef[ids], grid_traffic[gid])
            else:
                raise NotImplementedError
            solution1[ids] = temp
    return solution1

if __name__ == '__main__':
    setproctitle.setproctitle('basestation_sleep@huangwenzhen')
    _t_iter = time.time()

    device = 'cuda:1'
    single_step = True
    auto_cell_off_control = True
    gamma = 0.99 if not single_step else 0.
    w_entropy = None
    n_input = 15 if not single_step else 10
    reward_scale = 1.
    learning_rate = 0.0003
    buffer_size = 100
    batchsize = 512
    n_action = 3
    n_selected_action = 2 if auto_cell_off_control else 3      # agent only can select in the first two actions if true


    env = BaseStationSleepingProblem(
        unsatisfied_penalty_type='action_constraint',
        auto_cell_off_control=auto_cell_off_control,
        single_step=single_step
    )
    graph = get_sparse_graph([env.cell_bs_groups, env.cell_grid_groups],
                             [env.cell_in_bs_id, env.cell_in_grid_id],
                             env.n_cell, device)
    gtool = GraphTool(env.n_cell, graph)
    agent = CellControlMFAgent(
        n_input=n_input,
        n_output=n_selected_action,
        n_agent=env.n_cell,
        gamma=gamma,
        w_entropy=w_entropy,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        graph=graph,
        device=device)
    transition = {'state': None,
                  'action': None,
                  'reward': None,
                  'next_state': None,
                  'done': None}

    # env init
    state, info = env.init()

    n_iter = 0
    n_ep = 0
    done = False
    cum_reward = 0
    best_cum_reward = -np.inf
    ep_uns_flow, ep_value_loss, ep_policy_loss, ep_entropy_loss = 0., 0., 0., 0.
    # #debug
    # temp = np.load('input/typical_info.npy', allow_pickle=True)[()]
    # tempt = 0
    # #debug
    while True:
        transition['state'] = state
        state = torch.from_numpy(state).float().to(device)                          # [n_cell, n_in]
        action = agent.sample_action(state, eval_mode=n_ep%10==0).long()            # [n_cell]
        if n_ep == 0:
            action[:] = n_selected_action - 1
            # tmp = np.copy(env.cell_power_coef)
            # tmp[:, 1:2] -= env.cell_sleep_power
            # action1 = get_solution_for_all_grid(env.cell_grid_groups, env.cell_bs_groups,
            #                                     env.cell_capacity, env.cell_power_coef,
            #                                     env.grid_flow[env.timestep, :], method='optimal')
            # action = torch.from_numpy(action1).to(device)
            # #debug
            # temp1 = {cid: temp['Grid'][gid][0][tempt][cid] for gid in temp['Grid'] for cid in temp['Grid'][gid][0][tempt]}
            # action1 = torch.LongTensor([1 - temp1[cid]['Status'] for cid in env.cell_sids]).to(device=device)
            # # for t in range(48):
            # #     print('active cell num', [sum([temp['Grid'][gid][0][t][cid]['Status'] for cid in temp['Grid'][gid][0][t]])
            # #                               for gid in temp['Grid']])
            # print('power rru', sum([temp1[cid]['Power'] for cid in env.cell_sids]))
            # print('traffic', sum([temp1[cid]['Traffic'] for cid in env.cell_sids]))
            # tempt = (tempt + 1) % 48
            # #debug
        action_onehot = F.one_hot(action, n_action).cpu().numpy()                   # [n_cell, n_act]

        next_state, reward, done, info = env.step(action_onehot)
        transition['action'] = action.cpu().numpy()
        transition['reward'] = reward
        transition['next_state'] = next_state if next_state is not None else transition['state']
        transition['done'] = done
        if n_ep != 0:
            agent.buffer.append(transition)

        if agent.buffer.is_full():
            b_state, b_action, b_reward, b_next_state, b_done = agent.buffer.sample(batchsize)
            b_state = torch.from_numpy(b_state).to(device)
            b_act_onehot = to_onehot(b_action, n_action)                            # [n_batch, n_cell, n_act]
            b_next_state = torch.from_numpy(b_next_state).to(device)
            for _ in range(10):
                cell_idx = np.random.randint(low=0, high=env.n_cell, size=batchsize)
                cell_idx_tensor = torch.from_numpy(cell_idx).to(device)

                b_mask = get_mask(env, b_act_onehot, cell_idx, device)

                b_act_new = get_near_action_for_selected_agent(agent, gtool, b_state, cell_idx_tensor, 2,
                                                               batchsize, env.n_cell, device)
                b_act_new_onehot = F.one_hot(b_act_new, n_action).cpu().numpy()     # [n_batch, n_cell, n_act]
                b_mask_new = get_mask(env, b_act_new_onehot, cell_idx, device)

                b_next_act = get_near_action_for_selected_agent(agent, gtool, b_next_state, cell_idx_tensor, 2,
                                                                batchsize, env.n_cell, device)
                b_next_act_onehot = F.one_hot(b_next_act, n_action).cpu().numpy()   # [n_batch, n_cell, n_act]
                b_next_mask = get_mask(env, b_next_act_onehot, cell_idx, device)

                compressed_states, subgraph, _ \
                    = gtool.get_fea_graph(node_ids=cell_idx_tensor, fea=[b_state, b_next_state], order=1)
                bc_state, bc_next_state = compressed_states                         # [?, n_dim]
                selected_cell_id_in_batch_cell_ids = subgraph.coalesce().indices()[1].unique()
                value_loss, policy_loss, entropy_loss \
                    = agent.train(bc_state,
                                  torch.from_numpy(b_action[range(batchsize), cell_idx]).to(device),
                                  torch.from_numpy(b_reward[range(batchsize), cell_idx]).to(device),
                                  bc_next_state,
                                  torch.from_numpy(b_done).to(device),
                                  subgraph,
                                  selected_cell_id_in_batch_cell_ids,
                                  b_mask,
                                  b_act_new[range(batchsize), cell_idx],
                                  b_mask_new,
                                  b_next_act[range(batchsize), cell_idx],
                                  b_next_mask)
            print('n_iter={}, n_ep={}, reward={}, value_loss={}, policy_loss={}, entropy_loss={} time={}'
                  .format(n_iter, n_ep, reward.sum().item(), value_loss, policy_loss, entropy_loss, time.time()-_t_iter))
            _t_iter = time.time()

        state = next_state
        n_iter += 1
        cum_reward += reward
        if done:
            power_tmp = info['ep_record']['power_rru'] + info['ep_record']['power_bbu']\
                        + info['ep_record']['power_airconditioner']
            original_power_tmp = info['ep_record']['original_power_rru'] + info['ep_record']['original_power_bbu']\
                                 + info['ep_record']['original_power_airconditioner']
            print('power', power_tmp, 'original_power', original_power_tmp, 1 - power_tmp / original_power_tmp)
            print(info['ep_record'])
            if n_ep >= 1 and cum_reward.sum() > best_cum_reward:
                best_cum_reward = cum_reward.sum()

                agent.save()
                np.save('output/{}/sleep_records.npy'.format(agent.start_t), info)

            n_ep += 1
            n_iter = 0
            cum_reward = 0
            state, info = env.init()


    pass
