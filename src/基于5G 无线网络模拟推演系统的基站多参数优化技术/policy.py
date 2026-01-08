import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from ATSPModel import EncodingBlock, MultiGraphEncoderLayer
from ATSPModel_LIB import MultiHeadAttention, AddAndInstanceNormalization
from environment import BaseStationSleepingProblem


def attention(emb, w_qkv, attention_layer, mix_layer, norm, n_head, n_embed):
    n_batch, n_cell = emb.shape[:2]

    qkv = w_qkv(emb)                                                            # [n_batch, n_cell, n_embed * n_head * 3]
    qkv = qkv.reshape(n_batch, n_cell, 3, n_head, n_embed)                      # [n_batch, n_cell, 3, n_head, n_dim]
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]                          # [n_batch, n_cell, n_head, n_dim],...
    q = q.transpose(1, 2)                                                       # [n_batch, n_head, n_cell, n_dim]
    k = k.transpose(1, 2)                                                       # [n_batch, n_head, n_cell, n_dim]
    v = v.transpose(1, 2)                                                       # [n_batch, n_head, n_cell, n_dim]
    att = attention_layer(q, k, v)                                              # [n_batch, n_cell, n_dim * n_head]
    mix = mix_layer(att)                                                        # [n_batch, n_cell, n_dim]
    emb = norm(emb, mix)                                                        # [n_batch, n_cell, n_dim]

    return emb


class CellControlNetwork(nn.Module):
    def __init__(self,
                 n_input,
                 n_embed=16,
                 n_head=6,
                 device='cuda:0'):
        super(CellControlNetwork, self).__init__()
        self.n_input = n_input
        self.n_embed = n_embed
        self.n_head = n_head
        self.model_params = {
            'head_num': self.n_head,
            'qkv_dim': self.n_embed,
            'embedding_dim': self.n_embed
        }

        self.encoder = nn.Sequential(
            nn.Linear(n_input, self.n_embed),
            nn.ReLU(),
            nn.Linear(self.n_embed, self.n_embed),
        )

        self.w_qkv1 = nn.Linear(self.n_embed, self.n_embed * self.n_head * 3, bias=False)
        self.attention_layer1 = MultiHeadAttention(head_num=self.n_head, qkv_dim=self.n_embed)
        self.mix_layer1 = nn.Linear(self.n_embed * self.n_head, self.n_embed)
        self.norm1 = AddAndInstanceNormalization(embedding_dim=self.n_embed)

        self.w_qkv2 = nn.Linear(self.n_embed, self.n_embed * self.n_head * 3, bias=False)
        self.attention_layer2 = MultiHeadAttention(head_num=self.n_head, qkv_dim=self.n_embed)
        self.mix_layer2 = nn.Linear(self.n_embed * self.n_head, self.n_embed)
        self.norm2 = AddAndInstanceNormalization(embedding_dim=self.n_embed)

        self.activation_layer = nn.Linear(self.n_embed, 3)

        self.w_s = nn.Linear(self.n_embed, self.n_head, bias=False)
        self.w_v = nn.Linear(self.n_embed, self.n_embed * self.n_head, bias=False)
        self.mix_layer3 = nn.Linear(self.n_embed * self.n_head, self.n_embed)
        self.value_layer = nn.Sequential(
            nn.Linear(self.n_embed, self.n_embed),
            nn.ReLU(),
            nn.Linear(self.n_embed, 1)
        )

        self.device = device
        self.to(self.device)

    def forward(self, state):
        """

        :param state:           # [n_batch, n_cell, n_in]
        :return:
        """
        n_batch, n_cell = state.shape[:2]
        x = self.encoder(state)                                                 # [n_batch, n_cell, n_dim]
        x = attention(x, self.w_qkv1, self.attention_layer1, self.mix_layer1, self.norm1, self.n_head, self.n_embed)
                                                                                # [n_batch, n_cell, n_dim]
        x = attention(x, self.w_qkv2, self.attention_layer2, self.mix_layer2, self.norm2, self.n_head, self.n_embed)
                                                                                # [n_batch, n_cell, n_dim]

        activation = self.activation_layer(x)                                   # [n_batch, n_cell, 3]
        action_prob = torch.softmax(activation, dim=-1)                         # [n_batch, n_cell, 3]

        score = torch.softmax(self.w_s(x), dim=1)                               # [n_batch, n_cell, n_head]
        score = score.transpose(1, 2)                                           # [n_batch, n_head, n_cell]
        emb = self.w_v(x)                                                       # [n_batch, n_cell, n_dim * n_head]
        emb = emb.reshape([n_batch, n_cell, self.n_head, self.n_embed])         # [n_batch, n_cell, n_head, n_dim]
        emb = emb.transpose(1, 2)                                               # [n_batch, n_head, n_cell, n_dim]
        emb = (score.unsqueeze(-1) * emb).sum(-2)                               # [n_batch, n_head, n_dim]
        emb = self.mix_layer3(emb.view(n_batch, -1))                            # [n_batch, n_dim]
        value = self.value_layer(emb)                                           # [n_batch, 1]

        return action_prob, value                                               # [n_batch, n_cell, 3], [n_batch, 1]


class CellControlAgent:
    def __init__(self, n_input, gamma, w_entropy, learning_rate, device):
        self.n_input = n_input
        self.gamma = gamma
        self.w_entropy = w_entropy
        self.learning_rate = learning_rate
        self.control_network = CellControlNetwork(n_input=n_input, device=device)

        self.optim = torch.optim.Adam(self.control_network.parameters(), lr=self.learning_rate)

    def sample_action(self, state):
        """

        :param state:           [n_batch, n_cell, n_in]
        :return:
        """
        with torch.no_grad():
            action_prob, value = self.control_network(state)                    # [n_batch, n_cell, 3], [n_batch, 1]

            dist = torch.distributions.Categorical(action_prob)
            action = dist.sample()                                              # [n_batch, n_cell]

            return action, value                                                # [n_batch, n_cell], [n_batch, 1]

    def train(self, state, action, reward, next_state, done):
        """
        :param state:           [n_batch, n_cell, n_in]
        :param action:          [n_batch, n_cell]
        :param reward:          [n_batch, 1]
        :param next_state:      [n_batch, n_cell, n_in]
        :param done             [n_batch, 1]
        :return:                []
        """
        with torch.no_grad():
            if next_state is not None:
                _, next_value = self.control_network(next_state)            # _, [n_batch, 1]
            else:
                next_value = 0.
            target_value = next_value * self.gamma * done + reward          # [n_batch, 1]

        action_prob, value = self.control_network(state)                    # [n_batch, n_cell, 3], [n_batch, 1]

        value_loss = (0.5 * (value - target_value) ** 2).mean()             # []

        adv = (target_value - value).detach()                               # [n_batch, 1]
        dist = torch.distributions.Categorical(action_prob)
        action_logprob = dist.log_prob(action)                              # [n_batch, n_cell]
        policy_loss = (-adv * action_logprob).mean()                        # []

        entropy_loss = dist.entropy().mean()                                # []

        total_loss = value_loss + policy_loss +self.w_entropy * entropy_loss
        self.optim.zero_grad()
        total_loss.backward()
        self.optim.step()

        return value_loss.item(), policy_loss.item(), entropy_loss.item()

if __name__ == '__main__':
    device = 'cuda:2'
    gamma = 0.99
    w_entropy = 0.01
    n_input = 15
    reward_scale = 0.001
    learning_rate = 0.0003

    env = BaseStationSleepingProblem(
        unsatisfied_penalty_type='action_constraint',
    )
    env.set_environment_mode('traversal')
    agent = CellControlAgent(
        n_input=n_input,
        gamma=gamma,
        w_entropy=w_entropy,
        learning_rate=learning_rate,
        device=device)

    state, info = env.init()
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)         # [1, n_cell, n_in]
    n_iter = 0
    n_ep = 0
    total_cum_reward = 0
    cum_reward = 0
    infos = []
    done = False
    ep_uns_flow, ep_value_loss, ep_policy_loss, ep_entropy_loss = 0., 0., 0., 0.
    while True:
        action, value = agent.sample_action(state)                              # [1, n_cell], [1, 1]
        action_onehot = F.one_hot(action[0], 3).cpu().numpy()                   # [n_cell, 3]

        next_state, reward, done, info = env.step(action_onehot)
        next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(device) if next_state is not None else None
        reward = torch.Tensor([[reward * reward_scale]]).to(device)
        done = torch.Tensor([[done]]).float().to(device)
        action_onehot = info['adjusted_action']
        action = torch.argmax(torch.from_numpy(action_onehot).to(device), dim=1)

        ep_uns_flow += (info['unsatisfied_down_flow'] > 0)

        value_loss, policy_loss, entropy_loss \
            = agent.train(state, action, reward, next_state, done)
        # print('n_iter={}, reward={}, value_loss={}, policy_loss={}, entropy_loss={}'
        #       .format(n_iter, reward.item(), value_loss, policy_loss, entropy_loss))
        ep_value_loss += value_loss
        ep_policy_loss += policy_loss
        ep_entropy_loss += entropy_loss

        state = next_state
        n_iter += 1
        cum_reward += reward
        if done:
            n_ep += 1
            print('n_ep={}, n_iter={}, cum_reward={}, '
                  'ep_uns_flow/n_iter={}, ep_value_loss/n_iter={}, ep_policy_loss/n_iter={}, ep_entropy_loss/n_iter={}'
                  .format(n_ep, n_iter, cum_reward.item(),
                  ep_uns_flow/n_iter, ep_value_loss/n_iter, ep_policy_loss/n_iter, ep_entropy_loss/n_iter))
            infos.append(info['cell_record'])
            total_cum_reward += cum_reward

            cum_reward = 0
            state, info = env.init()
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)  # [1, n_cell, n_in]

            n_grid = len(env.cell_grid)
            if env.grid_id == n_grid - 1:
                print('--------', n_ep, total_cum_reward)
                np.save('cell_records.npy', infos)

                total_cum_reward = 0
                infos = []

    pass
