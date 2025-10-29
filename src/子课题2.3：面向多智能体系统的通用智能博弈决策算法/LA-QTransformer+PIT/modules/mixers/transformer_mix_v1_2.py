import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..utils.transformer import MultiHeadAttention
from ..utils.transformer import EncoderLayer


class TransformerMixerV1_2(nn.Module):
    """
    mean feature
    """
    def __init__(self, args):
        super(TransformerMixerV1_2, self).__init__()

        self.args = args
        self.n_agent = args.n_agents
        self.n_enemy = args.n_enemy
        self.n_action = 6 + self.n_enemy

        self.ally_dim = 4 + args.shield_bits_ally + args.unit_type_bits
        self.enemy_dim = 3 + args.shield_bits_enemy + args.unit_type_bits

        self.total_agent_dim = self.ally_dim * self.n_agent
        self.total_enemy_dim = self.enemy_dim * self.n_enemy

        self.state_dim = int(np.prod(args.state_shape))

        self.agent_transformer = EncoderLayer(in_dim=self.ally_dim,
                                              model_dim=args.transformer_mix_model_dim,
                                              num_heads=args.transformer_mix_num_heads,
                                              ffn_dim=args.transformer_mix_ffn_dim,
                                              dropout=args.transformer_mix_dropout)
        self.enemy_transformer = EncoderLayer(in_dim=self.enemy_dim,
                                              model_dim=args.transformer_mix_model_dim,
                                              num_heads=args.transformer_mix_num_heads,
                                              ffn_dim=args.transformer_mix_ffn_dim,
                                              dropout=args.transformer_mix_dropout)
        self.final_attention = MultiHeadAttention(in_dim=args.transformer_mix_model_dim*5,
                                                  model_dim=args.transformer_mix_model_dim*5,
                                                  n_heads=1,
                                                  dropout=args.transformer_mix_dropout,)

        self.fc_c1 = nn.Linear(self.state_dim, args.fc_c_dim)
        self.fc_c2 = nn.Linear(args.fc_c_dim, 1)

    def forward(self, agent_qs, states):

        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        bz = states.size(0)

        agent_states = states[:, :self.total_agent_dim].reshape(bz, self.n_agent, -1)
        enemy_states = states[:, self.total_agent_dim:self.total_agent_dim + self.total_enemy_dim].reshape(bz,
                                                                                                           self.n_enemy,
                                                                                                           -1)
        last_action_states = states[:, self.total_agent_dim + self.total_enemy_dim:].reshape(bz, self.n_agent,
                                                                                             -1)  # [bs, n_agent, n_action]

        agent_feats, _ = self.agent_transformer(agent_states)  # [bs, n_agent, model_dim]
        enemy_feats, _ = self.enemy_transformer(enemy_states)  # [bs, n_enemy, model_dim]

        # attention selector
        agent_enemy_attention_matrix = th.bmm(agent_feats, enemy_feats.transpose(1, 2))
        agent_enemy_attention = F.softmax(agent_enemy_attention_matrix, dim=2)
        agent_enemy_attention_index = agent_enemy_attention.argmax(-1)  # [bs, n_agent]
        agent_enemy_attention_index_repeat = agent_enemy_attention_index.unsqueeze(-1).repeat(1, 1,
                                                                                              self.args.transformer_mix_model_dim)

        enemy_embedding_attention_select = th.gather(enemy_feats, dim=1,
                                                     index=agent_enemy_attention_index_repeat)  # [bs, n_agent, model_dim]

        # last action selector
        extend_agent_feats = agent_feats.unsqueeze(2)
        extend_agent_feats = extend_agent_feats.repeat(1, 1, 6, 1)  # [bs, n_agent, 6, -1]
        extend_enemy_feats = enemy_feats.unsqueeze(1)
        extend_enemy_feats = extend_enemy_feats.repeat(1, self.n_agent, 1, 1)  # [bs, n_agent, n_enemy, -1]
        extend_feats = th.cat([extend_agent_feats, extend_enemy_feats], dim=-2)

        last_action_index = last_action_states.argmax(-1)  # [bs, n_agent]

        last_action_states_repeat = last_action_index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, self.args.transformer_mix_model_dim)

        enemy_embedding_last_action_select = th.gather(extend_feats, dim=-2, index=last_action_states_repeat).squeeze(
            -2)  # [bs, n_agent, model_dim]

        # mean feature
        id_mask = 1 - th.eye(self.n_agent, device=agent_feats.device).unsqueeze(0).repeat(bz, 1, 1) # [bs, n_agent, n_agent]
        mean_agent_feats = th.bmm(id_mask, agent_feats) / (self.n_agent-1)
        mean_enemy_feats = enemy_feats.mean(-2).unsqueeze(-2).repeat(1, self.n_agent, 1)

        # merge feats
        total_feats = th.cat([agent_feats, mean_agent_feats, mean_enemy_feats, enemy_embedding_attention_select, enemy_embedding_last_action_select], dim=-1) # [bs, n_agent, 5*model_dim]

        _, final_attention = self.final_attention(total_feats, total_feats, total_feats)

        # c layer
        c = F.relu(self.fc_c1(states))
        c = self.fc_c2(c)

        # Q_tot
        agent_qs = agent_qs.reshape(bz, -1).unsqueeze(-1)
        q = th.bmm(final_attention, agent_qs)
        q_tot = q.sum(-2) + c


        return q_tot.view(bs, -1, 1)

