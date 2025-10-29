import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..utils.transformer import ScaledDotProductAttention
from ..utils.transformer import MultiHeadAttention
from ..utils.transformer import MultiHeadAttentionV2
from ..utils.transformer import MultiHeadAttentionV3
from ..utils.transformer import EncoderLayer


class TransformerMixerV2(nn.Module):
    def __init__(self, args):
        super(TransformerMixerV2, self).__init__()

        self.args = args
        self.n_agent = args.n_agents
        self.n_enemy = args.n_enemy
        self.n_action = 6 + self.n_enemy

        self.ally_dim = 4 + args.shield_bits_ally + args.unit_type_bits
        self.enemy_dim = 3 + args.shield_bits_enemy + args.unit_type_bits

        self.total_ally_dim = self.ally_dim * self.n_agent
        self.total_enemy_dim = self.enemy_dim * self.n_enemy

        self.last_action_extend_dim = args.last_action_extend_dim

        self.state_dim = int(np.prod(args.state_shape))

        self.agent_attention = EncoderLayer(in_dim=self.ally_dim+self.last_action_extend_dim,
                                            model_dim=args.transformer_mix_model_dim,
                                            num_heads=args.transformer_mix_num_heads,
                                            ffn_dim=args.transformer_mix_ffn_dim,
                                            dropout=args.transformer_mix_dropout)

        self.enemy_attention = MultiHeadAttentionV2(k_in_dim=self.enemy_dim,
                                                    v_in_dim=self.enemy_dim,
                                                    q_in_dim=args.transformer_mix_model_dim,
                                                    model_dim=args.transformer_mix_model_dim,
                                                    n_heads=args.transformer_mix_num_heads,
                                                    dropout=args.transformer_mix_dropout)

        self.final_attention = MultiHeadAttentionV3(model_dim=args.transformer_mix_model_dim * 2,
                                                    n_heads=args.transformer_mix_num_heads,
                                                    dropout=args.transformer_mix_dropout, )

        self.fc_mul_1 = nn.Linear(args.transformer_mix_model_dim*4, args.fc_mul_dim)
        self.fc_mul_2 = nn.Linear(args.fc_mul_dim, args.transformer_mix_num_heads)

        self.fc_add_1 = nn.Linear(args.transformer_mix_model_dim*4, args.fc_add_dim)
        self.fc_add_2 = nn.Linear(args.fc_add_dim, 1)

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        bz = states.size(0)

        # raw state
        agent_states = states[:, :self.total_ally_dim].reshape(bz, self.n_agent, -1)
        enemy_states = states[:, self.total_ally_dim:self.total_ally_dim + self.total_enemy_dim].reshape(bz,
                                                                                                         self.n_enemy,
                                                                                                         -1)
        last_action_states = th.zeros(bz, self.n_agent, self.last_action_extend_dim).cuda()
        last_action_states[..., :self.n_action] = states[:, self.total_ally_dim + self.total_enemy_dim:].reshape(bz, self.n_agent, -1)


        # embedding layer
        agent_states = th.cat([agent_states, last_action_states], dim=-1)
        agent_feats, _ = self.agent_attention(agent_states)  # [bs, n_agent, model_dim]
        enemy_feats, _ = self.enemy_attention(enemy_states, enemy_states, agent_feats)

        # attention
        agent_embedding = th.cat([agent_feats, enemy_feats], dim=-1)
        final_embedding, final_attention = self.final_attention(agent_embedding, agent_embedding, agent_embedding)

        # Q_h
        agent_qs = agent_qs.reshape(bz, -1).unsqueeze(-1).unsqueeze(1)
        final_attention = final_attention.reshape(-1, self.args.transformer_mix_num_heads, self.n_agent, self.n_agent)

        q_h = th.matmul(final_attention, agent_qs)
        q_h = q_h.sum(-2)

        # weight
        merge_vector = th.cat([final_embedding.mean(-2), final_embedding.std(-2)], dim=-1)

        weight = F.relu(self.fc_mul_1(merge_vector))
        weight = th.abs(self.fc_mul_2(weight)).view(-1, 1, self.args.transformer_mix_num_heads)

        # bias
        bias = F.relu(self.fc_add_1(merge_vector))
        bias = self.fc_add_2(bias)

        # q_tot
        q_tot = th.bmm(weight, q_h).squeeze(-1) + bias
        return q_tot.view(bs, -1, 1)
