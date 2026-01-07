import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

from ..utils.transformer import EncoderLayer, DynamicEncoderLayer, MultiHeadAttention, ScaleEncoderLayer, HeterogeneousEncoderLayer


class METAQV5HETERMixer(nn.Module):
    def __init__(self, args):
        super(METAQV5HETERMixer, self).__init__()

        # args
        self.args = args
        self.state_dim = int(np.prod(args.state_shape))
        self.n_agent = args.n_agents
        self.n_enemy = args.n_enemy
        self.n_action = 6 + self.n_enemy

        self.ally_dim = 4 + args.shield_bits_ally + args.unit_type_bits
        self.enemy_dim = 3 + args.shield_bits_enemy + args.unit_type_bits

        self.total_ally_dim = self.ally_dim * self.n_agent
        self.total_enemy_dim = self.enemy_dim * self.n_enemy

        model_dim = args.metaq_transformer_model_dim
        ffn_dim = args.metaq_transformer_ffn_dim
        n_heads = args.metaq_transformer_n_heads
        dropout = args.metaq_transformer_dropout
        self.n_heads = n_heads
        self.model_dim = model_dim

        final_layer_dim = args.metaq_final_layer_dim

        # embedding layer
        self.ally_encoder = nn.Linear(self.ally_dim, model_dim)
        self.enemy_encoder = nn.Linear(self.enemy_dim, model_dim)
        self.base_transformer = ScaleEncoderLayer(None, model_dim, n_heads, ffn_dim, dropout)

        # dynamic layer
        # self.ally_dynamic_encoder = nn.Linear(model_dim + 6, model_dim)
        # self.enemy_dynamic_encoder = nn.Linear(model_dim + 1, model_dim)
        # self.dynamic_transformer = DynamicEncoderLayer(None, model_dim, n_heads, ffn_dim, dropout)
        # self.ally_dynamic_decoder_mean = nn.Linear(model_dim, self.ally_dim)
        # self.ally_dynamic_decoder_logstd = nn.Linear(model_dim, self.ally_dim)
        # self.enemy_dynamic_decoder_mean = nn.Linear(model_dim, self.enemy_dim)
        # self.enemy_dynamic_decoder_logstd = nn.Linear(model_dim, self.enemy_dim)

        # reward layer
        # self.reward_encoder = nn.Linear(model_dim, 1)

        # attention layer
        self.attention = HeterogeneousEncoderLayer(model_dim*2, model_dim*2, n_heads)

        # final mul & add
        self.final_mul_1 = nn.Linear(model_dim * 4, final_layer_dim)
        self.final_mul_2 = nn.Linear(final_layer_dim, n_heads)

        self.final_add_1 = nn.Linear(model_dim * 4, final_layer_dim)
        self.final_add_2 = nn.Linear(final_layer_dim, 1)

    def forward(self, agent_qs, states, actions):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.n_agent, self.args.n_actions).float()
        agent_qs = agent_qs.view(-1, 1, self.n_agent)

        bz = states.size(0)

        # raw state
        ally_states = states[:, :self.total_ally_dim].reshape(bz, self.n_agent, -1)
        enemy_states = states[:, self.total_ally_dim:self.total_ally_dim + self.total_enemy_dim].reshape(bz,
                                                                                                         self.n_enemy,
                                                                                                         -1)
        # embedding layer
        ally_embedding_pre = self.ally_encoder(ally_states)
        enemy_embedding_pre = self.enemy_encoder(enemy_states)
        embedding_pre = th.cat([ally_embedding_pre, enemy_embedding_pre], dim=-2)
        
        embeddings, _ = self.base_transformer(embedding_pre, embedding_pre, embedding_pre, embedding_pre)

        # dynamic layer
        # ally_embedding_pos = embeddings[:, :self.n_agent, :self.model_dim]
        # ally_embedding_action = actions[:, :, :6]
        # ally_embedding_pos = th.cat([ally_embedding_pos, ally_embedding_action], dim=-1)
        # enemy_embedding_action = actions[:, :, 6:].sum(-2).unsqueeze(-1)
        # enemy_embedding_pos = th.cat([embeddings[:, self.n_agent:, :self.model_dim], enemy_embedding_action], dim=-1)
        
        # ally_embedding_pos = self.ally_dynamic_encoder(ally_embedding_pos)
        # enemy_embedding_pos = self.enemy_dynamic_encoder(enemy_embedding_pos)
        # embedding_pos = th.cat([ally_embedding_pos, enemy_embedding_pos], dim=-2)

        # dynamic_embeddings, _ = self.dynamic_transformer(embedding_pos, embedding_pos, embedding_pos, embedding_pos)
        # dynamic_ally_embedding = dynamic_embeddings[:, :self.n_agent, :]
        # dynamic_enemy_embedding = dynamic_embeddings[:, self.n_agent:, :]

        # dynamic_ally_embedding_mean = self.ally_dynamic_decoder_mean(dynamic_ally_embedding)
        # dynamic_ally_embedding_logstd = self.ally_dynamic_decoder_logstd(dynamic_ally_embedding)
        # dynamic_enemy_embedding_mean = self.enemy_dynamic_decoder_mean(dynamic_enemy_embedding)
        # dynamic_enemy_embedding_logstd = self.enemy_dynamic_decoder_logstd(dynamic_enemy_embedding)

        # dynamic_ally = dynamic_ally_embedding_mean + dynamic_ally_embedding_logstd.exp() * th.randn_like(dynamic_ally_embedding_mean)
        # dynamic_enemy = dynamic_enemy_embedding_mean + dynamic_enemy_embedding_logstd.exp() * th.randn_like(dynamic_enemy_embedding_mean)

        # dynamic = th.cat([dynamic_ally.reshape(bz, -1), dynamic_enemy.reshape(bz, -1)], dim=-1)

        # reward layer
        # r = self.reward_encoder(embeddings[:, :, :self.model_dim].mean(-2))

        # attention layer
        mask = th.ones(embeddings.shape[0], self.n_heads, self.n_agent, self.n_agent, device=embeddings.device)
        mask[:, 0, :2, :2] = 0.
        mask[:, 0, 2:, 2:] = 0.
        mask[:, 1, :2, :2] = 1.
        mask[:, 1, 2:, 2:] = 1.
        mask = mask.reshape(-1, self.n_agent, self.n_agent)
        
        ally_embeddings = embeddings[:, :self.n_agent, :]
        _, attention_weight = self.attention(ally_embeddings, mask)

        agent_qs = agent_qs.reshape(bz, -1).unsqueeze(-1).unsqueeze(1)
        attention_weight = attention_weight.reshape(-1, self.n_heads, self.n_agent, self.n_agent)

        q_h = th.matmul(attention_weight, agent_qs)
        q_h = q_h.sum(-2)

        # final layer
        merge_embedding = th.cat([ally_embeddings.mean(-2), ally_embeddings.std(-2)], dim=-1)

        weight = F.relu(self.final_mul_1(merge_embedding))
        weight = th.abs(self.final_mul_2(weight)).view(-1, 1, self.n_heads)

        bias = F.relu(self.final_add_1(merge_embedding))
        bias = self.final_add_2(bias)

        q_tot = th.bmm(weight, q_h).squeeze(-1) + bias
        return q_tot.view(bs, -1, 1) #, dynamic















