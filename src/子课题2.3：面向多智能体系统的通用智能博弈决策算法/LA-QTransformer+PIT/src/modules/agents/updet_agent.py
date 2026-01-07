import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ..utils.transformer import EncoderLayer


class UpDetAgent(nn.Module):
    def __init__(self, inputZ_shape, args):
        super(UpDetAgent, self).__init__()
        self.args = args

        # ------------------------------------------
        #                net parameter
        # ------------------------------------------
        # net hyperparameter
        self.unit_type_bits = self.args.unit_type_bits
        self.model_dim = self.args.model_dim
        self.num_heads = self.args.num_heads
        self.ffn_dim = self.args.ffn_dim
        self.dropout = self.args.dropout

        # agent obs split(in order)
        self.n_move_feats = self.args.move_feats_dim
        self.n_enemy_feats = self.args.enemy_feats_dim
        self.n_ally_feats = self.args.ally_feats_dim
        self.n_own_feats = self.args.own_feats_dim
        self.n_self_action = self.args.n_self_action

        self.n_ally = self.args.n_ally
        self.n_enemy = self.args.n_enemy

        # ------------------------------------------
        #                net structure
        # ------------------------------------------
        self.n_own = self.n_move_feats + self.n_own_feats
        # if self.args.obs_last_action:
        #     self.n_own += self.args.last_action_dim
        # if self.args.obs_agent_id:
        #     self.n_own += self.n_ally + 1

        self.own_fc = nn.Linear(self.n_own + self.model_dim, self.model_dim * 2)
        self.ally_fc = nn.Linear(self.n_ally_feats // self.n_ally + self.model_dim, self.model_dim * 2)
        self.enemy_fc = nn.Linear(self.n_enemy_feats // self.n_enemy + self.model_dim, self.model_dim * 2)

        self.transformer = EncoderLayer(in_dim=self.model_dim*2,
                                        model_dim=self.model_dim*2,
                                        num_heads=self.num_heads,
                                        ffn_dim=self.ffn_dim*2,
                                        dropout=self.dropout)
        self.transformer2 = EncoderLayer(in_dim=self.model_dim*2,
                                        model_dim=self.model_dim*2,
                                        num_heads=self.num_heads,
                                        ffn_dim=self.ffn_dim*2,
                                        dropout=self.dropout)

        self.self_fc = nn.Linear(self.model_dim, self.n_self_action)
        self.attack_fc = nn.Linear(self.model_dim, 1)


    def init_hidden(self):
        return th.zeros(1, (1 + self.n_ally + self.n_enemy) * self.model_dim).cuda()

    def forward(self, inputs, hidden_state):
        bs = inputs.shape[0]

        own_input = th.cat([inputs[:, :self.n_move_feats],
                           inputs[:, self.n_move_feats + self.n_enemy_feats + self.n_ally_feats: self.n_move_feats + self.n_enemy_feats + self.n_ally_feats + self.n_own_feats]], -1)
      
        #if self.args.obs_last_action:
        # own_other = th.zeros(bs, self.args.last_action_dim).cuda()
        # own_other[:, :self.n_enemy + self.n_self_action] = inputs[:, self.n_enemy_feats + self.n_ally_feats + self.n_move_feats + self.n_own_feats:]

        # own_input = own_input, own_other], -1)

        enemy_input = inputs[:, self.n_move_feats:self.n_move_feats + self.n_enemy_feats].reshape(bs, self.n_enemy, -1)
        
        ally_input = inputs[:,
                     self.n_move_feats + self.n_enemy_feats:self.n_move_feats + self.n_enemy_feats + self.n_ally_feats].reshape(
            bs, self.n_ally, -1)


        h_feats = hidden_state.reshape(-1, (1 + self.n_ally + self.n_enemy), self.model_dim)

        own_input = th.cat([own_input, h_feats[:, 0,:]], dim=-1).unsqueeze(-2)
        ally_input = th.cat([ally_input, h_feats[:, 1:self.n_ally+1,:]], dim=-1)
        enemy_input = th.cat([enemy_input, h_feats[:, self.n_ally+1:,:]], dim=-1)
        
        own_input = self.own_fc(own_input)
        ally_input = self.ally_fc(ally_input)
        enemy_input = self.enemy_fc(enemy_input)
        tf_input = th.cat([own_input, ally_input, enemy_input], dim=-2)

        r_feats, _ = self.transformer(tf_input)
        r_feats, _ = self.transformer2(r_feats)
        
        own_feats = r_feats[:, 0, :].unsqueeze(-2)
        ally_feats = r_feats[:, 1:self.n_ally+1, :]
        enemy_feats = r_feats[:, self.n_ally+1:, :]

        own_embedding, own_h = own_feats[..., :self.model_dim], own_feats[..., self.model_dim:]
        ally_embedding, ally_h = ally_feats[..., :self.model_dim], ally_feats[..., self.model_dim:]
        enemy_embedding, enemy_h = enemy_feats[..., :self.model_dim], enemy_feats[..., self.model_dim:]

        h = th.cat([own_h, ally_h, enemy_h], dim=-2)
        h = h.reshape(h.shape[0], -1)

        self_action = self.self_fc(own_embedding).squeeze(-2)
        attack_action = self.attack_fc(enemy_embedding).squeeze(-1)
        #attack_action = enemy_embedding.mean(dim=-1).squeeze(-1)

        action = th.cat([self_action, attack_action], dim=-1)

        return action, h

class ScaledDotProductAttention(nn.Module):
    def __init__(
            self,
            attention_dropout=0.0
    ):
        super(ScaledDotProductAttention, self).__init__()

        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(
            self,
            q,
            k ,
            v,
            scale=None,
            attn_mask=None
    ):
        attention = th.bmm(q, k.transpose(1, 2))

        if scale:
            attention *= scale
        if attn_mask:
            attention = attn_mask.masked_fill_(attn_mask, -np.inf)

        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = th.bmm(attention, v)

        return context, attention


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            in_dim,
            model_dim=512,
            n_heads=8,
            dropout=0.0
    ):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // n_heads
        self.n_heads= n_heads

        self.residual_fc = nn.Linear(in_dim, model_dim)

        self.linear_k = nn.Linear(in_dim, self.dim_per_head * n_heads)
        self.linear_v = nn.Linear(in_dim, self.dim_per_head * n_heads)
        self.linear_q = nn.Linear(in_dim, self.dim_per_head * n_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)

        self.linear_final = nn.Linear(self.dim_per_head * n_heads, model_dim)
        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(
            self,
            k,
            v,
            q,
            attn_mask=None
    ):
        dim_per_head = self.dim_per_head
        n_heads = self.n_heads
        batch_size = k.size(0)

        residual = self.residual_fc(q)

        k = self.linear_k(k).view(batch_size * n_heads, -1, dim_per_head)
        v = self.linear_v(v).view(batch_size * n_heads, -1, dim_per_head)
        q = self.linear_q(q).view(batch_size * n_heads, -1, dim_per_head)

        if attn_mask:
            attn_mask = attn_mask.repeat(n_heads, 1, 1)

        scale = (k.size(-1)) ** (-0.5)
        context, attention = self.dot_product_attention(q, k, v, scale, attn_mask)

        context = context.view(batch_size, -1, dim_per_head * n_heads)
        output = self.linear_final(context)
        output = self.dropout(output)
        output = self.layer_norm(residual + output)
        

        return output, attention

class PositionalWiseFeedForward(nn.Module):
    def __init__(
            self,
            model_dim=512,
            ffn_dim=2048,
            dropout=0.0
    ):
        super(PositionalWiseFeedForward, self).__init__()

        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(
            self,
            x
    ):
        output = x.transpose(1, 2)
        output = self.w2(F.relu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))

        output = self.layer_norm(x + output)

        return output


class EncoderLayer(nn.Module):
    def __init__(
            self,
            in_dim,
            model_dim=512,
            num_heads=8,
            ffn_dim=2048,
            dropout=0.0
    ):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(in_dim, model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(
            self,
            inputs,
            attn_mask=None
    ):
        context, attention = self.attention(inputs, inputs, inputs, attn_mask)
        output = self.feed_forward(context)

        return output, attention
