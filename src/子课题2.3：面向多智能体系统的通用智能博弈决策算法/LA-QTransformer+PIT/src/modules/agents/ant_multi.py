import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ..utils.transformer import EncoderLayer


class ANTMultiAgent(nn.Module):
    """
    参数设置
    obs_agent_id: False

    """
    def __init__(self, input_shape, args):
        super(ANTMultiAgent, self).__init__()
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
        if self.args.obs_last_action:
            self.n_own += self.n_self_action
        if self.args.obs_agent_id:
            self.n_own += self.n_ally + 1
        self.fc1 = nn.Linear(self.n_own, args.rnn_hidden_dim // 2)
        self.fc_selector = nn.Linear(args.rnn_hidden_dim // 2, self.model_dim)

        self.ally_transformer1 = EncoderLayer(in_dim=self.n_ally_feats // self.n_ally,
                                             model_dim=self.model_dim,
                                             num_heads=self.num_heads,
                                             ffn_dim=self.ffn_dim,
                                             dropout=self.dropout)
        self.enemy_transformer1 = EncoderLayer(in_dim=self.n_enemy_feats // self.n_enemy + 1,
                                              model_dim=self.model_dim,
                                              num_heads=self.num_heads,
                                              ffn_dim=self.ffn_dim,
                                              dropout=self.dropout)
        
        self.ally_transformer2 = EncoderLayer(in_dim=self.n_ally_feats // self.n_ally,
                                             model_dim=self.model_dim,
                                             num_heads=self.num_heads,
                                             ffn_dim=self.ffn_dim,
                                             dropout=self.dropout)
        self.enemy_transformer2 = EncoderLayer(in_dim=self.n_enemy_feats // self.n_enemy + 1,
                                              model_dim=self.model_dim,
                                              num_heads=self.num_heads,
                                              ffn_dim=self.ffn_dim,
                                              dropout=self.dropout)

        self.rnn = nn.GRUCell(args.rnn_hidden_dim // 2 + 8 * self.model_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, self.model_dim)

        self.self_action_fc = nn.Linear(args.rnn_hidden_dim, self.n_self_action)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        # inputs [bz * n_agent, n_input]
        bs = inputs.shape[0]


        own_input = th.cat([inputs[:, :self.n_move_feats], inputs[:,
                                                           self.n_move_feats + self.n_enemy_feats + self.n_ally_feats: self.n_move_feats + self.n_enemy_feats + self.n_ally_feats + self.n_own_feats]],
                           -1)
        if self.args.obs_last_action:
            own_input = th.cat(
                [own_input, inputs[:, self.n_move_feats + self.n_enemy_feats + self.n_ally_feats + self.n_own_feats:self.n_move_feats + self.n_enemy_feats + self.n_ally_feats + self.n_own_feats+6]],
                 -1)

        enemy_input = inputs[:, self.n_move_feats:self.n_move_feats + self.n_enemy_feats].reshape(bs, self.n_enemy, -1)
        enemy_attack = inputs[:, self.n_move_feats + self.n_enemy_feats + self.n_ally_feats + self.n_own_feats+6:].reshape(bs, self.n_enemy, -1)
        enemy_input = th.cat([enemy_input, enemy_attack], -1)

        ally_input = inputs[:,
                     self.n_move_feats + self.n_enemy_feats:self.n_move_feats + self.n_enemy_feats + self.n_ally_feats].reshape(
            bs, self.n_ally, -1)
        

        enemy_type_1 = enemy_input[..., -3].gt(0.5)
        enemy_type_2 = enemy_input[..., -2].gt(0.5)
        ally_type_1 = ally_input[..., -2].gt(0.5)
        ally_type_2 = ally_input[..., -1].gt(0.5)

        n_enemy_type_1 = enemy_type_1.sum(-1)
        n_enemy_type_2 = enemy_type_2.sum(-1) 
        n_ally_type_1 = ally_type_1.sum(-1)
        n_ally_type_2 = ally_type_2.sum(-1)

        n_enemy_type_1 = th.where(n_enemy_type_1 > 0.5, n_enemy_type_1, th.ones_like(n_enemy_type_1))
        n_enemy_type_2 = th.where(n_enemy_type_2 > 0.5, n_enemy_type_2, th.ones_like(n_enemy_type_2))
        n_ally_type_1 = th.where(n_ally_type_1 > 0.5, n_ally_type_1, th.ones_like(n_ally_type_1))
        n_ally_type_2 = th.where(n_ally_type_2 > 0.5, n_ally_type_2, th.ones_like(n_ally_type_2))
 

        enemy_input_1 = enemy_input * enemy_type_1.unsqueeze(-1).float()
        enemy_input_2 = enemy_input * enemy_type_2.unsqueeze(-1).float()
        ally_input_1 = ally_input * ally_type_1.unsqueeze(-1).float()
        ally_input_2 = ally_input * ally_type_2.unsqueeze(-1).float()

        own_feats = F.relu(self.fc1(own_input))  # [bz * n_agent, n_own_feats]
        own_selector_feats = F.relu(self.fc_selector(own_feats)).unsqueeze(-1) # [bz * n_agent, model_dim, 1]

        enemy_embedding1, _ = self.enemy_transformer1(enemy_input_1)  # [bz * n_agent, n_enemy, model_dim]
        enemy_embedding2, _ = self.enemy_transformer2(enemy_input_2)  # [bz * n_agent, n_enemy, model_dim]
        ally_embedding1, _ = self.ally_transformer1(ally_input_1)  # [bz * n_agent, n_ally, model_dim]
        ally_embedding2, _ = self.ally_transformer2(ally_input_2)  # [bz * n_agent, n_ally, model_dim]


        # attention selector
        enemy_attention1 = F.softmax(th.bmm(enemy_embedding1, own_selector_feats).squeeze(-1), dim=-1)
        enemy_attention2 = F.softmax(th.bmm(enemy_embedding2, own_selector_feats).squeeze(-1), dim=-1)
        ally_attention1 = F.softmax(th.bmm(ally_embedding1, own_selector_feats).squeeze(-1), dim=-1)
        ally_attention2 = F.softmax(th.bmm(ally_embedding2, own_selector_feats).squeeze(-1), dim=-1)

        enemy_attention_index1 = enemy_attention1.argmax(-1)
        enemy_attention_index2 = enemy_attention2.argmax(-1)
        ally_attention_index1 = ally_attention1.argmax(-1)
        ally_attention_index2 = ally_attention2.argmax(-1)
        enemy_embedding_select1 = enemy_embedding1[[i for i in range(bs)], enemy_attention_index1]
        enemy_embedding_select2 = enemy_embedding2[[i for i in range(bs)], enemy_attention_index2]
        ally_attention_select1 = ally_embedding1[[i for i in range(bs)] ,ally_attention_index1]
        ally_attention_select2 = ally_embedding2[[i for i in range(bs)] ,ally_attention_index2]
       
        enemy_embedding_mean1 = enemy_embedding1.sum(dim=-2) / n_enemy_type_1.unsqueeze(-1).float()  # [bz * n_agent, model_dim]
        enemy_embedding_mean2 = enemy_embedding2.sum(dim=-2) / n_enemy_type_2.unsqueeze(-1).float() # [bz * n_agent, model_dim]
        ally_embedding_mean1 = ally_embedding1.sum(dim=-2) / n_ally_type_1.unsqueeze(-1).float()  # [bz * n_agent, model_dim]
        ally_embedding_mean2 = ally_embedding2.sum(dim=-2) / n_ally_type_2.unsqueeze(-1).float()  # [bz * n_agent, model_dim]

        rnn_input = th.cat([own_feats, ally_embedding_mean1, ally_attention_select1, ally_embedding_mean2, ally_attention_select2, 
                                      enemy_embedding_mean1, enemy_embedding_select1, enemy_embedding_mean2, enemy_embedding_select2], dim=-1)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        own_feats_h = self.rnn(rnn_input, h_in)  # [bz * n_agent, rnn_hidden_dim]

        # self action
        self_action = self.self_action_fc(own_feats_h)  # [bz * n_agent, n_self_action]

        # attack action
        own_feature_fc = self.fc2(own_feats_h)

        enemy_embedding = enemy_embedding1 * enemy_type_1.unsqueeze(-1).float() + enemy_embedding2 * enemy_type_2.unsqueeze(-1).float()


        attack_action = th.matmul(enemy_embedding, own_feature_fc.unsqueeze(-1)).squeeze(
            -1)  # [bz * n_agent, n_attack_action]

        return th.cat([self_action, attack_action], dim=-1), own_feats_h
