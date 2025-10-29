import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ..utils.transformer import EncoderLayer


class ANTAttentionAgent(nn.Module):
    """
    参数设置
    obs_agent_id: False

    """

    def __init__(self, input_shape, args):
        super(ANTAttentionAgent, self).__init__()
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
            self.n_own += args.agent_id_dim #self.n_ally + 1
        self.fc1 = nn.Linear(self.n_own, args.rnn_hidden_dim // 2)
        self.fc_selector = nn.Linear(args.rnn_hidden_dim // 2, self.model_dim)

        self.ally_transformer = EncoderLayer(in_dim=self.n_ally_feats // self.n_ally,
                                             model_dim=self.model_dim,
                                             num_heads=self.num_heads,
                                             ffn_dim=self.ffn_dim,
                                             dropout=self.dropout)
        self.enemy_transformer = EncoderLayer(in_dim=self.n_enemy_feats // self.n_enemy + 1,
                                              model_dim=self.model_dim,
                                              num_heads=self.num_heads,
                                              ffn_dim=self.ffn_dim,
                                              dropout=self.dropout)

        self.rnn = nn.GRUCell(args.rnn_hidden_dim // 2 + 4 * self.model_dim, args.rnn_hidden_dim)
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
                [own_input, inputs[:,
                            self.n_move_feats + self.n_enemy_feats + self.n_ally_feats + self.n_own_feats:self.n_move_feats + self.n_enemy_feats + self.n_ally_feats + self.n_own_feats + self.n_self_action]],
                -1)
        if self.args.obs_agent_id:
            agent_id_states = th.zeros(bs, self.args.agent_id_dim).cuda()
            agent_id_states[:, :self.n_ally+1] = inputs[:,
                            self.n_move_feats + self.n_enemy_feats + self.n_ally_feats + self.n_own_feats + self.n_self_action + self.n_enemy:]

            own_input = th.cat([own_input, agent_id_states], -1)

        enemy_input = inputs[:, self.n_move_feats:self.n_move_feats + self.n_enemy_feats].reshape(bs, self.n_enemy, -1)
        enemy_attack = inputs[:,
                       self.n_move_feats + self.n_enemy_feats + self.n_ally_feats + self.n_own_feats + self.n_self_action:self.n_move_feats + self.n_enemy_feats + self.n_ally_feats + self.n_own_feats + self.n_self_action + self.n_enemy].reshape(
            bs, self.n_enemy, -1)
        enemy_input = th.cat([enemy_input, enemy_attack], -1)

        ally_input = inputs[:,
                     self.n_move_feats + self.n_enemy_feats:self.n_move_feats + self.n_enemy_feats + self.n_ally_feats].reshape(
            bs, self.n_ally, -1)

        own_feats = F.relu(self.fc1(own_input))  # [bz * n_agent, n_own_feats]
        own_selector_feats = F.relu(self.fc_selector(own_feats)).unsqueeze(-1)  # [bz * n_agent, model_dim, 1]

        enemy_embedding, _ = self.enemy_transformer(enemy_input)  # [bz * n_agent, n_enemy, model_dim]
        ally_embedding, _ = self.ally_transformer(ally_input)  # [bz * n_agent, n_ally, model_dim]

        # attention selector
        enemy_attention = F.softmax(th.bmm(enemy_embedding, own_selector_feats).squeeze(-1), dim=-1)
        ally_attention = F.softmax(th.bmm(ally_embedding, own_selector_feats).squeeze(-1), dim=-1)

        enemy_attention_index = enemy_attention.argmax(-1)
        ally_attention_index = ally_attention.argmax(-1)
        enemy_embedding_select = enemy_embedding[[i for i in range(bs)], enemy_attention_index]
        ally_attention_select = ally_embedding[[i for i in range(bs)], ally_attention_index]

        # mean-field embedding
        enemy_embedding_mean = enemy_embedding.mean(dim=-2)  # [bz * n_agent, model_dim]
        ally_embedding_mean = ally_embedding.mean(dim=-2)  # [bz * n_agent, model_dim]

        rnn_input = th.cat(
            [own_feats, ally_embedding_mean, ally_attention_select, enemy_embedding_mean, enemy_embedding_select],
            dim=-1)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        own_feats_h = self.rnn(rnn_input, h_in)  # [bz * n_agent, rnn_hidden_dim]

        # self action
        self_action = self.self_action_fc(own_feats_h)  # [bz * n_agent, n_self_action]

        # attack action
        own_feature_fc = self.fc2(own_feats_h)
        attack_action = th.matmul(enemy_embedding, own_feature_fc.unsqueeze(-1)).squeeze(
            -1)  # [bz * n_agent, n_attack_action]

        return th.cat([self_action, attack_action], dim=-1), own_feats_h
