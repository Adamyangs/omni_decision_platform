import torch as th
import torch.nn as nn
import torch.nn.functional as F


class AsnRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(AsnRNNAgent, self).__init__()
        self.args = args

        self.n_ally = self.args.n_ally
        self.n_enemy = self.args.n_enemy

        # feature index
        self.n_move_feats = self.args.move_feats_dim
        self.n_enemy_feats = self.args.enemy_feats_dim // self.n_enemy
        self.n_ally_feats = self.args.ally_feats_dim // self.n_ally
        self.n_own_feats = self.args.own_feats_dim

        self.move_feat_end = self.n_move_feats

        self.blood_feat_start = self.n_move_feats + self.n_enemy_feats * self.n_enemy + self.n_ally_feats * self.n_ally
        # self.blood_feat_start = 5 + 5 * 8 + 5 * 8
        self.blood_feat_end = self.blood_feat_start + 1

        self.other_feat_start = self.n_move_feats + self.n_enemy_feats * self.n_enemy + self.n_ally_feats * self.n_ally + 1
        # self.other_feat_start = 5 + 5 * 8 + 5 * 8 + 1

        self.enemies_feat_start = self.n_move_feats

        self.agents_feat_start = self.n_move_feats + self.n_enemy_feats * self.n_enemy


        # network struct
        self.env_info_fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.env_info_fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.env_info_rnn3 = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

        # no-op + stop + up, down, left, right
        self.wo_action_fc = nn.Linear(args.rnn_hidden_dim, 6)

        self.enemies_info_fc1 = nn.Linear(self.n_enemy_feats, args.rnn_hidden_dim)
        self.enemies_info_fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.enemies_info_rnn3 = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.env_info_fc1.weight.new(1, self.args.rnn_hidden_dim * (1 + self.n_enemy)).zero_()

    def forward(self, inputs, hidden_state):
        # print(inputs.shape)
        # print(hidden_state.shape)

        enemies_feats = [inputs[:, self.enemies_feat_start + i * self.n_enemy_feats: self.enemies_feat_start + self.n_enemy_feats * (1 + i)]
                         for i in range(self.n_enemy)]

        # agents_feats = [inputs[:, self.agents_feat_start + i * 5: self.agents_feat_start + 5 * (1 + i)]
        #                 for i in range(self.args.agents_num - 1)]
        # self_input = th.cat([inputs[:, :self.move_feat_end],
        #                      inputs[:, self.blood_feat_start: self.blood_feat_end],
        #                      inputs[:, self.other_feat_start:]],
        #                     dim=1)

        h_in = th.split(hidden_state, self.args.rnn_hidden_dim, dim=-1)
        h_in_env = h_in[0].reshape(-1, self.args.rnn_hidden_dim)
        h_in_enemies = [_h.reshape(-1, self.args.rnn_hidden_dim) for _h in h_in[1:]]

        env_hidden_1 = F.relu(self.env_info_fc1(inputs))
        env_hidden_2 = self.env_info_fc2(env_hidden_1)
        h_env = self.env_info_rnn3(env_hidden_2, h_in_env)

        wo_action_fc_Q = self.wo_action_fc(h_env)

        enemies_hiddent_1 = [F.relu(self.enemies_info_fc1(enemy_info)) for enemy_info in enemies_feats]
        enemies_hiddent_2 = [self.enemies_info_fc2(enemy_info) for enemy_info in enemies_hiddent_1]
        enemies_h_hiddent_3 = [self.enemies_info_rnn3(enemy_info, enemy_h) for enemy_info, enemy_h in zip(enemies_hiddent_2, h_in_enemies)]

        attack_enemy_id_Q = [th.sum(h_env * enemy_info, dim=-1, keepdim=True) for enemy_info in enemies_h_hiddent_3]

        q = th.cat([wo_action_fc_Q, *attack_enemy_id_Q], dim=-1)
        hidden_state = th.cat([h_env, *enemies_h_hiddent_3], dim=-1)

        return q, hidden_state
