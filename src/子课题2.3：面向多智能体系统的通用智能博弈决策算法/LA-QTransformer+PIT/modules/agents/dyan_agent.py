import torch as th
import torch.nn as nn
import torch.nn.functional as F


class DyAN(nn.Module):
    def __init__(self, input_shape, args):
        super(DyAN, self).__init__()
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
            self.n_own += self.n_self_action + self.n_enemy
        if self.args.obs_agent_id:
            self.n_own += self.n_ally + 1

        
        self.self_fc = nn.Linear(self.n_own, self.args.hidden_feats)
        self.ally_fc = nn.Linear(self.n_ally_feats // self.n_ally, self.args.hidden_feats)
        self.enemy_fc = nn.Linear(self.n_enemy_feats // self.n_enemy, self.args.hidden_feats)
        
        self.final_fc = nn.Linear(self.args.hidden_feats * 3, self.args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.action_fc = nn.Linear(args.rnn_hidden_dim, self.n_self_action + self.n_enemy)

    def init_hidden(self):
        return self.final_fc.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        bs = inputs.shape[0]

        own_input = th.cat([inputs[:, :self.n_move_feats],
                           inputs[:, self.n_move_feats + self.n_enemy_feats + self.n_ally_feats: self.n_move_feats + self.n_enemy_feats + self.n_ally_feats + self.n_own_feats]], -1)

        if self.args.obs_last_action:
            own_input = th.cat(
                [own_input, inputs[:, self.n_move_feats + self.n_enemy_feats + self.n_ally_feats + self.n_own_feats:]],
                 -1)

        enemy_input = inputs[:, self.n_move_feats:self.n_move_feats + self.n_enemy_feats].reshape(bs, self.n_enemy, -1)
        
        ally_input = inputs[:,
                     self.n_move_feats + self.n_enemy_feats:self.n_move_feats + self.n_enemy_feats + self.n_ally_feats].reshape(
            bs, self.n_ally, -1)

        own_input = self.self_fc(own_input)
        ally_input = self.ally_fc(ally_input).sum(-2)
        enemy_input = self.enemy_fc(enemy_input).sum(-2)

        final_input = th.cat([own_input, ally_input, enemy_input], dim=-1)
        final_feats = F.relu(self.final_fc(final_input))

        hidden_state = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(final_feats, hidden_state)

        action = self.action_fc(h)

        return action, h






