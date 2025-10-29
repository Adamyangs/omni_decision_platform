import torch as th
import torch.nn as nn
import torch.nn.functional as F


class JointCentralV(nn.Module):
    def __init__(self, scheme, args):
        super(JointCentralV, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "v"

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)

    def forward(self, batch, t=None, joint_action_minus_i=None):
        inputs, bs, max_t, joint_actions_minus_i = self._build_inputs(batch, t=t, joint_action_minus_i=joint_action_minus_i)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q, joint_actions_minus_i

    def _build_inputs(self, batch, t=None, actions=None, joint_action_minus_i=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []

        # state
        inputs.append(batch["state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # observations
        if self.args.obs_individual_obs:
            inputs.append(batch["obs"][:, ts].view(bs, max_t, -1).unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # last actions
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1))
            elif isinstance(t, int):
                inputs.append(batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1))
            else:
                last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
                last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
                inputs.append(last_actions)
       
        # \vec{a}^{-i}
        # if isinstance(joint_action_minus_i, tuple):
        #     joint_action_minus_i = th.tensor(joint_action_minus_i, device=batch.device)
        #     joint_action_minus_i = joint_action_minus_i[None, None, None, ...]
        #     repeat_shape = list(batch["obs"].shape)
        #     repeat_shape[-1] = 1
        #     joint_action_minus_i = joint_action_minus_i.repeat(repeat_shape)
        #     inputs.append(joint_action_minus_i[:, ts])
        # else:
        inputs.append(joint_action_minus_i[:, ts])

        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        inputs = th.cat(inputs, dim=-1)
        return inputs, bs, max_t, joint_action_minus_i

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["state"]["vshape"]
        # observations
        if self.args.obs_individual_obs:
            input_shape += scheme["obs"]["vshape"] * self.n_agents
        # last actions
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
        
        # Make allowance for joint_action
        # num_actions = num_agents, so the below makes sense
        input_shape += self.n_agents - 1 #scheme["actions"]["vshape"][0]
        
        input_shape += self.n_agents
        
        return input_shape