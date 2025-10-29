import torch as th
import torch.nn as nn
import torch.nn.functional as F


class JointAC(nn.Module):
    def __init__(self, scheme, args):
        super(JointAC, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "v"

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)

    def forward(self, batch, t=None, joint_action_minus_i=None, type=None):
        inputs, bs, max_t, joint_action_minus_i = self._build_inputs(batch, t=t, joint_action_minus_i=joint_action_minus_i, type=type)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q, joint_action_minus_i

    def _build_inputs(self, batch, t=None, joint_action_minus_i=None, type=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []

        # observations
        # OLD
        inputs.append(batch["obs"][:, ts])

        # NEW
        # observations = th.unsqueeze(batch["obs"][:, ts], 0).repeat(self.args.n_samples, 1, 1, 1, 1)
        # inputs.append(observations)

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

        # Agent IDs
        # NEW
        # agent_ids = th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(self.args.n_samples, bs, max_t, -1, -1)
        # inputs.append(agent_ids)
        # OLD
        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        inputs = th.cat(inputs, dim=-1)

        return inputs, bs, max_t, joint_action_minus_i

    def _get_input_shape(self, scheme):
        # observations
        input_shape = scheme["obs"]["vshape"]

        # Make allowance for joint_action
        # num_actions = num_agents, so the below makes sense
        input_shape += self.n_agents - 1 #scheme["actions"]["vshape"][0]
        
        # agent id
        input_shape += self.n_agents
        
        return input_shape
