import torch as th
import torch.nn as nn
import torch.nn.functional as F


class ACCritic(nn.Module):
    # 说明: 独立智能体使用的集中价值网络, 依据个体观测估计 V 值
    def __init__(self, scheme, args):
        super(ACCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "v"

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)

    # 说明: 前向计算价值, 返回每个时间步的 V(s)
    def forward(self, batch, t=None):
        inputs, bs, max_t = self._build_inputs(batch, t=t)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

    # 说明: 拼接个体观测与身份信息
    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        # observations
        inputs.append(batch["obs"][:, ts])

        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        inputs = th.cat(inputs, dim=-1)
        return inputs, bs, max_t

    # 说明: 计算 critic 输入维度
    def _get_input_shape(self, scheme):
        # observations
        input_shape = scheme["obs"]["vshape"]
        # agent id
        input_shape += self.n_agents
        return input_shape
