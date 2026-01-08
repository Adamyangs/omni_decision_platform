import torch as th
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    # 说明: 简单三层感知机, 作为 critic 子网络复用
    def __init__(self, input_shape, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    # 说明: 标准前向传播, 输出单个标量或向量
    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q
