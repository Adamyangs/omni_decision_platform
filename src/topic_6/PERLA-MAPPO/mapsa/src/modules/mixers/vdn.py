import torch as th
import torch.nn as nn


class VDNMixer(nn.Module):
    # 说明: VDN 混合器, 直接对所有智能体的 Q 值求和
    def __init__(self):
        super(VDNMixer, self).__init__()

    # 说明: 对第三维求和得到全局 Q_tot
    def forward(self, agent_qs, batch):
        return th.sum(agent_qs, dim=2, keepdim=True)
