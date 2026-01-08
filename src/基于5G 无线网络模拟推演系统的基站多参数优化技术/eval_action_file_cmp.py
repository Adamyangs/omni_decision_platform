import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from ATSPModel import EncodingBlock, MultiGraphEncoderLayer
from ATSPModel_LIB import MultiHeadAttention, AddAndInstanceNormalization
from environment import BaseStationSleepingProblem
from mf_policy import CellControlMFAgent
from util import to_onehot


if __name__ == '__main__':
    device = 'cuda:2'
    gamma = 0.99
    w_entropy = 0.1
    n_input = 15
    reward_scale = 1.
    learning_rate = 0.0003
    buffer_size = 100
    batchsize = 512

    env = BaseStationSleepingProblem(
        network_power_5g_file='input/5G/Network_Power2022-05-26.npy',
        network_power_4g_file='input/4G/Network_Power2022-05-26.npy',
        unsatisfied_penalty_type='action_constraint',
    )
    agent = CellControlMFAgent(
        n_input=n_input,
        n_agent=env.n_cell,
        gamma=gamma,
        w_entropy=w_entropy,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        device=device)
    agent.load('sleep_controller.pt')

    bs_powers = {bid:[] for bid in env.bs_sids}
    bs_control = {bid: [] for bid in env.bs_sids}
    cell_control = {cid: [] for cid in env.cell_sids}
    original_total_powers, total_flows, total_powers = [], [], []
    for day in range(20, 27):
        env = BaseStationSleepingProblem(
            network_power_5g_file='5G/Network_Power2022-05-{}.npy'.format(day),
            network_power_4g_file='4G/Network_Power2022-05-{}.npy'.format(day),
            unsatisfied_penalty_type='action_constraint',
        )
        # env init
        state, info = env.init()
        done = False
        while not done:
            state = torch.from_numpy(state).float().to(device)  # [n_cell, n_in]
            action = agent.sample_action(state).long()  # [n_cell]
            action_onehot = F.one_hot(action, 3).cpu().numpy()  # [n_cell, 3]

            next_state, reward, done, info = env.step(action_onehot)

            state = next_state

            # bbu_powers = {bid: [] for bid in env.bs_sids}
            # rru_powers = {bid: [] for bid in env.bs_sids}
            # bs_control = {bid: [] for bid in env.bs_sids}
            # cell_control = {cid: [] for cid in env.cell_sids}
            # total_flow, total_powers = [], []
            original_total_power, total_power = 0., 0.
            for bid in info['debug_record']['bs_record']:
                if bid not in bs_powers:
                    print(bid)
                    continue
                bs_powers[bid].append([info['debug_record']['bs_record'][bid]['power_rru'],
                                       info['debug_record']['bs_record'][bid]['power_bbu']])
                total_power += info['debug_record']['bs_record'][bid]['power_bs']
                original_total_power += info['debug_record']['bs_record'][bid]['original_power_bs']
            total_powers.append(total_power)
            original_total_powers.append(original_total_power)
            for bid in info['bs_record'][-1]:
                bs_control[bid].append(info['bs_record'][-1][bid]['bs_control_state'])
            for cid in info['cell_record'][-1]:
                cell_control[cid].append(info['cell_record'][-1][cid]['cell_control_state'])
            total_flow = 0.
            for gid in info['debug_record']['grid_record']:
                total_flow += info['debug_record']['grid_record'][gid]['flow']
            total_flows.append(total_flow)

    np.save('output/0/bbu_rru_powers.npy', bs_powers)
    np.save('output/bs_control.npy', bs_control)
    np.save('output/cell_control.npy', cell_control)
    np.save('output/0/flow_powers.npy', {'total_flow': total_flows,
                                'total_powers': total_powers,
                                'original_total_powers': original_total_powers})
    pass

