import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from ATSPModel import EncodingBlock, MultiGraphEncoderLayer
from ATSPModel_LIB import MultiHeadAttention, AddAndInstanceNormalization
from environment import BaseStationSleepingProblem
from mf_policy import CellControlMFAgent, GraphTool, get_sparse_graph
from util import to_onehot
from global_parameters import agent_path

if __name__ == '__main__':
    _t = time.time()
    device = 'cuda:1'
    single_step = True
    auto_cell_off_control = True
    gamma = 0.99 if not single_step else 0.
    w_entropy = None
    n_input = 15 if not single_step else 10
    reward_scale = 1.
    learning_rate = 0.0003
    buffer_size = 100
    batchsize = 512
    n_action = 3
    n_selected_action = 2 if auto_cell_off_control else 3
    open_all = False
    assert not open_all

    env = BaseStationSleepingProblem(
        network_power_5g_file='input/5G/Network_Power2022-05-26.npy',
        network_power_4g_file='input/4G/Network_Power2022-05-26.npy',
        unsatisfied_penalty_type='action_constraint',
        auto_cell_off_control=auto_cell_off_control,
        single_step=single_step
    )
    graph = get_sparse_graph([env.cell_bs_groups, env.cell_grid_groups],
                             [env.cell_in_bs_id, env.cell_in_grid_id],
                             env.n_cell, device)
    gtool = GraphTool(env.n_cell, graph)
    agent = CellControlMFAgent(
        n_input=n_input,
        n_output=n_selected_action,
        n_agent=env.n_cell,
        gamma=gamma,
        w_entropy=w_entropy,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        graph=graph,
        device=device)
    agent.load('output/{}/sleep_controller.pt'.format(agent_path))

    whatif_conditions = ['Max0.4', 'Max0.5', 'Max0.6', 'Max0.7', 'Max0.8', 'Max0.9',]
    # whatif_conditions = ['Min-max0.4', 'Min-max0.5', 'Min-max0.6', 'Min-max0.7', 'Min-max0.8', 'Min-max0.9', ]
    # whatif_conditions = ['Min-max0.8', 'Min-max0.9']

    for whatif in whatif_conditions:
        bs_powers = {bid:[] for bid in env.bs_sids}
        bs_control = {bid: [] for bid in env.bs_sids}
        cell_control = {cid: [] for cid in env.cell_sids}
        cell_traffic = {cid: [] for cid in env.cell_sids}
        original_total_powers, total_flows, total_powers = [], [], []
        for day in range(20, 27):
            env = BaseStationSleepingProblem(
                network_power_5g_file='input/5G/Network_Power2022-05-{}.npy'.format(day),
                network_power_4g_file='input/4G/Network_Power2022-05-{}.npy'.format(day),
                what_if_file='input/Whatif_Traffic/Whatif_Traffic{}.npy'.format(whatif),
                unsatisfied_penalty_type='action_constraint',
                auto_cell_off_control=auto_cell_off_control,
                single_step=single_step
            )
            # env init
            state, info = env.init()
            done = False
            while not done:
                state = torch.from_numpy(state).float().to(device)  # [n_cell, n_in]
                action = agent.sample_action(state).long()  # [n_cell]
                if open_all:
                    action = action * 0
                action_onehot = F.one_hot(action, 3).cpu().numpy()  # [n_cell, 3]

                next_state, reward, done, info = env.step(action_onehot)
                print('{} {} {} {}'.format(whatif, day, env.timestep, time.time()-_t))
                _t = time.time()

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
                    cell_traffic[cid].append(info['cell_record'][-1][cid]['traffic'])
                total_flow = 0.
                for gid in info['debug_record']['grid_record']:
                    total_flow += info['debug_record']['grid_record'][gid]['flow']
                total_flows.append(total_flow)

        np.save('output/{}/bbu_rru_powers_{}{}.npy'.format(agent_path, whatif, '_oa' if open_all else ''), bs_powers)
        np.save('output/{}/bs_control_{}{}.npy'.format(agent_path, whatif, '_oa' if open_all else ''), bs_control)
        np.save('output/{}/cell_control_{}{}.npy'.format(agent_path, whatif, '_oa' if open_all else ''), cell_control)
        np.save('output/{}/cell_traffic_{}{}.npy'.format(agent_path, whatif, '_oa' if open_all else ''), cell_traffic)
        np.save('output/{}/flow_powers_{}{}.npy'.format(agent_path, whatif, '_oa' if open_all else ''),
                {'total_flow': total_flows,
                 'total_powers': total_powers,
                 'original_total_powers': original_total_powers})
    pass

