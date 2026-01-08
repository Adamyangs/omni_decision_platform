import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from environment import BaseStationSleepingProblem
from util import to_onehot


if __name__ == '__main__':

    # load and merge dataset
    network_5g_info = np.load('input/5G/Network_Power2022-05-20.npy', allow_pickle=True)[()]
    network_4g_info = np.load('input/4G/Network_Power2022-05-20.npy', allow_pickle=True)[()]
    # cell_sids
    cell_sids = [cell_sid for bs_sid in network_5g_info[0] for cell_sid in network_5g_info[0][bs_sid]]\
                + [cell_sid for bs_sid in network_4g_info[0] for cell_sid in network_4g_info[0][bs_sid]]
    cell_sids_to_idx = {sid:i for i, sid in enumerate(cell_sids)}
    n_cell = len(cell_sids)

    onoffs = np.load('output/data_onoff_PeAs.npy', allow_pickle=True)[()]
    actions = []
    for day in range(7):
        for t_hh in range(48):
            action = np.zeros([n_cell], dtype='int32')
            grid_onoffs = onoffs[day][t_hh]
            for grid_id in grid_onoffs:
                if not isinstance(grid_id, int):
                    break
                sid_onoffs = grid_onoffs[grid_id]
                for sid in sid_onoffs:
                    # print(day,t_hh,grid_id,sid)
                    action[cell_sids_to_idx[sid]] = sid_onoffs[sid]['Status']
            actions.append([action])
    actions = np.concatenate(actions)
    actions = (1-actions) * 2

    powers = []
    for day in [20]:
        env = BaseStationSleepingProblem(
            network_power_5g_file='5G/Network_Power2022-05-{}.npy'.format(day),
            network_power_4g_file='4G/Network_Power2022-05-{}.npy'.format(day),
            unsatisfied_penalty_type='action_constraint',
        )
        # env init
        state, info = env.init()
        done = False
        step = 0
        while not done:
            action = actions[step]
            action_onehot = to_onehot(action, 3)  # [n_cell, 3]

            next_state, reward, done, info = env.step(action_onehot)
            print(day, env.timestep)

            state = next_state
            step += 1
        # powers.append(info['bs_record'])
        print(info['ep_record'])
    pass
