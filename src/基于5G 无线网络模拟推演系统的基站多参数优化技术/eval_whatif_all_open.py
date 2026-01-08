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
from global_parameters import agent_path

if __name__ == '__main__':
    _t = time.time()
    device = 'cuda:3'
    open_all = True

    env = BaseStationSleepingProblem(
        network_power_5g_file='input/5G/Network_Power2022-05-26.npy',
        network_power_4g_file='input/4G/Network_Power2022-05-26.npy',
        unsatisfied_penalty_type='action_constraint',
    )

    whatif_conditions = ['Max0.4', 'Max0.5', 'Max0.6', 'Max0.7', 'Max0.8', 'Max0.9',
                         'Min-max0.4', 'Min-max0.5', 'Min-max0.6', 'Min-max0.7', 'Min-max0.8', 'Min-max0.9', ]

    # power_bbu = {k:[] for k in env.bs_sids}
    # for d in range(20, 27):
    #     network_5g_info = np.load('input/5G/Network_Power2022-05-{}.npy'.format(d), allow_pickle=True)[()]
    #     network_4g_info = np.load('input/4G/Network_Power2022-05-{}.npy'.format(d), allow_pickle=True)[()]
    #     network_info = [{**network_5g_info[k1], **network_4g_info[k2]} for k1, k2 in zip(network_5g_info, network_4g_info)]
    #     for k in env.bs_sids:
    #         power_bbu[k] += [next(iter(network_info[t][k].values()))['Power_BBU'] for t in range(48)]
    # np.save('output/eval_whatif_all_open_power_bbu.npy', power_bbu)
    power_bbu = np.load('output/eval_whatif_all_open_power_bbu.npy', allow_pickle=True)[()]


    for whatif in whatif_conditions:
        what_if_flow = np.load('input/Whatif_Traffic/Whatif_Traffic{}.npy'.format(whatif), allow_pickle=True)[()]
        what_if_flow = {t + (d-20) * 48:
                            {**{cell_sid: what_if_flow['2022-05-{}'.format(d)]['4G'][t][bs_sid][cell_sid]['down_flow']
                                for bs_sid in what_if_flow['2022-05-{}'.format(d)]['4G'][t]
                                for cell_sid in what_if_flow['2022-05-{}'.format(d)]['4G'][t][bs_sid]},
                             **{cell_sid: what_if_flow['2022-05-{}'.format(d)]['5G'][t][bs_sid][cell_sid]['down_flow']
                                for bs_sid in what_if_flow['2022-05-{}'.format(d)]['5G'][t]
                                for cell_sid in what_if_flow['2022-05-{}'.format(d)]['5G'][t][bs_sid]}}
                         for d in range(20, 27) for t in range(48)}

        power_rru = {cell_sid: [what_if_flow[t][cell_sid] * env.cell_info[cell_sid][-2] + env.cell_info[cell_sid][-1]
                                for t in range(7*48)]
                     for cell_sid in env.cell_sids}
        power_rru = {bs_sid: [sum([power_rru[env.cell_sids[cell_id]][t] for cell_id in env.cell_bs_groups[env.bs_sids_to_id[bs_sid]]])
                              for t in range(7*48)]
                     for bs_sid in env.bs_sids}

        infos = {
            'rru': power_rru,
            'bbu': power_bbu,
            'air': env.new_air_info
        }
        np.save('output/allopen_infos_{}.npy'.format(whatif), infos)

    pass

