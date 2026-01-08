import time

from pyproj import Geod
import numpy as np
from scipy import stats
import torch
import torch.nn.functional as F
import math
import json
import os
import scipy
import itertools
from util import load_from_list_of_dict, cal_switch_cost, to_onehot
# from Cell_RRUPower import Cell_Power, cell_power_debug

MINI_FILE = False

class BaseStationSleepingProblem:
    def __init__(self,
                 network_power_5g_file:str='input/5G/Network_Power2022-05-20_26.npy',    # '5G_Network_Power_mini.npy' '5G/Network_Power2022-05-20.npy'
                 network_power_4g_file:str='input/4G/Network_Power2022-05-20_26.npy',    # '4G_Network_Power_mini.npy' '4G/Network_Power2022-05-20.npy'
                 basestation_5g_file:str='input/Cell_info_5G.npy',          # 'Cell_info_5G_mini.npy' 'Cell_info_5G.npy'
                 basestation_4g_file:str='input/Cell_info_4G.npy',           # 'Cell_4G_LTE_mini.npy' 'Cell_4G_LTE.npy'
                 basestation_5g_new_air_file: str = 'input/data_as_5G.npy',  # '' 'data_as_5G.npy'
                 basestation_4g_new_air_file: str = 'input/data_as_4G.npy',  # '' 'data_as_4G.npy'
                 cell_capacity_5g_file:str='input/Cell_Capacity_5G.npy',    # 'Cell_Capacity_5G_mini.npy' 'Cell_Capacity_5G.npy'
                 cell_capacity_4g_file:str='input/Cell_Capacity_4G.npy',    # 'Cell_Capacity_4G_mini.npy' 'Cell_Capacity_4G.npy'
                 what_if_file:str='',                                   # '' 'Whatif_Traffic/Whatif_TrafficMax0.4.npy'
                 cell_power_coef_file:str='input/cell_rru_power.npy',       # 'cell_rru_power_mini.npy' 'cell_rru_power.npy'
                 cell_sleep_power_file:str='input/sleep_power.npy',               # 'sleep_power_typical.npy' 'sleep_power.npy'
                 cell_grid_file:str='input/Grid_45G_300m.npy',                                 # 'Grid_Cell_mini.npy', 'Grid_45G_300m.npy'
                 unsatisfied_penalty_type:str='action_constraint',                   # {'single_step_penalty', 'action_constraint'}
                 auto_cell_off_control:bool=True,
                 single_step:bool=True,
                 ):
        """
        :param network_power_5g_file:   {'input/5G/Network_Power2022-05-2i.npy' | 0<=i<=6]}
        :param network_power_4g_file:   {'input/4G/Network_Power2022-05-2i.npy' | 0<=i<=6]}
        :param basestation_5g_file:     'input/Cell_info_5G.npy'
        :param basestation_4g_file:     'input/Cell_4G_LTE.npy'
        :param basestation_5g_new_air_file: 'input/data_as_5G.npy'
        :param basestation_4g_new_air_file: 'input/data_as_4G.npy'
        :param cell_capacity_5g_file    'input/Cell_Capacity_5G.npy'
        :param cell_capacity_4g_file    'input/Cell_Capacity_4G.npy'
        :param what_if_file             ''
        :param cell_power_coef_file     'input/cell_rru_power.npy'
        :param cell_sleep_power_file    'input/sleep_power.npy'
        :param cell_grid_file           'input/Grid_45G_300m.npy'
        :param n_cell:                  1<=n_bs<=8562
        :param unsatisfied_penalty_type: {'single_step_penalty', 'action_constraint'}
        """
        # hyper-parameter
        self.history_queue_length = 4
        self.scale_down_flow = 1.e7
        self.scale_user_cnt = 1.e2
        self.scale_power = 1.e3
        self.scale_courious_reward = 0.003
        self.scale_reward = 1.e3
        self.cell_loc_min = np.array([[115.4, 28.1]])
        self.cell_loc_max = np.array([[116.6, 29.1]])
        self.scale_cell_power_coef = np.array([1e-4, 1.e3])

        self.bs_switch_cost = np.array([[0., 10.],
                                        [10., 0.]])                     # {[1,0]: on, [0,1]: off}
        self.cell_switch_cost = np.array([[0., 1., 5.],
                                          [1., 0., 5.],
                                          [5., 5., 0.]])                # {[1,0,0]: on, [0,1,0]: sleep, [0,0,1]: off}

        self.reward_type = 'independent'                                # {'independent'}
        assert self.reward_type == 'independent'
        self.unsatisfied_penalty_type = unsatisfied_penalty_type
        self.unsatisfied_penalty = 0
        self.sleep_power_4g = 17
        self.sleep_power_5g = 80

        # input parameter
        self.auto_cell_off_control = auto_cell_off_control
        self.single_step = single_step

        # input file
        if MINI_FILE:
            network_power_5g_file = 'input/5G/Network_Power2022-05-20_typical.npy'
            network_power_4g_file = 'input/4G/Network_Power2022-05-20_typical.npy'
            basestation_5g_file = 'input/Cell_info_5G_typical.npy'
            basestation_4g_file = 'input/Cell_info_4G_typical.npy'
            basestation_5g_new_air_file = 'input/data_as_5G_typical.npy'
            basestation_4g_new_air_file = 'input/data_as_4G_typical.npy'
            cell_capacity_5g_file = 'input/Cell_Capacity_5G_typical.npy'
            cell_capacity_4g_file = 'input/Cell_Capacity_4G_typical.npy'
            what_if_file = ''
            cell_power_coef_file = 'input/cell_rru_power_typical.npy'
            cell_sleep_power_file = 'input/sleep_power_typical.npy'
            cell_grid_file = 'input/Grid_45G_300m_typical.npy'
        self.network_power_5g_file = network_power_5g_file
        self.network_power_4g_file = network_power_4g_file
        self.basestation_5g_file = basestation_5g_file
        self.basestation_4g_file = basestation_4g_file
        self.basestation_5g_new_air_file = basestation_5g_new_air_file
        self.basestation_4g_new_air_file = basestation_4g_new_air_file
        self.cell_capacity_5g_file = cell_capacity_5g_file
        self.cell_capacity_4g_file = cell_capacity_4g_file
        self.what_if_file = what_if_file
        self.cell_power_coef_file = cell_power_coef_file
        self.cell_sleep_power_file = cell_sleep_power_file
        self.cell_grid_file = cell_grid_file

        self.bs_sids = None
        self.cell_sids = None
        self.n_bs = None
        self.n_cell = None
        self.n_grid = None
        self.horizon = None
        self.cell_loc = None
        self.cell_capacity = None
        self.cell_sleep_power = None
        self.cell_power_coef = None
        self.down_flow = None
        self.user_cnt = None
        self.power_transmit_cell = None
        self.power_rru = None
        self.power_bbu = None
        self.power_airconditioner = None
        self.cell_bs_mask = None
        self.cell_grid_mask = None
        self.cell_bs_groups = None
        self.cell_grid_groups = None
        self.cell_in_bs_id = None
        self.cell_in_grid_id = None
        self.grid_flow = None
        self.cell_grid_flow = None
        self.what_if_flow = None
        self._load_network_power_file()
        self._get_array_infos()

        # environment state
        self.timestep = None
        self.last_down_flow = None
        self.last_user_cnt = None
        self.prev_cell_control_state = None
        self.cell_control_state = None                                  # {[1,0,0]: on, [0,1,0]: sleep, [0,0,1]: off}
        self.cell_on_id, self.cell_sleep_id, self.cell_off_id = 0, 1, 2
        self.prev_bs_control_state = None
        self.bs_control_state = None                                    # {[1,0]: on, [0,1]: off}
        self.bs_on_id, self.bs_off_id = 0, 1
        self.cur_down_flow = None
        self.unsatisfied_down_flow_grid = None

        # record_info
        self.cell_record = []                                            # [{cell_id:{'cell_control_state':cell_control_state,
                                                                         #            'traffic': traffic}}]
        self.bs_record = []
        self.grid_record = []
        self.total_record = []
        self.ep_record = {}
        self.reward_related_record = {}
        self.debug_record = {}

    def _load_network_power_file(self):
        # load and merge dataset
        network_5g_info = np.load(self.network_power_5g_file, allow_pickle=True)[()]
        if self.basestation_5g_new_air_file != '':
            new_air_5g_info = np.load(self.basestation_5g_new_air_file, allow_pickle=True)[()]
            new_air_4g_info = np.load(self.basestation_4g_new_air_file, allow_pickle=True)[()]
            self.new_air_info = new_air_5g_info
            self.new_air_info.update(new_air_4g_info)
        else:
            self.new_air_info = None
        if self.network_power_5g_file[-6:-4].isdigit():
            self.day = int(self.network_power_5g_file[-6:-4]) - 20
        else:
            self.day = 0
        if self.what_if_file != '':
            self.what_if_flow = np.load(self.what_if_file, allow_pickle=True)[()]
            self.what_if_flow = self.what_if_flow['2022-05-{}'.format(20 + self.day)]
            self.what_if_flow = {t: {**{cell_sid:self.what_if_flow['4G'][t][bs_sid][cell_sid]
                                        for bs_sid in self.what_if_flow['4G'][t]
                                        for cell_sid in self.what_if_flow['4G'][t][bs_sid]},
                                     **{cell_sid: self.what_if_flow['5G'][t][bs_sid][cell_sid]
                                        for bs_sid in self.what_if_flow['5G'][t]
                                        for cell_sid in self.what_if_flow['5G'][t][bs_sid]}}
                                 for t in range(48)}
        basestation_5g_info = np.load(self.basestation_5g_file, allow_pickle=True)[()]
        cell_capacity_5g = np.load(self.cell_capacity_5g_file, allow_pickle=True)[()]
        network_4g_info = np.load(self.network_power_4g_file, allow_pickle=True)[()]
        basestation_4g_info = np.load(self.basestation_4g_file, allow_pickle=True)[()]
        cell_capacity_4g = np.load(self.cell_capacity_4g_file, allow_pickle=True)[()]
        cell_power_coef = np.load(self.cell_power_coef_file, allow_pickle=True)[()]
        cell_sleep_power = np.load(self.cell_sleep_power_file, allow_pickle=True)[()]
        self.network_info = [{**network_5g_info[k1], **network_4g_info[k2]} for k1, k2 in zip(network_5g_info, network_4g_info)]

        # bs_sids
        self.bs_sids = [bs_sid for bs_sid in network_5g_info[0]] + [bs_sid for bs_sid in network_4g_info[0]]
        self.n_bs = len(self.bs_sids)
        print("Number of Base Stations:", self.n_bs)

        # cell_sids
        self.cell_sids = [cell_sid for bs_sid in network_5g_info[0] for cell_sid in network_5g_info[0][bs_sid]] \
                         + [cell_sid for bs_sid in network_4g_info[0] for cell_sid in network_4g_info[0][bs_sid]]
        self.n_cell = len(self.cell_sids)
        print("Number of Cells:", self.n_cell)

        # cell_to_bs and cell_info
        self.cell_to_bs = {}
        self.cell_info = {}
        for bs_id in network_5g_info[0]:
            for cell_id in network_5g_info[0][bs_id]:
                self.cell_info[cell_id] = basestation_5g_info[cell_id][11:13] \
                                          + [cell_capacity_5g[cell_id]['Capacity'], cell_sleep_power[cell_id]] \
                                          + [*cell_power_coef[cell_id]] \
                                          + [cell_capacity_5g[cell_id]['Capacity']]
                assert len(self.cell_info[cell_id]) == 7
                self.cell_to_bs[cell_id] = bs_id
        for bs_id in network_4g_info[0]:
            for cell_id in network_4g_info[0][bs_id]:
                self.cell_info[cell_id] = list(basestation_4g_info[cell_id][6:8]) \
                                          + [cell_capacity_4g[cell_id]['Capacity'], cell_sleep_power[cell_id]] \
                                          + [*cell_power_coef[cell_id]] \
                                          + [cell_capacity_4g[cell_id]['Capacity']]
                assert len(self.cell_info[cell_id]) == 7
                self.cell_to_bs[cell_id] = bs_id
        self.horizon = len(self.network_info)

        # cell_to_grid
        cell_grid = np.load(self.cell_grid_file, allow_pickle=True)[()]
        self.cell_grid = list(cell_grid.values())
        self.n_grid = len(self.cell_grid)
        self.cell_sid_to_grid = {cell_sid:i for i in range(len(self.cell_grid)) for cell_sid in self.cell_grid[i]}

    def _get_array_infos(self):
        # string id to index id
        self.bs_sids_to_id = {sid:i for i, sid in enumerate(self.bs_sids)}

        # cell_info
        self.cell_loc = np.array([self.cell_info[i][:2] for i in self.cell_sids]).astype('float32')        # [n_cell, 2]
        self.cell_capacity = np.array([self.cell_info[i][2:3] for i in self.cell_sids]).astype('float32')  # [n_cell, 1]
        self.cell_priority = np.array([self.cell_info[i][6:7] for i in self.cell_sids]).astype('float32')  # [n_cell, 1]
        self.cell_sleep_power = np.array([self.cell_info[i][3:4] for i in self.cell_sids]).astype('float32')
                                                                                                          # [n_cell, 1]
        self.cell_power_coef = np.array([self.cell_info[i][4:6] for i in self.cell_sids]).astype('float32')# [n_cell, 2]

        # group_mask
        self.cell_bs_mask = np.zeros([self.n_cell, self.n_bs], dtype='bool')
        self.cell_grid_mask = np.zeros([self.n_cell, self.n_grid], dtype='bool')
        self.cell_bs_groups = {i:[] for i in range(self.n_bs)}
        self.cell_grid_groups = {i:[] for i in range(self.n_grid)}
        self.cell_in_bs_id = {i:None for i in range(self.n_cell)}
        self.cell_in_grid_id = {i:None for i in range(self.n_cell)}
        for i in range(self.n_cell):
            cell_sid = self.cell_sids[i]
            self.cell_bs_mask[i, self.bs_sids_to_id[self.cell_to_bs[cell_sid]]] = 1
            self.cell_grid_mask[i, self.cell_sid_to_grid[cell_sid]] = 1
            self.cell_bs_groups[self.bs_sids_to_id[self.cell_to_bs[cell_sid]]].append(i)
            self.cell_grid_groups[self.cell_sid_to_grid[cell_sid]].append(i)
            self.cell_in_bs_id[i] = self.bs_sids_to_id[self.cell_to_bs[cell_sid]]
            self.cell_in_grid_id[i] = self.cell_sid_to_grid[cell_sid]

        # flow and power
        self.down_flow = np.zeros([self.horizon, self.n_cell])
        self.user_cnt = np.zeros([self.horizon, self.n_cell])
        self.power_transmit_cell = np.zeros([self.horizon, self.n_cell])
        self.power_rru = np.zeros([self.horizon, self.n_cell])
        for t in range(self.horizon):
            temp = [self.network_info[t][self.cell_to_bs[cell_sid]][cell_sid] for cell_sid in self.cell_sids]
            items = load_from_list_of_dict(temp, ['down_flow', 'user_cnt', 'Power_Transmit_Cell', 'Power_RRU'])
            for var, key in zip([self.down_flow, self.user_cnt, self.power_transmit_cell, self.power_rru],
                                ['down_flow', 'user_cnt', 'Power_Transmit_Cell', 'Power_RRU']):
                var[t] = np.array(items[key])
        if self.what_if_flow is not None:
            self.down_flow = np.array([[self.what_if_flow[t][cell_sid]['down_flow'] if cell_sid in self.what_if_flow[t] else 0.
                                        for cell_sid in self.cell_sids]
                                       for t in self.what_if_flow])

        self.power_bbu = np.zeros([self.horizon, self.n_bs])
        self.power_airconditioner = np.zeros([self.horizon, self.n_bs])
        for t in range(self.horizon):
            temp = [next(iter(self.network_info[t][bs_sid].values())) for bs_sid in self.bs_sids]
            items = load_from_list_of_dict(temp, ['Power_BBU', 'Power_airconditioner'])
            for var, key in zip([self.power_bbu, self.power_airconditioner],
                                ['Power_BBU', 'Power_airconditioner']):
                var[t] = np.array(items[key])
        if self.new_air_info:
            temp = np.array([self.new_air_info[bs_sid][self.day*48:(1+self.day)*48] for bs_sid in self.bs_sids])
                                                                                                    # [horizon, n_bs]
            self.power_airconditioner = np.transpose(temp, axes=[1, 0])                             # [horizon, n_bs]

        # air_power_coef
        temp = np.concatenate([self.power_rru[:, self.cell_bs_groups[bid]].sum(-1, keepdims=True)
                               for bid in range(self.n_bs)], axis=-1) + self.power_bbu
        self.air_power_coef = np.zeros([48, 2])
        for i in range(48):
            res = stats.linregress(temp[i], self.power_airconditioner[i])
            self.air_power_coef[i, :] = [res.slope, res.intercept]

        # grid_flow
        self.grid_flow = np.matmul(self.down_flow, self.cell_grid_mask)                             # [horizon, n_grid]
        self.cell_grid_flow = np.matmul(self.grid_flow, np.transpose(self.cell_grid_mask, axes=[1, 0]))
                                                                                                    # [horizon, n_cell]

        # num cotrol status
        self.n_cell_control_status = np.zeros([48, self.n_cell, 3])
        self.n_basestation_control_status = np.zeros([48, self.n_bs, 2])

    def _cal_state(self):
        cell_bs_control_state = self.bs_control_state[[self.cell_in_bs_id[i] for i in range(self.n_cell)]]
                                                                                                    # [n_cell, 2]
        if self.single_step:
            state = {
                'cell_loc': (self.cell_loc - self.cell_loc_min) / (self.cell_loc_max - self.cell_loc_min),
                                                                                                        # [n_cell, 2]
                'cell_capacity': self.cell_capacity / self.scale_down_flow,                             # [n_cell, 1]
                'timestep': np.ones([self.n_cell, 1]) * self.timestep / self.horizon,                   # [n_cell, 1]
                'last_down_flow': self.last_down_flow / self.scale_down_flow,                           # [n_cell, hq_len]
                'cell_power_coef': self.cell_power_coef / self.scale_cell_power_coef,                   # [n_cell, 2]
            }
        else:
            state = {
                'cell_loc': (self.cell_loc - self.cell_loc_min) / (self.cell_loc_max - self.cell_loc_min),
                                                                                                        # [n_cell, 2]
                'cell_capacity': self.cell_capacity / self.scale_down_flow,                             # [n_cell, 1]
                # 'group_mask': self.group_mask,                                                          # [n_cell, n_bs]
                'timestep': np.ones([self.n_cell, 1]) * self.timestep / self.horizon,                   # [n_cell, 1]
                'last_down_flow': self.last_down_flow / self.scale_down_flow,                           # [n_cell, hq_len]
                # 'last_user_cnt': self.last_user_cnt / self.scale_user_cnt,                              # [n_cell, hq_len]
                'bs_control_state': cell_bs_control_state,                                              # [n_cell, 2]
                'cell_control_state': self.cell_control_state,                                          # [n_cell, 3]
                'cell_power_coef': self.cell_power_coef / self.scale_cell_power_coef,                   # [n_cell, 2]
            }
        return state

    def can_turn_off_bs(self, cell_control_state, idx):
        """

        :param cell_control_state:          [n_batch, n_cell, 3]
        :param idx:                         [n_batch]
        :return:
        """
        mask = np.zeros([idx.shape[0]], dtype='float32')
        for n, i in enumerate(idx):
            # org_cell_state = cell_control_state[n, i, -1]
            # cell_control_state[n, i, -1] = 1
            # cell_ids = self.cell_bs_groups[self.cell_in_bs_id[i]]                 # [n_cell_in_bs]
            # mask[n] = cell_control_state[n, cell_ids, -1].sum() == len(cell_ids)
            # cell_control_state[n, i, -1] = org_cell_state

            cell_ids = self.cell_bs_groups[self.cell_in_bs_id[i]]  # [n_cell_in_bs]
            if self.auto_cell_off_control:
                mask[n] = cell_control_state[n, cell_ids, 1].sum() - cell_control_state[n, i, 1] == len(cell_ids) - 1
            else:
                mask[n] = cell_control_state[n, cell_ids, -1].sum() - cell_control_state[n, i, -1] == len(cell_ids) - 1
        return mask

    def cell_grid_total_capacity(self, cell_control_state, idx):

        """

        :param cell_control_state:          [n_batch, n_cell, 3]
        :param idx:                         [n_batch], selected_cell_id
        :return:
        """
        capacity = np.zeros([idx.shape[0]], dtype='float32')
        for n, i in enumerate(idx):
            # org_cell_state = cell_control_state[n, i, 0]
            # cell_control_state[n, i, 0] = 0
            # cell_ids = self.cell_grid_groups[self.cell_in_grid_id[i]]  # [n_cell_in_bs]
            # capacity[n] = (cell_control_state[n, cell_ids, 0] * self.cell_capacity[cell_ids, 0]).sum() / self.scale_down_flow
            # cell_control_state[n, i, 0] = org_cell_state

            cell_ids = self.cell_grid_groups[self.cell_in_grid_id[i]]  # [n_cell_in_bs]
            capacity[n] = ((cell_control_state[n, cell_ids, 0] * self.cell_capacity[cell_ids, 0]).sum()
                           - cell_control_state[n, i, 0] * self.cell_capacity[i, 0]) / self.scale_down_flow
        return capacity

    def _update_bs_control_state(self):
        if self.auto_cell_off_control:
            bs_need_open = np.array([self.cell_control_state[self.cell_bs_groups[i], 0].sum() > 0
                                     for i in range(self.n_bs)]).astype('float32')[:, None]         # [n_bs, 1]
            cell_can_off = np.array([self.cell_control_state[i, 1] * (1 - bs_need_open[self.cell_in_bs_id[i], 0])
                                     for i in range(self.n_cell)])
            self.cell_control_state[:, 1] = self.cell_control_state[:, 1] - cell_can_off
            self.cell_control_state[:, 2] = cell_can_off
        else:
            bs_need_open = np.array([self.cell_control_state[self.cell_bs_groups[i], :2].sum()>0
                                     for i in range(self.n_bs)]).astype('float32')[:, None]         # [n_bs, 1]
        self.bs_control_state = np.concatenate([bs_need_open, 1-bs_need_open], axis=1)              # [n_bs, 2]

    def _adjust_action(self, action):
        """

        :param action:  [n_cell, 3]
        :return:        [n_cell, 3]
        """
        if self.unsatisfied_penalty_type == 'action_constraint':
            down_flow_all = self.down_flow[self.timestep]  # [n_cell]
            cell_capacity_all = self.cell_capacity[:, 0]  # [n_cell]
            # idx = np.argsort(-cell_capacity_all)  # [n_cell]
            for i in range(self.n_grid):
                down_flow_in_grid = down_flow_all[self.cell_grid_groups[i]]  # [n_cell_in_grid]
                cell_capacity_in_grid = cell_capacity_all[self.cell_grid_groups[i]]  # [n_cell_in_grid]
                unsatisfied_down_flow = down_flow_in_grid.sum() - (cell_capacity_in_grid * action[self.cell_grid_groups[i], 0]).sum()
                # [1]
                if unsatisfied_down_flow > 0.:
                    # print('grid_id', i, 'unsatisfied_down_flow', unsatisfied_down_flow)
                    idx = np.array(self.cell_grid_groups[i])
                    idx_ls = np.argsort(-self.cell_priority[idx, 0])  # [n_cell]
                    # idx_ls = np.random.permutation(range(idx.shape[0]))  # [n_cell]
                    # large to small
                    for j in idx_ls:
                        if unsatisfied_down_flow < 0:
                            break
                        if not action[idx[j], 0]:
                            action[idx[j], :] = [1, 0, 0]
                            unsatisfied_down_flow -= self.cell_capacity[idx[j], 0]

        return action

    def _reallocate_flow(self):
        self.cur_down_flow = np.zeros([self.n_cell])
        self.unsatisfied_down_flow_grid = np.zeros([self.n_grid])

        down_flow_all = self.down_flow[self.timestep]  # [n_cell]
        cell_capacity_all = self.cell_capacity[:, 0]  # [n_cell]
        for i in range(self.n_grid):
            total_down_flow_in_grid = down_flow_all[self.cell_grid_groups[i]].sum()  # [n_cell_in_grid]
            total_activate_cell_capacity_in_grid = \
                (cell_capacity_all[self.cell_grid_groups[i]] * self.cell_control_state[self.cell_grid_groups[i], 0]).sum()  # [n_cell_in_grid]

            if total_down_flow_in_grid > total_activate_cell_capacity_in_grid:
                self.cur_down_flow[self.cell_grid_groups[i]] \
                    = self.cell_capacity[self.cell_grid_groups[i], 0] * self.cell_control_state[self.cell_grid_groups[i], 0]
                self.unsatisfied_down_flow_grid[i] = total_down_flow_in_grid - total_activate_cell_capacity_in_grid
            else:
                # # --large to small--
                # idx = np.arange(self.n_cell)[self.cell_grid_mask[:, i]]
                # idx_ls = np.argsort(-cell_capacity_all[idx])  # [n_cell]
                # for j in idx_ls:
                #     if total_down_flow_in_grid <= 0:
                #         break
                #     if self.cell_control_state[idx[j], 0]:
                #         self.cur_down_flow[idx[j]] = min(self.cell_capacity[idx[j], 0], total_down_flow_in_grid)
                #         total_down_flow_in_grid -= self.cur_down_flow[idx[j]]
                # self.unsatisfied_down_flow_grid[i] = 0.
                # # --proportional to cell_capacity--
                self.cur_down_flow[self.cell_grid_groups[i]] \
                    = total_down_flow_in_grid * ((cell_capacity_all[self.cell_grid_groups[i]] * self.cell_control_state[self.cell_grid_groups[i], 0]) / (total_activate_cell_capacity_in_grid + 1e-6))
                self.unsatisfied_down_flow_grid[i] = 0.
                # --small coef to large coef--
                # idx = np.arange(self.n_cell)[self.cell_grid_mask[:, i]]
                # idx_ls = np.argsort(self.cell_power_coef[idx, 0])  # [n_cell]
                # for j in idx_ls:
                #     if total_down_flow_in_grid <= 0:
                #         break
                #     if self.cell_control_state[idx[j], 0]:
                #         self.cur_down_flow[idx[j]] = min(self.cell_capacity[idx[j], 0], total_down_flow_in_grid)
                #         total_down_flow_in_grid -= self.cur_down_flow[idx[j]]
                # self.unsatisfied_down_flow_grid[i] = 0.

            assert (self.cur_down_flow <= self.cell_capacity[:, 0] * self.cell_control_state[:, 0]).all()
            assert self.cur_down_flow.sum() <= self.down_flow[self.timestep].sum() + 1e-6

    def _update_control_status(self):
        self.n_cell_control_status[self.timestep] += self.cell_control_state
        self.n_basestation_control_status[self.timestep] += self.bs_control_state

    def _cal_switch_cost(self):
        bs_cost = cal_switch_cost(self.prev_bs_control_state, self.bs_control_state, self.bs_switch_cost)
        cell_cost = cal_switch_cost(self.prev_cell_control_state, self.cell_control_state, self.cell_switch_cost)

        self.reward_related_record['cell_cost'] = cell_cost
        self.reward_related_record['cell_switch_num'] = (cell_cost>0).astype('int32')
        self.reward_related_record['cell_open_num'] = self.cell_control_state[:, 0]
        self.reward_related_record['cell_sleep_num'] = self.cell_control_state[:, 1]
        self.reward_related_record['cell_close_num'] = self.cell_control_state[:, 2]
        self.reward_related_record['bs_cost'] = bs_cost
        self.reward_related_record['bs_switch_num'] = (bs_cost > 0).astype('int32')
        self.reward_related_record['bs_open_num'] = self.bs_control_state[:, 0]
        self.reward_related_record['bs_close_num'] = self.bs_control_state[:, 1]

        if self.reward_type == 'independent':
            cell_bs_cost = bs_cost[[self.cell_in_bs_id[i] for i in range(self.n_cell)]]
            return cell_bs_cost + cell_cost

    def _cal_power(self):
        power_rru = (self.cell_power_coef[:, 0] * self.cur_down_flow + self.cell_power_coef[:, 1]) * self.cell_control_state[:, 0]
        power_rru_sleep = self.cell_sleep_power[:, 0] * self.cell_control_state[:, 1]
        power_bbu = self.bs_control_state[:, 0] * self.power_bbu[self.timestep]

        temp = np.array([(power_rru + power_rru_sleep)[self.cell_bs_groups[i]].sum() for i in range(self.n_bs)]) + power_bbu
        power_airconditioner = self.bs_control_state[:, 0] * (self.air_power_coef[self.timestep][0] * temp + self.air_power_coef[self.timestep][1])

        self.reward_related_record['power_rru'] = power_rru + power_rru_sleep
        self.reward_related_record['sleep_rru_power'] = power_rru_sleep
        self.reward_related_record['activate_cell'] = self.cell_control_state[:, 0]
        self.reward_related_record['original_power_rru'] = self.power_rru[self.timestep]
        self.reward_related_record['power_bbu'] = power_bbu
        self.reward_related_record['original_power_bbu'] = self.power_bbu[self.timestep]
        self.reward_related_record['power_airconditioner'] = power_airconditioner
        self.reward_related_record['original_power_airconditioner'] = self.power_airconditioner[self.timestep]
        self.reward_related_record['total_power'] = \
            power_bbu + power_airconditioner + \
            np.array([(power_rru + power_rru_sleep)[self.cell_bs_groups[i]].sum() for i in range(self.n_bs)])

        if self.reward_type == 'independent':
            cell_bs_power = (power_bbu + power_airconditioner)[[self.cell_in_bs_id[i] for i in range(self.n_cell)]]
            power_rru = power_rru + power_rru_sleep
            power_rru_in_grid = np.array([power_rru[self.cell_grid_groups[cell_id]].sum() for cell_id in range(self.n_grid)])
            power_rru_in_grid = np.array([power_rru_in_grid[self.cell_in_grid_id[cell_id]] for cell_id in range(self.n_cell)])
            return power_rru_in_grid + cell_bs_power

    def _cal_qos_degradation_cost(self):
        qos_cost = 1. / (self.cell_capacity[:, 0] - self.cur_down_flow + 1e-2) * self.cell_control_state[:, 0]
        qos_cost = qos_cost + \
                   (self.unsatisfied_down_flow_grid > 0.)[[self.cell_in_grid_id[i] for i in range(self.n_cell)]] \
                   * self.unsatisfied_penalty

        self.reward_related_record['qos_cost'] = qos_cost
        self.reward_related_record['original_qos_cost'] = \
            1. / (self.cell_capacity[:, 0] - self.down_flow[self.timestep] + 1e-2)

        if self.reward_type == 'independent':
            return qos_cost

    def _cal_courious_reward(self):
        reward_cell = ((self.n_cell_control_status[self.timestep] * self.cell_control_state).sum(-1) + 1) ** (-0.5)
        reward_bs = ((self.n_basestation_control_status[self.timestep] * self.bs_control_state).sum(-1) + 1) ** (-0.5)

        if self.reward_type == 'independent':
            cell_bs_reward = reward_bs[[self.cell_in_bs_id[i] for i in range(self.n_cell)]]
            return cell_bs_reward + reward_cell

    def _cal_reward(self):
        switch_cost = self._cal_switch_cost()
        power = self._cal_power()
        qos_cost = self._cal_qos_degradation_cost()
        courious_reward = self._cal_courious_reward()
        print('courious_reward', np.median(power), np.median(courious_reward) / self.scale_courious_reward, np.max(courious_reward) / self.scale_courious_reward)

        return -(switch_cost*0. + power + qos_cost*0.) + courious_reward / self.scale_courious_reward

    def _record(self):
        # record
        self.cell_record.append(
            {cid: {'cell_control_state': c_cs,
                   'traffic': traf,
                   'cell_cost': c_cost,
                   'switch_num': n_sw,
                   'open_num': n_op,
                   'sleep_num': n_sl,
                   'close_num': n_cl,
                   'power_rru': p_rru,
                   'original_power_rru': o_p_rru,
                   'qos_cost': qc,
                   'original_qos_cost': o_qc}
             for cid, c_cs, traf, c_cost, n_sw, n_op, n_sl, n_cl, p_rru, o_p_rru, qc, o_qc in zip(
                self.cell_sids,
                self.cell_control_state,
                self.cur_down_flow,
                self.reward_related_record['cell_cost'],
                self.reward_related_record['cell_switch_num'],
                self.reward_related_record['cell_open_num'],
                self.reward_related_record['cell_sleep_num'],
                self.reward_related_record['cell_close_num'],
                self.reward_related_record['power_rru'],
                self.reward_related_record['original_power_rru'],
                self.reward_related_record['qos_cost'],
                self.reward_related_record['original_qos_cost']
            )}
        )
        self.debug_record['cell_record'] = {
            cid: {'onoff': c_cs[0],
                  'traffic': traf,
                  'power_rru': p_rru,}
            for cid, c_cs, traf, p_rru in zip(
                self.cell_sids,
                self.cell_control_state,
                self.cur_down_flow,
                self.reward_related_record['power_rru'],
            )
        }

        total_cell_power_in_bs = np.array([self.reward_related_record['power_rru'][self.cell_bs_groups[i]].sum() for i in range(self.n_bs)])
        total_original_cell_power_in_bs = np.array([self.reward_related_record['original_power_rru'][self.cell_bs_groups[i]].sum() for i in range(self.n_bs)])
        self.bs_record.append(
            {bid: {'bs_control_state': b_cs,
                   'bs_cost': b_cost,
                   'switch_num': n_sw,
                   'open_num': n_op,
                   'close_num': n_cl,
                   'power_bbu': p_bbu,
                   'original_power_bbu': o_p_bbu,
                   'power_air': p_air,
                   'original_power_air': o_p_air,
                   'power': t_p}
             for bid, b_cs, b_cost, n_sw, n_op, n_cl, p_bbu, o_p_bbu, p_air, o_p_air, t_p in zip(
                self.bs_sids,
                self.bs_control_state,
                self.reward_related_record['bs_cost'],
                self.reward_related_record['bs_switch_num'],
                self.reward_related_record['bs_open_num'],
                self.reward_related_record['bs_close_num'],
                self.reward_related_record['power_bbu'],
                self.reward_related_record['original_power_bbu'],
                self.reward_related_record['power_airconditioner'],
                self.reward_related_record['original_power_airconditioner'],
                self.reward_related_record['total_power']
            )}
        )
        self.debug_record['bs_record'] = {
            bid: {'onoff': float(b_cs[0]),
                  'power_rru': total_cell_power_in_bs[i],
                  'power_bbu': p_bbu,
                  'power_air': p_air,
                  'power_bs': total_cell_power_in_bs[i] + p_bbu + p_air,
                  'original_power_bs': o_p_all}
            for i, bid, b_cs, p_bbu, p_air, o_p_all in zip(
                range(self.n_bs),
                self.bs_sids,
                self.bs_control_state,
                self.reward_related_record['power_bbu'],
                self.reward_related_record['power_airconditioner'],
                self.reward_related_record['original_power_bbu'] +
                    self.reward_related_record['original_power_airconditioner'] + total_original_cell_power_in_bs
            )
        }

        down_flow_all = self.down_flow[self.timestep]  # [n_cell]
        cell_capacity_all = self.cell_capacity[:, 0]  # [n_cell]
        total_down_flow_in_grid = \
            np.array([down_flow_all[self.cell_grid_groups[i]].sum() for i in range(self.n_grid)])
        total_activate_cell_capacity_in_grid = \
            np.array([(cell_capacity_all[self.cell_grid_groups[i]]
                       * self.cell_control_state[self.cell_grid_groups[i], 0]).sum()
                      for i in range(self.n_grid)])
        self.grid_record.append({
            gid: {
                'flow': total_down_flow_in_grid[gid],
                'capacity': total_activate_cell_capacity_in_grid[gid]
            }
            for gid in range(self.n_grid)
        })
        self.debug_record['grid_record'] = self.grid_record[-1]

        self.total_record.append({
            'cell_switch_num': self.reward_related_record['cell_switch_num'].sum(),
            'bs_switch_num': self.reward_related_record['bs_switch_num'].sum(),
            'power_rru': self.reward_related_record['power_rru'].sum(),
            'original_power_rru': self.reward_related_record['original_power_rru'].sum(),
            'power_bbu': self.reward_related_record['power_bbu'].sum(),
            'original_power_bbu': self.reward_related_record['original_power_bbu'].sum(),
            'power_airconditioner': self.reward_related_record['power_airconditioner'].sum(),
            'original_power_airconditioner': self.reward_related_record['original_power_airconditioner'].sum(),
            'sleep_rru_power': self.reward_related_record['sleep_rru_power'].sum(),
            'activate_cell': self.reward_related_record['activate_cell'].sum(),
        })

        for k in self.ep_record:
            self.ep_record[k] = self.ep_record[k] + self.total_record[-1][k]

    def init(self):
        self.timestep = 0
        self.last_down_flow = -1 * np.ones([self.n_cell, self.history_queue_length]) # [n_cell, hq_len]
        self.last_user_cnt = -1 * np.ones([self.n_cell, self.history_queue_length])   # [n_cell, hq_len]
        self.bs_control_state = np.array([[1, 0] for _ in range(self.n_bs)])      # {[1,0]: on, [0,1]: off}
        self.cell_control_state = np.array([[1, 0, 0] for _ in range(self.n_cell)])     # {[1,0,0]: on, [0,1,0]: sleep, [0,0,1]: off}

        state = np.concatenate(list(self._cal_state().values()), axis=1)

        self.cell_record = []
        self.bs_record = []
        self.total_record = []
        self.ep_record = {
            'cell_switch_num': 0,
            'bs_switch_num': 0,
            'power_rru': 0,
            'original_power_rru': 0,
            'power_bbu': 0,
            'original_power_bbu': 0,
            'power_airconditioner': 0,
            'original_power_airconditioner': 0,
            'sleep_rru_power': 0,
            'activate_cell': 0,
        }
        info = {}
        return state, info

    def step(self, action):
        """
        :param action:          np.array([n_cell, 3])       action[i]\in{[1,0,0]: on, [0,1,0]: sleep, [0,0,1]: off}
        :return:
        """
        assert self.timestep <= self.horizon        # only cal reward for t=self.horizon
        assert (action >= 0).all() and (action <= 1).all() and (action.sum(-1) == 1).all()

        # update control sates of base stations and cells
        self.prev_cell_control_state = self.cell_control_state                              # [n_cell, 3]
        self.prev_bs_control_state = self.bs_control_state                                  # [n_bs, 2]
        action = self._adjust_action(action)
        self.cell_control_state = action                                                    # [n_cell, 3]
        self._update_bs_control_state()
        # reallocate flow
        self._reallocate_flow()
        # update control status
        self._update_control_status()

        # reward
        self.reward_related_record = {}
        reward = self._cal_reward()
        reward = reward / self.scale_reward

        # record
        self._record()

        # new state
        self.timestep += 1
        if self.timestep < self.horizon:
            average_down_flow_in_grid = \
                np.array([self.down_flow[self.timestep][self.cell_grid_groups[i]].sum() / len(self.cell_grid_groups[i])
                          for i in range(self.n_grid)])
            cell_average_down_flow = np.array([average_down_flow_in_grid[self.cell_in_grid_id[i]] for i in range(self.n_cell)])
            self.last_down_flow[:, :-1] = self.last_down_flow[:, 1:]
            self.last_down_flow[:, -1] = cell_average_down_flow / self.scale_down_flow            # [n_cell, hq_len]
            # self.last_user_cnt[:, :-1] = self.last_user_cnt[:, 1:]
            # self.last_user_cnt[:, -1] = self.user_cnt[self.timestep]                            # [n_cell, hq_len]
            state = np.concatenate(list(self._cal_state().values()), axis=1)
        else:
            state = None
        info = {'cell_record': self.cell_record,
                'bs_record': self.bs_record,
                'total_record': self.total_record,
                'ep_record': self.ep_record,
                'unsatisfied_down_flow_grid': self.unsatisfied_down_flow_grid,
                'adjusted_action': action,
                'debug_record': self.debug_record}

        done = (self.timestep == self.horizon)
        return state, reward, done, info


def analysis_record(record_file):
    env = BaseStationSleepingProblem()
    env.set_environment_mode('traversal')
    cell_records = np.load('cell_records.npy', allow_pickle=True)
    cell_records1 = [{} for i in range(48)]
    for r_grid in cell_records:
        for i in range(48):
            cell_records1[i].update(r_grid[i])

    state, info = env.init()
    n_grid = 0
    n_timestep = 0
    done = False
    infos = []
    total_cell_cost = 0.
    total_n_sw = 0.
    total_n_op = 0.
    total_n_sl = 0.
    total_n_cl = 0.
    total_n_p_rru = 0.
    total_n_o_p_rru = 0.
    total_n_qc = 0.
    total_n_o_qc = 0.
    while True:
        action_onehot = np.array([cell_records1[n_timestep][cid]['cell_control_state'] for cid in env.cell_ids])
        next_state, reward, done, info = env.step(action_onehot)

        n_timestep += 1
        if done:
            n_timestep = 0
            done = False
            infos.append(info['cell_record'])

            temp = np.array([[list(record[cid].values()) for cid in record] for record in info['cell_record']])
            total_cell_cost += temp[:, :, 2].sum()
            total_n_sw += temp[:, :, 3].sum()
            total_n_op += temp[:, :, 4].sum()
            total_n_sl += temp[:, :, 5].sum()
            total_n_cl += temp[:, :, 6].sum()
            total_n_p_rru += temp[:, :, 7].sum()
            total_n_o_p_rru += temp[:, :, 8].sum()
            total_n_qc += temp[:, :, 9].sum()
            total_n_o_qc += temp[:, :, 10].sum()


            state, info = env.init()

            n_grid = len(env.cell_grid)
            print(env.grid_id)
            if env.grid_id == n_grid - 1:
                np.save('cell_records_analysised.npy', infos)
                infos = []

                print(total_cell_cost, total_n_sw, total_n_op, total_n_sl, total_n_cl, total_n_p_rru,
                      total_n_o_p_rru, total_n_qc, total_n_o_qc)


def test_bs_sleeping_problem():
    env = BaseStationSleepingProblem()

    state, info = env.init()

    done = False
    while not done:
        action = np.random.randint(0, 3, size=env.n_cell)
        action = to_onehot(action, num_classes=3)
        state, reward, done, info = env.step(action)

    print(info['cell_record'], info['bs_record'], info['total_record'])
    pass


if __name__ == '__main__':

    # analysis_record('cell_records.npy')

    test_bs_sleeping_problem()
    pass