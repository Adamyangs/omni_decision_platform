import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from environment import BaseStationSleepingProblem
from util import to_onehot

if __name__ == '__main__':
    mbc_result = np.load('output/data_onoff_PeAs_45G52000.npy', allow_pickle=True)[()]
    rl_result = np.load('cmp.npy', allow_pickle=True)[()]
    mbc_result = np.load('output/info_52000.npy', allow_pickle=True)[()]
    onoffs = np.load('output/data_onoff_PeAs.npy', allow_pickle=True)[()]

    # # cell_record
    cell_ids = rl_result['cell_record'].keys()
    for cell_id in cell_ids:
        if abs(rl_result['cell_record'][cell_id]['onoff']-mbc_result['cell_record'][cell_id]['Status']) > 0.:
            for gid in onoffs[0][0]:
                if cell_id in onoffs[0][0][gid]:
                    print(cell_id,
                          onoffs[0][0][gid][cell_id],
                          rl_result['cell_record'][cell_id]['onoff'],
                          mbc_result['cell_record'][cell_id]['Status'])
            print('diff onoff = {} traffic = {} power_rru = {}'.format(
                rl_result['cell_record'][cell_id]['onoff']-mbc_result['cell_record'][cell_id]['Status'],
                rl_result['cell_record'][cell_id]['traffic']-mbc_result['cell_record'][cell_id]['Traffic'],
                rl_result['cell_record'][cell_id]['power_rru']-mbc_result['cell_record'][cell_id]['Power'],))


    # bs_ids = rl_result['bs_record'].keys()
    # for bs_id in bs_ids:
    #     # if abs(rl_result['bs_record'][bs_id]['traffic']-mbc_result['bs_record'][bs_id]['Traffic']) > 1.:
    #         print('diff onoff = {} power_bbu = {}'.format(
    #             rl_result['bs_record'][bs_id]['onoff']-mbc_result['bs_record'][bs_id]['onoff'],
    #             rl_result['bs_record'][bs_id]['power_bbu']-mbc_result['bs_record'][bs_id]['power_bbu']))


    # grid_ids = rl_result['grid_record'].keys()
    # for grid_id in grid_ids:
    #     if abs(rl_result['grid_record'][grid_id]['capacity']-mbc_result['grid_record'][grid_id]['Capacity']) > 1.:
    #         print(grid_id, 'diff flow = {} capacity = {}'.format(
    #             rl_result['grid_record'][grid_id]['flow']-mbc_result['grid_record'][grid_id]['flow'],
    #             rl_result['grid_record'][grid_id]['capacity']-mbc_result['grid_record'][grid_id]['Capacity']))

    pass