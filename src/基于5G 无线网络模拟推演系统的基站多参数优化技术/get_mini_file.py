import numpy as np
import random


if __name__ == '__main__':
    temp = np.load('input/typical.npy', allow_pickle=True)[()]
    pass

    day = 20
    suffix = 'typical'

    # network_power_5g = np.load('input/5G/Network_Power2022-05-{}.npy'.format(day), allow_pickle=True)[()]
    # network_power_5g_mini = {}
    # for t in network_power_5g:
    #     network_power_5g_t = {}
    #     for bs_id in temp['BS']:
    #         if bs_id in network_power_5g[t]:
    #             network_power_5g_t[bs_id] = network_power_5g[t][bs_id]
    #     network_power_5g_mini[t] = network_power_5g_t
    # np.save('input/5G/Network_Power2022-05-{}_{}.npy'.format(day, suffix), network_power_5g_mini)
    # network_power_4g = np.load('input/4G/Network_Power2022-05-{}.npy'.format(day), allow_pickle=True)[()]
    # network_power_4g_mini = {}
    # for t in network_power_4g:
    #     network_power_4g_t = {}
    #     for bs_id in temp['BS']:
    #         if bs_id in network_power_4g[t]:
    #             network_power_4g_t[bs_id] = network_power_4g[t][bs_id]
    #     network_power_4g_mini[t] = network_power_4g_t
    # np.save('input/4G/Network_Power2022-05-{}_{}.npy'.format(day, suffix), network_power_4g_mini)
    #
    # basestation_5g = np.load('input/Cell_info_5G.npy', allow_pickle=True)[()]
    # basestation_5g_mini = {}
    # for i in temp['Cell']:
    #     if i in basestation_5g:
    #         basestation_5g_mini[i] = basestation_5g[i]
    # np.save('input/Cell_info_5G_{}.npy'.format(suffix), basestation_5g_mini)
    # basestation_4g = np.load('input/Cell_info_4G.npy', allow_pickle=True)[()]
    # basestation_4g_mini = {}
    # for i in temp['Cell']:
    #     if i in basestation_4g:
    #         basestation_4g_mini[i] = basestation_4g[i]
    # np.save('input/Cell_info_4G_{}.npy'.format(suffix), basestation_4g_mini)
    #
    # basestation_5g_new_air = np.load('input/data_as_5G.npy', allow_pickle=True)[()]
    # basestation_5g_new_air_mini = {}
    # for i in temp['BS']:
    #     if i in basestation_5g_new_air:
    #         basestation_5g_new_air_mini[i] = basestation_5g_new_air[i]
    # np.save('input/data_as_5G_{}.npy'.format(suffix), basestation_5g_new_air_mini)
    # basestation_4g_new_air = np.load('input/data_as_4G.npy', allow_pickle=True)[()]
    # basestation_4g_new_air_mini = {}
    # for i in temp['BS']:
    #     if i in basestation_4g_new_air:
    #         basestation_4g_new_air_mini[i] = basestation_4g_new_air[i]
    # np.save('input/data_as_4G_{}.npy'.format(suffix), basestation_4g_new_air_mini)
    #
    # cell_capacity_5g = np.load('input/Cell_Capacity_5G.npy', allow_pickle=True)[()]
    # cell_capacity_5g_mini = {}
    # for i in temp['Cell']:
    #     if i in cell_capacity_5g:
    #         cell_capacity_5g_mini[i] = cell_capacity_5g[i]
    # np.save('input/Cell_Capacity_5G_{}.npy'.format(suffix), cell_capacity_5g_mini)
    # cell_capacity_4g = np.load('input/Cell_Capacity_4G.npy', allow_pickle=True)[()]
    # cell_capacity_4g_mini = {}
    # for i in temp['Cell']:
    #     if i in cell_capacity_4g:
    #         cell_capacity_4g_mini[i] = cell_capacity_4g[i]
    # np.save('input/Cell_Capacity_4G_{}.npy'.format(suffix), cell_capacity_4g_mini)
    #
    # cell_power_coef = np.load('input/cell_rru_power.npy', allow_pickle=True)[()]
    # cell_power_coef_mini = {}
    # for i in temp['Cell']:
    #     if i in cell_power_coef:
    #         cell_power_coef_mini[i] = cell_power_coef[i]
    # np.save('input/cell_rru_power_{}.npy'.format(suffix), cell_power_coef_mini)
    #
    # cell_sleep_power = np.load('input/sleep_power.npy', allow_pickle=True)[()]
    # cell_sleep_power_mini = {}
    # for i in temp['Cell']:
    #     if i in cell_sleep_power:
    #         cell_sleep_power_mini[i] = cell_sleep_power[i]
    # np.save('input/sleep_power_{}.npy'.format(suffix), cell_sleep_power_mini)
    #
    # cell_grid = np.load('input/Grid_45G_300m.npy', allow_pickle=True)[()]
    # cell_grid_mini = {}
    # for i, gid in enumerate(temp['Grid']):
    #     if gid in cell_grid:
    #         cell_grid_mini[i] = cell_grid[gid]
    #         for cell_sid in cell_grid[gid]:
    #             if cell_sid not in temp['Cell']:
    #                 print('error')
    # np.save('input/Grid_45G_300m_{}.npy'.format(suffix), cell_grid_mini)
    #
    #
    temp1 = np.load('input/Power_mobicom_300m.npy', allow_pickle=True)[()]
    mobicom_rru_power = sum([temp1['Power_RRU'][cid][:48].sum() for cid in temp['Cell']])
    mobicom_bbu_power = sum([temp1['Power_BBU'][bid][:48].sum() for bid in temp['BS']])
    mobicom_air_power = sum([temp1['Power_Airconditioner'][bid][:48].sum() for bid in temp['BS']])

    print(mobicom_rru_power, + mobicom_bbu_power, + mobicom_air_power)

    pass
