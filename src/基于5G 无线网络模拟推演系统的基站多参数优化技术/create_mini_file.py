import numpy as np
import random


if __name__ == '__main__':
    grid_cell = np.load('input/Grid_Cell.npy', allow_pickle=True)[()]
    pass

    network_5g_info = np.load('input/5G/Network_Power2022-05-20.npy', allow_pickle=True).item()
    basestation_5g_info = np.load('input/Cell_info_5G.npy', allow_pickle=True).item()
    Cell_Capacity_5G = np.load('input/Cell_Capacity_5G.npy', allow_pickle=True).item()
    network_4g_info = np.load('input/4G/Network_Power2022-05-20.npy', allow_pickle=True).item()
    basestation_4g_info = np.load('input/Cell_4G_LTE.npy', allow_pickle=True).item()
    Cell_Capacity_4G = np.load('input/Cell_Capacity_4G.npy', allow_pickle=True).item()

    n_bs_5g = 4
    n_bs_4g = 7

    network_5g_info_mini = {}
    for t in network_5g_info:
        network_5g_info_mini_t = {}
        for i, bs_id in enumerate(network_5g_info[t]):
            if i == n_bs_5g:
                break
            network_5g_info_mini_t[bs_id] = network_5g_info[t][bs_id]
        network_5g_info_mini[t] = network_5g_info_mini_t
    np.save('input/5G_Network_Power_mini.npy', network_5g_info_mini)

    basestation_5g_info_mini = {}
    for id in [cell_id for bs_id in network_5g_info_mini[0] for cell_id in network_5g_info_mini[0][bs_id]]:
        basestation_5g_info_mini[id] = basestation_5g_info[id]
    np.save('input/Cell_info_5G_mini.npy', basestation_5g_info_mini)

    Cell_Capacity_5G_mini = {}
    for id in [cell_id for bs_id in network_5g_info_mini[0] for cell_id in network_5g_info_mini[0][bs_id]]:
        Cell_Capacity_5G_mini[id] = Cell_Capacity_5G[id]
    np.save('input/Cell_Capacity_5G_mini.npy', Cell_Capacity_5G_mini)


    network_4g_info_mini = {}
    for t in network_4g_info:
        network_4g_info_mini_t = {}
        for i, bs_id in enumerate(network_4g_info[t]):
            if i == n_bs_4g:
                break
            network_4g_info_mini_t[bs_id] = network_4g_info[t][bs_id]
        network_4g_info_mini[t] = network_4g_info_mini_t
    np.save('input/4G_Network_Power_mini.npy', network_4g_info_mini)

    basestation_4g_info_mini = {}
    for id in [cell_id for bs_id in network_4g_info_mini[0] for cell_id in network_4g_info_mini[0][bs_id]]:
        basestation_4g_info_mini[id] = list(basestation_4g_info[id])
    np.save('input/Cell_4G_LTE_mini.npy', basestation_4g_info_mini)

    Cell_Capacity_4G_mini = {}
    for id in [cell_id for bs_id in network_4g_info_mini[0] for cell_id in network_4g_info_mini[0][bs_id]]:
        Cell_Capacity_4G_mini[id] = Cell_Capacity_4G[id]
    np.save('input/Cell_Capacity_4G_mini.npy', Cell_Capacity_4G_mini)

    Cell_ids = list(Cell_Capacity_5G_mini.keys()) + list(Cell_Capacity_4G_mini.keys())
    rand_cell_ids = np.random.permutation(Cell_ids).tolist()
    cell_grid_mini = {}
    n, i = 0, 0
    while n < len(rand_cell_ids):
        n_cell_to_add = min(np.random.randint(1, 10), len(rand_cell_ids) - n)
        cell_grid_mini[i] = rand_cell_ids[n:n+n_cell_to_add]
        n += n_cell_to_add
        i += 1
    np.save('input/Grid_Cell_mini.npy', cell_grid_mini)
    pass
