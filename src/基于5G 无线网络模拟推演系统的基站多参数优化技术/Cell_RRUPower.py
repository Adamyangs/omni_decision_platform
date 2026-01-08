import numpy as np


def Cell_Power(Cell_ID, Traffic):
    Cell_Traffic_PRB_Model = np.load('output/Cell_Traffic_PRB_Model.npy', allow_pickle=True).item()
    Cell_5G = np.load('input/Cell_info_5G.npy', allow_pickle=True).item()
    Power_Max_Cell = np.load('output/Power_Max_Cell.npy', allow_pickle=True).item()
    if Cell_ID in Cell_5G:
        Attena = Cell_5G[Cell_ID][4]
        Mode = Cell_5G[Cell_ID][18]
        if Attena == '32TR':
            if Mode =='SA':
                Power_cell_max = 244.75921581
                P_0 = 389.7694737
                delta_P = 1.69277579
            else:
                Power_cell_max = 259.95527467
                P_0 = 287.35462756
                delta_P = 1.44737735
        elif Attena == '64TR':
            if Mode =='SA':
                Power_cell_max = 203.08416447
                P_0 = 702.56410063
                delta_P = 1.86843048
            else:
                Power_cell_max = 159.18477044
                P_0 = 741.6785062
                delta_P = 1.45236084
        Power_Transmit_Cell = Power_cell_max * Cell_Traffic_PRB_Model[Cell_ID]['Coef']*Traffic
        Power_RRU = P_0 + delta_P*Power_Transmit_Cell

        # P_0 + delta_P * (Power_cell_max * Cell_Traffic_PRB_Model[Cell_ID]['Coef']*Traffic)
        # P_0 + delta_P * Power_cell_max * Cell_Traffic_PRB_Model[Cell_ID]['Coef'] * Traffic

    elif Cell_ID in Power_Max_Cell:
        # pdb.set_trace()
        if Power_Max_Cell[Cell_ID] == 19.87:
            P_0 = 156.62338972
            delta_P = 6.62833641

        elif Power_Max_Cell[Cell_ID] == 79.09:
            P_0 = 103.07570764
            delta_P = 4.9461373

        elif Power_Max_Cell[Cell_ID] ==39.64:
            P_0 = 138.01200241
            delta_P = 5.75720085

        Power_Ratio = 0.3333075584 + 0.6665727 * Cell_Traffic_PRB_Model[Cell_ID]['Coef']*Traffic
        Power_Transmit_Cell = Power_Ratio * Power_Max_Cell[Cell_ID]
        Power_RRU = P_0 + delta_P*Power_Transmit_Cell

        # P_0 + delta_P*Power_Ratio * Power_Max_Cell[Cell_ID] * (0.3333075584 + 0.6665727 * Cell_Traffic_PRB_Model[Cell_ID]['Coef']*Traffic)
        # P_0 + delta_P * Power_Max_Cell[Cell_ID] * 0.3333075584 + delta_P * Power_Max_Cell[Cell_ID] * 0.6665727 * Cell_Traffic_PRB_Model[Cell_ID]['Coef']*Traffic
    else:
        # pdb.set_trace()
        print('Cell Not Found!!!')

    return Power_RRU


def get_weight_bias():
    Cell_Traffic_PRB_Model = np.load('output/Cell_Traffic_PRB_Model.npy', allow_pickle=True).item()
    Cell_5G = np.load('input/Cell_info_5G.npy', allow_pickle=True).item()
    Power_Max_Cell = np.load('output/Power_Max_Cell.npy', allow_pickle=True).item()

    network_5g_info = np.load('input/5G/Network_Power2022-05-20.npy', allow_pickle=True)[()]
    network_4g_info = np.load('input/4G/Network_Power2022-05-20.npy', allow_pickle=True)[()]
    network_info = [{**network_5g_info[k1], **network_4g_info[k2]} for k1, k2 in zip(network_5g_info, network_4g_info)]
    cell_ids = [cell_id for bs_id in network_info[0] for cell_id in network_info[0][bs_id]]  # [n_cell]

    weights_and_biases = {}
    sleep_powers = {}
    for Cell_ID in cell_ids:
        if Cell_ID in Cell_5G:
            Attena = Cell_5G[Cell_ID][4]
            Mode = Cell_5G[Cell_ID][18]
            if Attena == '32TR':
                if Mode == 'SA':
                    Power_cell_max = 244.75921581
                    P_0 = 389.7694737
                    delta_P = 1.69277579
                else:
                    Power_cell_max = 259.95527467
                    P_0 = 287.35462756
                    delta_P = 1.44737735
            elif Attena == '64TR':
                if Mode == 'SA':
                    Power_cell_max = 203.08416447
                    P_0 = 702.56410063
                    delta_P = 1.86843048
                else:
                    Power_cell_max = 159.18477044
                    P_0 = 741.6785062
                    delta_P = 1.45236084

            def cal_power_5g(Traffic):
                Power_Transmit_Cell = Power_cell_max * Cell_Traffic_PRB_Model[Cell_ID]['Coef'] * Traffic
                Power_RRU = P_0 + delta_P * Power_Transmit_Cell
                return Power_RRU
            bias = cal_power_5g(0)
            weight = cal_power_5g(1) - bias
            weights_and_biases[Cell_ID] = [weight, bias]

            # P_0 + delta_P * (Power_cell_max * Cell_Traffic_PRB_Model[Cell_ID]['Coef']*Traffic)
            # P_0 + delta_P * Power_cell_max * Cell_Traffic_PRB_Model[Cell_ID]['Coef'] * Traffic

        elif Cell_ID in Power_Max_Cell:
            # pdb.set_trace()
            if Power_Max_Cell[Cell_ID] == 19.87:
                P_0 = 156.62338972
                delta_P = 6.62833641

            elif Power_Max_Cell[Cell_ID] == 79.09:
                P_0 = 103.07570764
                delta_P = 4.9461373

            elif Power_Max_Cell[Cell_ID] == 39.64:
                P_0 = 138.01200241
                delta_P = 5.75720085

            def cal_power_4g(Traffic):
                Power_Ratio = 0.3333075584 + 0.6665727 * Cell_Traffic_PRB_Model[Cell_ID]['Coef'] * Traffic
                Power_Transmit_Cell = Power_Ratio * Power_Max_Cell[Cell_ID]
                Power_RRU = P_0 + delta_P * Power_Transmit_Cell
                return Power_RRU

            bias = cal_power_4g(0)
            weight = cal_power_4g(1) - bias
            weights_and_biases[Cell_ID] = [weight, bias]

            # P_0 + delta_P*Power_Ratio * Power_Max_Cell[Cell_ID] * (0.3333075584 + 0.6665727 * Cell_Traffic_PRB_Model[Cell_ID]['Coef']*Traffic)
            # P_0 + delta_P * Power_Max_Cell[Cell_ID] * 0.3333075584 + delta_P * Power_Max_Cell[Cell_ID] * 0.6665727 * Cell_Traffic_PRB_Model[Cell_ID]['Coef']*Traffic
        else:
            # pdb.set_trace()
            print('Cell Not Found!!!')

        if Cell_ID in Cell_5G:
            Attena = Cell_5G[Cell_ID][4]
            Mode = Cell_5G[Cell_ID][18]
            if Attena == '32TR':
                if Mode == 'SA':
                    sleep_power = 78.9992
                else:
                    sleep_power = 69.4302
            elif Attena == '64TR':
                if Mode == 'SA':
                    sleep_power = 88.4698
                else:
                    sleep_power = 90.5637

        elif Cell_ID in Power_Max_Cell:
            if Power_Max_Cell[Cell_ID] == 19.87:
                sleep_power = 119.0310
            if Power_Max_Cell[Cell_ID] == 39.64:
                sleep_power = 127.9319
            if Power_Max_Cell[Cell_ID] == 79.09:
                sleep_power = 133.9013
        else:
            print('Cell Not Found!!!')
        sleep_powers[Cell_ID] = sleep_power

    np.save('input/cell_rru_power.npy', weights_and_biases)
    np.save('input/sleep_power.npy', sleep_powers)
    return weights_and_biases, sleep_powers



if __name__ == '__main__':
    weights_and_biases, sleep_powers = get_weight_bias()

    network_5g_info = np.load('input/5G/Network_Power2022-05-20.npy', allow_pickle=True)[()]
    network_4g_info = np.load('input/4G/Network_Power2022-05-20.npy', allow_pickle=True)[()]
    network_info = [{**network_5g_info[k1], **network_4g_info[k2]} for k1, k2 in zip(network_5g_info, network_4g_info)]
    cell_ids = [cell_id for bs_id in network_info[0] for cell_id in network_info[0][bs_id]]  # [n_cell]
    rnd_cell_ids = np.random.randint(0, len(cell_ids), [20])
    for i in rnd_cell_ids:
        traffic = np.random.rand() * 1e8
        power1 = Cell_Power(cell_ids[i], traffic)
        weight, bias = weights_and_biases[cell_ids[i]]
        power2 = weight * traffic + bias

        print(np.abs(power1-power2), power1, power2, traffic, weight, bias)