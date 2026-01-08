import pandas as pd
import os
import pdb
from itertools import islice
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.ticker as mtick
import random


def Cell_Power(Cell_ID, Cell_Status, Traffic, Cell_Traffic_PRB_Model, Cell_5G, Power_Max_Cell):

    if Cell_Status == 'Active':
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
                    P_0 = 729.73145801
                    delta_P = 1.67607051
            Power_Transmit_Cell = Power_cell_max * Cell_Traffic_PRB_Model[Cell_ID]['Coef']*Traffic
            Power_RRU = P_0 + delta_P*Power_Transmit_Cell

        elif Cell_ID in Power_Max_Cell:
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
        else:
            print('Cell Not Found!!!')
    elif Cell_Status == 'Sleep':
        assert Traffic ==0, 'Error!!!'
        if Cell_ID in Cell_5G:
            Attena = Cell_5G[Cell_ID][4]
            Mode = Cell_5G[Cell_ID][18]
            if Attena == '32TR':
                if Mode =='SA':
                    Power_RRU = 78.9992
                else:
                    Power_RRU = 69.4302
            elif Attena == '64TR':
                if Mode =='SA':
                    Power_RRU = 88.4698
                else:
                    Power_RRU = 90.5637

        elif Cell_ID in Power_Max_Cell:
            if Power_Max_Cell[Cell_ID] == 19.87:
                Power_RRU = 119.0310
            if Power_Max_Cell[Cell_ID] == 39.64:
                Power_RRU = 127.9319
            if Power_Max_Cell[Cell_ID] == 79.09:
                Power_RRU = 133.9013
        else:
            print('Cell Not Found!!!')
    elif Cell_Status == 'OFF':
        assert Traffic ==0, 'Error!!!'
        Power_RRU = 0
    else:
        print('Status not found!!!')
    return Power_RRU



Date_list =  ['2022-05-20', '2022-05-21', '2022-05-22', '2022-05-23', '2022-05-24', '2022-05-25', '2022-05-26']
Cell_Traffic_PRB_Model = np.load('Cell_Traffic_PRB_Model.npy', allow_pickle=True).item()
Cell_5G = np.load('Cell_info_5G.npy', allow_pickle=True).item()
Power_Max_Cell = np.load('Power_Max_Cell.npy', allow_pickle=True).item()
# # pdb.set_trace()
for Date in Date_list:
    print(Date)
    Network_Power_4G = {}
    Network_Power_5G = {}

    Network_Power_4G_Sleep = np.load('Network_4G_Sleep'+Date+'.npy', allow_pickle=True).item()
    Network_Power_5G_Sleep = np.load('Network_5G_Sleep'+Date+'.npy', allow_pickle=True).item()

    for time_slot in Network_Power_4G_Sleep:
        print(time_slot)

        Network_Power_5G[time_slot] = {}
        Network_Power_4G[time_slot] = {}

        index = len(Network_Power_5G_Sleep[time_slot])
        for BS_ID in Network_Power_5G_Sleep[time_slot]:
            # print(index)
            index =index -1

            Network_Power_5G[time_slot][BS_ID] = {}
            Power_BBU = 115.44450236 + 82.44883492* len(Network_Power_5G_Sleep[time_slot][BS_ID])
            Power_Element = Power_BBU

            for Cell_ID in Network_Power_5G_Sleep[time_slot][BS_ID]:
                Network_Power_5G[time_slot][BS_ID][Cell_ID] = {}

                Power_RRU = Cell_Power(Cell_ID, Network_Power_5G_Sleep[time_slot][BS_ID][Cell_ID]['Status'], Network_Power_5G_Sleep[time_slot][BS_ID][Cell_ID]['down_flow'], Cell_Traffic_PRB_Model, Cell_5G, Power_Max_Cell)

                Network_Power_5G[time_slot][BS_ID][Cell_ID]['Power_RRU'] = Power_RRU
                Network_Power_5G[time_slot][BS_ID][Cell_ID]['Power_BBU'] = Power_BBU
                Network_Power_5G[time_slot][BS_ID][Cell_ID]['Status'] =  Network_Power_5G_Sleep[time_slot][BS_ID][Cell_ID]['Status'] 
                Network_Power_5G[time_slot][BS_ID][Cell_ID]['user_cnt'] = Network_Power_5G_Sleep[time_slot][BS_ID][Cell_ID]['user_cnt']
                Network_Power_5G[time_slot][BS_ID][Cell_ID]['down_flow'] = Network_Power_5G_Sleep[time_slot][BS_ID][Cell_ID]['down_flow']
                
                Power_Element = Power_Element + Power_RRU
                
            for Cell_ID in Network_Power_5G_Sleep[time_slot][BS_ID]:
                Network_Power_5G[time_slot][BS_ID][Cell_ID]['Power_Element'] = Power_Element


        index = len(Network_Power_4G_Sleep[time_slot])
        for BS_ID in Network_Power_4G_Sleep[time_slot]:
            # print(index)
            index =index -1
            Network_Power_4G[time_slot][BS_ID] = {}
            Power_BBU = 82.35517255 + 2.20383226* len(Network_Power_4G_Sleep[time_slot][BS_ID])
            Power_Element = Power_BBU

            for Cell_ID in Network_Power_4G_Sleep[time_slot][BS_ID]:
                Network_Power_4G[time_slot][BS_ID][Cell_ID] = {}
                Power_RRU = Cell_Power(Cell_ID, Network_Power_4G_Sleep[time_slot][BS_ID][Cell_ID]['Status'], Network_Power_4G_Sleep[time_slot][BS_ID][Cell_ID]['down_flow'], Cell_Traffic_PRB_Model, Cell_5G, Power_Max_Cell)

                Network_Power_4G[time_slot][BS_ID][Cell_ID]['Power_RRU'] = Power_RRU
                Network_Power_4G[time_slot][BS_ID][Cell_ID]['Power_BBU'] = Power_BBU

                Network_Power_4G[time_slot][BS_ID][Cell_ID]['Status'] =  Network_Power_4G_Sleep[time_slot][BS_ID][Cell_ID]['Status'] 
                Network_Power_4G[time_slot][BS_ID][Cell_ID]['user_cnt'] = Network_Power_4G_Sleep[time_slot][BS_ID][Cell_ID]['user_cnt']
                Network_Power_4G[time_slot][BS_ID][Cell_ID]['down_flow'] = Network_Power_4G_Sleep[time_slot][BS_ID][Cell_ID]['down_flow']
                
                Power_Element = Power_Element + Power_RRU
                
            for Cell_ID in Network_Power_4G_Sleep[time_slot][BS_ID]:
                Network_Power_4G[time_slot][BS_ID][Cell_ID]['Power_Element'] = Power_Element
    
    np.save('Network_Power_4G'+Date+'.npy', Network_Power_4G)
    np.save('Network_Power_5G'+Date+'.npy', Network_Power_5G)