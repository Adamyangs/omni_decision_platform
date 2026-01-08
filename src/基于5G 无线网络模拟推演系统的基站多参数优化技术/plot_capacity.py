import numpy as np
import matplotlib.pyplot as plt
from environment import BaseStationSleepingProblem
from global_parameters import agent_path

colors = ["#7aa0c4", "#8bcd50", "#ca82e1", "#df9f53", "#64b9a1", "#745ea6", "#db7e76"]
color_id = 0
DEBUG=0

def load_data_file(horizon=7*48):
    # env = BaseStationSleepingProblem(
    #     network_power_5g_file='5G/Network_Power2022-05-26.npy',
    #     network_power_4g_file='4G/Network_Power2022-05-26.npy',
    #     unsatisfied_penalty_type='action_constraint',
    # )

    flow_power = np.load('output/{}/flow_powers.npy'.format(agent_path), allow_pickle=True)[()]
    flow = flow_power['total_flow']
    cell_capacity = np.load('input/cell_capacity.npy')[()]
    cell_sids = np.load('input/cell_sids.npy')[()] # [56524]
    cell_sid_to_id = {sid:i for i, sid in enumerate(cell_sids)}
    bs_sids = np.load('input/bs_sids.npy')[()]    # [14243]
    bs_sid_to_id = {sid:i for i, sid in enumerate(bs_sids)}

    # rl_info
    bbu_rru_power = np.load('output/{}/bbu_rru_powers.npy'.format(agent_path), allow_pickle=True)[()]  # sum=1530396801.2035217
    br_power = np.sum([bbu_rru_power[k] for k in bbu_rru_power], axis=0)
    rl_bbu, rl_rru = br_power[:, 1], br_power[:, 0]
    air_power = np.load('output/{}/Pa_RL.npy'.format(agent_path), allow_pickle=True)[()]
    rl_air = np.sum([air_power[k] for k in air_power], axis=0)
    bs_control = np.load('output/{}/bs_control.npy'.format(agent_path), allow_pickle=True)[()]
    rl_bs_num = np.sum(np.array([bs_control[k] for k in bs_control])[..., 0], axis=0)
    cell_control = np.load('output/{}/cell_control.npy'.format(agent_path), allow_pickle=True)[()]
    cell_control = np.array([cell_control[k] for k in cell_control])[..., 0]
    rl_capacity = np.matmul(cell_capacity, cell_control)
    rl_cell_num = np.sum(cell_control, axis=0)

    # mobicom info
    tmp = np.load('output/data_onoff_PeAs_45G.npy', allow_pickle=True)[()]
    mobicom_bs_num = [len(tmp[w][h]['Active_BS_num']) for w in tmp for h in tmp[w]]
    bs_control = np.zeros([bs_sids.size, 48*7])
    cell_control = np.zeros([cell_sids.size, 48*7])
    mobicom_rru = np.zeros([48*7])
    mobicom_capacity = np.zeros([48*7])

    for w in tmp:
        for h in tmp[w]:
            for grid in list(tmp[w][h].values())[:-1]:
                for cell_sid in grid:
                    cell_control[cell_sid_to_id[cell_sid], w*48+h] = grid[cell_sid]['Status']
                    mobicom_rru[w*48+h] += grid[cell_sid]['Status'] * grid[cell_sid]['Power']
                    mobicom_capacity[w*48+h] += grid[cell_sid]['Status'] * grid[cell_sid]['Capacity']
            # for sid in tmp[w][h]['Active_BS_num']:
            #     bs_control[bs_sid_to_id[sid], w*48+h] = 1
    m_cell_num = np.sum(cell_control, axis=0)
    m_air_power = np.load('output/Pa_45G.npy', allow_pickle=True)[()]
    mobicom_air = np.sum([m_air_power[k] for k in m_air_power], axis=0)
    m_br_power = np.load('output/Pe_45G.npy', allow_pickle=True)[()]           # sum=1779648534.0894184
    m_br_power = np.sum([m_br_power[k] for k in m_br_power], axis=0)
    mobicom_bbu = m_br_power - mobicom_rru

    rl_info = {
        'capacity': rl_capacity,
        'bs_num': rl_bs_num,
        'cell_num': rl_cell_num,
        'rru_power': rl_rru,
        'bbu_power': rl_bbu,
        'air_power': rl_air,
        'name': 'RL Method'
    }
    mobicom_info = {
        'capacity': mobicom_capacity,
        'bs_num': mobicom_bs_num,
        'cell_num': m_cell_num,
        'rru_power': mobicom_rru,
        'bbu_power': mobicom_bbu,
        'air_power': mobicom_air,
        'name': 'Mobicom Method'
    }
    return flow, mobicom_info, rl_info


def plot(flow, infos, cid):
    org_cid = cid

    # flow and capacity
    cid = org_cid
    fig, ax = plt.subplots()
    # make a plot
    ax.plot(range(48*7), flow, color=colors[cid], linestyle=':', label='traffic')
    cid += 1
    for info in infos:
        ax.plot(range(48*7), info['capacity'], color=colors[cid], linestyle='-', label=info['name'])
        cid += 1
    ax.legend(loc='center right')
    # set x-axis label
    ax.set_xlabel("Time(0.5h)", fontsize=14)
    # set y-axis label
    ax.set_ylabel("Traffic or Capacity(kByte)", fontsize=14)
    ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    plt.show()
    # save the plot as a file
    fig.savefig('output/{}/figure/flow_capacity.pdf'.format(agent_path), format='pdf', dpi=100, bbox_inches='tight')

    # cell_num
    cid = org_cid+1
    fig, ax = plt.subplots()
    # make a plot
    for info in infos:
        ax.plot(range(48*7), info['cell_num'], color=colors[cid], linestyle='-', label=info['name'])
        cid += 1
    ax.legend()
    # set x-axis label
    ax.set_xlabel("Time(0.5h)", fontsize=14)
    # set y-axis label
    ax.set_ylabel("Number Of Opened Cells", fontsize=14)
    ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    plt.show()
    # save the plot as a file
    fig.savefig('output/{}/figure/cell_num.pdf'.format(agent_path), format='pdf', dpi=100, bbox_inches='tight')

    # bs_num
    cid = org_cid+1
    fig, ax = plt.subplots()
    # make a plot
    for info in infos:
        ax.plot(range(48*7), info['cell_num']/info['bs_num'], color=colors[cid], linestyle='-', label=info['name'])
        cid += 1
    ax.legend()
    # set x-axis label
    ax.set_xlabel("Time(0.5h)", fontsize=14)
    # set y-axis label
    ax.set_ylabel("Number Of Opened Base Stations", fontsize=14)
    ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    plt.show()
    # save the plot as a file
    fig.savefig('output/{}/figure/bs_num.pdf'.format(agent_path), format='pdf', dpi=100, bbox_inches='tight')

    # air_power
    cid = org_cid+1
    fig, ax = plt.subplots()
    # make a plot
    for info in infos:
        ax.plot(range(48*7), info['air_power'], color=colors[cid], linestyle='-', label=info['name'])
        cid += 1
    ax.legend()
    # set x-axis label
    ax.set_xlabel("Time(0.5h)", fontsize=14)
    # set y-axis label
    ax.set_ylabel("Air Conditioner Power(W)", fontsize=14)
    ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    plt.show()
    # save the plot as a file
    fig.savefig('output/{}/figure/air_power.pdf'.format(agent_path), format='pdf', dpi=100, bbox_inches='tight')

    # rru_power
    cid = org_cid+1
    fig, ax = plt.subplots()
    # make a plot
    for info in infos:
        ax.plot(range(48*7), info['rru_power'], color=colors[cid], linestyle='-', label=info['name'])
        cid += 1
    ax.legend()
    # set x-axis label
    ax.set_xlabel("Time(0.5h)", fontsize=14)
    # set y-axis label
    ax.set_ylabel("RRU Power(W)", fontsize=14)
    ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    plt.show()
    # save the plot as a file
    fig.savefig('output/{}/figure/rru_power.pdf'.format(agent_path), format='pdf', dpi=100, bbox_inches='tight')

    # bbu_power
    cid = org_cid+1
    fig, ax = plt.subplots()
    # make a plot
    for info in infos:
        ax.plot(range(48*7), info['bbu_power'], color=colors[cid], linestyle='-', label=info['name'])
        cid += 1
    ax.legend()
    # set x-axis label
    ax.set_xlabel("Time(0.5h)", fontsize=14)
    # set y-axis label
    ax.set_ylabel("BBU Power(W)", fontsize=14)
    ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    plt.show()
    # save the plot as a file
    fig.savefig('output/{}/figure/bbu_power.pdf'.format(agent_path), format='pdf', dpi=100, bbox_inches='tight')


if __name__ == '__main__':
    flow, mobicom_info, rl_info = load_data_file()
    np.save('output/{}/figure/flow_and_power.npy'.format(agent_path), [flow, mobicom_info, rl_info])
    flow, mobicom_info, rl_info = np.load('output/{}/figure/flow_and_power.npy'.format(agent_path), allow_pickle=True)[()]

    plot(flow, [mobicom_info, rl_info], color_id)

