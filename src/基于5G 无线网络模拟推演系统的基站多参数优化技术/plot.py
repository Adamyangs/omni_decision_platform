import numpy as np
import matplotlib.pyplot as plt
from global_parameters import agent_path

colors = ["#7aa0c4", "#ca82e1", "#8bcd50", "#df9f53", "#64b9a1", "#745ea6", "#db7e76"]
color_id = 0

def load_data_file():

    flow_power = np.load('output/{}/flow_powers.npy'.format(agent_path), allow_pickle=True)[()]
    flow = flow_power['total_flow']
    rl_power = flow_power['total_powers']
    org_power = flow_power['original_total_powers']

    bbu_rru_power = np.load('output/{}/bbu_rru_powers.npy'.format(agent_path), allow_pickle=True)[()]  # sum=1530396801.2035217
    br_power = np.sum([bbu_rru_power[k] for k in bbu_rru_power], axis=(0, -1))
    # air_power = np.load('output/{}/Pa_RL.npy'.format(agent_path), allow_pickle=True)[()]
    # air_power = np.sum([air_power[k] for k in air_power], axis=0)
    rl_power = br_power * 2 #+ air_power

    m_air_power = np.load('output/Pa_45G.npy', allow_pickle=True)[()]
    m_air_power = np.sum([m_air_power[k] for k in m_air_power], axis=0)
    m_br_power = np.load('output/Pe_45G.npy', allow_pickle=True)[()]           # sum=1779648534.0894184
    m_br_power = np.sum([m_br_power[k] for k in m_br_power], axis=0)
    m_power = m_air_power + m_br_power

    print(1-sum(rl_power)/sum(org_power), 1-sum(m_power)/sum(org_power), sum(org_power))

    return flow, rl_power, m_power, org_power


def twin_plot(flow, powers, cid):
    # create figure and axis objects with subplots()
    fig, ax = plt.subplots()
    # make a plot
    ax.plot(range(48*7), flow, color=colors[cid], linestyle='dotted', label='traffic')
    cid += 1
    ax.legend(loc='center left')
    ax.set_ylim([0, 1e11])
    # set x-axis label
    ax.set_xlabel("Time(0.5h)", fontsize=14)
    # set y-axis label
    ax.set_ylabel("Traffic(kByte)",
                  fontsize=14)

    # twin object for two different y-axis on the sample plot
    ax2 = ax.twinx()
    for k in powers:
        # make a plot with different y-axis using second axis object
        ax2.plot(range(48*7), powers[k], color=colors[cid], label=k)
        cid += 1
    ax2.legend()
    ax2.set_ylabel("Power(W)", fontsize=14)
    plt.show()
    # save the plot as a file
    fig.savefig('output/{}/figure/flow_power.pdf'.format(agent_path),
                format='pdf',
                dpi=100,
                bbox_inches='tight')

if __name__ == '__main__':
    flow, rl_power, m_power, org_power = load_data_file()
    twin_plot(flow, {'our method': rl_power, 'minimal number': m_power, 'original':org_power}, color_id)

