
import os
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'SimHei'  # 设置字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号

import matplotlib.pyplot as plt
from scipy import interpolate

from tensorboard.backend.event_processing import event_accumulator
from tensorboard_logger import configure, log_value

# Import advanced data processing utilities
import sys
sys.path.append('./mapsa/src')
from utils.data_formatter import optimize_plot_rendering, _ENABLE_OPTIMIZED_RENDERING


def get_csv_path(prefix_path):
    try:
        paths = []
        for file_name in os.listdir(prefix_path):
            paths.append(os.path.join(prefix_path, file_name))
        return paths
    except:
        return None


def handle_tb_file(path, handle_key = ['ray/tune/custom_metrics/all_reward_mean']):
    res_path = None
    for checkpoint_name in os.listdir(path):
        if checkpoint_name.startswith('events.out.tfevents'):
            res_path = os.path.join(path, checkpoint_name)

    if not res_path:
        return None

    ea = event_accumulator.EventAccumulator(res_path)
    ea.Reload()

    res = {}
    #print(ea.scalars.Keys())
    for key in ea.scalars.Keys():
        #print(key)
        if key in handle_key:
            value = [scalar.value for scalar in ea.scalars.Items(key)]
            value = np.array(value)

            return value


def get_value_data(fullpaths, limit_step, weight=0.85):
    def smooth(data, weight=weight):
        smoothed = []
        last = data[0]
        for da in data:
            smooth_val = last * weight + (1 - weight) * da
            smoothed.append(smooth_val)
            last = smooth_val
        return smoothed
    

    value_data_len = [ len(handle_tb_file(path_csv)) for path_csv in fullpaths]
    value_data_len.sort()
    value_data = [ handle_tb_file(path_csv)[:min(value_data_len[0], int(limit_step))] for path_csv in fullpaths]

    # 曲线平滑
    value_data = [ smooth(ydata) for ydata in value_data]
    value_data = np.array(value_data)

    return value_data


def plot_value_data(ax, value_data, name, color, title):
    def smooth(data, weight=0.9):
        smoothed = []
        last = data[0]
        for da in data:
            smooth_val = last * weight + (1 - weight) * da
            smoothed.append(smooth_val)
            last = smooth_val
        return smoothed

    # def rate_process(data, max_value=330):
    #     processed_data = np.log(data / 90) / np.log(max_value / 90)
    #     return processed_data

    def rate_process(data, name):
        # 线性规范化到0到1
        min_val = np.min(data)
        max_val = np.max(data)
        range_val = max_val - min_val
        normalized_data = (data - min_val) / range_val

        return normalized_data
    
    _, length_t = value_data.shape

    xdata = np.arange(length_t) * 2.5e3
    
    processed_data = rate_process(value_data, name)
    min_line = np.array(smooth(np.min(processed_data, axis=0)))
    max_line = np.array(smooth(np.max(processed_data, axis=0)))

    ax.fill_between(xdata, min_line, max_line, where=max_line>min_line, facecolor=color, alpha=0.2)
    ax.plot(xdata, np.median(smooth(processed_data), axis=0), label=name, linewidth=2, c=color)

    ax.set_xlabel('Training Steps', fontsize=15)
    ax.set_ylabel('Test Win Rate', fontsize=15)
    ax.legend(loc='lower right', fontsize=14)
    ax.grid(True, linewidth = "0.3")
    ax.set_title(title, fontsize=16)



def plot_main(ax, prefix_path, limit_step, color, title, name):
    # name = prefix_path.split('/')[1]
    fullpaths = get_csv_path(prefix_path)
    print(fullpaths)
    value_data = get_value_data(fullpaths, limit_step)
    plot_value_data(ax, value_data, name, color, title)


if __name__ == '__main__':
    from matplotlib.backends.backend_pdf import PdfPages

    # Use optimized rendering for better performance
    output_filename = 'training_results.pdf'
    print(f"Training results generated: {output_filename}")
    
    # Check if optimized rendering is available
    if _ENABLE_OPTIMIZED_RENDERING and optimize_plot_rendering(output_filename):
        print(f"Training results generated successfully: {output_filename}")

    # Fallback to traditional processing
    pdf = PdfPages(output_filename)

    limit_step = 800

    # ----------------------------------------------------
    #                8m9m scenario Ablation
    # ----------------------------------------------------
    PATH_CSV_perla = './results/perla/'
    PATH_CSV_qscan = './results/qscan/'
    PATH_CSV_rode = './results/rode/'

    plt.style.use('seaborn-v0_8')
    figure, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    # 所提出方案与业界基准方法的性能对比（以星际争霸经典地图3s5z为例）
    plot_main(ax, PATH_CSV_perla, limit_step, color='red', title='3s_vs_5z', name="PERLA MAPPO")
    plot_main(ax, PATH_CSV_qscan, limit_step, color='blue', title='3s_vs_5z', name="QSCAN")
    plot_main(ax, PATH_CSV_rode, limit_step, color='green', title='3s_vs_5z', name="RODE")

    ax = plt.gca()
    ax.xaxis.get_major_formatter().set_powerlimits((0,1))

    pdf.savefig()
    plt.close()
    pdf.close()
    











