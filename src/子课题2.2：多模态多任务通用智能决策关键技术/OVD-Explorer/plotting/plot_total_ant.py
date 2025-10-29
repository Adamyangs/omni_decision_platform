import pandas as pd
import os
import os.path as osp
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("Agg")
import numpy as np
import copy
import csv
import time
from main import get_cmd_args, get_log_dir
from utils.env_utils import domain_to_epoch
time.sleep(2)
plt.rcParams['font.size'] = '18'
t = "2"
# define dicts and corresponding args
DOMAINS = ['ant']
DATASIZE = 2500
VERSION = f"1010{t}" 
RLKIT_BASE_LOG_DIR_BASELINE = RLKIT_BASE_LOG_DIR_ALGO = './data'

def get_plot_title(args):

    if FORMAL_FIG:
        title = args.env
    else:
        title = '\n'.join([
            args.env,
            f'num_run: {NUM_RUN}', '---'

            f'beta_UB: {args.beta_UB}',
            f'delta: {args.delta}',
            f'train/env step ratio: {int(args.num_trains_per_train_loop / args.num_expl_steps_per_train_loop)}'
        ])
    return title

def set_attr_with_dict(target, source_dict):

    for k, v in source_dict.items():
        setattr(target, k, v)

    return target

def get_tick_space(domain):
    return 500

def get_one_domain_one_run_res(domain, seed=0, hyper_params=None, path=None):

    args = get_cmd_args()

    args.domain = domain
    if path is None:
        args.base_log_dir = RLKIT_BASE_LOG_DIR
        args.seed = seed
        for k, v in hyper_params.items():
            setattr(args, k, v)

        res_path = get_log_dir(args)

        csv_path = osp.join(
            res_path, 'progress.csv'
        )
    else:
        csv_path = osp.join(
            path, 'progress.csv'
        )

    values = []

    with open(csv_path, 'r') as csv_file:
        reader = csv.reader(csv_file)

        col_names = next(reader)

        # Assume that the index of epoch is the last one
        # Not sure why the csv file is missing one col header
        # epoch_col_idx = col_names.index('Epoch')
        epoch_col_idx = -1
        val_col_idx = col_names.index('remote_evaluation/Average Returns')

        # print(csv_path)
        for row in reader:


            # If this equals Epoch, it means the header
            # was written to the csv file again
            # and we reset everything
            if row[epoch_col_idx] == 'Epoch':
                values = []
                continue

            epoch = int(row[epoch_col_idx])
            val = float(row[val_col_idx])

            # We need to check if the row contains the values
            # of the correct epoch
            # because after reloading from checkpoint,
            # we are writing the result to the same csv file
            if epoch == len(values):
                values.append(val)
            else:
                # print(
                    # f'Reloaded row found at epoch {len(values), epoch} found for', domain, seed, hyper_params)
                pass

    # Reshape the return value
    # to accomodate downstream api
    values = np.array(values)
    values = np.expand_dims(values, axis=-1)

    return values

def get_one_domain_all_run_res(domain, run_idxes=None, hyper_params=None, path=None):

    results = []
    if path is None:
        for idx in run_idxes:
            res = get_one_domain_one_run_res(domain, seed=idx, hyper_params=hyper_params)
            results.append(res)
    else:
        for p in path:
            res = get_one_domain_one_run_res(domain, path=p)
            results.append(res)
    print([len(col) for col in results])
    min_rows = min([len(col) for col in results])
    results = [col[0:min_rows] for col in results]

    results = np.hstack(results)

    return results

def smooth_results(results, smoothing_window=100):
    smoothed = np.zeros((results.shape[0], results.shape[1]))

    for idx in range(len(smoothed)):

        if idx == 0:
            smoothed[idx] = results[idx]
            continue

        start_idx = max(0, idx - smoothing_window)

        smoothed[idx] = np.mean(results[start_idx:idx], axis=0)

    return smoothed

def plot(values, label, color=[0, 0, 1, 1]):
    if DATASIZE != -1:
        values = values[:DATASIZE]
    mean = np.mean(values, axis=1)
    std = np.std(values, axis=1)

    x_vals = np.arange(len(mean))

    blur = copy.deepcopy(color)
    blur[-1] = 0.1

    plt.plot(x_vals, mean, label=label, color=color)
    plt.fill_between(x_vals, mean - std/2, mean + std/2, color=blur)
    index = 2499 if len(x_vals) >= 2500 else -1
    print(x_vals[index], label, mean[index], std[index], np.max(values, 0).mean())

    plt.legend()

pathes = [
    dict(
        path=[
            "./data/master/num_expl_steps_per_train_loop_1000_num_trains_per_train_loop_1000_beta_UB_0.0_delta_0.0_alpha_0.0_alpha_2_0.1_beta_0.0_sigma_0.0_nor_3.0_z_0.0_ee_False_version_112590/ant/seed_0",
            "./data/master/num_expl_steps_per_train_loop_1000_num_trains_per_train_loop_1000_beta_UB_0.0_delta_0.0_alpha_0.0_alpha_2_0.1_beta_0.0_sigma_0.0_nor_3.0_z_0.0_ee_False_version_112590/ant/seed_1",
            "./data/master/num_expl_steps_per_train_loop_1000_num_trains_per_train_loop_1000_beta_UB_0.0_delta_0.0_alpha_0.0_alpha_2_0.1_beta_0.0_sigma_0.0_nor_3.0_z_0.0_ee_False_version_112590/ant/seed_2",
            "./data/master/num_expl_steps_per_train_loop_1000_num_trains_per_train_loop_1000_beta_UB_0.0_delta_0.0_alpha_0.0_alpha_2_0.1_beta_0.0_sigma_0.0_nor_3.0_z_0.0_ee_False_version_112590/ant/seed_3",
            "./data/master/num_expl_steps_per_train_loop_1000_num_trains_per_train_loop_1000_beta_UB_0.0_delta_0.0_alpha_0.0_alpha_2_0.1_beta_0.0_sigma_0.0_nor_3.0_z_0.0_ee_False_version_112590/ant/seed_4",
        ],
        label="SAC",
        color=[0.6, 0.5, 0.1, 1],
    ),
    dict(
        path=[
            "./data/master/num_expl_steps_per_train_loop_1000_num_trains_per_train_loop_1000_beta_UB_0.0_delta_0.0_alpha_0.0_alpha_2_0.1_beta_0.0_sigma_0.0_nor_3.0_z_0.0_ee_False_version_11250/ant/seed_0",
            "./data/master/num_expl_steps_per_train_loop_1000_num_trains_per_train_loop_1000_beta_UB_0.0_delta_0.0_alpha_0.0_alpha_2_0.1_beta_0.0_sigma_0.0_nor_3.0_z_0.0_ee_False_version_11250/ant/seed_1",
            "./data/master/num_expl_steps_per_train_loop_1000_num_trains_per_train_loop_1000_beta_UB_0.0_delta_0.0_alpha_0.0_alpha_2_0.1_beta_0.0_sigma_0.0_nor_3.0_z_0.0_ee_False_version_11250/ant/seed_2",
            "./data/master/num_expl_steps_per_train_loop_1000_num_trains_per_train_loop_1000_beta_UB_0.0_delta_0.0_alpha_0.0_alpha_2_0.1_beta_0.0_sigma_0.0_nor_3.0_z_0.0_ee_False_version_11250/ant/seed_3",
            "./data/master/num_expl_steps_per_train_loop_1000_num_trains_per_train_loop_1000_beta_UB_0.0_delta_0.0_alpha_0.0_alpha_2_0.1_beta_0.0_sigma_0.0_nor_3.0_z_0.0_ee_False_version_11250/ant/seed_4",
        ],
        label="DSAC",
        color=[0, 1, 0, 1],
    ),
    dict(
        path=[
            "./data/master/num_expl_steps_per_train_loop_1000_num_trains_per_train_loop_1000_beta_UB_4.66_delta_23.53_alpha_0.0_alpha_2_0.1_beta_0.0_sigma_0.0_nor_3.0_z_0.0_ee_False_version_11250/ant/seed_0",
            "./data/master/num_expl_steps_per_train_loop_1000_num_trains_per_train_loop_1000_beta_UB_4.66_delta_23.53_alpha_0.0_alpha_2_0.1_beta_0.0_sigma_0.0_nor_3.0_z_0.0_ee_False_version_11250/ant/seed_1",
            "./data/master/num_expl_steps_per_train_loop_1000_num_trains_per_train_loop_1000_beta_UB_4.66_delta_23.53_alpha_0.0_alpha_2_0.1_beta_0.0_sigma_0.0_nor_3.0_z_0.0_ee_False_version_11250/ant/seed_2",
            "./data/master/num_expl_steps_per_train_loop_1000_num_trains_per_train_loop_1000_beta_UB_4.66_delta_23.53_alpha_0.0_alpha_2_0.1_beta_0.0_sigma_0.0_nor_3.0_z_0.0_ee_False_version_11250/ant/seed_3",
            "./data/master/num_expl_steps_per_train_loop_1000_num_trains_per_train_loop_1000_beta_UB_4.66_delta_23.53_alpha_0.0_alpha_2_0.1_beta_0.0_sigma_0.0_nor_3.0_z_0.0_ee_False_version_11250/ant/seed_4",
        ],
        label="DOAC",
        color=[0, 1, 1, 1],
    ),
    dict(
        path=[
            "./data/master/num_expl_steps_per_train_loop_1000_num_trains_per_train_loop_1000_beta_UB_0.0_delta_0.0_alpha_0.05_alpha_2_0.1_beta_3.2_sigma_0.0_nor_3.0_z_0.5_ee_True_version_11253/ant/seed_0",
            "./data/master/num_expl_steps_per_train_loop_1000_num_trains_per_train_loop_1000_beta_UB_0.0_delta_0.0_alpha_0.05_alpha_2_0.1_beta_3.2_sigma_0.0_nor_3.0_z_0.5_ee_True_version_11253/ant/seed_1",
            "./data/master/num_expl_steps_per_train_loop_1000_num_trains_per_train_loop_1000_beta_UB_0.0_delta_0.0_alpha_0.05_alpha_2_0.1_beta_3.2_sigma_0.0_nor_3.0_z_0.5_ee_True_version_11253/ant/seed_2",
            "./data/master/num_expl_steps_per_train_loop_1000_num_trains_per_train_loop_1000_beta_UB_0.0_delta_0.0_alpha_0.05_alpha_2_0.1_beta_3.2_sigma_0.0_nor_3.0_z_0.5_ee_True_version_11253/ant/seed_3",
            "./data/master/num_expl_steps_per_train_loop_1000_num_trains_per_train_loop_1000_beta_UB_0.0_delta_0.0_alpha_0.05_alpha_2_0.1_beta_3.2_sigma_0.0_nor_3.0_z_0.5_ee_True_version_11253/ant/seed_4",
        ],
        label="OVDE_G",
        color=[1, 0, 0, 1],
    ),
    dict(
        path=[
            "./data/master/num_expl_steps_per_train_loop_1000_num_trains_per_train_loop_1000_beta_UB_0.0_delta_0.0_alpha_0.05_alpha_2_0.1_beta_3.2_sigma_0.0_nor_3.0_z_0.5_ee_True_version_11254/ant/seed_0",
            "./data/master/num_expl_steps_per_train_loop_1000_num_trains_per_train_loop_1000_beta_UB_0.0_delta_0.0_alpha_0.05_alpha_2_0.1_beta_3.2_sigma_0.0_nor_3.0_z_0.5_ee_True_version_11254/ant/seed_1",
            "./data/master/num_expl_steps_per_train_loop_1000_num_trains_per_train_loop_1000_beta_UB_0.0_delta_0.0_alpha_0.05_alpha_2_0.1_beta_3.2_sigma_0.0_nor_3.0_z_0.5_ee_True_version_11254/ant/seed_2",
            "./data/master/num_expl_steps_per_train_loop_1000_num_trains_per_train_loop_1000_beta_UB_0.0_delta_0.0_alpha_0.05_alpha_2_0.1_beta_3.2_sigma_0.0_nor_3.0_z_0.5_ee_True_version_11254/ant/seed_3",
            "./data/master/num_expl_steps_per_train_loop_1000_num_trains_per_train_loop_1000_beta_UB_0.0_delta_0.0_alpha_0.05_alpha_2_0.1_beta_3.2_sigma_0.0_nor_3.0_z_0.5_ee_True_version_11254/ant/seed_4",
        ],
        label="OVDE_Q",
        color=[0, 0, 1, 1],
    ),
]

# pathes = None
dict_num = 0
for domain in DOMAINS:
    plt.clf()
    i = 0
    if pathes is None:
        for hyper_params in all_hyper_params_dict:
            """
            Set up
            """
            # We need to do this so that jupyter notebook
            # works with argparse
            import sys
            sys.argv = ['']
            del sys

            args = get_cmd_args()

            set_attr_with_dict(args, hyper_params)

            args.env = f'{domain}-v2'

            relative_log_dir = get_log_dir(
                args, should_include_base_log_dir=False, should_include_seed=False, should_include_domain=False)

            graph_base_path = osp.join(
                RLKIT_BASE_LOG_DIR_ALGO, 'plot', relative_log_dir)

            os.makedirs(graph_base_path, exist_ok=True)

            """
            Obtain Result
            """
            RLKIT_BASE_LOG_DIR = RLKIT_BASE_LOG_DIR_ALGO
            results = get_one_domain_all_run_res(domain, run_idxes = RUN_IDXES[dict_num], hyper_params=hyper_params)
            results = smooth_results(results)
        

            # if domain == 'ant' and FORMAL_FIG:
            #     mean = np.mean(results, axis=1)
            #     x_vals = np.arange(len(mean))

            #     # This is the index where OAC has
            #     # the same performance as SAC with 10 million steps
            #     # Plus 200 so that we are not overstating our claim
            #     magic_idx = np.argmax(mean > 8000) + 300

            #     plt.plot(8000 * np.ones(magic_idx), linestyle='--',
            #              color=[0, 0, 1, 1], linewidth=3, label='Soft Actor Critic 10 million steps performance')
            #     plt.vlines(x=magic_idx,
            #                ymin=0, ymax=8000, linestyle='--',
            #                color=[0, 0, 1, 1],)

            """
            Plot result
            """

            plot(results, label=hyper_params["label"],
                color=hyper_params["color"])
            print("Finished plotting ", hyper_params["label"])

            RLKIT_BASE_LOG_DIR = RLKIT_BASE_LOG_DIR_BASELINE
            dict_num += 1
            i += 1
    else:
        for path in pathes:
            results = get_one_domain_all_run_res(domain, path=path["path"])
            results = smooth_results(results)
            """
            Plot result
            """
            print(results[-1])
            plot(results, label=path["label"],
                color=path["color"])
            # print("Finished plotting ", path["label"])

            RLKIT_BASE_LOG_DIR = RLKIT_BASE_LOG_DIR_BASELINE
            dict_num += 1
            i += 1


    # plt.title("ant-v2")

    plt.ylabel('Average Return')

    xticks = np.arange(0, domain_to_epoch(
            domain) + 1, get_tick_space(domain))

    plt.xticks(xticks, xticks)

    plt.xlabel('Number of Training Epoch')
    # plt.legend(loc='upper left')
    plt.legend(loc='lower right')

    fig_path = osp.join(
            "logs/fig", f'ant.pdf')

    plt.savefig(fig_path, bbox_inches='tight')

    print(f'Saved fig at {fig_path}')

    print('Finish plotting')
