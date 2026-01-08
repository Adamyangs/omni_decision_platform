# Please install the following packages
# pip install pandas
# pip install seaborn
# pip install SciencePlots
# Note: you also need Latex installed on your machine, see here for details:
#   https://github.com/garrettj403/SciencePlots/wiki/FAQ#installing-latex

import os
import json

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('classic')
plt.style.use('science')

def load_json(
    f_name: str
):
    """
        Loads relevant fields in metrics.json and makes dataframe.
    """
    with open(f_name) as f:
        raw_data = json.load(f)

    test_return_mean = pd.DataFrame.from_dict(raw_data["test_return_mean"])['values']
    test_return_mean_T = pd.DataFrame.from_dict(raw_data["test_return_mean"])['steps']

    # battle_won_mean = raw_data["battle_won_mean"]
    # battle_won_mean_T = raw_data["battle_won_mean_T"]

    return pd.DataFrame(
        {
            'steps':test_return_mean_T,
            'values':test_return_mean
        })

def load_experiment(
    experiment_directory: str
):      
    """
        Makes Pandas dataframe for seaborn to use for plotting.
    """
    df = None
    runs = os.listdir(experiment_directory)
    if '_sources' in runs:
        runs.remove('_sources')

    for i, run in enumerate(runs):
        path = os.path.join(experiment_directory, run, 'metrics.json')
        run_df = load_json(path)

        step_size = run_df['steps'].iloc[1] - run_df['steps'].iloc[0] # Make col 'steps' be uniform across runs
        run_df['steps'] = np.arange(run_df['steps'].iloc[0], run_df['steps'].iloc[-1], step_size)
        
        run_df['i'] = i
        df = pd.concat([df, run_df], ignore_index=True)

    return df


def plot(experiments):
    # fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.725))
    fig, ax = plt.subplots(1, 1)

    steps = None
    for idx, experiment in enumerate(experiments):
        data = load_experiment(experiment[0])
        sns.lineplot(x=data['steps'] / 100000, y=data['values'], ax=ax, label=experiment[1])

    ax.set_title("LBF 3p 5f")
    ax.set_ylabel('Evaluation Return')
    ax.set_xlabel('Steps')
    plt.savefig('comparison_18.png', format='png')



if __name__ == '__main__':
    david_idea = \
        zip(
            [          
                "./LBF/nips_10x10_3p_5f/ippo/Foraging-10x10-3p-5f-v2",
                # "./LBF/nips_10x10_3p_5f/mappo/Foraging-10x10-3p-5f-v2",
                # "./LBF/nips_10x10_3p_5f/c_mappo/Foraging-10x10-3p-5f-v2",
                # "./LBF/nips_10x10_3p_5f/c_ippo/Foraging-10x10-3p-5f-v2",

            ],
            [
                "IPPO",
                # "MAPPO",
                # "C MAPPO",
                # "C IPPO",
            ]
        )

    plot(david_idea)
