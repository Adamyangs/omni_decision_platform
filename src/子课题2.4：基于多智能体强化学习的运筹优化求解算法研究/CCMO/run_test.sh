#! /bin/bash

bash run_code.sh 3 3m qmix_act_rew checkpoint_path="./results/model/qmix_act__2026-01-05_15-50-40",evaluate=True,wandb_enabled=False,save_replay=False,runner="episode",batch_size_run=1,test_nepisode=32