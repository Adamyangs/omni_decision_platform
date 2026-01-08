#!/bin/bash

maps=(Foraging-5x5-2p-1f-v2 Foraging-5x5-2p-1f-coop-v2 Foraging-grid-2s-5x5-2p-1f-coop-v2 Foraging-5x5-2p-2f-v2 Foraging-8x8-3p-3f-v2 Foraging-8x8-2p-2f-coop-v2 Foraging-10x10-5p-1f-coop-v2 Foraging-10x10-3p-5f-v2 Foraging-grid-2s-10x10-3p-3f-v2 Foraging-10x10-5p-3f-v2 Foraging-15x15-3p-5f-v2 Foraging-15x15-5p-5f-v2 Foraging-15x15-5p-3f-v2)

for map in "${maps[@]}"
do
    for seed in {1..3}
    do
        # echo "Running IPPO on ${map} ${hidden_dim} ${learning_rate} ${reward_standardisation} ${target_update} ${n_step}"
        python3 src/main.py --config=ippo --env-config=gymma with\
            name="ippo"\
            t_max=2055000\
            hidden_dim=128\
            lr=0.0003\
            use_rnn=False\
            target_update_interval_or_tau=200\
            q_nstep=5\
            env_args.key=${map}\
            seed=${seed} &

        # echo "Running MAPPO on ${map} ${hidden_dim} ${learning_rate} ${reward_standardisation} ${target_update} ${n_step}"
        python3 src/main.py --config=mappo --env-config=gymma with\
            name="mappo"\
            t_max=2055000\
            hidden_dim=128\
            lr=0.0003\
            use_rnn=False\
            q_nstep=5\
            env_args.key=${map}\
            seed=${seed} &
        wait 
    done 
done 