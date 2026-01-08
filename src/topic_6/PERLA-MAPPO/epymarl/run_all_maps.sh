#!/bin/bash

easy_maps=(3m 8m 25m 2s3z 1c3s5z 10m_vs_11m 2m_vs_1z 2s_vs_1sc 3s_vs_3z 3s_vs_4)
hard_maps=(so_many_baneling MMM 3s5z 5m_vs_6m 8m_vs_9m 3s_vs_5z 2c_vs_64zg bane_vs_bane)
super_hard_maps=(corridor MMM2 6h_vs_8z 27m_vs_30m 3s5z_vs_3s6z)

for map in "${easy_maps[@]}"
do
    for seed in {1..3}
    do
        # echo "Running IPPO on ${map} ${hidden_dim} ${learning_rate} ${reward_standardisation} ${target_update} ${n_step}"
        python3 src/main.py --config=ippo --env-config=sc2 with\
            name="ippo"\
            t_max=1555000\
            hidden_dim=128\
            env_args.map_name=${map}\
            seed=${seed}

        # echo "Running MAPPO on ${map} ${hidden_dim} ${learning_rate} ${reward_standardisation} ${target_update} ${n_step}"
        python3 src/main.py --config=mappo --env-config=sc2 with\
            name="mappo"\
            t_max=1555000\
            hidden_dim=64\
            env_args.map_name=${map}\
            seed=${seed}
    done 
done 

for map in "${hard_maps[@]}"
do
    for seed in {1..3}
    do
        # echo "Running IPPO on ${map} ${hidden_dim} ${learning_rate} ${reward_standardisation} ${target_update} ${n_step}"
        python3 src/main.py --config=ippo --env-config=sc2 with\
            name="ippo"\
            t_max=2555000\
            hidden_dim=128\
            env_args.map_name=${map}\
            seed=${seed}

        # echo "Running MAPPO on ${map} ${hidden_dim} ${learning_rate} ${reward_standardisation} ${target_update} ${n_step}"
        python3 src/main.py --config=mappo --env-config=sc2 with\
            name="mappo"\
            t_max=2555000\
            hidden_dim=64\
            env_args.map_name=${map}\
            seed=${seed}
    done 
done 

for map in "${super_hard_maps[@]}"
do
    for seed in {1..3}
    do
        # echo "Running IPPO on ${map} ${hidden_dim} ${learning_rate} ${reward_standardisation} ${target_update} ${n_step}"
        python3 src/main.py --config=ippo --env-config=sc2 with\
            name="ippo"\
            t_max=5555000\
            hidden_dim=128\
            env_args.map_name=${map}\
            seed=${seed}

        # echo "Running MAPPO on ${map} ${hidden_dim} ${learning_rate} ${reward_standardisation} ${target_update} ${n_step}"
        python3 src/main.py --config=mappo --env-config=sc2 with\
            name="mappo"\
            t_max=5555000\
            hidden_dim=64\
            env_args.map_name=${map}\
            seed=${seed}
    done 
done 