#!/bin/bash

STEPS=2555000

maps=(3m 5m_vs_6m corridor)
hidden_dims=(64 128 256)
learning_rates=(0.00035 0.0007 0.001)
reward_standardisations=(True False)
target_updates=(0.01)
# target_updates=(200 0.01)
n_steps=(10)
# n_steps=(5 10)
n_samples=(10 100 250)

for map in "${maps[@]}"
do
    for hidden_dim in "${hidden_dims[@]}"
    do
        for learning_rate in "${learning_rates[@]}"
        do
            for reward_standardisation in "${reward_standardisations[@]}"
            do
                for target_update in "${target_updates[@]}"
                do
                    for n_step in "${n_steps[@]}"
                    do
                        for n_sample in "${n_samples[@]}"
                        do
                            for seed in {1..2}
                            do
                                echo "Running IPPO on ${map} ${hidden_dim} ${learning_rate} ${reward_standardisation} ${target_update} ${n_step} ${n_sample}"
                                python3 src/main.py --config=joint_ippo --env-config=sc2 with\
                                    name="ippo_${hidden_dim}_${learning_rate}_${reward_standardisation}_${target_update}_${n_steps}_${n_sample}"\
                                    t_max=$STEPS\
                                    hidden_dim=${hidden_dim}\
                                    lr=${learning_rate}\
                                    standardise_rewards=${reward_standardisation}\
                                    target_update_interval_or_tau=${target_update}\
                                    q_nstep=${n_step}\
                                    n_samples=${n_sample}\
                                    env_args.map_name=${map}\
                                    seed=${seed}

                                echo "Running MAPPO on ${map} ${hidden_dim} ${learning_rate} ${reward_standardisation} ${target_update} ${n_step} ${n_sample}"
                                python3 src/main.py --config=joint_mappo --env-config=sc2 with\
                                    name="mappo_${hidden_dim}_${learning_rate}_${reward_standardisation}_${target_update}_${n_steps}_${n_sample}"\
                                    t_max=$STEPS\
                                    hidden_dim=${hidden_dim}\
                                    lr=${learning_rate}\
                                    standardise_rewards=${reward_standardisation}\
                                    target_update_interval_or_tau=${target_update}\
                                    q_nstep=${n_step}\
                                    n_samples=${n_sample}\
                                    env_args.map_name=${map}\
                                    seed=${seed}
                            done
                        done
                    done
                done
            done        
        done
    done
done