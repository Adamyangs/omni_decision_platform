#!/bin/bash

STEPS=1055000

maps=(Foraging-8x8-2p-2f-coop-v2 Foraging-10x10-3p-5f-v2)
hidden_dims=(64 128 256)
learning_rates=(0.00035 0.0005 0.001)
reward_standardisations=(True False)
target_updates=(200 0.01)
n_steps=(5 10)

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
                        for seed in {1..2}
                        do
                            # echo "Running IPPO on ${map} ${hidden_dim} ${learning_rate} ${reward_standardisation} ${target_update} ${n_step}"
                            python3 src/main.py --config=ippo --env-config=gymma with\
                                name="ippo_${hidden_dim}_${learning_rate}_${reward_standardisation}_${target_update}_${n_steps}"\
                                t_max=$STEPS\
                                hidden_dim=${hidden_dim}\
                                lr=${learning_rate}\
                                standardise_rewards=${reward_standardisation}\
                                target_update_interval_or_tau=${target_update}\
                                q_nstep=${n_step}\
                                env_args.key=${map}\
                                seed=${seed}

                            # echo "Running MAPPO on ${map} ${hidden_dim} ${learning_rate} ${reward_standardisation} ${target_update} ${n_step}"
                            # python3 src/main.py --config=mappo --env-config=gymma with\
                            #     name="mappo_${hidden_dim}_${learning_rate}_${reward_standardisation}_${target_update}_${n_steps}"\
                            #     t_max=$STEPS\
                            #     hidden_dim=${hidden_dim}\
                            #     lr=${learning_rate}\
                            #     standardise_rewards=${reward_standardisation}\
                            #     target_update_interval_or_tau=${target_update}\
                            #     q_nstep=${n_step}\
                            #     env_args.key=${map}\
                            #     seed=${seed}
                        done
                    done
                done
            done        
        done
    done
done