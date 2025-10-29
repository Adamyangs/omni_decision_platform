#!/bin/bash

STEPS=555000

maps=(Foraging-8x8-2p-2f-coop-v2 Foraging-10x10-3p-5f-v2)
# maps=(Foraging-8x8-2p-2f-coop-v2)

# hidden_dims=(128 256)
hidden_dims=(256)
learning_rates=(0.0003 0.0005)
# learning_rates=(0.0003)
reward_standardisations=(False)
# entropy_coef=(0.001 0.01)
entropy_coef=(0.001)
# target_updates=(200 0.01)
target_updates=(0.01)
# n_steps=(5 10)
n_steps=(5 10)
# n_samples=(10 100 250)
n_samples=(100)


for map in "${maps[@]}"
do
    for hidden_dim in "${hidden_dims[@]}"
    do
        for learning_rate in "${learning_rates[@]}"
        do
            for reward_standardisation in "${reward_standardisations[@]}"
            do
                for ent_coef in "${entropy_coef[@]}"
                do
                    for target_update in "${target_updates[@]}"
                    do
                        for n_step in "${n_steps[@]}"
                        do
                            for n_sample in "${n_samples[@]}"
                            do
                                for seed in {1..2}
                                do
                                    echo "Running IPPO on ${map} ${hidden_dim} ${learning_rate} ${reward_standardisation} ${ent_coef} ${target_update} ${n_step} ${n_sample}"
                                    CUDA_VISIBLE_DEVICES=0 taskset -c 0-9 python3 src/main.py --config=joint_ippo --env-config=gymma with\
                                        name="ippo_${hidden_dim}_${learning_rate}_${reward_standardisation}_${ent_coef}_${target_update}_${n_step}_${n_sample}"\
                                        t_max=$STEPS\
                                        hidden_dim=${hidden_dim}\
                                        lr=${learning_rate}\
                                        entropy_coef=${ent_coef}\
                                        standardise_rewards=${reward_standardisation}\
                                        target_update_interval_or_tau=${target_update}\
                                        q_nstep=${n_step}\
                                        n_samples=${n_sample}\
                                        env_args.key=${map}\
                                        seed=${seed} &

                                    echo "Running MAPPO on ${map} ${hidden_dim} ${learning_rate} ${ent_coef} ${reward_standardisation} ${target_update} ${n_step} ${n_sample}"
                                    CUDA_VISIBLE_DEVICES=1 taskset -c 10-19 python3 src/main.py --config=joint_mappo --env-config=gymma with\
                                        name="mappo_${hidden_dim}_${learning_rate}_${reward_standardisation}_${ent_coef}_${target_update}_${n_step}_${n_sample}"\
                                        t_max=$STEPS\
                                        hidden_dim=${hidden_dim}\
                                        lr=${learning_rate}\
                                        entropy_coef=${ent_coef}\
                                        standardise_rewards=${reward_standardisation}\
                                        target_update_interval_or_tau=${target_update}\
                                        q_nstep=${n_step}\
                                        n_samples=${n_sample}\
                                        env_args.key=${map}\
                                        seed=${seed} &
                                    wait
                                done
                            done
                        done
                    done
                done
            done        
        done
    done
done