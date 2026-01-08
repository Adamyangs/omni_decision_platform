#!/bin/bash

STEPS=2055000

maps=(Foraging-5x5-2p-1f-v2 Foraging-5x5-2p-1f-coop-v2 Foraging-grid-2s-5x5-2p-1f-coop-v2 Foraging-5x5-2p-2f-v2 Foraging-8x8-3p-3f-v2 Foraging-8x8-2p-2f-coop-v2 Foraging-10x10-5p-1f-coop-v2 Foraging-10x10-3p-5f-v2 Foraging-grid-2s-10x10-3p-3f-v2 Foraging-10x10-5p-3f-v2 Foraging-15x15-3p-5f-v2 Foraging-15x15-5p-5f-v2 Foraging-15x15-5p-3f-v2)

for map in "${maps[@]}"
do

    CUDA_VISIBLE_DEVICES=0 taskset -c 0-9 python3 src/main.py --config=joint_ippo --env-config=gymma with\
        name="perla_critic_ippo"\
        t_max=$STEPS\
        env_args.key=${map}\
        seed=1 &
    CUDA_VISIBLE_DEVICES=1 taskset -c 10-19 python3 src/main.py --config=joint_mappo --env-config=gymma with\
        name="perla_critic_mappo"\
        t_max=$STEPS\
        env_args.key=${map}\
        seed=1 &

    CUDA_VISIBLE_DEVICES=0 taskset -c 0-9 python3 src/main.py --config=joint_ippo --env-config=gymma with\
        name="perla_critic_ippo"\
        t_max=$STEPS\
        env_args.key=${map}\
        seed=2 &
    CUDA_VISIBLE_DEVICES=1 taskset -c 10-19 python3 src/main.py --config=joint_mappo --env-config=gymma with\
        name="perla_critic_mappo"\
        t_max=$STEPS\
        env_args.key=${map}\
        seed=2 &

    CUDA_VISIBLE_DEVICES=0 taskset -c 0-9 python3 src/main.py --config=joint_ippo --env-config=gymma with\
        name="perla_critic_ippo"\
        t_max=$STEPS\
        env_args.key=${map}\
        seed=3 &
    CUDA_VISIBLE_DEVICES=1 taskset -c 10-19 python3 src/main.py --config=joint_mappo --env-config=gymma with\
        name="perla_critic_mappo"\
        t_max=$STEPS\
        env_args.key=${map}\
        seed=3 &
    wait
done