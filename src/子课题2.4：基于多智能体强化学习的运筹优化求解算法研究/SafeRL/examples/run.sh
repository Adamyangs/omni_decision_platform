#!/bin/bash

time=$(date +"%Y-%m-%d_%T")

algo=FCPO
env=SafetyHumanoidVelocity-v1

args=" \
--algo ${algo} \
--env-id "${env}" \
--parallel 1 \
--total-steps 10000000 \
--device cpu \
--vector-env-nums 20
"
nohup python train_policy.py $args > ./logs/log_${algo}_${env}_${time}.out 2>&1 &