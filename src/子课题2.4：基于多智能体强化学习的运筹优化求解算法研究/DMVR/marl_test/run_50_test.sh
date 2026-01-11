#! /bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python marl_test2.py --customer_nodes 50 --agent-num 10 --n_test_start 0 --n-test 2000 & 
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,0
python marl_test2.py --customer_nodes 50 --agent-num 10 --n_test_start 2000 --n-test 2000 &
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,0,1
python marl_test2.py --customer_nodes 50 --agent-num 10 --n_test_start 4000 --n-test 2000 &
export CUDA_VISIBLE_DEVICES=4,5,6,7,0,2,3,1
python marl_test2.py --customer_nodes 200 --agent-num 28 --n_test_start 6000 --n-test 2000 &
export CUDA_VISIBLE_DEVICES=5,6,7,0,2,3,1,4
python marl_test2.py --customer_nodes 200 --agent-num 28 --n_test_start 8000 --n-test 2000 &