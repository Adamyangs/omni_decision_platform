# DMVR

This repository is open-source code for Decomposed Combinatorial Optimization for Multi-Vehicle Routing via Cooperative Learning (DMVR). 



## Installation instructions

### Conda
```shell
conda env create -n <env name> -f environment.yml
# The environment.yml was generated from
# conda env export --no-builds > environment.yml
```
It can take a few minutes.

## Command Line Tool

### Train
```python
export CUDA_VISIBLE_DEVICES=0,1
python marl_main.py  --customer_nodes 50  --agent-num  10 --num-envs 1024 --num-minibatches 4  --learning-rate 0.0001
```
* customer_nodes means the problem size
* agent-num means the max number of vehicle
* num-envs means the environment number for training
* num-minibatches means the ppo update minibatch
* learning-rate means the learning rate


### Test

#### DMVR
```python
cd marl_test
bash run_50_test.sh
```

#### OR-Tools
```python
cd OR-Tools
bash run_ortools.sh
```


