# CCMO

This repository is open-source code for Consistency-Constrained Cooperative Multi-Agent Optimization (CCMO). Our Experiment is conducted on the StarCraft 2 SMAC, with version SC2.4.10.



## Installation instructions

Install Python packages

```shell
# require Anaconda 3 or Miniconda 3
conda create -n pymarl python=3.8 -y
conda activate pymarl

bash install_dependecies.sh
```

Set up StarCraft II (2.4.10) and SMAC:

```shell
bash install_sc2.sh
```

This will download SC2.4.10 into the 3rdparty folder and copy the maps necessary to run over.

## Command Line Tool

### Train
```python
bash run_code.sh 0 3m qmix_act_rew  wandb_enabled=False
```
* 0 means the GPU id, which can be set as 0,1,2,...
* 3m means the SMAC map
* qmix_act_rew is our method
* wandb_enabled=False means the setting for wandb

### Test

```python
bash run_test.sh 
```
* The run_test.sh details the test code for our method qmix_act_rew for 3m againt enemies.


