

# Reinforcement Learning Based Sleep Mechanism


## Installation instructions

1. Create conda-env using 
```shell
conda env create -f env.yml
```
 
2. Import Datafile
```shell
cp path/4G/Network_Power2022-05-2*.npy input/4G/
cp path/4G/Network_Power2022-05-2*.npy input/5G/
cp path/Whatif_Traffic/Whatif_TrafficMax0.*.npy input/Whatif_Traffic/
cp path/Whatif_Traffic/Whatif_TrafficMin-ax0.*.npy input/Whatif_Traffic/
```

## Run an experiment 

1. Train the policy network
```shell
python mf_policy.py
```
The checkpoint of policy network is saved in 'output/{%Y-%m-%d_%H:%M:%S}/', where ‘{%Y-%m-%d_%H:%M:%S}’ is the timestamp.

2. Evaluate the trained policy network

Change the agent_path in 'global_parameters.py' to the timestamp '{%Y-%m-%d_%H:%M:%S}'.

```shell
python eval.py
python eval_whatif.py
```




