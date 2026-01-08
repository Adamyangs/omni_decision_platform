## Project Overview

HGCN2SP is an innovative framework that combines a hierarchical graph convolutional network with reinforcement learning methods to solve two-stage stochastic programming problems. This project focuses on the CFLP (Capacitated Facility Location Problem) and achieves efficient solving by learning scenario features.

## Environment Requirements

- Python 3.8+
- PyTorch 1.10+
- PyG (PyTorch Geometric)
- Gurobi
- NumPy, tqdm, wandb (optional)

## Project Structure

```
HGCN2SP/
├── configs              # Configuration file directory
├── data                 # Storage for training and testing data
├── eval_instance        # Storage for validation data
├── model_path           # Storage for model weights
├── models               # Network model components
├── test_csv             # Storage for test CSV files
├── train/test_scenarios # Storage for training/testing instances
├── train/test_results   # Storage for standard training/testing results
├── utils                # Utility libraries
├── agent.py             # Agent model definition
├── env.py               # Environment interaction interface
├── sample.py            # Data sampler
├── trainer.py           # PPO trainer
├── generate_data.py     # Data generation script
├── process_data_new.py  # Data preprocessing script
├── run.py               # Main running script
```

## Running Process

### Step 1: Generate Problem Instance Data

First, generate CFLP problem instances and their optimal solutions:

```bash
python generate_data.py --seed 0 --num_total_ins 100 --file_path ./train_scenarios --result_path ./train_results
```

Parameter Description:
- `--seed`: Random seed
- `--num_total_ins`: Number of problem instances to generate
- `--file_path`: Storage path for problem instances
- `--result_path`: Storage path for solving results

### Step 2: Data Preprocessing

Process raw data to create datasets:

```bash
python process_data_new.py
```

This script reads problem instances and solving results from the paths specified in the configuration file and constructs a data structure suitable for model input. Please modify the parameters within the script before use to avoid errors.

### Step 3: Model Training and Evaluation

Run the main script for model training/testing:

```bash
python run.py --config_file ./configs/cflp_config.json
```

Parameter Description:
- `--config_file`: Path to the configuration file

During training, set the `mode` in `Args` to "train"; during testing, change it to "test" and update the `model_test_path` field.

## Configuration File Explanation

The configuration file `cflp_config.json` contains the following main sections:

1. **Policy**: Controls the policy network structure
   - `var_dim`, `con_dim`: Feature dimensions for variables and constraints
   - `l_hid_dim`, `h_hid_dim`: Hidden layer dimensions for local and global features
   - `n_heads`: Number of attention heads

2. **TrainData/TestData**: Data configuration
   - `n_scenarios`: Number of scenarios
   - `pkl_folder`, `result_folder`: Paths for data and results
   - `save_path`, `cls_path`: Storage paths for processed data

3. **train**: Training configuration
   - `sel_num`: Number of selected scenarios
   - `decode_type`: Decoding strategy type
   - `eval_cls_loc`: Path for validation data
   - `eval_result`: Path for validation data results
   - `eval_epoch`: Evaluation frequency

4. **test**: Testing configuration
   - `sel_num`: Number of selected scenarios during testing
   - `decode_type`: Testing decoding strategy

The configuration file can be modified to adjust the network structure, training parameters, data paths, and other settings. Please update the configuration according to your actual paths.
