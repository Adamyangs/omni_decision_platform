# Subproject 2.2 — Key Technologies for Multimodal, Multi-task General Intelligent Decision-Making

## Overview
This subproject focuses on foundational research and engineering for general-purpose intelligent decision-making across multiple tasks and modalities. The goal is to develop algorithms and systems that can learn from heterogeneous sensory inputs (e.g., vision, proprioception, language) and transfer knowledge across tasks, supporting both large-scale pretraining and task-specific fine-tuning. The repository collects code, configuration files, training/evaluation scripts, and experimental results for methods including pretraining, multi-contrastive learning (MCL), fine-tuning pipelines, and exploration-driven RL approaches. Emphasis is placed on scalability, reproducibility, and benchmarking across continuous-control and manipulation tasks.

## Contents
- `euclid/`
  - Core research code for pretraining, MCL-based pretraining/fine-tuning, and standard fine-tuning pipelines.
  - Subfolders:
    - `pretrain_cfgs/`, `finetune_cfgs/`, `mcl_pretrain_cfgs/`, `mcl_finetune_cfgs/`: default example YAML configurations.
    - `euclid_results/`: CSV results from benchmark tasks (e.g., `Humanoid-Run.csv`, `Walker-Run.csv`, etc.).
    - `src/`: implementation modules such as `pretrain.py`, `finetune.py`, `mcl_pretrain.py`, `mcl_finetune.py`, `env.py`, `cfg.py`, `logger.py`, and algorithm components.
  - Typical usage: pretraining large models, then running fine-tuning or MCL experiments using provided configs.

- `OVD-Explorer/`
  - Research code for exploration-oriented RL and optimistic exploration strategies.
  - Contains main training/evaluation scripts (`main.py`, `run.sh`, `rl_algorithm.py`), neural network implementations (`networks.py`), replay buffer and collectors, utilities, and plotting tools.
  - `environment.yml` present for conda-based environment setup.

- `PADDLE/`
  - Contains scripts for knowledge extraction and transfer experiments (`get_knowledge.py`, `run_transfer_1.py`), plus `core/` and `env_transfer/` modules.

- Top-level utilities
  - `scripts/run.sh`: convenience run script.
  - Top-level documents and READMEs linking the different submodules.

## Getting Started

Prerequisites (assumed)
- Linux or macOS.
- Python 3.8+.
- PyTorch (for `euclid` and `OVD-Explorer`) — exact version depends on configuration.
- PaddlePaddle if you intend to run code inside `PADDLE/`.
- Conda is recommended for reproducible environment setup (see `OVD-Explorer/environment.yml`).

Example environment setup (OVD-Explorer):
```bash
cd src/子课题2.2：多模态多任务通用智能决策关键技术/OVD-Explorer
conda env create -f environment.yml
conda activate ovd-explorer  # or the environment name defined in the YAML
```

If no requirements file is provided, install typical packages:
```bash
pip install torch torchvision torchaudio numpy scipy pyyaml tensorboard
# Add or adjust packages according to submodule needs.
```

## Typical Workflows

Pretraining (euclid)
```bash
# from repository root
cd src/子课题2.2：多模态多任务通用智能决策关键技术/euclid
python src/pretrain.py --config pretrain_cfgs/default.yaml
```

MCL pretraining / finetuning
```bash
python src/mcl_pretrain.py --config mcl_pretrain_cfgs/default.yaml
python src/mcl_finetune.py --config mcl_finetune_cfgs/default.yaml
```

Run OVD-Explorer experiments
```bash
cd src/子课题2.2：多模态多任务通用智能决策关键技术/OVD-Explorer
bash run.sh          # wrapper script if provided
python main.py --config configs/your_experiment.yaml
```

Notes:
- Most scripts expect a configuration file path or use sensible defaults located in `*_cfgs/` folders.
- Logging and TensorBoard outputs are typically stored under `logs/` in each submodule.

## Results & Benchmarks
- Results CSVs for the `euclid` experiments are stored in `euclid/euclid_results/` (e.g., `Humanoid-Run.csv`, `Walker-Walk.csv`).
- `OVD-Explorer/logs/` and `OVD-Explorer/plotting/` contain training logs and visualization scripts for analyzing experiments.

## Reproducibility & Configuration
- Use the YAML configuration files found under `*_cfgs/` to reproduce experiments.
- Seed and RNG settings are typically controlled in each module's `cfg.py` or main script; check `src/cfg.py` in `euclid` for defaults.

## Contributing
- Please open issues for reproducibility problems or feature requests.
- For code contributions, follow the repository's coding style and include unit or integration tests when changing core logic.
- Add new experiment configs under the appropriate `*_cfgs/` directory and document the results in `euclid_results/` or `OVD-Explorer/logs/`.

## Citation
If you use this code in your research, please cite the relevant papers or this project per the instructions in the top-level README (add citation details here if available).

## Contact
For questions or collaboration, please open an issue or contact the maintainers listed in the top-level project README.
