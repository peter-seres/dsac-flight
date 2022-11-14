# Distributional Reinforcement Learning for Flight Control

This repository contains the source code used to conduct my MSc thesis project on applying distributional RL for 
flight control tasks.

The agent used is the Distributional Soft Actor-Critic (DSAC), which uses IQN critics to estimate the return distribution.

The environment is the validated aerodynamic model of the PH-LAB (Cessna Citation II) research aircraft.

## Code structure

- `agents/`: soft actor-critic (SAC) and distibutional soft actor-critic (DSAC) using implicit quantile networks (IQN)
- `data_management/`: Logging agents to `.pth` files, configs to `.toml` files and episodes to `.csv` files.
- `environments/`: contains the PH-LAB environment using a `gym`-like interface.
- `train_and_eval/`: training and evaluation routines

An example training script has been added: `train_dsac.py` which trains a risk-averse DSAC agents using the 
settings contained in `config.toml`.

## Dependencies

Setup conda environment:
```bash
conda env create --file environment.yml
```

### Dependency Justification

|          Package          |                      Usage                       |
|:-------------------------:|:------------------------------------------------:|
|         `pytorch`         |                  deep learning                   |
|    `cudatoolkit=11.6`     |               deep learning on gpu               |          
|          `tqdm`           |                  progress bars                   |            
|           `gym`           |                   gym wrappers                   |           
|          `numpy`          |                vector and linalg                 |            
|          `toml`           |                   config file                    |           
|         `pandas`          |            dataframes and csv saving             |     
|         `fsspec`          |             pandas to csv needs this             |          
|        `gitpython`        |              log git hash metadata               | 
| `peter-seres/signals.git` |           reference signal generation            | 
