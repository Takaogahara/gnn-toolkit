# **Config File Explanation**

### TASK
The first option is about what task the model will do. There are 2 options available in the moment: **Classification** and **Regression** 

```yaml
# Task configuration
    # - Classification
    # - Regression
TASK: ["Classification"]
```

### Run

In this section, the settings for the run are defined.

```yaml
# Run configuration
RUN:
# Experiment name
- RUN_NAME: ["None"]
# MLflow tracking URI
- RUN_MLFLOW_URI: ["http://localhost:5000"]
# Number of hyperparameters search samples
- RUN_RAY_SAMPLES: [100]
# Max epoch number for Hyperband
- RUN_RAY_MAX_EPOCH: [5]
# CPU core number per trial (int or float)
- RUN_RAY_CPU: [2]
# GPU core number per trial (int or float)
- RUN_RAY_GPU: [0]
# Time budget to kill trials
- RUN_RAY_TIME_BUDGET_S: [None]
# Resume Ray Tune run
    # - False
    # - "Auto"
- RUN_RAY_RESUME: [False]
# Telegram updates verbose
    # - 0 - None
    # - 1 - Start/end
    # - 2 - Start/end and MLflow run
- RUN_TELEGRAM_VERBOSE: [0]
  ```

### DATA

In this section, the settings for the data are defined.

```yaml
# Data configuration
DATA:
# Dataloader to be used
    # - Default
    # - MoleculeNet
- DATA_DATALOADER: ["Default"]
# Dataset separation already made
- DATA_PREMADE: [True]
# Path to files root directory
- DATA_ROOT_PATH: ["path/to/folder"]
# Data filename
- DATA_FILE_NAME: ["filename.csv"]
  ```

### MODEL

The following section is about model architecture.

```yaml
# Model configuration
MODEL:
# Use transfer learning
- MODEL_USE_TRANSFER: [False]
# Transfer learning model path
- MODEL_TRANSFER_PATH:  ["path/to/model"]
# Model architecture
    # - AttentiveFP
    # - CGC
    # - GAT
    # - GCN
    # - GIN
    # - GINE
    # - GraphSage
    # - Transformer
- MODEL_ARCHITECTURE: ["AttentiveFP"]
# Internal embedding size
- MODEL_EMBEDDING_SIZE: [64]
# Number of architecture layers
- MODEL_NUM_LAYERS: [3]
# Dropout rate
- MODEL_DROPOUT_RATE: [0.2]
# Number of heads
    # - Transformer
    # - GAT
- MODEL_NUM_HEADS: [1]
# Attention V2 usage
    # - GAT
- MODEL_ATT_V2: [True]
# Number of timesteps
    # - AttentiveFP
- MODEL_NUM_TIMESTEPS: [1]
# GCN improved usage
    # - GCN
- MODEL_GCN_IMPROVED: [True]
  ```

### SOLVER

This section is about solver hyperparameters.

```yaml
# Solver configuration
SOLVER:
# Batch size
- SOLVER_BATCH_SIZE: [64]
# Number of epochs
- SOLVER_NUM_EPOCH: [25]
# Optimizer to be used
    # - SGD
    # - Adam
- SOLVER_OPTIMIZER: ["SGD"]
# Learning rate
- SOLVER_LEARNING_RATE: [0.01]
# L2 penalty
- SOLVER_L2_PENALTY: [0.001]
# Scheduler gamma (https://pytorch.org/docs/stable/optim.html)
- SOLVER_SCHEDULER_GAMMA: [0.8]
# Positive class weight
    # - "Auto"
    # - int or float value
- SOLVER_LOSS_FN_POS_WEIGHT: ["Auto"]
# SGD momentum
- SOLVER_SGD_MOMENTUM: [0.8]
# Adam beta 1
- SOLVER_ADAM_BETA_1: [0.9]
# Adam beta 2
- SOLVER_ADAM_BETA_2: [0.999]
```
_____________________________________________________________________________________
<br/>
