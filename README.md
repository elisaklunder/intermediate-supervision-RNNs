# Deep Thinking for Maze Solving with Intermediate Supervision

This repository contains my thesis research extending the DeepThinking framework for maze solving tasks. This codebase is forked and significantly modified from the original [DeepThinking](https://github.com/aks2203/deep-thinking) repository by Schwarzschild et al., with additional contributions for intermediate supervision training, stable testing protocol and novel maze exploration strategies.

## Attribution

The original DeepThinking work is described in:

1. **End-to-end Algorithm Synthesis with Recurrent Networks: Logical Extrapolation Without Overthinking** (NeurIPS '22)
   - [arXiv](https://arxiv.org/abs/2202.05826)
   - Authors: Arpit Bansal, Avi Schwarzschild, Eitan Borgnia, Zeyad Emam, Furong Huang, Micah Goldblum, Tom Goldstein

2. **Can You Learn an Algorithm? Generalizing from Easy to Hard Problems with Recurrent Networks** (NeurIPS '21)
   - [Proceedings](https://proceedings.neurips.cc/paper/2021/file/3501672ebc68a5524629080e3ef60aef-Paper.pdf)
   - Authors: Avi Schwarzschild, Eitan Borgnia, Arjun Gupta, Furong Huang, Uzi Vishkin, Micah Goldblum, Tom Goldstein


## Contributions

This thesis project extends DeepThinking with the following additions:

### 1. Intermediate Supervision Training

- **New training mode**: `train_mode=intermediate` 
- Provides supervision signals at intermediate steps of the network's iterative reasoning process
- Combines final-output loss with intermediate-step supervision via a weighted loss (`alpha` parameter)


### 2. Novel Maze Exploration Strategies

Implemented multiple maze solver modes for generating intermediate supervision masks:

- **Incremental** (`mazesolver_mode=incremental`): Provides masks following the optimal path incrementally
- **Reverse Exploration** (`mazesolver_mode=reverse`): Explores from goal backward, removing dead ends
- **Bidirectional** (`mazesolver_mode=bidirectional`): Explores from both start and goal simultaneously
- **DFS** (`mazesolver_mode=dfs`): Uses depth-first search exploration patterns

These modes are implemented in the `MazeSolver` class (`deepthinking/utils/maze_solver.py`) and can be controlled via the `mazesolver_mode` hyperparameter.

### 3. Stable Testing Protocol

- **New testing mode**: `test_mode=stable`
- Implements a per-sample early stopping criterion based on prediction stability
- Stops inference when a sample has produced 5 consecutive identical predictions with high confidence

### 4. Additional Features

- **Step parameter**: Controls the granularity of intermediate supervision steps
- **Extended analysis tools**: Additional scripts for analyzing model behavior, convergence patterns, and performance across different maze sizes and for visualizing intermediate reasoning steps and maze exploration patterns

## Requirements

This code was developed and tested with Python 3.10.17. All package versions are pinned in `requirements.txt` for reproducibility.

To install requirements:

```bash
pip install -r requirements.txt
```

## Getting Started

### Training

To train models with intermediate supervision, run:

```bash
python train_model.py \
    problem=mazes \
    problem/model=dt_net_recall_outputspace_2d \
    problem.train_data=9 \
    problem.test_data=9 \
    problem.hyp.train_mode=intermediate \
    problem.hyp.mazesolver_mode=reverse \
    problem.hyp.step=3
```

Key hyperparameters:
- `problem.hyp.train_mode=intermediate`: Enables intermediate supervision training
- `problem/model=dt_net_recall_outputspace_2d`: Enables conditioning the network on the previous step output during training
- `problem.hyp.mazesolver_mode`: Choose from `incremental`, `reverse`, `bidirectional`, or `dfs`
- `problem.hyp.step`: Step size for intermediate supervision (e.g., `step=3` means supervision every 3 solver steps)
- `problem.hyp.alpha`: Weight for intermediate supervision loss (0 = only final loss, 1 = only intermediate loss)

For more examples, see the scripts in the `launch/` directory.

### Testing

To test a saved model:

```bash
python test_model.py \
    problem.model.model_path=<path_to_checkpoint> \
    problem.test_data=<size> \
    problem.hyp.test_mode=stable
```

Testing modes:
- `test_mode=stable`: Uses the stable testing protocol (per-sample early stopping when predictions stabilize)
- `test_mode=default`: Uses fixed number of iterations for all samples
- `test_mode=max_conf`: Stops at the iteration with maximum confidence

The stable testing protocol is recommended for this repository as it allows models to use adaptive reasoning steps based on problem difficulty.

## Project Structure

```
deepthinking/
├── models/          # Network architectures
│   └── blocks.py   # Residual blocks
├── utils/
│   ├── training.py              # Training routines
│   ├── testing.py               # Testing and evaluation
│   ├── maze_solver.py          # Maze solver with multiple exploration modes
│   └── ...
├── data_analysis/  # Analysis and visualization scripts
└── ...

config/
└── problem/        # Hydra configuration files
    ├── hyp/        # Hyperparameter configurations
    └── model/      # Model architecture configurations

launch/             # Example training/testing scripts
outputs/            # Training outputs, checkpoints, and results
```

## Saving Protocol

### During Training

Each training run creates a unique `run_id` (adjective-Name combination) to avoid overwriting previous runs. Outputs are saved in:

```
outputs/
└── <name>/
    └── training-<run_id>/
        ├── model_best.pth          # Best model checkpoint
        ├── stats.json              # Training metrics
        ├── tensorboard/            # TensorBoard logs
        └── .hydra/                 # Configuration used for this run
```

### During Testing

Test results are saved with a separate `run_id`:

```
outputs/
└── <name>/
    └── testing-<run_id>/
        ├── stats.json              # Test metrics
        └── ...
```

## Analysis Tools

Several analysis scripts are available in `deepthinking/data_analysis/`:

- `make_table.py`: Generate pivot tables of average accuracies across trials
- `make_schoop.py`: Create "schoopy plots" showing accuracy over iterations
- `analyze_traces.py`: Analyze model reasoning traces
- `plot_convergence_stablemode.py`: Plot convergence patterns
- `compare_models.py`: Compare different model configurations

Usage example:

```bash
python deepthinking/data_analysis/make_table.py outputs/my_experiment
```

## Notes

- The neural network blocks (`deepthinking/models/blocks.py`) are borrowed from ResNet architectures and were collaboratively developed by the original DeepThinking authors (Avi Schwarzschild, Eitan Borgnia, Arpit Bansal, Zeyad Emam).
- This codebase maintains compatibility with the original DeepThinking experiments while adding new capabilities for maze solving with intermediate supervision.
