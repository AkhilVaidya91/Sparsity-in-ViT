# Vision Transformer Pruning Experiments

## Purpose

This repository trains a Vision Transformer (ViT) for traffic sign recognition. It studies unstructured pruning in two parts of the model: attention and the multilayer perceptron (MLP). It compares the dense model with pruned models. It also tests whether retraining works better when it starts from the original weights or from new random weights. The pipeline measures clean accuracy and adversarial robustness.

## How the code is organized

The code is split into small Python modules. `config.py` stores model settings, training values, data locations, output paths, and MLflow settings. `data.py` builds image transformations, prepares the GTSRB dataset, creates train and validation splits, and provides data loaders. `model.py` defines the masked ViT layers and the full model. `pruning.py` collects Wanda activation statistics, scores model parameters, creates masks, and reports model sparsity. `training.py` contains the optimizer, learning-rate schedule, checkpoint saving, early stopping, and training loop. `evaluation.py` measures clean performance and inference time. `adversarial.py` runs FGSM and PGD attacks. `experiments.py` handles dense training, pruning-only evaluation, and retraining jobs. `results.py` writes result tables to CSV files. `pipeline.py` joins these steps into one experiment.

## Requirements

Use Python 3.10 or newer. UV manages the virtual environment, lockfile, and editable project installation:

```bash
uv sync
```

PyTorch must be installed with a build that supports the CUDA version on the target machine. The main pipeline requires CUDA and at least two visible GPUs. This requirement is kept because the original retraining stage starts two GPU workers. A single-GPU setup will stop with an error before training begins.

## Dataset and output paths

The code downloads the GTSRB dataset when needed. By default, it stores data in `data/gtsrb` within the repository. It writes CSV files, logs, checkpoints, and the local MLflow database under `outputs/`. These locations are general defaults. They do not depend on a Kaggle profile or a Kaggle-only directory.

You can change the locations with environment variables. Set `GTSRB_DATA_ROOT` to change the dataset directory. Set `OUTPUT_DIR` to change the output directory. Set `MLFLOW_TRACKING_URI` to use a different MLflow tracking store. Set `MLFLOW_EXPERIMENT_NAME` to change the experiment name. For example, on Linux or macOS:

```bash
export GTSRB_DATA_ROOT="$HOME/datasets/gtsrb"
export OUTPUT_DIR="$HOME/experiments/vit-pruning"
```

## Run the pipeline

Start the complete experiment with either:

```bash
uv run vit-pruning
# or:
uv run python -m vit_pruning
```

The pipeline trains the dense model first. It then prepares Wanda pruning masks for the attention and MLP parameter groups. It evaluates the masked models, runs the adversarial checks, and starts the retraining jobs. The configured pruning ratios apply to parameters inside the selected group, not to all parameters in the model. The code saves its results as CSV files. It records training history and saves checkpoints at the configured intervals.

## MLflow experiment tracking

The code logs each training run to MLflow. It records run settings, training and validation metrics for each epoch, final test metrics, early-stopping details, and training time. It also records clean evaluation metrics for pruning-only models and clean and adversarial metrics for attack runs. A summary run stores the result CSV files as artifacts.

The default tracking database is `outputs/mlflow.db`. After a run, open a second terminal in the repository and start the MLflow interface:

```bash
uv run mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db
```

Then open the local address printed by MLflow. If you use a custom `OUTPUT_DIR` or `MLFLOW_TRACKING_URI`, point the interface to the same tracking store. The experiment name defaults to `vit-structured-pruning`.