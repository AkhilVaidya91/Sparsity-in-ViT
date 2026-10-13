"""Small MLflow helpers used by training and evaluation code."""

from contextlib import contextmanager
import math
from pathlib import Path

import mlflow

from .config import (
    ATTENTION_HEADS, BATCH_SIZE, DROPOUT, EARLY_STOPPING_PATIENCE,
    HIDDEN_SIZE, LABEL_SMOOTHING, LEARNING_RATE, MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI, MLP_DIMENSION, PATCH_SIZE, PRUNING_METHOD,
    TRANSFORMER_BLOCKS, WARMUP_PCT, WEIGHT_DECAY,
    ADVERSARIAL_RESULTS_CSV, ALL_RESULTS_CSV, DENSE_RESULTS_CSV,
    PRUNING_RESULTS_CSV, RETRAINING_RESULTS_CSV, TRAINING_HISTORY_CSV,
)


def configure_tracking():
    """Configure the shared tracking store and experiment."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def _safe_metrics(metrics):
    safe = {}
    for key, value in metrics.items():
        if value is None or isinstance(value, str):
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric_value):
            safe[key] = numeric_value
    return safe


def _string_params(parameters):
    return {str(key): str(value) for key, value in parameters.items() if value is not None}


@contextmanager
def training_tracking_run(run_metadata, epochs, use_amp, use_fused_adamw):
    """Open an MLflow run around one dense or retraining job."""
    configure_tracking()
    parameters = dict(run_metadata)
    parameters.update({
        "epochs_requested": epochs,
        "use_amp": use_amp,
        "use_fused_adamw": use_fused_adamw,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "dropout": DROPOUT,
        "label_smoothing": LABEL_SMOOTHING,
        "warmup_fraction": WARMUP_PCT,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "hidden_size": HIDDEN_SIZE,
        "patch_size": PATCH_SIZE,
        "transformer_blocks": TRANSFORMER_BLOCKS,
        "attention_heads": ATTENTION_HEADS,
        "mlp_dimension": MLP_DIMENSION,
        "pruning_method": PRUNING_METHOD,
    })
    with mlflow.start_run(run_name=str(run_metadata.get("tag", "training"))):
        mlflow.log_params(_string_params(parameters))
        yield


def log_epoch_metrics(row):
    """Log numeric metrics for the current training epoch, when a run is active."""
    if mlflow.active_run() is None:
        return
    metrics = _safe_metrics({
        "train_loss": row.get("train_loss"),
        "train_accuracy": row.get("train_accuracy"),
        "val_loss": row.get("val_loss"),
        "val_accuracy": row.get("val_accuracy"),
        "val_f1": row.get("val_f1"),
        "val_roc_auc": row.get("val_roc_auc"),
        "learning_rate": row.get("learning_rate"),
    })
    if metrics:
        mlflow.log_metrics(metrics, step=int(row.get("epoch", 0)))


def log_training_result(training):
    """Log final test and early-stopping metrics for a completed training run."""
    test_metrics = training.get("test_metrics", {})
    metrics = {f"test_{key}": value for key, value in test_metrics.items()}
    metrics.update({
        "best_val_loss": training.get("best_val_loss"),
        "best_epoch": training.get("best_epoch"),
        "early_stopped": int(bool(training.get("early_stopped"))),
        "early_stopping_epoch": training.get("early_stopping_epoch"),
        "epochs_trained": training.get("epochs_trained"),
        "training_seconds": training.get("training_seconds"),
    })
    safe = _safe_metrics(metrics)
    if safe:
        mlflow.log_metrics(safe)
    optimizer_name = training.get("optimizer_name")
    if optimizer_name:
        mlflow.log_param("optimizer", optimizer_name)


def log_evaluation_run(run_name, parameters, metrics):
    """Store clean or adversarial evaluation results as a separate run."""
    configure_tracking()
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(_string_params(parameters))
        safe = _safe_metrics(metrics)
        if safe:
            mlflow.log_metrics(safe)


def log_result_artifacts(paths=None):
    """Attach final CSV outputs to an MLflow summary run."""
    configure_tracking()
    if paths is None:
        paths = [
            DENSE_RESULTS_CSV,
            PRUNING_RESULTS_CSV,
            RETRAINING_RESULTS_CSV,
            TRAINING_HISTORY_CSV,
            ALL_RESULTS_CSV,
            ADVERSARIAL_RESULTS_CSV,
        ]
    with mlflow.start_run(run_name="experiment_results_summary"):
        for path in paths:
            path = Path(path)
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path="results")
