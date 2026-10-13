"""CSV output schemas and experiment bookkeeping."""

import csv
import fcntl
import os

from .config import (
    ADVERSARIAL_RESULTS_CSV, DENSE_RESULTS_CSV, PRUNING_RESULTS_CSV,
    RETRAINING_RESULTS_CSV, TRAINING_HISTORY_CSV,
)


HISTORY_COLUMNS = [
    "run_type",
    "tag",
    "prune_type",
    "requested_ratio",
    "actual_group_pruning_fraction",
    "init_method",
    "parameters_pruned",
    "group_parameters",
    "epoch",
    "learning_rate",
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
    "val_f1",
    "val_roc_auc",
    "best_epoch_so_far",
    "patience_counter",
    "is_best_epoch",
]


def initialize_csv(
    path,
    columns,
):
    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
        )

        writer.writeheader()


def append_csv_row(
    path,
    columns,
    row,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        fcntl.flock(
            file.fileno(),
            fcntl.LOCK_EX,
        )

        try:
            writer = csv.DictWriter(
                file,
                fieldnames=columns,
                extrasaction="ignore",
            )

            writer.writerow(row)

            file.flush()
            os.fsync(
                file.fileno()
            )

        finally:
            fcntl.flock(
                file.fileno(),
                fcntl.LOCK_UN,
            )


def append_history_rows(rows):
    for row in rows:
        append_csv_row(
            TRAINING_HISTORY_CSV,
            HISTORY_COLUMNS,
            row,
        )


def make_tag(
    prune_type,
    ratio,
    init_method=None,
):
    if init_method is None:
        return (
            f"{prune_type}_r{ratio:g}_pruned"
        )

    return (
        f"{prune_type}_r{ratio:g}_"
        f"{init_method}"
    )


def make_run_metadata(
    run_type,
    tag,
    prune_type,
    requested_ratio,
    init_method,
    mask_stats,
):
    return {
        "run_type": run_type,
        "tag": tag,
        "prune_type": prune_type,
        "requested_ratio": requested_ratio,
        "actual_group_pruning_fraction": (
            mask_stats["actual_group_pruning_fraction"]
        ),
        "init_method": init_method,
        "parameters_pruned": (
            mask_stats["parameters_pruned"]
        ),
        "group_parameters": (
            mask_stats["group_parameters"]
        ),
    }


RESULT_COLUMNS = [
    "run_type",
    "tag",
    "prune_type",
    "requested_ratio",
    "actual_group_pruning_fraction",
    "init_method",
    "parameters_pruned",
    "group_parameters",
    "test_loss",
    "test_accuracy",
    "test_f1",
    "test_roc_auc",
    "test_correct",
    "test_samples",
    "inference_time_100_images_sec",
    "best_val_loss",
    "best_epoch",
    "early_stopped",
    "early_stopping_epoch",
    "epochs_trained",
    "training_seconds",
    "optimizer",
    "use_amp",
    "gpu_id",
    "random_reinit_seed",
    "original_init_sha256",
    "total_parameters",
    "attention_parameters",
    "mlp_parameters",
    "other_parameters",
    "mlp_to_attention_parameter_ratio",
    "pruned_attention_parameters",
    "pruned_mlp_parameters",
    "active_attention_parameters",
    "active_mlp_parameters",
    "active_other_parameters",
    "effective_active_parameters",
    "effective_pruned_parameters",
    "effective_parameter_sparsity",
]


DENSE_RESULT_COLUMNS = RESULT_COLUMNS + [
    "validation_samples",
    "training_samples",
    "calibration_samples",
]


PRUNING_RESULT_COLUMNS = RESULT_COLUMNS


RETRAINING_COLUMNS = RESULT_COLUMNS


ADVERSARIAL_RESULT_COLUMNS = [
    "run_type",
    "tag",
    "prune_type",
    "requested_ratio",
    "actual_group_pruning_fraction",
    "init_method",
    "parameters_pruned",
    "group_parameters",
    "attack",
    "epsilon",
    "epsilon_x255",
    "pgd_steps",
    "pgd_alpha",
    "random_start",
    "clean_loss",
    "clean_accuracy",
    "clean_f1",
    "clean_roc_auc",
    "adversarial_loss",
    "adversarial_accuracy",
    "adversarial_f1",
    "adversarial_roc_auc",
    "accuracy_drop",
    "attack_success_rate",
    "clean_correct",
    "successful_attacks",
    "clean_correct_count",
    "samples",
    "original_init_sha256",
    "gpu_id",
]
