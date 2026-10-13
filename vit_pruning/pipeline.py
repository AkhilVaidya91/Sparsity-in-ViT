"""Top-level orchestration for the full training and evaluation pipeline."""

import multiprocessing as mp

import pandas as pd
import torch

from .config import (
    ADV_ATTACKS, ADV_EPSILONS, INIT_METHODS, PRUNE_TYPES,
    REQUESTED_RATIOS,
)
from .adversarial import adversarial_row
from .data import make_inference_batch, prepare_datasets_and_indices
from .evaluation import measure_inference_time
from .experiments import (
    evaluate_pruned_models, train_dense_model, training_worker,
)
from .pruning import effective_parameter_breakdown, prepare_unstructured_wanda_pruning, validate_prune_plan
from .reproducibility import state_dict_sha256
from .results import (
    ADVERSARIAL_RESULT_COLUMNS, ADVERSARIAL_RESULTS_CSV,
    ALL_RESULTS_CSV, DENSE_RESULT_COLUMNS, DENSE_RESULTS_CSV,
    HISTORY_COLUMNS, PRUNING_RESULT_COLUMNS, PRUNING_RESULTS_CSV,
    RETRAINING_COLUMNS, RETRAINING_RESULTS_CSV, TRAINING_HISTORY_CSV,
    append_csv_row, initialize_csv,
)
from .tracking import configure_tracking, log_result_artifacts


def main():
    configure_tracking()
    validate_prune_plan()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "This pipeline requires CUDA."
        )

    if torch.cuda.device_count() < 2:
        raise RuntimeError(
            "This pipeline retains the existing "
            "two-GPU retraining setup and requires "
            "at least 2 visible GPUs."
        )

    print(
        "PyTorch:",
        torch.__version__,
        flush=True,
    )

    print(
        "CUDA available:",
        torch.cuda.is_available(),
        flush=True,
    )

    print(
        "Visible GPUs:",
        torch.cuda.device_count(),
        flush=True,
    )

    for gpu_id in range(
        torch.cuda.device_count()
    ):
        print(
            f"GPU {gpu_id}: "
            f"{torch.cuda.get_device_name(gpu_id)}",
            flush=True,
        )

    initialize_csv(
        DENSE_RESULTS_CSV,
        DENSE_RESULT_COLUMNS,
    )

    initialize_csv(
        PRUNING_RESULTS_CSV,
        PRUNING_RESULT_COLUMNS,
    )

    initialize_csv(
        RETRAINING_RESULTS_CSV,
        RETRAINING_COLUMNS,
    )

    initialize_csv(
        TRAINING_HISTORY_CSV,
        HISTORY_COLUMNS,
    )

    initialize_csv(
        ADVERSARIAL_RESULTS_CSV,
        ADVERSARIAL_RESULT_COLUMNS,
    )

    # --------------------------------------------------------
    # 1. Data
    # --------------------------------------------------------
    (
        train_dataset,
        val_dataset,
        calibration_dataset,
        test_dataset,
        train_indices,
        val_indices,
    ) = prepare_datasets_and_indices()

    # --------------------------------------------------------
    # 2. Dense model training
    # --------------------------------------------------------
    device = torch.device(
        "cuda:0"
    )

    torch.cuda.set_device(0)

    (
        dense_model,
        dense_best_state,
        original_init_state,
        dense_init_metadata,
        dense_training,
        _dense_train_loader,
        _dense_val_loader,
        _dense_test_loader,
    ) = train_dense_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        device=device,
    )

    original_init_hash = (
        dense_init_metadata[
            "original_init_sha256"
        ]
    )

    if (
        state_dict_sha256(
            original_init_state
        )
        != original_init_hash
    ):
        raise AssertionError(
            "Original initialization hash "
            "changed unexpectedly."
        )

    dense_params = (
        effective_parameter_breakdown(
            dense_model
        )
    )

    dense_inference_images = (
        make_inference_batch(
            test_dataset,
            device,
        )
    )

    dense_inf_time = (
        measure_inference_time(
            dense_model,
            device,
            dense_inference_images,
        )
    )

    dense_metrics = (
        dense_training["test_metrics"]
    )

    dense_row = {
        "run_type":
            "dense",
        "tag":
            "dense_baseline",
        "prune_type":
            "none",
        "requested_ratio":
            0.0,
        "actual_group_pruning_fraction":
            0.0,
        "init_method":
            "dense_baseline",
        "parameters_pruned":
            0,
        "group_parameters":
            0,
        "test_loss":
            dense_metrics["loss"],
        "test_accuracy":
            dense_metrics["accuracy"],
        "test_f1":
            dense_metrics["f1"],
        "test_roc_auc":
            dense_metrics["roc_auc"],
        "test_correct":
            dense_metrics["correct"],
        "test_samples":
            dense_metrics["samples"],
        "inference_time_100_images_sec":
            dense_inf_time,
        "best_val_loss":
            dense_training["best_val_loss"],
        "best_epoch":
            dense_training["best_epoch"],
        "early_stopped":
            dense_training["early_stopped"],
        "early_stopping_epoch":
            dense_training["early_stopping_epoch"],
        "epochs_trained":
            dense_training["epochs_trained"],
        "training_seconds":
            dense_training["training_seconds"],
        "optimizer":
            dense_training["optimizer_name"],
        "use_amp":
            False,
        "gpu_id":
            0,
        "random_reinit_seed":
            None,
        "original_init_sha256":
            original_init_hash,
        **dense_params,
        "validation_samples":
            len(val_dataset),
        "training_samples":
            len(train_dataset),
        "calibration_samples":
            len(calibration_dataset),
    }

    append_csv_row(
        DENSE_RESULTS_CSV,
        DENSE_RESULT_COLUMNS,
        dense_row,
    )

    print(
        f"Dense baseline | "
        f"Acc={dense_metrics['accuracy']:.4f} | "
        f"F1={dense_metrics['f1']:.4f} | "
        f"ROC-AUC={dense_metrics['roc_auc']:.4f} | "
        f"StopEpoch="
        f"{dense_training['early_stopping_epoch']}",
        flush=True,
    )

    # --------------------------------------------------------
    # 2b. Dense-model adversarial robustness
    # --------------------------------------------------------
    print(
        "Running dense-model adversarial evaluation...",
        flush=True,
    )

    for attack_name in ADV_ATTACKS:
        for epsilon_pixel in ADV_EPSILONS:
            row = adversarial_row(
                model=dense_model,
                test_dataset=test_dataset,
                device=device,
                run_type="dense_adversarial",
                tag="dense_baseline",
                prune_type="none",
                requested_ratio=0.0,
                init_method="dense_baseline",
                actual_group_pruning_fraction=0.0,
                parameters_pruned=0,
                group_parameters=0,
                original_init_hash=original_init_hash,
                gpu_id=0,
                attack_name=attack_name,
                epsilon_pixel=epsilon_pixel,
            )

            append_csv_row(
                ADVERSARIAL_RESULTS_CSV,
                ADVERSARIAL_RESULT_COLUMNS,
                row,
            )

            print(
                f"[dense | {attack_name} | "
                f"eps={epsilon_pixel * 255:.1f}/255] "
                f"AdvAcc={row['adversarial_accuracy']:.4f} | "
                f"Drop={row['accuracy_drop']:.4f} | "
                f"ASR={row['attack_success_rate']:.4f}",
                flush=True,
            )

    # --------------------------------------------------------
    # 3. Wanda calibration and independent unstructured masks
    # --------------------------------------------------------
    pruning_masks, pruning_stats = (
        prepare_unstructured_wanda_pruning(
            dense_model=dense_model,
            calib_dataset=calibration_dataset,
            device=device,
        )
    )

    # --------------------------------------------------------
    # 4. Pruning-only evaluation
    # --------------------------------------------------------
    pruning_rows = (
        evaluate_pruned_models(
            dense_best_state=dense_best_state,
            pruning_masks=pruning_masks,
            pruning_stats=pruning_stats,
            test_dataset=test_dataset,
            device=device,
            original_init_hash=original_init_hash,
        )
    )

    for row in pruning_rows:
        append_csv_row(
            PRUNING_RESULTS_CSV,
            PRUNING_RESULT_COLUMNS,
            row,
        )

    # --------------------------------------------------------
    # Release GPU 0 before creating workers.
    # The original initialization, CPU masks, and pruning stats
    # remain in memory.
    # --------------------------------------------------------
    del dense_model
    torch.cuda.empty_cache()

    # --------------------------------------------------------
    # 5. Build retraining task list
    #
    # 8 pruning conditions:
    #   Attention: 20/40/60/80
    #   MLP:       20/40/60/80
    #
    # Each condition is retrained with:
    #   random_reinit
    #   original_init
    #
    # Total retraining runs = 16.
    # --------------------------------------------------------
    tasks = []

    for prune_type in PRUNE_TYPES:
        for requested_ratio in REQUESTED_RATIOS:
            for init_method in INIT_METHODS:
                tasks.append(
                    (
                        prune_type,
                        requested_ratio,
                        init_method,
                    )
                )

    # --------------------------------------------------------
    # 6. Existing two-GPU retraining strategy
    # --------------------------------------------------------
    tasks_by_gpu = [
        tasks[0::2],
        tasks[1::2],
    ]

    print(
        f"GPU 0 assignments: "
        f"{len(tasks_by_gpu[0])}",
        flush=True,
    )

    print(
        f"GPU 1 assignments: "
        f"{len(tasks_by_gpu[1])}",
        flush=True,
    )

    ctx = mp.get_context(
        "spawn"
    )

    processes = []

    for gpu_id in range(2):
        process = ctx.Process(
            target=training_worker,
            args=(
                gpu_id,
                tasks_by_gpu[gpu_id],
                train_indices,
                val_indices,
                original_init_state,
                pruning_masks,
                pruning_stats,
            ),
            name=f"gpu{gpu_id}_trainer",
        )

        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    failed_processes = [
        (
            process.name,
            process.exitcode,
        )
        for process in processes
        if process.exitcode != 0
    ]

    if failed_processes:
        raise RuntimeError(
            f"Worker process failures: "
            f"{failed_processes}. "
            f"Inspect the per-run logs in "
            f"the logs directory."
        )

    # --------------------------------------------------------
    # 7. Combined CSVs
    # --------------------------------------------------------
    dense_df = pd.read_csv(
        DENSE_RESULTS_CSV
    )

    pruning_df = pd.read_csv(
        PRUNING_RESULTS_CSV
    )

    retraining_df = pd.read_csv(
        RETRAINING_RESULTS_CSV
    )

    all_df = pd.concat(
        [
            dense_df,
            pruning_df,
            retraining_df,
        ],
        ignore_index=True,
        sort=False,
    )

    all_df.to_csv(
        ALL_RESULTS_CSV,
        index=False,
    )

    print(
        "\nDense results:",
        flush=True,
    )

    print(
        dense_df.to_string(
            index=False
        ),
        flush=True,
    )

    print(
        "\nPruning results:",
        flush=True,
    )

    print(
        pruning_df.to_string(
            index=False
        ),
        flush=True,
    )

    print(
        "\nRetraining results:",
        flush=True,
    )

    print(
        retraining_df.to_string(
            index=False
        ),
        flush=True,
    )

    print(
        "\nOutput CSVs:",
        flush=True,
    )

    print(
        f"Dense:       "
        f"{DENSE_RESULTS_CSV}",
        flush=True,
    )

    print(
        f"Pruning:     "
        f"{PRUNING_RESULTS_CSV}",
        flush=True,
    )

    print(
        f"Retraining:  "
        f"{RETRAINING_RESULTS_CSV}",
        flush=True,
    )

    print(
        f"History:     "
        f"{TRAINING_HISTORY_CSV}",
        flush=True,
    )

    print(
        f"All results: "
        f"{ALL_RESULTS_CSV}",
        flush=True,
    )

    print(
        f"Adversarial: "
        f"{ADVERSARIAL_RESULTS_CSV}",
        flush=True,
    )

    log_result_artifacts()

    print(
        "Done.",
        flush=True,
    )
