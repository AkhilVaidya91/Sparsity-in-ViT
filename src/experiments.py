"""Dense training, pruning evaluation, and retraining workers."""

import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import GTSRB

from .config import (
    ADVERSARIAL_RESULTS_CSV, ADV_ATTACKS, ADV_EPSILONS, BATCH_SIZE,
    DATA_ROOT, DATA_SEED, DENSE_EVAL_NUM_WORKERS, DENSE_TRAIN_NUM_WORKERS,
    DROPOUT, EPOCHS, LABEL_SMOOTHING, LOG_DIR, PRUNE_TYPES,
    PRUNING_EVAL_NUM_WORKERS, REQUESTED_RATIOS, RETRAINING_RESULTS_CSV,
    RETRAIN_EPOCHS, RETRAIN_EVAL_NUM_WORKERS, RETRAIN_TRAIN_NUM_WORKERS,
    USE_AMP,
)
from .adversarial import adversarial_row
from .data import (
    make_dataloaders, make_inference_batch, seed_worker,
    test_transformation, train_transformation,
)
from .evaluation import evaluate, measure_inference_time
from .model import VisionTransformer
from .pruning import apply_unstructured_mask, effective_parameter_breakdown
from .reproducibility import config_seed, set_seed, state_dict_cpu, state_dict_sha256
from .results import (
    ADVERSARIAL_RESULT_COLUMNS, DENSE_RESULTS_CSV, DENSE_RESULT_COLUMNS,
    PRUNING_RESULTS_CSV, PRUNING_RESULT_COLUMNS, RETRAINING_RESULTS_CSV,
    RETRAINING_COLUMNS, make_run_metadata, make_tag, append_csv_row,
    append_history_rows,
)
from .training import train_model


def train_dense_model(
    train_dataset,
    val_dataset,
    test_dataset,
    device,
):
    # Capture the exact initial parameter state immediately after
    # model construction and before any optimizer operation.
    set_seed(42)

    model = VisionTransformer(
        dropout=DROPOUT
    ).to(device)

    original_init_state = (
        state_dict_cpu(
            model.state_dict()
        )
    )

    original_init_hash = (
        state_dict_sha256(
            original_init_state
        )
    )

    dense_init_metadata = {
        "original_init_sha256":
            original_init_hash,
        "seed": 42,
    }

    train_loader, val_loader, test_loader = (
        make_dataloaders(
            train_dataset,
            val_dataset,
            test_dataset,
            seed=DATA_SEED,
            train_num_workers=(
                DENSE_TRAIN_NUM_WORKERS
            ),
            eval_num_workers=(
                DENSE_EVAL_NUM_WORKERS
            ),
        )
    )

    dense_metadata = {
        "run_type": "dense",
        "tag": "dense_baseline",
        "prune_type": "none",
        "requested_ratio": 0.0,
        "actual_group_pruning_fraction": 0.0,
        "init_method": "dense_baseline",
        "parameters_pruned": 0,
        "group_parameters": 0,
    }

    training = train_model(
        model=model,
        run_metadata=dense_metadata,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        epochs=EPOCHS,
        log_path=(
            LOG_DIR
            / "dense_baseline.log"
        ),
        use_amp=False,
        use_fused_adamw=False,
    )

    append_history_rows(
        training["history"]
    )

    # Verify that the pristine initialization state was never mutated.
    if (
        state_dict_sha256(
            original_init_state
        )
        != original_init_hash
    ):
        raise AssertionError(
            "The in-memory original initialization "
            "changed during dense training."
        )

    dense_model = training["model"]

    dense_best_state = (
        state_dict_cpu(
            dense_model.state_dict()
        )
    )

    return (
        dense_model,
        dense_best_state,
        original_init_state,
        dense_init_metadata,
        training,
        train_loader,
        val_loader,
        test_loader,
    )


def evaluate_pruned_models(
    dense_best_state,
    pruning_masks,
    pruning_stats,
    test_dataset,
    device,
    original_init_hash,
):
    criterion = nn.CrossEntropyLoss(
        label_smoothing=LABEL_SMOOTHING
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=PRUNING_EVAL_NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(
            PRUNING_EVAL_NUM_WORKERS > 0
        ),
        prefetch_factor=(
            2
            if PRUNING_EVAL_NUM_WORKERS > 0
            else None
        ),
        worker_init_fn=(
            seed_worker
            if PRUNING_EVAL_NUM_WORKERS > 0
            else None
        ),
    )

    inference_images = (
        make_inference_batch(
            test_dataset,
            device,
        )
    )

    rows = []

    for prune_type in PRUNE_TYPES:
        for requested_ratio in REQUESTED_RATIOS:
            key = (
                prune_type,
                requested_ratio,
            )

            model = VisionTransformer(
                dropout=DROPOUT
            ).to(device)

            model.load_state_dict(
                dense_best_state
            )

            mask_dict = pruning_masks[key]
            stats = pruning_stats[key]

            apply_unstructured_mask(
                model,
                mask_dict,
            )

            metrics = evaluate(
                model,
                test_loader,
                device,
                criterion,
            )

            inf_time = measure_inference_time(
                model,
                device,
                inference_images,
            )

            params = (
                effective_parameter_breakdown(
                    model
                )
            )

            row = {
                "run_type":
                    "pruning_evaluation",
                "tag":
                    make_tag(
                        prune_type,
                        requested_ratio,
                    ),
                "prune_type":
                    prune_type,
                "requested_ratio":
                    requested_ratio,
                "actual_group_pruning_fraction":
                    stats[
                        "actual_group_pruning_fraction"
                    ],
                "init_method":
                    "dense_trained_weights",
                "parameters_pruned":
                    stats["parameters_pruned"],
                "group_parameters":
                    stats["group_parameters"],
                "test_loss":
                    metrics["loss"],
                "test_accuracy":
                    metrics["accuracy"],
                "test_f1":
                    metrics["f1"],
                "test_roc_auc":
                    metrics["roc_auc"],
                "test_correct":
                    metrics["correct"],
                "test_samples":
                    metrics["samples"],
                "inference_time_100_images_sec":
                    inf_time,
                "best_val_loss":
                    None,
                "best_epoch":
                    None,
                "early_stopped":
                    None,
                "early_stopping_epoch":
                    None,
                "epochs_trained":
                    None,
                "training_seconds":
                    None,
                "optimizer":
                    None,
                "use_amp":
                    False,
                "gpu_id":
                    0,
                "random_reinit_seed":
                    None,
                "original_init_sha256":
                    original_init_hash,
                **params,
            }

            rows.append(row)

            from .tracking import log_evaluation_run
            log_evaluation_run(
                run_name=f"pruned_{prune_type}_r{requested_ratio:g}",
                parameters={
                    "run_type": "pruning_evaluation",
                    "prune_type": prune_type,
                    "requested_ratio": requested_ratio,
                    "actual_group_pruning_fraction": stats["actual_group_pruning_fraction"],
                    "original_init_sha256": original_init_hash,
                },
                metrics={
                    "test_loss": metrics["loss"],
                    "test_accuracy": metrics["accuracy"],
                    "test_f1": metrics["f1"],
                    "test_roc_auc": metrics["roc_auc"],
                    "inference_time_100_images_sec": inf_time,
                    "effective_parameter_sparsity": params["effective_parameter_sparsity"],
                },
            )

            print(
                f"[{prune_type:>9s} | "
                f"ratio={requested_ratio:.2f}] "
                f"Pruned="
                f"{stats['parameters_pruned']:,}/"
                f"{stats['group_parameters']:,} "
                f"({stats['actual_group_pruning_fraction']:.4f}) | "
                f"Acc={metrics['accuracy']:.4f} | "
                f"F1={metrics['f1']:.4f} | "
                f"ROC-AUC={metrics['roc_auc']:.4f} | "
                f"Overall sparsity="
                f"{params['effective_parameter_sparsity']:.4f}",
                flush=True,
            )

            # Adversarial robustness is evaluated on the pruned,
            # dense-trained-weight model before retraining.
            for attack_name in ADV_ATTACKS:
                for epsilon_pixel in ADV_EPSILONS:
                    adv_row = adversarial_row(
                        model=model,
                        test_dataset=test_dataset,
                        device=device,
                        run_type="pruning_adversarial",
                        tag=make_tag(
                            prune_type,
                            requested_ratio,
                        ),
                        prune_type=prune_type,
                        requested_ratio=requested_ratio,
                        init_method="dense_trained_weights",
                        actual_group_pruning_fraction=(
                            stats[
                                "actual_group_pruning_fraction"
                            ]
                        ),
                        parameters_pruned=stats[
                            "parameters_pruned"
                        ],
                        group_parameters=stats[
                            "group_parameters"
                        ],
                        original_init_hash=original_init_hash,
                        gpu_id=0,
                        attack_name=attack_name,
                        epsilon_pixel=epsilon_pixel,
                    )

                    append_csv_row(
                        ADVERSARIAL_RESULTS_CSV,
                        ADVERSARIAL_RESULT_COLUMNS,
                        adv_row,
                    )

                    print(
                        f"[{prune_type:>9s} | "
                        f"ratio={requested_ratio:.2f} | "
                        f"{attack_name} | "
                        f"eps={epsilon_pixel * 255:.1f}/255] "
                        f"AdvAcc={adv_row['adversarial_accuracy']:.4f} | "
                        f"Drop={adv_row['accuracy_drop']:.4f} | "
                        f"ASR={adv_row['attack_success_rate']:.4f}",
                        flush=True,
                    )

            del model
            torch.cuda.empty_cache()

    return rows


def build_retraining_model(
    init_method,
    prune_type,
    ratio,
    original_init_state,
    device,
):
    if init_method == "random_reinit":
        set_seed(
            config_seed(
                prune_type,
                ratio,
            )
        )

        model = VisionTransformer(
            dropout=DROPOUT
        ).to(device)

    elif init_method == "original_init":
        model = VisionTransformer(
            dropout=DROPOUT
        ).to(device)

        model.load_state_dict(
            original_init_state
        )

    else:
        raise ValueError(
            f"Unknown init_method: {init_method}"
        )

    return model


def training_worker(
    gpu_id,
    tasks,
    train_indices,
    val_indices,
    original_init_state,
    pruning_masks,
    pruning_stats,
):
    torch.set_num_threads(1)
    cv2.setNumThreads(1)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable inside worker."
        )

    if gpu_id >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested GPU {gpu_id}, "
            f"but only "
            f"{torch.cuda.device_count()} "
            f"GPU(s) are visible."
        )

    torch.cuda.set_device(
        gpu_id
    )

    device = torch.device(
        f"cuda:{gpu_id}"
    )

    torch.backends.cudnn.benchmark = True

    print(
        f"[GPU {gpu_id}] Starting worker on "
        f"{torch.cuda.get_device_name(gpu_id)} "
        f"with {len(tasks)} tasks.",
        flush=True,
    )

    set_seed(
        DATA_SEED + gpu_id
    )

    train_aug_dataset = GTSRB(
        root=str(DATA_ROOT),
        split="train",
        download=False,
        transform=train_transformation,
    )

    train_eval_dataset = GTSRB(
        root=str(DATA_ROOT),
        split="train",
        download=False,
        transform=test_transformation,
    )

    test_dataset = GTSRB(
        root=str(DATA_ROOT),
        split="test",
        download=False,
        transform=test_transformation,
    )

    train_dataset = Subset(
        train_aug_dataset,
        train_indices,
    )

    val_dataset = Subset(
        train_eval_dataset,
        val_indices,
    )

    train_loader, val_loader, test_loader = (
        make_dataloaders(
            train_dataset,
            val_dataset,
            test_dataset,
            seed=DATA_SEED,
            train_num_workers=(
                RETRAIN_TRAIN_NUM_WORKERS
            ),
            eval_num_workers=(
                RETRAIN_EVAL_NUM_WORKERS
            ),
        )
    )

    inference_images = (
        make_inference_batch(
            test_dataset,
            device,
        )
    )

    original_init_hash = (
        state_dict_sha256(
            original_init_state
        )
    )

    for (
        prune_type,
        requested_ratio,
        init_method,
    ) in tasks:
        key = (
            prune_type,
            requested_ratio,
        )

        tag = make_tag(
            prune_type,
            requested_ratio,
            init_method,
        )

        log_path = (
            LOG_DIR
            / f"{tag}.log"
        )

        print(
            f"\n[GPU {gpu_id}] === "
            f"Starting {tag} ===",
            flush=True,
        )

        try:
            model = build_retraining_model(
                init_method=init_method,
                prune_type=prune_type,
                ratio=requested_ratio,
                original_init_state=(
                    original_init_state
                ),
                device=device,
            )

            # Verify exact original initialization before masks
            # are applied. This protects the rewinding experiment.
            if init_method == "original_init":
                loaded_hash = (
                    state_dict_sha256(
                        state_dict_cpu(
                            model.state_dict()
                        )
                    )
                )

                if loaded_hash != original_init_hash:
                    raise AssertionError(
                        f"Original initialization "
                        f"mismatch for {tag}."
                    )

            mask_dict = pruning_masks[key]
            stats = pruning_stats[key]

            # The same dense-derived unstructured mask is used for
            # both initialization conditions.
            apply_unstructured_mask(
                model,
                mask_dict,
            )

            params = (
                effective_parameter_breakdown(
                    model
                )
            )

            run_metadata = (
                make_run_metadata(
                    run_type="retraining",
                    tag=tag,
                    prune_type=prune_type,
                    requested_ratio=requested_ratio,
                    init_method=init_method,
                    mask_stats=stats,
                )
            )

            training = train_model(
                model=model,
                run_metadata=run_metadata,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                epochs=RETRAIN_EPOCHS,
                log_path=log_path,
                use_amp=USE_AMP,
                use_fused_adamw=True,
            )

            append_history_rows(
                training["history"]
            )

            metrics = (
                training["test_metrics"]
            )

            inf_time = measure_inference_time(
                training["model"],
                device,
                inference_images,
            )

            final_params = (
                effective_parameter_breakdown(
                    training["model"]
                )
            )

            row = {
                "run_type":
                    "retraining",
                "tag":
                    tag,
                "prune_type":
                    prune_type,
                "requested_ratio":
                    requested_ratio,
                "actual_group_pruning_fraction":
                    stats[
                        "actual_group_pruning_fraction"
                    ],
                "init_method":
                    init_method,
                "parameters_pruned":
                    stats["parameters_pruned"],
                "group_parameters":
                    stats["group_parameters"],
                "test_loss":
                    metrics["loss"],
                "test_accuracy":
                    metrics["accuracy"],
                "test_f1":
                    metrics["f1"],
                "test_roc_auc":
                    metrics["roc_auc"],
                "test_correct":
                    metrics["correct"],
                "test_samples":
                    metrics["samples"],
                "inference_time_100_images_sec":
                    inf_time,
                "best_val_loss":
                    training["best_val_loss"],
                "best_epoch":
                    training["best_epoch"],
                "early_stopped":
                    training["early_stopped"],
                "early_stopping_epoch":
                    training["early_stopping_epoch"],
                "epochs_trained":
                    training["epochs_trained"],
                "training_seconds":
                    training["training_seconds"],
                "optimizer":
                    training["optimizer_name"],
                "use_amp":
                    training["use_amp"],
                "gpu_id":
                    gpu_id,
                "random_reinit_seed":
                    (
                        config_seed(
                            prune_type,
                            requested_ratio,
                        )
                        if init_method
                        == "random_reinit"
                        else None
                    ),
                "original_init_sha256":
                    original_init_hash,
                **final_params,
            }

            append_csv_row(
                RETRAINING_RESULTS_CSV,
                RETRAINING_COLUMNS,
                row,
            )

            print(
                f"[GPU {gpu_id}] "
                f"COMPLETED {tag} | "
                f"Acc={metrics['accuracy']:.4f} | "
                f"F1={metrics['f1']:.4f} | "
                f"ROC-AUC={metrics['roc_auc']:.4f} | "
                f"StopEpoch="
                f"{training['early_stopping_epoch']}",
                flush=True,
            )

            # Evaluate the final best retrained model under the same
            # adversarial settings used for the dense and pruning-only
            # models. This keeps the attack protocol identical across
            # experimental conditions.
            for attack_name in ADV_ATTACKS:
                for epsilon_pixel in ADV_EPSILONS:
                    adv_row = adversarial_row(
                        model=training["model"],
                        test_dataset=test_dataset,
                        device=device,
                        run_type="retraining_adversarial",
                        tag=tag,
                        prune_type=prune_type,
                        requested_ratio=requested_ratio,
                        init_method=init_method,
                        actual_group_pruning_fraction=(
                            stats[
                                "actual_group_pruning_fraction"
                            ]
                        ),
                        parameters_pruned=stats[
                            "parameters_pruned"
                        ],
                        group_parameters=stats[
                            "group_parameters"
                        ],
                        original_init_hash=original_init_hash,
                        gpu_id=gpu_id,
                        attack_name=attack_name,
                        epsilon_pixel=epsilon_pixel,
                    )

                    append_csv_row(
                        ADVERSARIAL_RESULTS_CSV,
                        ADVERSARIAL_RESULT_COLUMNS,
                        adv_row,
                    )

                    print(
                        f"[GPU {gpu_id}] "
                        f"[{attack_name} | "
                        f"eps={epsilon_pixel * 255:.1f}/255] "
                        f"{tag} | "
                        f"AdvAcc={adv_row['adversarial_accuracy']:.4f} | "
                        f"Drop={adv_row['accuracy_drop']:.4f} | "
                        f"ASR={adv_row['attack_success_rate']:.4f}",
                        flush=True,
                    )

        except Exception as exc:
            print(
                f"[GPU {gpu_id}] "
                f"FAILED {tag}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

            with open(
                log_path,
                "a",
                encoding="utf-8",
            ) as log_file:
                log_file.write(
                    f"FAILED: "
                    f"{type(exc).__name__}: "
                    f"{exc}\n"
                )

            raise

        finally:
            if "model" in locals():
                del model

            torch.cuda.empty_cache()

    print(
        f"[GPU {gpu_id}] Worker finished.",
        flush=True,
    )
