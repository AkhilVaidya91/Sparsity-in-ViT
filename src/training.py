"""Optimization, checkpointing, and model training."""

import math
import os
import time

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from .config import (
    CHECKPOINT_DIR, CHECKPOINT_EVERY_EPOCHS, EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_PATIENCE, LABEL_SMOOTHING, LEARNING_RATE,
    PRUNING_METHOD, WARMUP_PCT, WEIGHT_DECAY,
)
from .evaluation import evaluate
from .pruning import enforce_unstructured_masks
from .reproducibility import state_dict_cpu
from .tracking import log_epoch_metrics, log_training_result, training_tracking_run


def create_optimizer_and_scheduler(
    model,
    total_steps,
    use_fused_adamw,
):
    if use_fused_adamw:
        try:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                fused=True,
            )

            optimizer_name = (
                "AdamW(fused=True)"
            )

        except (
            TypeError,
            RuntimeError,
        ):
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
            )

            optimizer_name = (
                "AdamW(default)"
            )

    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        optimizer_name = (
            "AdamW(default)"
        )

    warmup_steps = max(
        1,
        int(
            WARMUP_PCT
            * total_steps
        ),
    )

    def lr_lambda(step):
        if step < warmup_steps:
            return (
                float(step)
                / float(
                    max(
                        1,
                        warmup_steps,
                    )
                )
            )

        progress = (
            float(
                step - warmup_steps
            )
            / float(
                max(
                    1,
                    total_steps
                    - warmup_steps,
                )
            )
        )

        return 0.5 * (
            1.0
            + math.cos(
                math.pi
                * progress
            )
        )

    scheduler = LambdaLR(
        optimizer,
        lr_lambda,
    )

    return (
        optimizer,
        scheduler,
        optimizer_name,
    )


def checkpoint_path(tag):
    return (
        CHECKPOINT_DIR
        / f"{tag}_latest.pt"
    )


def save_training_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
    best_val_loss,
    best_epoch,
    patience_counter,
):
    state = {
        "epoch": epoch,
        "model_state_dict": state_dict_cpu(
            model.state_dict()
        ),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": (
            scaler.state_dict()
            if scaler is not None
            else None
        ),
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "patience_counter": patience_counter,
    }

    tmp = path.with_suffix(
        ".tmp"
    )

    torch.save(
        state,
        tmp,
    )

    os.replace(
        tmp,
        path,
    )


def _train_model_impl(
    model,
    run_metadata,
    train_loader,
    val_loader,
    test_loader,
    device,
    epochs,
    log_path,
    use_amp,
    use_fused_adamw,
):
    criterion = nn.CrossEntropyLoss(
        label_smoothing=LABEL_SMOOTHING
    )

    total_steps = (
        len(train_loader)
        * epochs
    )

    (
        optimizer,
        scheduler,
        optimizer_name,
    ) = create_optimizer_and_scheduler(
        model=model,
        total_steps=total_steps,
        use_fused_adamw=use_fused_adamw,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            use_amp
            and device.type == "cuda"
        ),
    )

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_state = None

    history = []
    epochs_trained = 0
    early_stopped = False
    stop_epoch = epochs

    training_start = time.perf_counter()

    with open(
        log_path,
        "a",
        encoding="utf-8",
    ) as log_file:
        log_file.write(
            f"Optimizer: {optimizer_name} | "
            f"AMP: {use_amp} | "
            f"EarlyStopping: "
            f"patience={EARLY_STOPPING_PATIENCE}, "
            f"min_delta={EARLY_STOPPING_MIN_DELTA} | "
            f"PruningMethod={PRUNING_METHOD}\n"
        )

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in train_loader:
            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(
                    use_amp
                    and device.type == "cuda"
                ),
            ):
                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels,
                )

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            # AdamW decoupled weight decay can alter masked parameters
            # even if their gradients are zero, so re-enforce the mask.
            enforce_unstructured_masks(
                model
            )

            scheduler.step()

            total_loss += loss.item()

            predicted = outputs.argmax(
                dim=1
            )

            total_correct += (
                predicted == labels
            ).sum().item()

            total_samples += labels.size(0)

        train_loss = (
            total_loss
            / len(train_loader)
        )

        train_accuracy = (
            total_correct
            / total_samples
        )

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            criterion,
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        improved = (
            val_metrics["loss"]
            < best_val_loss
            - EARLY_STOPPING_MIN_DELTA
        )

        if improved:
            best_val_loss = (
                val_metrics["loss"]
            )

            best_epoch = (
                epoch + 1
            )

            patience_counter = 0

            best_state = state_dict_cpu(
                model.state_dict()
            )

        else:
            patience_counter += 1

        epochs_trained = (
            epoch + 1
        )

        row = {
            **run_metadata,
            "epoch": epoch + 1,
            "learning_rate": current_lr,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_roc_auc": val_metrics["roc_auc"],
            "best_epoch_so_far": best_epoch,
            "patience_counter": patience_counter,
            "is_best_epoch": int(improved),
        }

        history.append(row)
        log_epoch_metrics(row)

        message = (
            f"[{run_metadata['tag']}] "
            f"Epoch [{epoch + 1:03d}/{epochs}] | "
            f"LR: {current_lr:.7g} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Patience: "
            f"{patience_counter}/"
            f"{EARLY_STOPPING_PATIENCE}"
        )

        print(
            message,
            flush=True,
        )

        with open(
            log_path,
            "a",
            encoding="utf-8",
        ) as log_file:
            log_file.write(
                message
                + "\n"
            )

        if (
            (epoch + 1)
            % CHECKPOINT_EVERY_EPOCHS
            == 0
            or (epoch + 1) == epochs
            or patience_counter
            >= EARLY_STOPPING_PATIENCE
        ):
            save_training_checkpoint(
                checkpoint_path(
                    run_metadata["tag"]
                ),
                epoch + 1,
                model,
                optimizer,
                scheduler,
                scaler,
                best_val_loss,
                best_epoch,
                patience_counter,
            )

        if (
            patience_counter
            >= EARLY_STOPPING_PATIENCE
        ):
            early_stopped = True
            stop_epoch = (
                epoch + 1
            )
            break

    if best_state is None:
        raise RuntimeError(
            "No best model state was recorded."
        )

    model.load_state_dict(
        best_state
    )

    # Re-enforce masks after loading the best state.
    enforce_unstructured_masks(
        model
    )

    final_test_metrics = evaluate(
        model,
        test_loader,
        device,
        criterion,
    )

    training_seconds = (
        time.perf_counter()
        - training_start
    )

    if not early_stopped:
        stop_epoch = epochs_trained

    return {
        "model": model,
        "test_metrics": final_test_metrics,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "early_stopped": early_stopped,
        "early_stopping_epoch": stop_epoch,
        "epochs_trained": epochs_trained,
        "training_seconds": training_seconds,
        "history": history,
        "optimizer_name": optimizer_name,
        "use_amp": use_amp,
    }


def train_model(
    model,
    run_metadata,
    train_loader,
    val_loader,
    test_loader,
    device,
    epochs,
    log_path,
    use_amp,
    use_fused_adamw,
):
    """Train one model and record its parameters and metrics in MLflow."""
    with training_tracking_run(
        run_metadata=run_metadata,
        epochs=epochs,
        use_amp=use_amp,
        use_fused_adamw=use_fused_adamw,
    ):
        training = _train_model_impl(
            model=model,
            run_metadata=run_metadata,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            epochs=epochs,
            log_path=log_path,
            use_amp=use_amp,
            use_fused_adamw=use_fused_adamw,
        )
        log_training_result(training)
        return training
