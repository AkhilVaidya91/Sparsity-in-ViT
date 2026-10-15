"""FGSM and PGD adversarial evaluation."""

import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset

from .config import (
    ADV_EVAL_BATCH_SIZE, ADV_EVAL_NUM_WORKERS, ADV_EVAL_SAMPLES,
    ADV_EVAL_SEED, MEAN, NUM_CHANNELS, PGD_ALPHA_RATIO,
    PGD_RANDOM_START, PGD_STEPS, STD,
)
from .data import seed_worker
from .evaluation import _safe_roc_auc
from .reproducibility import set_seed
from .tracking import log_evaluation_run


def make_adversarial_loader(test_dataset):
    """
    Build a deterministic loader for adversarial evaluation.

    Adversarial evaluation uses a smaller batch size than clean evaluation
    because PGD requires gradients and therefore stores intermediate
    activations for each attack step.

    ADV_EVAL_SAMPLES can be set to an integer for a fixed deterministic
    subset, or None to evaluate the complete held-out test set.
    """
    dataset = test_dataset

    if ADV_EVAL_SAMPLES is not None:
        count = min(
            int(ADV_EVAL_SAMPLES),
            len(test_dataset),
        )

        generator = torch.Generator().manual_seed(
            ADV_EVAL_SEED
        )

        indices = torch.randperm(
            len(test_dataset),
            generator=generator,
        )[:count].tolist()

        dataset = Subset(
            test_dataset,
            indices,
        )

    return DataLoader(
        dataset,
        batch_size=ADV_EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=ADV_EVAL_NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(
            ADV_EVAL_NUM_WORKERS > 0
        ),
        prefetch_factor=(
            2
            if ADV_EVAL_NUM_WORKERS > 0
            else None
        ),
        worker_init_fn=(
            seed_worker
            if ADV_EVAL_NUM_WORKERS > 0
            else None
        ),
    )


def _normalized_bounds(device):
    mean = torch.tensor(
        MEAN,
        device=device,
        dtype=torch.float32,
    ).view(1, NUM_CHANNELS, 1, 1)

    std = torch.tensor(
        STD,
        device=device,
        dtype=torch.float32,
    ).view(1, NUM_CHANNELS, 1, 1)

    lower = (0.0 - mean) / std
    upper = (1.0 - mean) / std

    return mean, std, lower, upper


def _random_uniform_like(tensor, low, high):
    return low + torch.rand_like(tensor) * (high - low)


def fgsm_attack(
    model,
    images,
    labels,
    epsilon_pixel,
    device,
):
    """
    Untargeted FGSM in input space.

    epsilon_pixel is measured in the unnormalized [0, 1] image space.
    Since the model receives channel-normalized inputs, epsilon is scaled
    by the per-channel standard deviation before updating the tensor.
    """
    model.eval()

    _, std, lower, upper = _normalized_bounds(device)
    epsilon = (
        torch.tensor(
            epsilon_pixel,
            device=device,
            dtype=torch.float32,
        )
        / std
    )

    x0 = images.detach().float()
    x_adv = x0.clone().requires_grad_(True)

    logits = model(x_adv)
    loss = F.cross_entropy(logits, labels)

    gradient = torch.autograd.grad(
        loss,
        x_adv,
        only_inputs=True,
    )[0]

    x_adv = x0 + epsilon * gradient.sign()
    x_adv = torch.max(
        torch.min(x_adv, upper),
        lower,
    )

    return x_adv.detach()


def pgd_attack(
    model,
    images,
    labels,
    epsilon_pixel,
    alpha_pixel,
    steps,
    random_start,
    device,
):
    """
    Untargeted projected-gradient-descent attack with an L_inf budget.

    epsilon_pixel and alpha_pixel are measured in the unnormalized [0, 1]
    image space. The attack internally converts them to normalized-input
    units using the channel-specific GTSRB standard deviations.
    """
    model.eval()

    _, std, lower, upper = _normalized_bounds(device)

    epsilon = (
        torch.tensor(
            epsilon_pixel,
            device=device,
            dtype=torch.float32,
        )
        / std
    )

    alpha = (
        torch.tensor(
            alpha_pixel,
            device=device,
            dtype=torch.float32,
        )
        / std
    )

    x0 = images.detach().float()

    if random_start:
        delta = _random_uniform_like(
            x0,
            -epsilon,
            epsilon,
        )
        x_adv = torch.max(
            torch.min(x0 + delta, upper),
            lower,
        ).detach()
    else:
        x_adv = x0.clone()

    for _ in range(max(1, int(steps))):
        x_adv.requires_grad_(True)

        logits = model(x_adv)
        loss = F.cross_entropy(logits, labels)

        gradient = torch.autograd.grad(
            loss,
            x_adv,
            only_inputs=True,
        )[0]

        x_adv = (
            x_adv.detach()
            + alpha * gradient.sign()
        )

        delta = torch.clamp(
            x_adv - x0,
            min=-epsilon,
            max=epsilon,
        )

        x_adv = torch.max(
            torch.min(x0 + delta, upper),
            lower,
        ).detach()

    return x_adv


def adversarial_attack_batch(
    model,
    images,
    labels,
    attack_name,
    epsilon_pixel,
    device,
):
    if attack_name == "fgsm":
        return fgsm_attack(
            model=model,
            images=images,
            labels=labels,
            epsilon_pixel=epsilon_pixel,
            device=device,
        )

    if attack_name == "pgd":
        alpha_pixel = (
            epsilon_pixel
            * PGD_ALPHA_RATIO
        )

        return pgd_attack(
            model=model,
            images=images,
            labels=labels,
            epsilon_pixel=epsilon_pixel,
            alpha_pixel=alpha_pixel,
            steps=PGD_STEPS,
            random_start=PGD_RANDOM_START,
            device=device,
        )

    raise ValueError(
        f"Unknown adversarial attack: {attack_name}"
    )


def evaluate_adversarial_model(
    model,
    data_loader,
    device,
    attack_name,
    epsilon_pixel,
):
    """
    Evaluate a model under one deterministic adversarial configuration.

    The attack loss is standard cross-entropy without label smoothing. This
    is deliberate: label smoothing is a training regularizer and should not
    redefine the objective used to find adversarial examples.
    """
    model.eval()

    attack_criterion = nn.CrossEntropyLoss()

    clean_total_loss = 0.0
    clean_total_correct = 0
    adv_total_loss = 0.0
    adv_total_correct = 0
    total_samples = 0
    clean_correct_and_adv_wrong = 0
    clean_correct_count = 0

    clean_labels = []
    clean_predictions = []
    clean_probabilities = []
    adv_labels = []
    adv_predictions = []
    adv_probabilities = []

    for images, labels in data_loader:
        images = images.to(
            device,
            non_blocking=True,
        ).float()

        labels = labels.to(
            device,
            non_blocking=True,
        )

        with torch.no_grad():
            clean_outputs = model(images)
            clean_loss = attack_criterion(
                clean_outputs,
                labels,
            )
            clean_predicted = clean_outputs.argmax(
                dim=1
            )

        adversarial_images = adversarial_attack_batch(
            model=model,
            images=images,
            labels=labels,
            attack_name=attack_name,
            epsilon_pixel=epsilon_pixel,
            device=device,
        )

        with torch.no_grad():
            adversarial_outputs = model(
                adversarial_images
            )
            adversarial_loss = attack_criterion(
                adversarial_outputs,
                labels,
            )
            adversarial_predicted = (
                adversarial_outputs.argmax(dim=1)
            )

        batch_size = labels.size(0)

        clean_total_loss += (
            clean_loss.item() * batch_size
        )
        adv_total_loss += (
            adversarial_loss.item() * batch_size
        )

        clean_correct = (
            clean_predicted == labels
        )
        adversarial_wrong = (
            adversarial_predicted != labels
        )

        clean_correct_count += (
            clean_correct.sum().item()
        )
        clean_correct_and_adv_wrong += (
            (clean_correct & adversarial_wrong)
            .sum()
            .item()
        )

        clean_total_correct += (
            clean_correct.sum().item()
        )
        adv_total_correct += (
            (~adversarial_wrong).sum().item()
        )
        total_samples += batch_size

        clean_labels.append(labels.cpu())
        clean_predictions.append(
            clean_predicted.cpu()
        )
        clean_probabilities.append(
            torch.softmax(
                clean_outputs,
                dim=1,
            ).cpu()
        )

        adv_labels.append(labels.cpu())
        adv_predictions.append(
            adversarial_predicted.cpu()
        )
        adv_probabilities.append(
            torch.softmax(
                adversarial_outputs,
                dim=1,
            ).cpu()
        )

    clean_labels_np = torch.cat(
        clean_labels
    ).numpy()
    clean_predictions_np = torch.cat(
        clean_predictions
    ).numpy()
    clean_probabilities_np = torch.cat(
        clean_probabilities
    ).numpy()

    adv_labels_np = torch.cat(
        adv_labels
    ).numpy()
    adv_predictions_np = torch.cat(
        adv_predictions
    ).numpy()
    adv_probabilities_np = torch.cat(
        adv_probabilities
    ).numpy()

    clean_accuracy = (
        clean_total_correct
        / total_samples
    )

    adversarial_accuracy = (
        adv_total_correct
        / total_samples
    )

    clean_f1 = f1_score(
        clean_labels_np,
        clean_predictions_np,
        average="macro",
    )

    adversarial_f1 = f1_score(
        adv_labels_np,
        adv_predictions_np,
        average="macro",
    )

    clean_roc_auc = _safe_roc_auc(
        clean_labels_np,
        clean_probabilities_np,
    )

    adversarial_roc_auc = _safe_roc_auc(
        adv_labels_np,
        adv_probabilities_np,
    )

    attack_success_rate = (
        clean_correct_and_adv_wrong
        / clean_correct_count
        if clean_correct_count > 0
        else float("nan")
    )

    return {
        "clean_loss": (
            clean_total_loss
            / total_samples
        ),
        "clean_accuracy": clean_accuracy,
        "clean_f1": clean_f1,
        "clean_roc_auc": clean_roc_auc,
        "adversarial_loss": (
            adv_total_loss
            / total_samples
        ),
        "adversarial_accuracy": (
            adversarial_accuracy
        ),
        "adversarial_f1": adversarial_f1,
        "adversarial_roc_auc": (
            adversarial_roc_auc
        ),
        "accuracy_drop": (
            clean_accuracy
            - adversarial_accuracy
        ),
        "attack_success_rate": (
            attack_success_rate
        ),
        "clean_correct": clean_total_correct,
        "samples": total_samples,
        "clean_correct_count": clean_correct_count,
        "successful_attacks": clean_correct_and_adv_wrong,
    }


def adversarial_row(
    model,
    test_dataset,
    device,
    run_type,
    tag,
    prune_type,
    requested_ratio,
    init_method,
    actual_group_pruning_fraction,
    parameters_pruned,
    group_parameters,
    original_init_hash,
    gpu_id,
    attack_name,
    epsilon_pixel,
):
    seed_key = (
        f"{tag}:{attack_name}:{epsilon_pixel:.12f}"
    )
    digest = hashlib.md5(
        seed_key.encode("utf-8")
    ).hexdigest()
    adv_seed = (
        ADV_EVAL_SEED
        + int(digest, 16) % 1000000
    ) % (2**31 - 1)

    set_seed(adv_seed)

    loader = make_adversarial_loader(
        test_dataset
    )

    metrics = evaluate_adversarial_model(
        model=model,
        data_loader=loader,
        device=device,
        attack_name=attack_name,
        epsilon_pixel=epsilon_pixel,
    )

    row = {
        "run_type": run_type,
        "tag": tag,
        "prune_type": prune_type,
        "requested_ratio": requested_ratio,
        "actual_group_pruning_fraction": (
            actual_group_pruning_fraction
        ),
        "init_method": init_method,
        "parameters_pruned": parameters_pruned,
        "group_parameters": group_parameters,
        "attack": attack_name,
        "epsilon": epsilon_pixel,
        "epsilon_x255": epsilon_pixel * 255.0,
        "pgd_steps": (
            PGD_STEPS
            if attack_name == "pgd"
            else 1
        ),
        "pgd_alpha": (
            epsilon_pixel * PGD_ALPHA_RATIO
            if attack_name == "pgd"
            else epsilon_pixel
        ),
        "random_start": (
            int(PGD_RANDOM_START)
            if attack_name == "pgd"
            else 0
        ),
        "clean_loss": metrics["clean_loss"],
        "clean_accuracy": metrics["clean_accuracy"],
        "clean_f1": metrics["clean_f1"],
        "clean_roc_auc": metrics["clean_roc_auc"],
        "adversarial_loss": metrics["adversarial_loss"],
        "adversarial_accuracy": (
            metrics["adversarial_accuracy"]
        ),
        "adversarial_f1": metrics["adversarial_f1"],
        "adversarial_roc_auc": (
            metrics["adversarial_roc_auc"]
        ),
        "accuracy_drop": metrics["accuracy_drop"],
        "attack_success_rate": (
            metrics["attack_success_rate"]
        ),
        "clean_correct": metrics["clean_correct"],
        "successful_attacks": (
            metrics["successful_attacks"]
        ),
        "clean_correct_count": (
            metrics["clean_correct_count"]
        ),
        "samples": metrics["samples"],
        "original_init_sha256": original_init_hash,
        "gpu_id": gpu_id,
    }

    log_evaluation_run(
        run_name=(
            f"{run_type}_{tag}_{attack_name}_eps{epsilon_pixel * 255:g}"
        ),
        parameters={
            "run_type": run_type,
            "tag": tag,
            "prune_type": prune_type,
            "requested_ratio": requested_ratio,
            "init_method": init_method,
            "attack": attack_name,
            "epsilon": epsilon_pixel,
            "gpu_id": gpu_id,
            "original_init_sha256": original_init_hash,
        },
        metrics={
            "clean_loss": row["clean_loss"],
            "clean_accuracy": row["clean_accuracy"],
            "clean_f1": row["clean_f1"],
            "clean_roc_auc": row["clean_roc_auc"],
            "adversarial_loss": row["adversarial_loss"],
            "adversarial_accuracy": row["adversarial_accuracy"],
            "adversarial_f1": row["adversarial_f1"],
            "adversarial_roc_auc": row["adversarial_roc_auc"],
            "accuracy_drop": row["accuracy_drop"],
            "attack_success_rate": row["attack_success_rate"],
        },
    )

    return row
