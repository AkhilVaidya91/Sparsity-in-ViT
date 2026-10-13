"""Clean evaluation and inference timing."""

import time

import torch
from sklearn.metrics import f1_score, roc_auc_score



def _safe_roc_auc(
    labels,
    probabilities,
):
    try:
        return roc_auc_score(
            labels,
            probabilities,
            multi_class="ovr",
            average="macro",
        )
    except ValueError:
        return float("nan")


@torch.inference_mode()
def evaluate(
    model,
    data_loader,
    device,
    criterion,
):
    model.eval()

    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    num_batches = 0

    all_labels = []
    all_predictions = []
    all_probabilities = []

    for images, labels in data_loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

        total_loss += loss.item()
        num_batches += 1

        predicted = outputs.argmax(
            dim=1
        )

        total_correct += (
            predicted == labels
        ).sum().item()

        total_samples += labels.size(0)

        all_labels.append(
            labels.cpu()
        )

        all_predictions.append(
            predicted.cpu()
        )

        all_probabilities.append(
            torch.softmax(
                outputs,
                dim=1,
            ).cpu()
        )

    all_labels = torch.cat(
        all_labels
    ).numpy()

    all_predictions = torch.cat(
        all_predictions
    ).numpy()

    all_probabilities = torch.cat(
        all_probabilities
    ).numpy()

    avg_loss = (
        total_loss / num_batches
    )

    accuracy = (
        total_correct
        / total_samples
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
    )

    roc_auc = _safe_roc_auc(
        all_labels,
        all_probabilities,
    )

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "f1": f1,
        "roc_auc": roc_auc,
        "correct": total_correct,
        "samples": total_samples,
    }


@torch.inference_mode()
def measure_inference_time(
    model,
    device,
    images,
    n_warmup=5,
):
    model.eval()

    for _ in range(n_warmup):
        _ = model(images)

    if device.type == "cuda":
        torch.cuda.synchronize(
            device
        )

    start = time.perf_counter()

    _ = model(images)

    if device.type == "cuda":
        torch.cuda.synchronize(
            device
        )

    return (
        time.perf_counter()
        - start
    )
