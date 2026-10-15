"""Dataset setup, preprocessing, splits, and data loaders."""

import cv2
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import GTSRB

from .config import (
    ADV_EVAL_BATCH_SIZE, ADV_EVAL_NUM_WORKERS, ADV_EVAL_SAMPLES,
    ADV_EVAL_SEED, BATCH_SIZE, CALIB_SEED, CALIB_SAMPLES, DATA_ROOT,
    IMAGE_SIZE, MEAN, NUM_CHANNELS, SPLIT_SEED, STD, VAL_FRACTION,
)


def apply_clahe(img):
    img_np = np.array(img)

    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))

    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)


train_transformation = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Lambda(apply_clahe),
    transforms.ToPILImage(),
    transforms.RandomRotation(15),
    transforms.RandomCrop(IMAGE_SIZE, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


test_transformation = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Lambda(apply_clahe),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def seed_worker(worker_id):
    worker_seed = (
        torch.initial_seed()
        % (2**32)
    )

    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_split_indices(
    num_samples,
    seed=SPLIT_SEED,
):
    generator = (
        torch.Generator()
        .manual_seed(seed)
    )

    permutation = torch.randperm(
        num_samples,
        generator=generator,
    ).tolist()

    num_val = int(
        round(
            VAL_FRACTION
            * num_samples
        )
    )

    num_val = max(
        1,
        min(
            num_samples - 1,
            num_val,
        ),
    )

    val_indices = permutation[:num_val]
    train_indices = permutation[num_val:]

    return train_indices, val_indices


def make_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    seed,
    train_num_workers,
    eval_num_workers,
):
    generator = (
        torch.Generator()
        .manual_seed(seed)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=train_num_workers,
        pin_memory=True,
        persistent_workers=(
            train_num_workers > 0
        ),
        prefetch_factor=(
            2
            if train_num_workers > 0
            else None
        ),
        worker_init_fn=(
            seed_worker
            if train_num_workers > 0
            else None
        ),
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=eval_num_workers,
        pin_memory=True,
        persistent_workers=(
            eval_num_workers > 0
        ),
        prefetch_factor=(
            2
            if eval_num_workers > 0
            else None
        ),
        worker_init_fn=(
            seed_worker
            if eval_num_workers > 0
            else None
        ),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=eval_num_workers,
        pin_memory=True,
        persistent_workers=(
            eval_num_workers > 0
        ),
        prefetch_factor=(
            2
            if eval_num_workers > 0
            else None
        ),
        worker_init_fn=(
            seed_worker
            if eval_num_workers > 0
            else None
        ),
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )


def make_inference_batch(
    test_dataset,
    device,
):
    loader = DataLoader(
        test_dataset,
        batch_size=100,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    images, _ = next(iter(loader))

    return images.to(
        device,
        non_blocking=True,
    )


def prepare_datasets_and_indices():
    print(
        f"Preparing GTSRB under {DATA_ROOT}...",
        flush=True,
    )

    train_aug_dataset = GTSRB(
        root=str(DATA_ROOT),
        split="train",
        download=True,
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
        download=True,
        transform=test_transformation,
    )

    train_indices, val_indices = (
        make_split_indices(
            len(train_aug_dataset),
            seed=SPLIT_SEED,
        )
    )

    # Calibration is sampled from the training partition only and uses
    # deterministic evaluation preprocessing, matching the previous
    # pipeline's calibration design.
    generator = (
        torch.Generator()
        .manual_seed(CALIB_SEED)
    )

    shuffled_positions = torch.randperm(
        len(train_indices),
        generator=generator,
    )

    calibration_count = min(
        CALIB_SAMPLES,
        len(train_indices),
    )

    calibration_positions = (
        shuffled_positions[
            :calibration_count
        ].tolist()
    )

    calibration_indices = [
        train_indices[position]
        for position in calibration_positions
    ]

    train_dataset = Subset(
        train_aug_dataset,
        train_indices,
    )

    val_dataset = Subset(
        train_eval_dataset,
        val_indices,
    )

    calibration_dataset = Subset(
        train_eval_dataset,
        calibration_indices,
    )

    print(
        f"Train: {len(train_dataset)} | "
        f"Validation: {len(val_dataset)} | "
        f"Calibration: {len(calibration_dataset)} | "
        f"Test: {len(test_dataset)}",
        flush=True,
    )

    return (
        train_dataset,
        val_dataset,
        calibration_dataset,
        test_dataset,
        train_indices,
        val_indices,
    )
