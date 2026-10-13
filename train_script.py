# %% [code]
import os
import csv
import fcntl
import hashlib
import math
import multiprocessing as mp
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import GTSRB


# ============================================================
# Configuration
# ============================================================

MEAN = (0.3337, 0.3064, 0.3171)
STD = (0.2672, 0.2564, 0.2629)

IMAGE_SIZE = 32
NUM_CHANNELS = 3
NUM_CLASSES = 43

PATCH_SIZE = 4
HIDDEN_SIZE = 128
TRANSFORMER_BLOCKS = 6
ATTENTION_HEADS = 4

# Updated from the previous structured-pruning experiment.
# This gives approximately a 2:1 MLP-to-Attention parameter ratio.
MLP_DIMENSION = 4 * HIDDEN_SIZE

HEAD_DIM = HIDDEN_SIZE // ATTENTION_HEADS
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2

WARMUP_PCT = 0.10
EPOCHS = 80
RETRAIN_EPOCHS = 80
BATCH_SIZE = 512
LEARNING_RATE = 6e-4
WEIGHT_DECAY = 0.05
DROPOUT = 0.10
LABEL_SMOOTHING = 0.1

# Early stopping is based only on a held-out validation split.
VAL_FRACTION = 0.10
SPLIT_SEED = 12345
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 0.0

# Existing retraining performance controls.
USE_AMP = True
RETRAIN_TRAIN_NUM_WORKERS = 1
RETRAIN_EVAL_NUM_WORKERS = 1
DENSE_TRAIN_NUM_WORKERS = 2
DENSE_EVAL_NUM_WORKERS = 2
PRUNING_EVAL_NUM_WORKERS = 2
CHECKPOINT_EVERY_EPOCHS = 5

DATA_SEED = 12345

# The requested fraction is the fraction of parameters inside the
# selected functional location, not the fraction of all model parameters.
REQUESTED_RATIOS = [0.20, 0.40, 0.60, 0.80]
PRUNE_TYPES = ["attention", "mlp"]
INIT_METHODS = ["random_reinit", "original_init"]

# Importance criterion retained from the previous experiment, but now
# applied at individual parameter level rather than head/neuron level.
# Masks are global within each functional location:
#   Attention: all trainable parameters in transformer attention modules
#   MLP:      all trainable parameters in transformer MLP modules
PRUNING_METHOD = "wanda_unstructured"

CALIB_SEED = 0
CALIB_SAMPLES = 512
CALIB_BATCH_SIZE = 64

# Adversarial robustness evaluation.
# Epsilon is expressed in the original [0, 1] image-value space.
# The attack is applied to the deterministic 32 x 32 model input after
# preprocessing, with per-channel normalization handled internally.
ADV_ATTACKS = ["fgsm", "pgd"]
ADV_EPSILONS = [2 / 255, 4 / 255, 8 / 255]
PGD_STEPS = 10
PGD_ALPHA_RATIO = 0.25
PGD_RANDOM_START = True
ADV_EVAL_BATCH_SIZE = 128
ADV_EVAL_NUM_WORKERS = 0
ADV_EVAL_SAMPLES = None  # None = use the complete held-out test set.
ADV_EVAL_SEED = 20261006

WORK_DIR = (
    Path("/kaggle/working")
    if Path("/kaggle/working").exists()
    else Path.cwd() / "kaggle_working"
)

DATA_ROOT = (
    Path("/kaggle/temp/gtsrb_data")
    if Path("/kaggle/temp").exists()
    else WORK_DIR / "gtsrb_data"
)

DENSE_RESULTS_CSV = WORK_DIR / "dense_results.csv"
PRUNING_RESULTS_CSV = WORK_DIR / "pruning_results.csv"
RETRAINING_RESULTS_CSV = WORK_DIR / "retraining_results.csv"
TRAINING_HISTORY_CSV = WORK_DIR / "training_history.csv"
ALL_RESULTS_CSV = WORK_DIR / "all_results.csv"
ADVERSARIAL_RESULTS_CSV = WORK_DIR / "adversarial_results.csv"
CHECKPOINT_DIR = WORK_DIR / "training_checkpoints"
LOG_DIR = WORK_DIR / "logs"

WORK_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
DATA_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# Reproducibility / state handling
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def config_seed(prune_type: str, ratio: float) -> int:
    key = f"{prune_type}:{ratio:g}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest, 16) % (2**31 - 1)


def state_dict_cpu(state_dict):
    return {
        key: value.detach().cpu().clone()
        for key, value in state_dict.items()
    }


def state_dict_sha256(state_dict) -> str:
    digest = hashlib.sha256()

    for key in sorted(state_dict.keys()):
        value = state_dict[key].detach().cpu().contiguous()

        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())

    return digest.hexdigest()


def assert_state_dict_equal(state_a, state_b):
    if state_a.keys() != state_b.keys():
        raise AssertionError("State-dict keys do not match.")

    for key in state_a.keys():
        a = state_a[key].detach().cpu()
        b = state_b[key].detach().cpu()

        if not torch.equal(a, b):
            raise AssertionError(f"State-dict mismatch at key: {key}")


# ============================================================
# Data transforms
# ============================================================

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


# ============================================================
# Model
# ============================================================

class MaskedLinear(nn.Linear):
    """
    Linear layer with a fixed unstructured parameter mask.

    The underlying parameter tensors remain trainable, but masked
    entries are excluded from the forward pass. The optimizer step
    is followed by explicit mask enforcement so pruned entries remain
    exactly zero in storage as well.
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )

        self.register_buffer(
            "weight_mask",
            torch.ones_like(self.weight),
        )

        if self.bias is not None:
            self.register_buffer(
                "bias_mask",
                torch.ones_like(self.bias),
            )

    def forward(self, input):
        masked_weight = self.weight * self.weight_mask

        if self.bias is None:
            masked_bias = None
        else:
            masked_bias = self.bias * self.bias_mask

        return F.linear(
            input,
            masked_weight,
            masked_bias,
        )


class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

        self.patch_embed = nn.Conv2d(
            in_channels=NUM_CHANNELS,
            out_channels=HIDDEN_SIZE,
            kernel_size=PATCH_SIZE,
            stride=PATCH_SIZE,
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class MaskedMultiheadAttention(nn.MultiheadAttention):
    """
    Standard PyTorch MHA with elementwise unstructured parameter masks.

    The pruning mask covers:
      * in_proj_weight
      * in_proj_bias
      * out_proj.weight
      * out_proj.bias
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        batch_first=True,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first,
        )

        self.register_buffer(
            "in_proj_weight_mask",
            torch.ones_like(self.in_proj_weight),
        )

        if self.in_proj_bias is not None:
            self.register_buffer(
                "in_proj_bias_mask",
                torch.ones_like(self.in_proj_bias),
            )

        original_out_proj = self.out_proj

        self.out_proj = MaskedLinear(
            embed_dim,
            embed_dim,
            bias=(original_out_proj.bias is not None),
        )

        with torch.no_grad():
            self.out_proj.weight.copy_(original_out_proj.weight)

            if self.out_proj.bias is not None:
                self.out_proj.bias.copy_(original_out_proj.bias)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=False,
        attn_mask=None,
        average_attn_weights=True,
        is_causal=False,
    ):
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        masked_in_proj_weight = (
            self.in_proj_weight * self.in_proj_weight_mask
        )

        if self.in_proj_bias is None:
            masked_in_proj_bias = None
        else:
            masked_in_proj_bias = (
                self.in_proj_bias * self.in_proj_bias_mask
            )

        masked_out_proj_weight = (
            self.out_proj.weight * self.out_proj.weight_mask
        )

        if self.out_proj.bias is None:
            masked_out_proj_bias = None
        else:
            masked_out_proj_bias = (
                self.out_proj.bias * self.out_proj.bias_mask
            )

        attn_output, attn_weights = F.multi_head_attention_forward(
            query=query,
            key=key,
            value=value,
            embed_dim_to_check=self.embed_dim,
            num_heads=self.num_heads,
            in_proj_weight=masked_in_proj_weight,
            in_proj_bias=masked_in_proj_bias,
            bias_k=self.bias_k,
            bias_v=self.bias_v,
            add_zero_attn=self.add_zero_attn,
            dropout_p=self.dropout,
            out_proj_weight=masked_out_proj_weight,
            out_proj_bias=masked_out_proj_bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=False,
            q_proj_weight=None,
            k_proj_weight=None,
            v_proj_weight=None,
            static_k=None,
            static_v=None,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )

        if self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        return attn_output, attn_weights


class TransformerEncoder(nn.Module):
    def __init__(self, dropout=DROPOUT):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(HIDDEN_SIZE)
        self.layer_norm2 = nn.LayerNorm(HIDDEN_SIZE)

        self.multihead_attention = MaskedMultiheadAttention(
            embed_dim=HIDDEN_SIZE,
            num_heads=ATTENTION_HEADS,
            batch_first=True,
            dropout=dropout,
        )

        self.mlp = nn.Sequential(
            MaskedLinear(HIDDEN_SIZE, MLP_DIMENSION),
            nn.GELU(),
            nn.Dropout(dropout),
            MaskedLinear(MLP_DIMENSION, HIDDEN_SIZE),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.layer_norm1(x)

        attention_output, _ = self.multihead_attention(
            x_norm,
            x_norm,
            x_norm,
            need_weights=False,
        )

        x = x + attention_output

        x_norm = self.layer_norm2(x)

        mlp_output = self.mlp[0](x_norm)
        mlp_output = self.mlp[1](mlp_output)
        mlp_output = self.mlp[2](mlp_output)

        mlp_output = self.mlp[3](mlp_output)
        mlp_output = self.mlp[4](mlp_output)

        x = x + mlp_output

        return x


class MLPHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_norm = nn.LayerNorm(HIDDEN_SIZE)
        self.fc = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.fc(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, dropout=DROPOUT):
        super().__init__()

        self.patch_embedding = PatchEmbedding()

        self.cls_token = nn.Parameter(
            torch.randn(
                1,
                1,
                HIDDEN_SIZE,
            )
        )

        self.position_embeddings = nn.Parameter(
            torch.randn(
                1,
                NUM_PATCHES + 1,
                HIDDEN_SIZE,
            )
        )

        self.dropout = nn.Dropout(dropout)

        self.transformer_blocks = nn.Sequential(
            *[
                TransformerEncoder(dropout)
                for _ in range(TRANSFORMER_BLOCKS)
            ]
        )

        self.mlp_head = MLPHead()

    def forward(self, x):
        x = self.patch_embedding(x)

        batch_size = x.size(0)

        cls_tokens = self.cls_token.expand(
            batch_size,
            -1,
            -1,
        )

        x = torch.cat(
            (cls_tokens, x),
            dim=1,
        )

        x = x + self.position_embeddings
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        x = self.mlp_head(x[:, 0, :])

        return x


# ============================================================
# Unstructured pruning
# ============================================================

def prunable_parameter_names(model, prune_type):
    """
    Return parameter names belonging exclusively to the selected
    functional location.

    Attention group:
        transformer_blocks.*.multihead_attention.*

    MLP group:
        transformer_blocks.*.mlp.*
    """
    names = []

    for name, parameter in model.named_parameters():
        del parameter

        if prune_type == "attention":
            if ".multihead_attention." in name:
                names.append(name)

        elif prune_type == "mlp":
            if ".mlp." in name:
                names.append(name)

        else:
            raise ValueError(f"Unknown prune_type: {prune_type}")

    return names


def parameter_group_count(model, prune_type):
    names = set(prunable_parameter_names(model, prune_type))

    return sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name in names
    )


def parameter_breakdown(model):
    attention_parameters = parameter_group_count(
        model,
        "attention",
    )

    mlp_parameters = parameter_group_count(
        model,
        "mlp",
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    other_parameters = (
        total_parameters
        - attention_parameters
        - mlp_parameters
    )

    mlp_to_attention_ratio = (
        mlp_parameters / attention_parameters
        if attention_parameters > 0
        else float("nan")
    )

    return {
        "total_parameters": total_parameters,
        "attention_parameters": attention_parameters,
        "mlp_parameters": mlp_parameters,
        "other_parameters": other_parameters,
        "mlp_to_attention_parameter_ratio": mlp_to_attention_ratio,
    }


def mask_buffer_for_parameter(name):
    return f"{name}_mask"


def get_mask_buffers(model):
    return dict(model.named_buffers())


def apply_unstructured_mask(model, mask_dict):
    """
    Copy a CPU-side boolean mask dictionary into the model's mask
    buffers and immediately enforce the corresponding zero values.
    """
    buffers = get_mask_buffers(model)

    for parameter_name, mask in mask_dict.items():
        buffer_name = mask_buffer_for_parameter(parameter_name)

        if buffer_name not in buffers:
            raise KeyError(
                f"Mask buffer '{buffer_name}' was not found for "
                f"parameter '{parameter_name}'."
            )

        target = buffers[buffer_name]

        if tuple(target.shape) != tuple(mask.shape):
            raise ValueError(
                f"Mask shape mismatch for '{parameter_name}': "
                f"model={tuple(target.shape)} vs "
                f"mask={tuple(mask.shape)}"
            )

        target.copy_(
            mask.to(
                device=target.device,
                dtype=target.dtype,
            )
        )

    enforce_unstructured_masks(model)


@torch.no_grad()
def enforce_unstructured_masks(model):
    """
    Explicitly zero every pruned parameter after optimizer updates.

    This prevents AdamW's decoupled weight decay from causing masked
    parameters to drift away from zero in storage.
    """
    buffers = get_mask_buffers(model)

    for name, parameter in model.named_parameters():
        buffer_name = mask_buffer_for_parameter(name)

        if buffer_name not in buffers:
            continue

        parameter.mul_(
            buffers[buffer_name].to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
        )


@torch.inference_mode()
def collect_wanda_stats(
    model,
    loader,
    device,
):
    """
    Collect layer-input activation energy needed for unstructured
    Wanda scores.

    For a weight element W[i, j], the Wanda score is:
        |W[i, j]| * ||X[:, j]||_2

    Attention uses:
        * MHA input for in-projection weights
        * pre-output-projection attention activations for out-projection weights

    MLP uses:
        * fc1 input for fc1 weights
        * fc2 input for fc2 weights

    Biases do not have an input feature dimension, so their magnitude is
    used as the importance score. Biases are included in the parameter
    count because the experiment defines pruning over all parameters in
    the selected functional location.
    """
    model.eval()

    attention_input_sq = [
        torch.zeros(
            HIDDEN_SIZE,
            device=device,
        )
        for _ in range(TRANSFORMER_BLOCKS)
    ]

    attention_output_sq = [
        torch.zeros(
            HIDDEN_SIZE,
            device=device,
        )
        for _ in range(TRANSFORMER_BLOCKS)
    ]

    mlp_fc1_input_sq = [
        torch.zeros(
            HIDDEN_SIZE,
            device=device,
        )
        for _ in range(TRANSFORMER_BLOCKS)
    ]

    mlp_fc2_input_sq = [
        torch.zeros(
            MLP_DIMENSION,
            device=device,
        )
        for _ in range(TRANSFORMER_BLOCKS)
    ]

    def make_attention_hook(block_idx):
        def hook(module, args, kwargs):
            del kwargs

            x = args[0]

            attention_input_sq[block_idx] += (
                x.detach().float().square().sum(
                    dim=(0, 1)
                )
            )

            E = module.embed_dim
            H = module.num_heads
            d = module.head_dim

            qkv = F.linear(
                x,
                module.in_proj_weight,
                module.in_proj_bias,
            )

            q, k, v = qkv.split(
                E,
                dim=-1,
            )

            B, N, _ = q.shape

            q = q.view(
                B,
                N,
                H,
                d,
            ).transpose(1, 2)

            k = k.view(
                B,
                N,
                H,
                d,
            ).transpose(1, 2)

            v = v.view(
                B,
                N,
                H,
                d,
            ).transpose(1, 2)

            attention_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
            )

            attention_output = (
                attention_output
                .transpose(1, 2)
                .reshape(B, N, E)
            )

            attention_output_sq[block_idx] += (
                attention_output.detach()
                .float()
                .square()
                .sum(dim=(0, 1))
            )

        return hook

    def make_mlp_fc1_hook(block_idx):
        def hook(module, args):
            del module

            x = args[0]

            mlp_fc1_input_sq[block_idx] += (
                x.detach().float().square().sum(
                    dim=(0, 1)
                )
            )

        return hook

    def make_mlp_fc2_hook(block_idx):
        def hook(module, args):
            del module

            x = args[0]

            mlp_fc2_input_sq[block_idx] += (
                x.detach().float().square().sum(
                    dim=(0, 1)
                )
            )

        return hook

    handles = []

    for block_idx, block in enumerate(
        model.transformer_blocks
    ):
        handles.append(
            block.multihead_attention.register_forward_pre_hook(
                make_attention_hook(
                    block_idx
                ),
                with_kwargs=True,
            )
        )

        handles.append(
            block.mlp[0].register_forward_pre_hook(
                make_mlp_fc1_hook(
                    block_idx
                )
            )
        )

        handles.append(
            block.mlp[3].register_forward_pre_hook(
                make_mlp_fc2_hook(
                    block_idx
                )
            )
        )

    try:
        for images, _ in loader:
            images = images.to(
                device,
                non_blocking=True,
            )

            model(images)
    finally:
        for handle in handles:
            handle.remove()

    return {
        "attention_input_sq":
            attention_input_sq,
        "attention_output_sq":
            attention_output_sq,
        "mlp_fc1_input_sq":
            mlp_fc1_input_sq,
        "mlp_fc2_input_sq":
            mlp_fc2_input_sq,
    }


@torch.inference_mode()
def compute_unstructured_wanda_scores(
    model,
    wanda_stats,
):
    """
    Build a per-parameter Wanda score tensor for every prunable
    Attention and MLP parameter.
    """
    scores = {}

    named_parameters = dict(
        model.named_parameters()
    )

    for block_idx, block in enumerate(
        model.transformer_blocks
    ):
        attention = (
            block.multihead_attention
        )

        attention_input_norm = torch.sqrt(
            wanda_stats[
                "attention_input_sq"
            ][block_idx]
        )

        attention_output_norm = torch.sqrt(
            wanda_stats[
                "attention_output_sq"
            ][block_idx]
        )

        in_proj_weight = (
            attention.in_proj_weight
        )

        scores[
            f"transformer_blocks.{block_idx}."
            f"multihead_attention.in_proj_weight"
        ] = (
            in_proj_weight.detach()
            .abs()
            * attention_input_norm.unsqueeze(0)
        ).cpu()

        if attention.in_proj_bias is not None:
            scores[
                f"transformer_blocks.{block_idx}."
                f"multihead_attention.in_proj_bias"
            ] = (
                attention.in_proj_bias.detach()
                .abs()
                .cpu()
            )

        out_proj_weight = (
            attention.out_proj.weight
        )

        scores[
            f"transformer_blocks.{block_idx}."
            f"multihead_attention.out_proj.weight"
        ] = (
            out_proj_weight.detach()
            .abs()
            * attention_output_norm.unsqueeze(0)
        ).cpu()

        if attention.out_proj.bias is not None:
            scores[
                f"transformer_blocks.{block_idx}."
                f"multihead_attention.out_proj.bias"
            ] = (
                attention.out_proj.bias.detach()
                .abs()
                .cpu()
            )

        fc1 = block.mlp[0]
        fc2 = block.mlp[3]

        fc1_input_norm = torch.sqrt(
            wanda_stats[
                "mlp_fc1_input_sq"
            ][block_idx]
        )

        fc2_input_norm = torch.sqrt(
            wanda_stats[
                "mlp_fc2_input_sq"
            ][block_idx]
        )

        fc1_weight_name = (
            f"transformer_blocks.{block_idx}."
            f"mlp.0.weight"
        )

        scores[fc1_weight_name] = (
            fc1.weight.detach()
            .abs()
            * fc1_input_norm.unsqueeze(0)
        ).cpu()

        if fc1.bias is not None:
            scores[
                f"transformer_blocks.{block_idx}.mlp.0.bias"
            ] = (
                fc1.bias.detach()
                .abs()
                .cpu()
            )

        fc2_weight_name = (
            f"transformer_blocks.{block_idx}."
            f"mlp.3.weight"
        )

        scores[fc2_weight_name] = (
            fc2.weight.detach()
            .abs()
            * fc2_input_norm.unsqueeze(0)
        ).cpu()

        if fc2.bias is not None:
            scores[
                f"transformer_blocks.{block_idx}.mlp.3.bias"
            ] = (
                fc2.bias.detach()
                .abs()
                .cpu()
            )

    expected_names = set()

    for prune_type in PRUNE_TYPES:
        expected_names.update(
            prunable_parameter_names(
                model,
                prune_type,
            )
        )

    score_names = set(scores.keys())

    missing = expected_names - score_names

    if missing:
        raise KeyError(
            "Missing Wanda scores for parameters: "
            f"{sorted(missing)}"
        )

    extra = score_names - expected_names

    if extra:
        raise KeyError(
            "Unexpected Wanda scores for parameters: "
            f"{sorted(extra)}"
        )

    # Ensure all scores and parameter shapes match exactly.
    for name in expected_names:
        if tuple(scores[name].shape) != tuple(
            named_parameters[name].shape
        ):
            raise ValueError(
                f"Wanda score shape mismatch for "
                f"'{name}': "
                f"score={tuple(scores[name].shape)} "
                f"parameter={tuple(named_parameters[name].shape)}"
            )

    return scores


def build_unstructured_mask(
    dense_model,
    score_dict,
    prune_type,
    requested_ratio,
):
    """
    Globally unstructured-prune the requested fraction of parameters
    within one functional location using precomputed Wanda scores.

    Exact element counts are selected with top-k, so the realized
    pruning fraction is recorded explicitly.
    """
    parameter_names = prunable_parameter_names(
        dense_model,
        prune_type,
    )

    if not parameter_names:
        raise RuntimeError(
            f"No parameters found for "
            f"prune_type='{prune_type}'."
        )

    tensors = []
    offsets = {}
    offset = 0

    for name in parameter_names:
        score = score_dict[name].reshape(-1).float()

        numel = score.numel()

        tensors.append(score)

        offsets[name] = (
            offset,
            offset + numel,
            tuple(score_dict[name].shape),
        )

        offset += numel

    all_scores = torch.cat(
        tensors,
        dim=0,
    )

    total_parameters = (
        all_scores.numel()
    )

    requested_pruned = int(
        round(
            requested_ratio
            * total_parameters
        )
    )

    requested_pruned = max(
        0,
        min(
            requested_pruned,
            total_parameters,
        ),
    )

    flat_prune_mask = torch.ones(
        total_parameters,
        dtype=torch.bool,
    )

    if requested_pruned > 0:
        prune_indices = torch.topk(
            all_scores,
            k=requested_pruned,
            largest=False,
            sorted=False,
        ).indices

        flat_prune_mask[
            prune_indices
        ] = False

    mask_dict = {}

    for name, (
        start,
        end,
        shape,
    ) in offsets.items():
        mask_dict[name] = (
            flat_prune_mask[
                start:end
            ]
            .reshape(shape)
            .clone()
        )

    actual_pruned = sum(
        int(
            (~mask)
            .sum()
            .item()
        )
        for mask in mask_dict.values()
    )

    if actual_pruned != requested_pruned:
        raise AssertionError(
            f"Expected {requested_pruned} "
            f"pruned parameters, got "
            f"{actual_pruned}."
        )

    actual_fraction = (
        actual_pruned
        / total_parameters
    )

    return mask_dict, {
        "group_parameters":
            total_parameters,
        "parameters_pruned":
            actual_pruned,
        "parameters_active":
            total_parameters
            - actual_pruned,
        "actual_group_pruning_fraction":
            actual_fraction,
    }

def prepare_unstructured_wanda_pruning(
    dense_model,
    calib_dataset,
    device,
):
    calib_loader = DataLoader(
        calib_dataset,
        batch_size=CALIB_BATCH_SIZE,
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

    print(
        "Collecting Wanda calibration statistics...",
        flush=True,
    )

    wanda_stats = collect_wanda_stats(
        dense_model,
        calib_loader,
        device,
    )

    score_dict = (
        compute_unstructured_wanda_scores(
            dense_model,
            wanda_stats,
        )
    )

    pruning_masks = {}
    pruning_stats = {}

    for prune_type in PRUNE_TYPES:
        for requested_ratio in REQUESTED_RATIOS:
            key = (
                prune_type,
                requested_ratio,
            )

            mask_dict, stats = (
                build_unstructured_mask(
                    dense_model=dense_model,
                    score_dict=score_dict,
                    prune_type=prune_type,
                    requested_ratio=requested_ratio,
                )
            )

            pruning_masks[key] = mask_dict
            pruning_stats[key] = stats

            print(
                f"[Mask] {prune_type:>9s} | "
                f"ratio={requested_ratio:.2f} | "
                f"parameters={stats['group_parameters']:,} | "
                f"pruned={stats['parameters_pruned']:,} | "
                f"active={stats['parameters_active']:,} | "
                f"actual={stats['actual_group_pruning_fraction']:.8f}",
                flush=True,
            )

    return (
        pruning_masks,
        pruning_stats,
    )

def effective_parameter_breakdown(model):
    breakdown = parameter_breakdown(model)
    buffers = get_mask_buffers(model)

    pruned_by_group = {
        "attention": 0,
        "mlp": 0,
    }

    for name, parameter in model.named_parameters():
        del parameter

        if ".multihead_attention." in name:
            prune_type = "attention"
        elif ".mlp." in name:
            prune_type = "mlp"
        else:
            continue

        buffer_name = mask_buffer_for_parameter(name)

        if buffer_name not in buffers:
            raise KeyError(
                f"Expected mask buffer '{buffer_name}' "
                f"for prunable parameter '{name}'."
            )

        pruned_by_group[prune_type] += int(
            (~buffers[buffer_name].bool())
            .sum()
            .item()
        )

    effective_attention_parameters = (
        breakdown["attention_parameters"]
        - pruned_by_group["attention"]
    )

    effective_mlp_parameters = (
        breakdown["mlp_parameters"]
        - pruned_by_group["mlp"]
    )

    effective_other_parameters = (
        breakdown["other_parameters"]
    )

    effective_active_parameters = (
        effective_attention_parameters
        + effective_mlp_parameters
        + effective_other_parameters
    )

    effective_pruned_parameters = (
        breakdown["total_parameters"]
        - effective_active_parameters
    )

    effective_parameter_sparsity = (
        effective_pruned_parameters
        / breakdown["total_parameters"]
    )

    return {
        **breakdown,
        "pruned_attention_parameters": (
            pruned_by_group["attention"]
        ),
        "pruned_mlp_parameters": (
            pruned_by_group["mlp"]
        ),
        "active_attention_parameters": (
            effective_attention_parameters
        ),
        "active_mlp_parameters": (
            effective_mlp_parameters
        ),
        "active_other_parameters": (
            effective_other_parameters
        ),
        "effective_active_parameters": (
            effective_active_parameters
        ),
        "effective_pruned_parameters": (
            effective_pruned_parameters
        ),
        "effective_parameter_sparsity": (
            effective_parameter_sparsity
        ),
    }


def validate_prune_plan():
    if REQUESTED_RATIOS != sorted(
        REQUESTED_RATIOS
    ):
        raise ValueError(
            "REQUESTED_RATIOS must be ascending."
        )

    if any(
        ratio <= 0.0 or ratio >= 1.0
        for ratio in REQUESTED_RATIOS
    ):
        raise ValueError(
            "All pruning ratios must be strictly between 0 and 1."
        )

    if PRUNING_METHOD != "wanda_unstructured":
        raise ValueError(
            "This implementation expects "
            "PRUNING_METHOD='wanda_unstructured'."
        )

    model = VisionTransformer()

    breakdown = parameter_breakdown(model)

    ratio = (
        breakdown["mlp_parameters"]
        / breakdown["attention_parameters"]
    )

    print(
        "\nDense functional parameter breakdown:",
        flush=True,
    )
    print(
        f"  Attention parameters: "
        f"{breakdown['attention_parameters']:,}",
        flush=True,
    )
    print(
        f"  MLP parameters:       "
        f"{breakdown['mlp_parameters']:,}",
        flush=True,
    )
    print(
        f"  Other parameters:     "
        f"{breakdown['other_parameters']:,}",
        flush=True,
    )
    print(
        f"  MLP / Attention:      "
        f"{ratio:.8f}",
        flush=True,
    )

    # With MLP_DIMENSION = 4 * HIDDEN_SIZE this is approximately 2:1
    # when counting all trainable parameters, including biases.
    if not (1.95 <= ratio <= 2.05):
        raise AssertionError(
            "The Attention/MLP parameter ratio is outside "
            "the expected approximately 2:1 range."
        )

    del model


# ============================================================
# Data loading
# ============================================================

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


# ============================================================
# Evaluation / inference
# ============================================================

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


# ============================================================
# Adversarial robustness evaluation
# ============================================================

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

    return row


# ============================================================
# Optimizer / scheduler / training
# ============================================================

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


# ============================================================
# CSV persistence
# ============================================================

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


# ============================================================
# Experiment bookkeeping
# ============================================================

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


# ============================================================
# Data preparation
# ============================================================

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


# ============================================================
# Dense training
# ============================================================

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


# ============================================================
# Pruning preparation and evaluation
# ============================================================

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


# ============================================================
# Retraining workers
# ============================================================

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


# ============================================================
# Result schemas
# ============================================================

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


# ============================================================
# Main
# ============================================================

def main():
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

    print(
        "Done.",
        flush=True,
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
