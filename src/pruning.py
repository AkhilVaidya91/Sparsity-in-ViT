"""Wanda scoring and unstructured mask creation."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import (
    ATTENTION_HEADS, CALIB_BATCH_SIZE, CALIB_SAMPLES, DROPOUT, HIDDEN_SIZE,
    MLP_DIMENSION, PRUNE_TYPES, PRUNING_EVAL_NUM_WORKERS, PRUNING_METHOD,
    REQUESTED_RATIOS, TRANSFORMER_BLOCKS,
)
from .data import seed_worker
from .model import VisionTransformer


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
