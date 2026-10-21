# Pruning and retraining

## Concept

Pruning removes low-importance weight elements from a trained ViT. This
reduces the number of active parameters in either the attention blocks or the
MLP blocks. Retraining allows the remaining weights to adapt after pruning.

This project uses unstructured pruning. It removes individual elements, not
whole neurons, heads, or layers.

## Method

### Pruning

1. Collect activation statistics from a calibration subset.
2. Compute a Wanda score for each weight:

   `score = |weight| x L2 norm of the corresponding input activation`

3. Combine scores within the selected functional group.
4. Remove the lowest-scoring requested fraction.
5. Store the result as a fixed boolean mask.
6. Evaluate the masked dense-trained model before retraining.

The configured group ratios are 20%, 40%, 60%, and 80%. The groups are
`attention` and `mlp`. The ratio applies only to the selected group, not to
the complete model.

The mask is applied during the forward pass. After each optimizer update, the
code enforces the mask again. This keeps pruned parameters at zero, including
when AdamW weight decay is active.

### Retraining

The same dense-derived mask is used for both initialization methods:

- `original_init`: start from the saved dense model initialization.
- `random_reinit`: create a new model with a deterministic seed.

Each masked model trains for up to 80 epochs with the normal validation,
early-stopping, checkpoint, and test-evaluation flow. Results include group
sparsity, effective model sparsity, accuracy, F1, ROC-AUC, and inference time.

## Code reference

- Wanda statistics and scores: [`src/pruning.py`](../src/pruning.py),
  `collect_wanda_stats` and `compute_unstructured_wanda_scores`.
- Mask creation: [`src/pruning.py`](../src/pruning.py),
  `build_unstructured_mask`.
- Mask application and enforcement: [`src/pruning.py`](../src/pruning.py),
  `apply_unstructured_mask` and `enforce_unstructured_masks`.
- Masked layers: [`src/model.py`](../src/model.py), `MaskedLinear` and
  `MaskedMultiheadAttention`.
- Pruning evaluation: [`src/experiments.py`](../src/experiments.py),
  `evaluate_pruned_models`.
- Retraining setup: [`src/experiments.py`](../src/experiments.py),
  `build_retraining_model` and `training_worker`.
- Pipeline orchestration: [`src/pipeline.py`](../src/pipeline.py), `main`.