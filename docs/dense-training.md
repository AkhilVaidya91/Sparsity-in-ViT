# Dense training

## Concept

Dense training creates the reference Vision Transformer (ViT). Every model
parameter is active. The result is the baseline for pruning, retraining, and
robustness comparisons.

The model classifies 43 GTSRB traffic-sign classes. It uses 32 x 32 images,
4 x 4 patches, six transformer blocks, and a 128-dimensional hidden size.

## Method

1. Build a new `VisionTransformer`.
2. Save its initial state and its SHA-256 hash.
3. Train it on the GTSRB training split.
4. Check validation loss after each epoch.
5. Save the best validation state.
6. Stop early after 10 epochs without improvement.
7. Evaluate the best state on the test split.

Training uses:

- AdamW.
- 10% learning-rate warm-up.
- Cosine learning-rate decay.
- Cross-entropy loss with 0.1 label smoothing.
- Gradient clipping at a norm of 1.0.
- Up to 80 epochs.

The dense state is the source state for pruning. The pipeline also records
accuracy, macro F1, ROC-AUC, inference time, checkpoints, and MLflow metrics.

## Code reference

- Model definition: [`src/model.py`](../src/model.py), `VisionTransformer`.
- Training loop: [`src/training.py`](../src/training.py), `train_model` and
  `_train_model_impl`.
- Baseline setup: [`src/experiments.py`](../src/experiments.py),
  `train_dense_model`.
- Experiment settings: [`src/config.py`](../src/config.py), `EPOCHS`,
  `LEARNING_RATE`, `WARMUP_PCT`, and `EARLY_STOPPING_PATIENCE`.
- End-to-end order: [`src/pipeline.py`](../src/pipeline.py), `main`.

Run the full pipeline from the repository root:

```bash
uv run vit-pruning
```

