"""Central configuration for the Vision Transformer pruning experiments."""

import os
from pathlib import Path

MEAN = (0.3337, 0.3064, 0.3171)

STD = (0.2672, 0.2564, 0.2629)

IMAGE_SIZE = 32

NUM_CHANNELS = 3

NUM_CLASSES = 43

PATCH_SIZE = 4

HIDDEN_SIZE = 128

TRANSFORMER_BLOCKS = 6

ATTENTION_HEADS = 4

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

VAL_FRACTION = 0.10

SPLIT_SEED = 12345

EARLY_STOPPING_PATIENCE = 10

EARLY_STOPPING_MIN_DELTA = 0.0

USE_AMP = True

RETRAIN_TRAIN_NUM_WORKERS = 1

RETRAIN_EVAL_NUM_WORKERS = 1

DENSE_TRAIN_NUM_WORKERS = 2

DENSE_EVAL_NUM_WORKERS = 2

PRUNING_EVAL_NUM_WORKERS = 2

CHECKPOINT_EVERY_EPOCHS = 5

DATA_SEED = 12345

REQUESTED_RATIOS = [0.20, 0.40, 0.60, 0.80]

PRUNE_TYPES = ["attention", "mlp"]

INIT_METHODS = ["random_reinit", "original_init"]

PRUNING_METHOD = "wanda_unstructured"

CALIB_SEED = 0

CALIB_SAMPLES = 512

CALIB_BATCH_SIZE = 64

ADV_ATTACKS = ["fgsm", "pgd"]

ADV_EPSILONS = [2 / 255, 4 / 255, 8 / 255]

PGD_STEPS = 10

PGD_ALPHA_RATIO = 0.25

PGD_RANDOM_START = True

ADV_EVAL_BATCH_SIZE = 128

ADV_EVAL_NUM_WORKERS = 0

ADV_EVAL_SAMPLES = None  # None = use the complete held-out test set.

ADV_EVAL_SEED = 20261006

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORK_DIR = Path(os.environ.get("OUTPUT_DIR", PROJECT_ROOT / "outputs")).expanduser().resolve()
DATA_ROOT = Path(os.environ.get("GTSRB_DATA_ROOT", PROJECT_ROOT / "data" / "gtsrb")).expanduser().resolve()

DENSE_RESULTS_CSV = WORK_DIR / "dense_results.csv"
PRUNING_RESULTS_CSV = WORK_DIR / "pruning_results.csv"
RETRAINING_RESULTS_CSV = WORK_DIR / "retraining_results.csv"
TRAINING_HISTORY_CSV = WORK_DIR / "training_history.csv"
ALL_RESULTS_CSV = WORK_DIR / "all_results.csv"
ADVERSARIAL_RESULTS_CSV = WORK_DIR / "adversarial_results.csv"
CHECKPOINT_DIR = WORK_DIR / "training_checkpoints"
LOG_DIR = WORK_DIR / "logs"

MLFLOW_DB_PATH = Path(os.environ.get("MLFLOW_DB_PATH", WORK_DIR / "mlflow.db")).expanduser().resolve()
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{MLFLOW_DB_PATH.as_posix()}",
)
MLFLOW_EXPERIMENT_NAME = os.environ.get(
    "MLFLOW_EXPERIMENT_NAME",
    "vit-structured-pruning",
)

for directory in (WORK_DIR, DATA_ROOT, CHECKPOINT_DIR, LOG_DIR, MLFLOW_DB_PATH.parent):
    directory.mkdir(parents=True, exist_ok=True)
