"""Random seed and model-state helpers."""

import hashlib
import random

import numpy as np
import torch


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
