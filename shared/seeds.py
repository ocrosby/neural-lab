"""Reproducibility: seed every RNG the projects touch.

Every project's entry point should call `set_seed(N)` before any tensor op or
data shuffle. Deterministic runs are the difference between "I fixed a bug" and
"I got lucky."
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42, *, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA if available).

    Args:
        seed: Non-negative integer used for every RNG.
        deterministic_torch: When True, set PyTorch to use deterministic
            algorithms where available. Slows some ops (esp. cuDNN conv) but
            eliminates a class of "flaky loss curve" bugs.
    """
    assert seed >= 0, f"seed must be non-negative, got {seed}"

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
