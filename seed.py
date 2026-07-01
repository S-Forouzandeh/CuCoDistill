"""Reproducibility: a single base seed applied uniformly to every RNG."""
from __future__ import annotations

import random
import numpy as np
import torch


def set_seed(seed: int = 5, deterministic: bool = True) -> None:
    """Seed every RNG used in training for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_list(n: int, base: int = 5) -> list:
    """Deterministic seed list anchored at `base` (default 5).
    Main results use n=5 -> [5,6,7,8,9]; significance uses n=10 -> [5..14].
    Each run seeds every RNG with one value via set_seed."""
    return [base + i for i in range(n)]
