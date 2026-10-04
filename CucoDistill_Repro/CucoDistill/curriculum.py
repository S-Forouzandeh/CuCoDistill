"""Spectral curriculum: difficulties, quantile gates, lambda schedule."""
from __future__ import annotations

import math
from typing import Tuple
import torch
import torch.nn.functional as F
from torch import Tensor
from config import Config


class SpectralCurriculum:
    """Difficulty metrics, quantile thresholds, and the lambda schedule (Eq. 21-26)."""

    def __init__(self, cfg: Config):
        self.T = cfg.epochs
        self.lambda3 = cfg.lambda3_task

    def loss_weights(self, ep: int) -> Tuple[float, float, float]:
        t = ep / max(1, self.T)
        lam1 = 0.5 * math.sqrt(t)             # distillation grows  (Eq. 25)
        lam2 = 0.3 * math.exp(-t)             # contrastive decays
        return lam1, lam2, self.lambda3

    @staticmethod
    def contrastive_difficulty(e_clean: Tensor, e_aug: Tensor) -> Tensor:
        return 1.0 - F.cosine_similarity(e_clean, e_aug, dim=-1)   # D_contrast (Eq. 21)

    @staticmethod
    def distill_difficulty(e_t: Tensor, e_s: Tensor) -> Tensor:
        return (e_t - e_s).norm(dim=-1)                            # D_distill  (Eq. 22)

    def gate(self, difficulty: Tensor, ep: int, kind: str) -> Tensor:
        """Easy-to-hard quantile gate: include nodes with difficulty <= tau(ep)."""
        t = ep / max(1, self.T)
        if kind == "distill":          # tighter window 0.2 -> 0.4 (paper beta_ep)
            q = 0.2 + 0.2 * math.sqrt(t)
        else:                          # contrastive: admit progressively harder
            q = 0.5 + 0.5 * t
        tau = torch.quantile(difficulty.detach(), min(1.0, q))
        return (difficulty.detach() <= tau).float()
