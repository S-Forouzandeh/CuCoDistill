"""Adaptive Knowledge-guided Edge Dropping (AKED)."""
from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor
from config import Config


class AKED(nn.Module):
    """Adaptive Knowledge-guided Edge Dropping + feature augmentation (Eq. 3-5)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.mu1 = cfg.aked_mu1
        self.mu2 = cfg.aked_mu2
        self.rho0 = cfg.aked_rho0
        self.p_feat = cfg.aug_feat_mask
        self.noise = cfg.aug_feat_noise          # 0.01, the paper's value

    def edge_signals(self, H: Tensor, hybrid_attn: Tensor,
                     t_emb: Tensor, s_emb: Tensor) -> Tuple[Tensor, Tensor]:
        deg_e = H.sum(0).clamp(min=1.0)                       # (M,)
        # s_attn(e): mean hybrid attention among members (Eq. 3)
        num = torch.einsum("ne,nm,me->e", H, hybrid_attn, H)  # sum_{i,j in e} a_ij
        s_attn = num / (deg_e * deg_e)
        # s_kd(e): mean teacher-student embedding gap among members (Eq. 4)
        gap = (t_emb - s_emb).norm(dim=-1)                    # (N,)
        s_kd = (H.t() @ gap) / deg_e
        return s_attn, s_kd

    def forward(self, X: Tensor, H: Tensor, hybrid_attn: Tensor,
                t_emb: Tensor, s_emb: Tensor, ep: int, total: int,
                training: bool = True) -> Tuple[Tensor, Tensor]:
        s_attn, s_kd = self.edge_signals(H, hybrid_attn, t_emb, s_emb)
        rho = self.rho0 * (1.0 - ep / max(1, total))          # decaying threshold
        p_retain = torch.sigmoid(self.mu1 * s_attn + self.mu2 * s_kd - rho)  # (M,)
        if training:
            keep = torch.bernoulli(p_retain)
        else:
            keep = (p_retain > 0.5).float()
        H_aug = H * keep.unsqueeze(0)

        # feature masking + small Gaussian noise -----------------------------
        if training:
            fmask = torch.bernoulli(torch.full_like(X, 1.0 - self.p_feat))
            X_aug = X * fmask + torch.randn_like(X) * self.noise
        else:
            X_aug = X
        return X_aug, H_aug
