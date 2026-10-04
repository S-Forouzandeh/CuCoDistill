"""Multi-level distillation, contrastive InfoNCE, spectral consistency."""
from __future__ import annotations

from typing import List
import torch
import torch.nn.functional as F
from torch import Tensor


class Losses:
    """Multi-level distillation, contrastive InfoNCE, and spectral consistency."""

    @staticmethod
    def soft_kd(s_logits: Tensor, t_logits: Tensor, T: float) -> Tensor:
        p_s = F.log_softmax(s_logits / T, -1)
        p_t = F.softmax(t_logits.detach() / T, -1)
        return F.kl_div(p_s, p_t, reduction="batchmean") * (T * T)

    @staticmethod
    def embed_align(s_emb: Tensor, t_emb: Tensor, w: Tensor) -> Tensor:
        """L_embed = sum_i w_i ||e_S - sg(e_T)||^2  (Eq. 15), curriculum-gated by w."""
        per_node = ((s_emb - t_emb.detach()) ** 2).sum(-1)
        return (w * per_node).sum() / (w.sum() + 1e-9)

    @staticmethod
    def attn_transfer(t_attn: Tensor, s_attn: Tensor, mask: Tensor) -> Tensor:
        """L_attn = KL(alpha_hybrid^T || beta^S) over neighbours (Eq. 16)."""
        P = t_attn.detach().clamp_min(1e-9)
        Q = s_attn.clamp_min(1e-9)
        kl = (P * (P.log() - Q.log()) * mask).sum(-1)
        return kl.mean()

    @staticmethod
    def feat_match(s_feats: List[Tensor], t_feats: List[Tensor], gamma: float) -> Tensor:
        """L_feat = sum_l gamma_l ||F_S - F_T||^2 (Eq. 17); deeper layers up-weighted."""
        L = min(len(s_feats), len(t_feats))
        loss = s_feats[0].new_zeros(())
        for l in range(L):
            w = gamma ** l
            loss = loss + w * F.mse_loss(s_feats[l], t_feats[l].detach())
        return loss / L

    @staticmethod
    def info_nce(z_clean: Tensor, z_aug: Tensor, tau: float) -> Tensor:
        """Per-node InfoNCE (positives = same node across views) (Eq. 26)."""
        a = F.normalize(z_clean, dim=-1)
        b = F.normalize(z_aug, dim=-1)
        logits = (a @ b.t()) / tau
        labels = torch.arange(a.size(0), device=a.device)
        return F.cross_entropy(logits, labels, reduction="none")  # (N,)
