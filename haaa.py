"""HAAA: shared three-head hypergraph attention layer."""
from __future__ import annotations

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from hypergraph import Hypergraph


class HAAALayer(nn.Module):
    """
    Hypergraph-Aware Adaptive Attention layer (Eq. 6-10).

    Three messages, gated per-node by a context-adaptive MLP:
        m_local  : cosine node-node attention over co-membership neighbours
        m_set    : set-aware hyperedge-mediated aggregation
        m_global : spectral attention using Z = ReLU((2I - Theta) X W_g)
    For the student, the node-node attention is restricted to the Top-K
    neighbours per row (N_i^K), giving O(K|V|d) cost.

    Returns the updated features and the (row-stochastic) hybrid node-node
    attention matrix, reused by AKED and by the attention-transfer loss.
    """

    def __init__(self, in_dim: int, out_dim: int, num_heads: int,
                 tau: float, dropout: float):
        super().__init__()
        assert out_dim % num_heads == 0
        self.tau = tau
        self.Wv = nn.Linear(in_dim, out_dim, bias=False)   # value / local-global proj
        self.Wg = nn.Linear(in_dim, out_dim, bias=False)   # spectral projection
        self.gate = nn.Sequential(                          # context-adaptive omega_i
            nn.Linear(out_dim + 2, 32), nn.ReLU(), nn.Linear(32, 3))
        self.res = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _masked_softmax(scores: Tensor, mask: Tensor) -> Tensor:
        scores = scores.masked_fill(~mask, float("-inf"))
        return torch.softmax(scores, dim=-1)

    def forward(self, x: Tensor, hg: "Hypergraph", spec_op: Tensor,
                A_mask: Tensor, deg_feats: Tensor,
                top_k: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        h = self.Wv(x)                                       # (N, d)

        # (a) local node-node cosine attention -------------------------------
        hn = F.normalize(h, dim=-1)
        local_score = (hn @ hn.t()) / self.tau               # (N,N)
        local_attn = self._masked_softmax(local_score, A_mask)

        # (c) global spectral attention --------------------------------------
        z = F.relu(spec_op @ self.Wg(x))                     # (N, d)
        zn = F.normalize(z, dim=-1)
        global_attn = self._masked_softmax(zn @ zn.t(), A_mask)

        # context-adaptive gate omega_i in the 2-simplex ---------------------
        omega = torch.softmax(self.gate(torch.cat([h, deg_feats], -1)), dim=-1)  # (N,3)

        # hybrid node-node attention (used for Top-K, AKED, L_attn) ----------
        hybrid = omega[:, 0:1] * local_attn + omega[:, 2:3] * global_attn
        hybrid = hybrid / (hybrid.sum(-1, keepdim=True) + 1e-9)

        # Top-K sparsification for the student (N_i^K) -----------------------
        if top_k is not None and top_k < hybrid.size(0):
            k = min(top_k, hybrid.size(0))
            topv, topi = hybrid.topk(k, dim=-1)
            keep = torch.zeros_like(hybrid).scatter_(1, topi, 1.0)
            local_attn = local_attn * keep
            global_attn = global_attn * keep
            hybrid = hybrid * keep
            local_attn = local_attn / (local_attn.sum(-1, keepdim=True) + 1e-9)
            global_attn = global_attn / (global_attn.sum(-1, keepdim=True) + 1e-9)
            hybrid = hybrid / (hybrid.sum(-1, keepdim=True) + 1e-9)

        # (b) set-aware hyperedge aggregation --------------------------------
        h_e = (hg.H.t() @ h) / hg.deg_e.unsqueeze(-1)        # (M, d) mean of members
        m_set = (hg.H @ h_e) / hg.deg_v.unsqueeze(-1)        # (N, d) scatter back

        # combine the three gated messages -----------------------------------
        m_local = local_attn @ h
        m_global = global_attn @ h
        out = omega[:, 0:1] * m_local + omega[:, 1:2] * m_set + omega[:, 2:3] * m_global
        out = self.norm(out + self.res(x))
        out = self.drop(F.relu(out))
        return out, hybrid
