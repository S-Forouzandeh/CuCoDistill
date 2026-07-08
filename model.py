"""CuCoModel: shared backbone + full teacher + Top-K student."""
from __future__ import annotations

from typing import Dict
import torch
import torch.nn as nn
from torch import Tensor
from config import Config
from haaa import HAAALayer
from hypergraph import Hypergraph


class CuCoModel(nn.Module):
    """
    Shared input projection + shared first HAAA layer, then a full-neighbourhood
    teacher path and a Top-K student path.  Sharing the backbone is what realises
    the bidirectional co-evolutionary feedback: gradients from both paths update
    the shared parameters.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim
        self.in_proj = nn.Linear(cfg.num_features, d)
        self.shared = HAAALayer(d, d, cfg.num_heads, cfg.tau_node, cfg.dropout)

        self.teacher_layers = nn.ModuleList(
            HAAALayer(d, d, cfg.num_heads, cfg.tau_node, cfg.dropout)
            for _ in range(cfg.teacher_layers - 1))
        self.student_layers = nn.ModuleList(
            HAAALayer(d, d, max(1, cfg.num_heads // 2), cfg.tau_node, cfg.dropout)
            for _ in range(cfg.student_layers - 1))

        self.teacher_cls = nn.Linear(d, cfg.num_classes)
        self.student_cls = nn.Linear(d, cfg.num_classes)
        self.align = nn.Linear(d, d)                          # student -> teacher emb

    # ---- shared backbone ----
    def _backbone(self, X, hg, spec_op, A, degf):
        h0 = self.in_proj(X)
        h, _ = self.shared(h0, hg, spec_op, A, degf, top_k=None)
        return h

    def teacher_forward(self, X, hg, spec_op, A, degf) -> Dict[str, Tensor]:
        h = self._backbone(X, hg, spec_op, A, degf)
        feats, attn = [h], None
        for layer in self.teacher_layers:
            h, attn = layer(h, hg, spec_op, A, degf, top_k=None)
            feats.append(h)
        return {"emb": h, "logits": self.teacher_cls(h), "feats": feats, "attn": attn}

    def student_forward(self, X, hg, spec_op, A, degf, K) -> Dict[str, Tensor]:
        h = self._backbone(X, hg, spec_op, A, degf)
        feats, attn = [h], None
        for layer in self.student_layers:
            h, attn = layer(h, hg, spec_op, A, degf, top_k=K)
            feats.append(h)
        return {"emb": h, "logits": self.student_cls(h),
                "feats": feats, "attn": attn, "aligned": self.align(h)}
