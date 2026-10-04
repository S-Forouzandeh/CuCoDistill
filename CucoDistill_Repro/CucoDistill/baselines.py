"""Baseline models with a unified interface for the reproducibility package.

Each baseline implements ``forward(X, hg) -> logits`` so it can be trained and
evaluated by the same loop (``train_baseline``).  These are clean reference
implementations of the standard families the paper compares against; swap in the
authors' official code where you need exact parity.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from config import Config
from hypergraph import Hypergraph
from official import OFFICIAL, HAVE_DHG


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def clique_adjacency(hg: Hypergraph, normalise: bool = True) -> Tensor:
    """Symmetric clique-expansion adjacency A = H H^T (with self-loops)."""
    A = (hg.H @ hg.H.t())
    A.fill_diagonal_(0.0)
    A = (A > 0).float() + torch.eye(hg.N, device=hg.H.device)
    if normalise:
        d = A.sum(1).clamp(min=1.0).pow(-0.5)
        A = A * d.unsqueeze(0) * d.unsqueeze(1)
    return A


class MLP(nn.Module):
    """Pure-MLP student (GLNN-style): ignores structure, uses node features only."""

    def __init__(self, cfg: Config, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.num_features, hidden), nn.ReLU(), nn.Dropout(cfg.dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(cfg.dropout),
            nn.Linear(hidden, cfg.num_classes))

    def forward(self, X: Tensor, hg: Hypergraph) -> Tensor:
        return self.net(X)


class HGNN(nn.Module):
    """Feng et al. (2019): X' = sigma(Theta X W) using the hypergraph Laplacian."""

    def __init__(self, cfg: Config, hidden: int = 64, layers: int = 2):
        super().__init__()
        dims = [cfg.num_features] + [hidden] * (layers - 1) + [cfg.num_classes]
        self.lins = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(layers))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, X: Tensor, hg: Hypergraph) -> Tensor:
        theta = hg.theta()
        h = X
        for i, lin in enumerate(self.lins):
            h = theta @ lin(h)
            if i < len(self.lins) - 1:
                h = self.drop(F.relu(h))
        return h


class HyperGCN(nn.Module):
    """Yadati et al. (2019), simplified: GCN over the clique expansion of H."""

    def __init__(self, cfg: Config, hidden: int = 64, layers: int = 2):
        super().__init__()
        dims = [cfg.num_features] + [hidden] * (layers - 1) + [cfg.num_classes]
        self.lins = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(layers))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, X: Tensor, hg: Hypergraph) -> Tensor:
        A = clique_adjacency(hg, normalise=True)
        h = X
        for i, lin in enumerate(self.lins):
            h = A @ lin(h)
            if i < len(self.lins) - 1:
                h = self.drop(F.relu(h))
        return h


class HyperGAT(nn.Module):
    """Bai et al. (2021), simplified: single-head attention over clique neighbours."""

    def __init__(self, cfg: Config, hidden: int = 64, layers: int = 2):
        super().__init__()
        dims = [cfg.num_features] + [hidden] * (layers - 1) + [cfg.num_classes]
        self.lins = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(layers))
        self.att = nn.ModuleList(nn.Linear(2 * dims[i + 1], 1) for i in range(layers))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, X: Tensor, hg: Hypergraph) -> Tensor:
        mask = ((hg.H @ hg.H.t()) > 0)
        mask.fill_diagonal_(True)
        h = X
        for i, lin in enumerate(self.lins):
            z = lin(h)                                   # (N, d')
            n = z.size(0)
            zi = z.unsqueeze(1).expand(n, n, z.size(1))
            zj = z.unsqueeze(0).expand(n, n, z.size(1))
            e = F.leaky_relu(self.att[i](torch.cat([zi, zj], -1)).squeeze(-1), 0.2)
            e = e.masked_fill(~mask, float("-inf"))
            a = torch.softmax(e, dim=-1)
            h = a @ z
            if i < len(self.lins) - 1:
                h = self.drop(F.relu(h))
        return h


BASELINES = {"mlp": MLP, "hgnn": HGNN, "hypergcn": HyperGCN, "hypergat": HyperGAT}

# Prefer the official DHG implementations when the library is installed:
# this overrides 'hgnn'/'hypergcn' with the authors' code and adds 'hgnnp',
# 'hnhn', and the Uni-* family.  The in-repo reference versions remain available
# under a '_ref' suffix for side-by-side comparison.  If DHG is absent, the
# reference implementations above are used unchanged.
_REFERENCE = dict(BASELINES)
if OFFICIAL:
    for name in OFFICIAL:
        if name in _REFERENCE:
            BASELINES[f"{name}_ref"] = _REFERENCE[name]
    BASELINES.update(OFFICIAL)


def is_official(name: str) -> bool:
    """True if `name` resolves to an official DHG-backed implementation."""
    return name in OFFICIAL


def backend_label(name: str) -> str:
    return "official (DHG)" if is_official(name) else "reference (in-repo)"


# ---------------------------------------------------------------------------
# Generic baseline training loop
# ---------------------------------------------------------------------------
def train_baseline(name: str, X: Tensor, hg: Hypergraph, labels: Tensor,
                   masks: Dict[str, Tensor], cfg: Config,
                   epochs: int = 200, verbose: bool = False) -> Dict[str, float]:
    model = BASELINES[name](cfg).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_val, best_test = 0.0, 0.0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(X, hg)
        loss = F.cross_entropy(out[masks["train"]], labels[masks["train"]])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        model.eval()
        with torch.no_grad():
            out = model(X, hg)
            va = (out[masks["val"]].argmax(-1) == labels[masks["val"]]).float().mean().item()
            te = (out[masks["test"]].argmax(-1) == labels[masks["test"]]).float().mean().item()
        if va >= best_val:
            best_val, best_test = va, te
        if verbose and (ep + 1) % 50 == 0:
            print(f"    [{name}] ep {ep+1} loss {loss.item():.3f} val {va:.3f}")
    return {"val_acc": best_val, "test_acc": best_test}
