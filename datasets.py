"""Dataset interface: synthetic generator + real-benchmark loader.

Every dataset returns the same tuple (X, labels, Hypergraph, masks) so the
trainer and baselines are dataset-agnostic.  The real benchmarks (DBLP, IMDB,
Yelp, Cora, Citeseer, the DBLP variants, and the higher-order sets) are NOT
redistributed here; load_real reads them from a standard on-disk format that
you populate from each dataset's original source (see README).
"""
from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor

from config import Config
from hypergraph import Hypergraph


def make_splits(n: int, seed: int, frac=(0.6, 0.2, 0.2)) -> Dict[str, Tensor]:
    """Deterministic 60/20/20 train/val/test masks for a given seed."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_tr, n_va = int(frac[0] * n), int(frac[1] * n)
    masks = {k: torch.zeros(n, dtype=torch.bool) for k in ("train", "val", "test")}
    masks["train"][perm[:n_tr]] = True
    masks["val"][perm[n_tr:n_tr + n_va]] = True
    masks["test"][perm[n_tr + n_va:]] = True
    return masks


class SyntheticHypergraph:
    """
    Cora-like generator with *controllable feature redundancy and noise*, so the
    student-superiority regime (R(X) > R*) can actually arise.  Augmentation noise
    is applied later by AKED at std 0.01; this is the generative noise.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def build(self) -> Tuple[Tensor, Tensor, Hypergraph, Dict[str, Tensor]]:
        c = self.cfg
        # low-dimensional class signal + many independent noisy/redundant
        # dimensions: this is the regime where the full-capacity teacher overfits
        # the noise dims and the Top-K student generalises better (R(X) > R*).
        sig_dim = max(4, 2 * c.num_classes)
        centers = torch.randn(c.num_classes, sig_dim)
        labels = torch.randint(0, c.num_classes, (c.num_nodes,))
        signal = centers[labels] + torch.randn(c.num_nodes, sig_dim) * c.data_feature_noise
        noise_dim = max(0, c.num_features - sig_dim)
        noise = torch.randn(c.num_nodes, noise_dim)          # uninformative, full-rank
        X = torch.cat([signal, noise], dim=-1)[:, :c.num_features]

        # label noise on a few nodes
        n_flip = int(c.label_noise * c.num_nodes)
        flip = torch.randperm(c.num_nodes)[:n_flip]
        labels[flip] = torch.randint(0, c.num_classes, (n_flip,))

        # hyperedges: mostly homophilous groups of size 3-6
        H = torch.zeros(c.num_nodes, c.num_hyperedges)
        for e in range(c.num_hyperedges):
            cls = e % c.num_classes
            pool = (labels == cls).nonzero(as_tuple=True)[0]
            if len(pool) == 0:
                pool = torch.arange(c.num_nodes)
            size = int(torch.randint(6, 13, (1,)).item())
            members = pool[torch.randint(0, len(pool), (size,))]
            H[members, e] = 1.0
        H[H.sum(1) == 0, torch.randint(0, c.num_hyperedges, ((H.sum(1) == 0).sum(),))] = 1.0

        # 60/20/20 split
        perm = torch.randperm(c.num_nodes)
        n_tr, n_va = int(0.6 * c.num_nodes), int(0.2 * c.num_nodes)
        masks = {k: torch.zeros(c.num_nodes, dtype=torch.bool)
                 for k in ("train", "val", "test")}
        masks["train"][perm[:n_tr]] = True
        masks["val"][perm[n_tr:n_tr + n_va]] = True
        masks["test"][perm[n_tr + n_va:]] = True
        return X, labels, Hypergraph(H), masks


def _incidence_from_hyperedges(n_nodes: int, hyperedges) -> Tensor:
    M = len(hyperedges)
    H = torch.zeros(n_nodes, M)
    for e, nodes in enumerate(hyperedges):
        for v in nodes:
            H[int(v), e] = 1.0
    H[H.sum(1) == 0] = 0.0
    return H


def load_real(name: str, root: str = "data_files",
              seed: int = 5) -> Tuple[Tensor, Tensor, Hypergraph, Dict[str, Tensor]]:
    """
    Load a benchmark from {root}/{name}/ in the standard format:
        features.npy     float array  (N, F)
        labels.npy       int   array  (N,)
        hyperedges.txt   one hyperedge per line: space-separated node indices
        splits/{seed}.npz  (optional) arrays 'train','val','test' (bool, len N)
    If a split file is absent, a deterministic 60/20/20 split is generated from
    .  See README for where to obtain each dataset.
    """
    d = os.path.join(root, name)
    if not os.path.isdir(d):
        raise FileNotFoundError(
            f"Dataset '{name}' not found at {d}. Place features.npy, labels.npy and "
            f"hyperedges.txt there (see README for sources), or use --dataset synthetic.")
    X = torch.from_numpy(np.load(os.path.join(d, "features.npy"))).float()
    labels = torch.from_numpy(np.load(os.path.join(d, "labels.npy"))).long()
    with open(os.path.join(d, "hyperedges.txt")) as f:
        hyperedges = [[int(x) for x in line.split()] for line in f if line.strip()]
    H = _incidence_from_hyperedges(X.size(0), hyperedges)

    split_path = os.path.join(d, "splits", f"{seed}.npz")
    if os.path.exists(split_path):
        z = np.load(split_path)
        masks = {k: torch.from_numpy(z[k]).bool() for k in ("train", "val", "test")}
    else:
        masks = make_splits(X.size(0), seed)
    return X, labels, Hypergraph(H), masks


def load_dataset(name: str, cfg: Config, seed: int = 5):
    """Dispatch: 'synthetic' uses the generator; anything else uses load_real."""
    if name == "synthetic":
        return SyntheticHypergraph(cfg).build()
    return load_real(name, seed=seed)
