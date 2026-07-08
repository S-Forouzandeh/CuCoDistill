"""H-SBM-RF: Hypergraph Stochastic Block Model with Redundant Features.

The controllable generator from the revised manuscript's synthetic section. It
independently varies the quantities that the theory predicts should govern
student superiority:

  * feature redundancy  R(X)  -- via the number of uninformative noise dimensions
  * spectral gap              -- via hyperedge homophily (cluster separation)
  * hyperedge cardinality     -- fixed or scale-free (heavy-tailed) sizes
  * density                   -- via the number of hyperedges
  * label / feature noise

Closed-form prediction (revised paper): the student attains lower risk than the
teacher exactly when ``R(X) > R* = K / d_eff`` and spectral coverage holds
(``K >= d_eff``).  Use :mod:`run_sweep` to vary one axis at a time and check the
prediction with or without training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor

from hypergraph import Hypergraph
from datasets import make_splits


@dataclass
class HSBMRFParams:
    n_nodes: int = 600
    n_classes: int = 5
    n_hyperedges: int = 320          # density knob
    signal_dim: int = 10             # informative feature dimensions
    noise_dim: int = 54              # redundant/uninformative dimensions (redundancy knob)
    homophily: float = 0.90          # P(a hyperedge is drawn within one class) -> spectral gap
    card_mode: str = "fixed"         # 'fixed' | 'scalefree'
    card_min: int = 6
    card_max: int = 12
    card_alpha: float = 2.5          # power-law exponent for scale-free cardinalities
    feature_noise: float = 0.35      # std of Gaussian noise on the signal block
    label_noise: float = 0.04        # fraction of flipped labels


def _cardinalities(p: HSBMRFParams, rng: np.random.Generator) -> np.ndarray:
    if p.card_mode == "scalefree":
        sizes = (rng.pareto(p.card_alpha, p.n_hyperedges) + 1.0) * p.card_min
    else:
        sizes = rng.integers(p.card_min, p.card_max + 1, p.n_hyperedges).astype(float)
    return np.clip(sizes, p.card_min, p.card_max).astype(int)


def generate(p: HSBMRFParams, seed: int = 5
             ) -> Tuple[Tensor, Tensor, Hypergraph, Dict[str, Tensor]]:
    rng = np.random.default_rng(seed)
    g = torch.Generator().manual_seed(seed)

    labels = torch.from_numpy(rng.integers(0, p.n_classes, p.n_nodes)).long()

    # ---- features: low-dim class signal + many independent noise dims ----
    centers = torch.randn(p.n_classes, p.signal_dim, generator=g)
    signal = centers[labels] + torch.randn(p.n_nodes, p.signal_dim, generator=g) * p.feature_noise
    noise = torch.randn(p.n_nodes, max(0, p.noise_dim), generator=g)
    X = torch.cat([signal, noise], dim=-1)

    # ---- hyperedges: homophilous with prob `homophily`, else random mix ----
    sizes = _cardinalities(p, rng)
    by_class = [np.where(labels.numpy() == c)[0] for c in range(p.n_classes)]
    H = torch.zeros(p.n_nodes, p.n_hyperedges)
    for e, sz in enumerate(sizes):
        sz = int(min(sz, p.n_nodes))
        if rng.random() < p.homophily:
            c = int(rng.integers(0, p.n_classes))
            pool = by_class[c] if len(by_class[c]) > 0 else np.arange(p.n_nodes)
            members = rng.choice(pool, size=min(sz, len(pool)),
                                 replace=len(pool) < sz)
        else:
            members = rng.choice(p.n_nodes, size=sz, replace=False)
        H[members, e] = 1.0

    # connect any isolated node
    for v in (H.sum(1) == 0).nonzero(as_tuple=True)[0].tolist():
        H[v, int(rng.integers(0, p.n_hyperedges))] = 1.0

    # ---- label noise ----
    n_flip = int(p.label_noise * p.n_nodes)
    if n_flip > 0:
        idx = rng.choice(p.n_nodes, n_flip, replace=False)
        labels[idx] = torch.from_numpy(rng.integers(0, p.n_classes, n_flip)).long()

    masks = make_splits(p.n_nodes, seed)
    return X.float(), labels, Hypergraph(H), masks
