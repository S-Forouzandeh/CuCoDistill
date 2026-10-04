"""Official baseline implementations via the DHG (DeepHypergraph) library.

DHG (https://github.com/iMoonLab/DeepHypergraph) is maintained by the original
HGNN authors' group (Gao, Feng et al.), so its ``HGNN``, ``HGNNP``, ``HyperGCN``
and ``HNHN`` are the *official* reference implementations of those methods, and
``UniGCN/UniGAT/UniSAGE/UniGIN`` (Huang & Yang, 2021) are official too.

This module adapts them to the package's ``forward(X, hg) -> logits`` interface
by converting our :class:`hypergraph.Hypergraph` (incidence matrix) into a
``dhg.Hypergraph`` (hyperedge list).  If DHG is not installed, ``OFFICIAL`` is
empty and :mod:`baselines` falls back to the in-repo reference implementations.

Install:  ``pip install dhg``  (DHG pins numpy<2 / torch<2 conservatively, but
the models run on newer versions; see README).
"""
from __future__ import annotations

from functools import partial
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from config import Config
from hypergraph import Hypergraph

try:
    import warnings
    warnings.filterwarnings("ignore", message=".*[Ss]parse.*")
    import dhg
    from dhg import Hypergraph as _DHGHypergraph
    HAVE_DHG = True
except Exception:                       # pragma: no cover
    HAVE_DHG = False


def to_dhg_hypergraph(hg: Hypergraph) -> "object":
    """Convert an incidence-matrix Hypergraph to a dhg.Hypergraph (edge list)."""
    H = hg.H
    e_list = []
    for e in range(hg.M):
        nodes = torch.nonzero(H[:, e] > 0, as_tuple=False).flatten().tolist()
        if len(nodes) >= 2:             # dhg expects hyperedges of size >= 2
            e_list.append(nodes)
    return _DHGHypergraph(hg.N, e_list, device=H.device)


class _OfficialWrapper(nn.Module):
    """Wrap a built dhg model; cache the converted hypergraph across epochs."""

    def __init__(self, builder: Callable[[], nn.Module]):
        super().__init__()
        self.net = builder()
        self._dhg = None
        self._key = None

    def _graph(self, hg: Hypergraph):
        key = (id(hg), hg.M, hg.N)
        if self._dhg is None or self._key != key:
            self._dhg = to_dhg_hypergraph(hg)
            self._key = key
        return self._dhg

    def forward(self, X: Tensor, hg: Hypergraph) -> Tensor:
        return self.net(X, self._graph(hg))


# ---- builders for each official model (in_dim, hidden, classes, drop) ----
def _hgnn(F, H, C, dr):     from dhg.models import HGNN;     return HGNN(F, H, C, use_bn=True, drop_rate=dr)
def _hgnnp(F, H, C, dr):    from dhg.models import HGNNP;    return HGNNP(F, H, C, use_bn=True, drop_rate=dr)
def _hypergcn(F, H, C, dr): from dhg.models import HyperGCN; return HyperGCN(F, H, C, use_mediator=True, drop_rate=dr)
def _hnhn(F, H, C, dr):     from dhg.models import HNHN;     return HNHN(F, H, C, drop_rate=dr)


def _make(builder, cfg: Config, hidden: int = 64) -> _OfficialWrapper:
    return _OfficialWrapper(lambda: builder(cfg.num_features, hidden, cfg.num_classes, cfg.dropout))


def official_backends() -> Dict[str, Callable[[Config], nn.Module]]:
    """name -> factory(cfg) -> nn.Module, for every official model available."""
    if not HAVE_DHG:
        return {}
    reg: Dict[str, Callable] = {
        "hgnn":     partial(_make, _hgnn),
        "hgnnp":    partial(_make, _hgnnp),
        "hypergcn": partial(_make, _hypergcn),
        "hnhn":     partial(_make, _hnhn),
    }
    # optional Uni-* family (official) — signatures vary, so probe-build each
    import dhg.models as _M
    candidates = [("unigcn", "UniGCN", {}), ("unigat", "UniGAT", {"num_heads": 4}),
                  ("unisage", "UniSAGE", {}), ("unigin", "UniGIN", {})]
    for name, cname, extra in candidates:
        cls = getattr(_M, cname, None)
        if cls is None:
            continue
        try:                            # verify the (in, hid, classes, **extra) signature works
            cls(4, 8, 3, drop_rate=0.5, **extra)
        except Exception:
            continue
        def _b(F, H, C, dr, _cls=cls, _extra=extra):
            return _cls(F, H, C, drop_rate=dr, **_extra)
        reg[name] = partial(_make, _b)
    return reg


OFFICIAL: Dict[str, Callable[[Config], nn.Module]] = official_backends()
