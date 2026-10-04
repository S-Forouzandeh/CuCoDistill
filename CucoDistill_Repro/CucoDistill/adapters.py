"""Adapters for plugging external baseline repositories behind the package's
``forward(X, hg) -> logits`` interface.

These cover the published methods *not* in DHG (Hyper-SAGNN, CHGNN, HyGCL-AdT)
and provide the substrate used by :mod:`kd` for the KD baselines. Third-party
code is **not** bundled: each :class:`ExternalSpec` records the official repo,
the class to import, and the graph representation it expects. ``build_external``
imports and wraps the model if its repo is on ``PYTHONPATH``; otherwise it raises
a clear, step-by-step error (mirrored in ``INTEGRATION.md``). Edit each spec's
``builder`` to match the repo's actual constructor.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from config import Config
from hypergraph import Hypergraph


# --------------------------------------------------------------------------- #
# Graph-representation converters (external repos expect different inputs)      #
# --------------------------------------------------------------------------- #
def to_incidence(hg: Hypergraph) -> Tensor:
    return hg.H


def to_clique_adj(hg: Hypergraph, normalise: bool = True) -> Tensor:
    A = (hg.H @ hg.H.t())
    A.fill_diagonal_(0.0)
    A = (A > 0).float() + torch.eye(hg.N, device=hg.H.device)
    if normalise:
        d = A.sum(1).clamp(min=1.0).pow(-0.5)
        A = A * d.unsqueeze(0) * d.unsqueeze(1)
    return A


def to_edge_index(hg: Hypergraph) -> Tensor:
    A = (hg.H @ hg.H.t())
    A.fill_diagonal_(0.0)
    return (A > 0).nonzero(as_tuple=False).t().contiguous()   # (2, E)


def to_dhg(hg: Hypergraph):
    from official import to_dhg_hypergraph
    return to_dhg_hypergraph(hg)


CONVERTERS: Dict[str, Callable] = {
    "incidence": to_incidence, "clique_adj": to_clique_adj,
    "edge_index": to_edge_index, "dhg": to_dhg,
}


# --------------------------------------------------------------------------- #
@dataclass
class ExternalSpec:
    name: str
    paper: str
    repo: str
    module: str                                  # import path once repo is on PYTHONPATH
    cls: str                                     # class to import
    graph: str                                   # key into CONVERTERS
    note: str = ""
    builder: Optional[Callable[[Config, int], nn.Module]] = None
    call: Optional[Callable] = None              # call(net, X, graph) -> logits


# Official repos for the published baselines DHG does not provide.  Verify the
# URLs against each paper; for the authors' own methods use their release.
EXTERNAL_SPECS: Dict[str, ExternalSpec] = {
    "hyper_sagnn": ExternalSpec(
        "hyper_sagnn", "Zhang et al., 2019",
        "https://github.com/ma-compbio/Hyper-SAGNN",
        module="hyper_sagnn", cls="Hyper_SAGNN", graph="edge_index",
        note="Self-attention HGNN over variable-sized hyperedges."),
    "chgnn": ExternalSpec(
        "chgnn", "Song et al., 2024 (IEEE TKDE)",
        "official repository linked in the paper",
        module="chgnn", cls="CHGNN", graph="incidence",
        note="Contrastive hypergraph net; needs incidence + augmented views."),
    "hygcl_adt": ExternalSpec(
        "hygcl_adt", "Qian et al., 2024 (WWW Companion)",
        "official repository linked in the paper",
        module="hygcl_adt", cls="HyGCL_AdT", graph="incidence",
        note="Dual-channel adaptive-topology contrastive learning."),
}


class ExternalModel(nn.Module):
    """Wrap an external model; cache the converted graph across epochs."""

    def __init__(self, net: nn.Module, spec: ExternalSpec):
        super().__init__()
        self.net = net
        self.spec = spec
        self._conv = CONVERTERS[spec.graph]
        self._g = None
        self._key = None

    def _graph(self, hg: Hypergraph):
        key = (id(hg), hg.M, hg.N)
        if self._g is None or self._key != key:
            self._g = self._conv(hg)
            self._key = key
        return self._g

    def forward(self, X: Tensor, hg: Hypergraph) -> Tensor:
        g = self._graph(hg)
        if self.spec.call is not None:
            return self.spec.call(self.net, X, g)
        return self.net(X, g)


def integration_help(name: str) -> str:
    s = EXTERNAL_SPECS.get(name)
    if s is None:
        return f"No external spec registered for '{name}'."
    return (
        f"\nTo enable baseline '{name}' ({s.paper}):\n"
        f"  1. Clone the official repo:\n        {s.repo}\n"
        f"     into  external/{name}/  (or anywhere importable).\n"
        f"  2. export PYTHONPATH=$PYTHONPATH:external/{name}\n"
        f"  3. In adapters.py, set EXTERNAL_SPECS['{name}'].module / .cls to the\n"
        f"     repo's model module/class, choose graph in {list(CONVERTERS)},\n"
        f"     and edit `.builder` to match its constructor.\n"
        f"     Model note: {s.note}\n"
        f"  4. Re-run. Third-party code is not bundled here; please cite the paper.\n")


def build_external(name: str, cfg: Config, hidden: int = 64) -> ExternalModel:
    """Import and wrap an external model, or raise with integration instructions."""
    s = EXTERNAL_SPECS.get(name)
    if s is None:
        raise KeyError(f"Unknown external baseline '{name}'. "
                       f"Known: {list(EXTERNAL_SPECS)}")
    try:
        mod = importlib.import_module(s.module)
        cls = getattr(mod, s.cls)
    except Exception as e:                        # repo not on PYTHONPATH yet
        raise ImportError(integration_help(name)) from e
    if s.builder is not None:
        net = s.builder(cfg, hidden)
    else:                                         # default signature; edit per repo
        net = cls(cfg.num_features, hidden, cfg.num_classes)
    return ExternalModel(net, s)
