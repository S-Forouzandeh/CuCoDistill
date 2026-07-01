"""Knowledge-distillation baselines.

``glnn_reference`` is a runnable, faithful GLNN-style distillation — a hypergraph
teacher (the official DHG model when available) distilled into an MLP student via
soft-label KL + CE — exposed as method ``glnn_ref``. The named methods
(``glnn``, ``krd``, ``lighthgnn``, ``distillhgnn``, ``ssgnn``, ``lad_gnn``) are
official-repo pointers: :func:`run_kd` runs them once their repo is wired in
(see ``INTEGRATION.md``), otherwise it raises with the repo URL and steps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from config import Config
from hypergraph import Hypergraph
from baselines import BASELINES, MLP


@dataclass
class KDSpec:
    name: str
    paper: str
    repo: str
    note: str = ""


# Verify URLs against each paper; for the authors' own methods use their release.
KD_SPECS: Dict[str, KDSpec] = {
    "glnn": KDSpec("glnn", "Tian et al., 2022",
                   "https://github.com/snap-stanford/graphless-neural-networks",
                   "GNN->MLP soft-label distillation; 'glnn_ref' is a runnable hypergraph version."),
    "krd": KDSpec("krd", "Wu et al., 2023", "https://github.com/LirongWu/RKD",
                  "Reliable KD: weights nodes by teacher-knowledge reliability."),
    "lighthgnn": KDSpec("lighthgnn", "Feng et al., 2024",
                        "https://github.com/iMoonLab/LightHGNN",
                        "Distil HGNN into MLP using hypergraph structural cues."),
    "distillhgnn": KDSpec("distillhgnn", "Forouzandeh et al., 2025",
                          "authors' official repository",
                          "Contrastive hypergraph KD (your prior work)."),
    "ssgnn": KDSpec("ssgnn", "Wu et al., 2024", "authors' official repository",
                    "Teacher-free dual self-distillation."),
    "lad_gnn": KDSpec("lad_gnn", "Hong et al., 2024", "authors' official repository",
                      "Label-attentive distillation for GNN classification."),
}


def _acc(logits, labels, mask):
    return (logits[mask].argmax(-1) == labels[mask]).float().mean().item()


def glnn_reference(X: Tensor, hg: Hypergraph, labels: Tensor,
                   masks: Dict[str, Tensor], cfg: Config, teacher: str = "hgnn",
                   t_epochs: int = 200, s_epochs: int = 200,
                   T: float = 4.0, alpha: float = 0.5) -> Dict[str, float]:
    """Train a hypergraph teacher, then distil it into an MLP student (GLNN-style)."""
    dev = cfg.device
    # ---- teacher (official DHG model if installed, else reference) ----
    Tnet = BASELINES[teacher](cfg).to(dev)
    opt = torch.optim.AdamW(Tnet.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    for _ in range(t_epochs):
        Tnet.train(); opt.zero_grad()
        out = Tnet(X, hg)
        F.cross_entropy(out[masks["train"]], labels[masks["train"]]).backward()
        nn.utils.clip_grad_norm_(Tnet.parameters(), cfg.grad_clip); opt.step()
    Tnet.eval()
    with torch.no_grad():
        t_logits = Tnet(X, hg)
        t_test = _acc(t_logits, labels, masks["test"])
    soft_t = F.softmax(t_logits.detach() / T, -1)

    # ---- student MLP distilled from soft labels ----
    S = MLP(cfg).to(dev)
    opt = torch.optim.AdamW(S.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_val, best_test = 0.0, 0.0
    for _ in range(s_epochs):
        S.train(); opt.zero_grad()
        out = S(X, hg)
        ce = F.cross_entropy(out[masks["train"]], labels[masks["train"]])
        kd = F.kl_div(F.log_softmax(out / T, -1), soft_t, reduction="batchmean") * (T * T)
        (alpha * kd + (1 - alpha) * ce).backward()
        nn.utils.clip_grad_norm_(S.parameters(), cfg.grad_clip); opt.step()
        S.eval()
        with torch.no_grad():
            o = S(X, hg)
            va, te = _acc(o, labels, masks["val"]), _acc(o, labels, masks["test"])
        if va >= best_val:
            best_val, best_test = va, te
    return {"teacher_test": t_test, "student_test": best_test}


def run_kd(name: str, X: Tensor, hg: Hypergraph, labels: Tensor,
           masks: Dict[str, Tensor], cfg: Config) -> Dict[str, float]:
    """Run a KD baseline; 'glnn_ref' is runnable, named methods need their repo."""
    if name in ("glnn_ref", "softlabel"):
        return glnn_reference(X, hg, labels, masks, cfg)
    spec = KD_SPECS.get(name)
    if spec is None:
        raise KeyError(f"Unknown KD method '{name}'. Known: "
                       f"{['glnn_ref'] + list(KD_SPECS)}")
    raise NotImplementedError(
        f"\nKD baseline '{name}' ({spec.paper}) needs the official repo:\n"
        f"  repo: {spec.repo}\n  note: {spec.note}\n"
        f"  Wire it in per INTEGRATION.md: clone -> add to PYTHONPATH -> implement a\n"
        f"  run_kd hook returning dict(teacher_test=..., student_test=...).\n"
        f"  A runnable hypergraph GLNN-style reference is available as 'glnn_ref'.\n")
