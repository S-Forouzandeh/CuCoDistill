"""Centralised configuration + per-dataset overrides."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Config:
    """Centralised, documented configuration."""

    # ---- data ----
    num_nodes: int = 600
    num_features: int = 64
    num_classes: int = 5
    num_hyperedges: int = 320
    redundancy_copies: int = 3        # duplicated feature blocks -> R(X) > R*
    data_feature_noise: float = 0.35  # generative feature noise (NOT augmentation)
    label_noise: float = 0.04         # fraction of flipped training labels

    # ---- shared backbone / encoder ----
    hidden_dim: int = 64              # uniform width (compression is via Top-K + depth)
    num_heads: int = 4
    tau_node: float = 0.10            # temperature for cosine attention (Eq. 6)
    dropout: float = 0.5

    # ---- teacher / student depth ----
    teacher_layers: int = 3           # includes the shared layer
    student_layers: int = 2           # fewer layers + Top-K -> lighter
    topk_alpha: float = 0.5           # K = ceil(alpha * max_i |E_i|), alpha in [0.3,0.7]

    # ---- AKED ----
    aked_rho0: float = 1.0            # initial retention threshold rho(0)
    aked_mu1: float = 2.0            # weight on attention salience s_attn
    aked_mu2: float = 2.0            # weight on knowledge disparity s_kd
    aug_feat_mask: float = 0.10       # p_feat (Eq. 3.2)
    aug_feat_noise: float = 0.01      # N(0, 0.01^2)  <-- paper value (NOT 5)

    # ---- distillation / contrastive ----
    kd_temperature: float = 4.0       # T for soft-label KL
    infonce_tau: float = 0.5
    feat_layer_gamma: float = 1.5     # up-weight deeper layers in L_feat

    # ---- curriculum (Eq. 21-26) ----
    epochs: int = 120
    lambda3_task: float = 0.2         # constant task weight

    # ---- optimisation ----
    lr: float = 1e-3
    weight_decay: float = 5e-4
    grad_clip: float = 5.0
    pretrain_epochs: int = 40

    # ---- spectral ----
    spectral_rank: int = 32           # rank for the Theorem-1 approximation
    spectral_eps: float = 0.05        # epsilon in Theorem 1
    deff_threshold: float = 0.90      # eigen-energy kept to define d_eff

    device: str = "cpu"
    seed: int = 5                      # single base seed, applied to every RNG
    num_seeds: int = 5                 # runs for main results (mean +/- std)
    num_seeds_significance: int = 10   # runs for paired significance tests


# Per-dataset overrides for the nine benchmarks and the higher-order sets.
# Fill in num_features / num_classes from each dataset's metadata; the Top-K
# alpha and learning rate follow the guidelines in the paper (Table 14).
DATASET_CONFIGS: Dict[str, dict] = {
    # ---- original nine benchmarks (noisy/redundant -> smaller alpha) ----
    "dblp":        dict(num_classes=4, topk_alpha=0.40, lr=1e-3),
    "imdb":        dict(num_classes=3, topk_alpha=0.40, lr=1e-3),
    "yelp":        dict(num_classes=5, topk_alpha=0.40, lr=1e-3),
    "dblp_term":   dict(num_classes=4, topk_alpha=0.40, lr=1e-3),
    "cc_cora":     dict(num_classes=7, topk_alpha=0.50, lr=1e-3),
    "cc_citeseer": dict(num_classes=6, topk_alpha=0.50, lr=1e-3),
    "dblp_paper":  dict(num_classes=4, topk_alpha=0.55, lr=5e-4),
    "dblp_conf":   dict(num_classes=4, topk_alpha=0.55, lr=5e-4),
    "imdb_aw":     dict(num_classes=3, topk_alpha=0.55, lr=5e-4),
    # ---- genuinely higher-order sets added for R1.2 ----
    "senate_bills":  dict(num_classes=2, topk_alpha=0.50, lr=1e-3),
    "house_bills":   dict(num_classes=2, topk_alpha=0.55, lr=5e-4),
    "contact_primary": dict(num_classes=11, topk_alpha=0.45, lr=1e-3),
    "email_enron":   dict(num_classes=2, topk_alpha=0.50, lr=1e-3),
    "modelnet40":    dict(num_classes=40, topk_alpha=0.55, lr=5e-4),
    "ntu2012":       dict(num_classes=67, topk_alpha=0.55, lr=5e-4),
}


def config_for(dataset: str, **overrides) -> "Config":
    """Build a Config for a named dataset, applying registry + ad-hoc overrides."""
    base = dict(DATASET_CONFIGS.get(dataset, {}))
    base.update(overrides)
    return Config(**base)
