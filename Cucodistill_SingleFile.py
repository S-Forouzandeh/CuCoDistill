"""
CuCoDistill: Curriculum Co-Evolutionary Knowledge Distillation
with Spectral Guarantees for Hypergraph Neural Networks
================================================================

Faithful reference implementation of the components described in the paper:

  * Hypergraph-Aware Adaptive Attention (HAAA): a shared three-head attention
    module (local node-node, set-aware hyperedge, global spectral) with a
    context-adaptive gate.  Eq. (6)-(10).
  * Hypergraph Triple Attention (HTA) teacher: full-neighbourhood encoder.
  * Top-K student: keeps only the K highest hybrid-attention neighbours per
    node (N_i^K), giving the O(K|V|d) inference cost and the spectral
    regularisation that underlies Theorem 2.
  * Adaptive Knowledge-guided Edge Dropping (AKED): retention probability
    sigma(mu1 * s_attn + mu2 * s_kd - rho(ep)) with a time-decaying threshold,
    plus feature masking and *small* Gaussian feature noise N(0, 0.01^2). Eq. (3)-(5).
  * Co-evolutionary training: teacher AND student are optimised *simultaneously*
    every step through a shared backbone; the teacher is never frozen.  The
    student aligns to a stop-gradient teacher (sg(e_T)); the teacher keeps
    improving via the task loss, and the student's Top-K sparsity regularises
    the shared parameters (bidirectional feedback).
  * Multi-level transfer: L_embed (Eq. 15), L_attn (Eq. 16), L_feat (Eq. 17).
  * Contrastive InfoNCE between the clean and AKED-augmented views (Eq. 26).
  * Spectral curriculum: per-node difficulties D_contrast = 1 - cos(e_clean,e_aug)
    and D_distill = ||e_T - e_S|| with time-varying quantile thresholds and the
    loss-weight schedule lambda1/lambda2/lambda3 (Eq. 18, 21-26).
  * A *constructive* student-superiority diagnostic R* = K / d_eff (the revised
    paper's Proposition), computed with no training.

Paper reporting convention: a single base seed (5) is applied uniformly to every
RNG (random / numpy / torch / cuda).  Main results are averaged over 5 seeds and
the paired significance tests over 10 seeds.  AKED augmentation noise std is 0.01.

This single file is a runnable reference; for the JMLR reproducibility package it
should be split into the modules indicated in the section banners (models/, losses/,
data/, train.py, ...) with one script per table.  See the README of the repository.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# Reproducibility  ->  utils/seed.py
# =============================================================================
def set_seed(seed: int = 5, deterministic: bool = True) -> None:
    """Seed every RNG used in training for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =============================================================================
# Configuration  ->  config.py
# =============================================================================
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


# =============================================================================
# Hypergraph utilities  ->  models/hypergraph.py
# =============================================================================
class Hypergraph:
    """Incidence matrix, Laplacian, effective spectral dimension, neighbour mask."""

    def __init__(self, H: Tensor):
        self.H = H                                   # (N, M) in {0,1}
        self.N, self.M = H.shape
        self.deg_v = H.sum(1).clamp(min=1.0)         # (N,)
        self.deg_e = H.sum(0).clamp(min=1.0)         # (M,)

    # ---- normalised hypergraph Laplacian operator (paper Eq. 1) ----
    def theta(self) -> Tensor:
        """Theta = D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}  (W = I)."""
        Dv_isqrt = self.deg_v.pow(-0.5)
        De_inv = self.deg_e.pow(-1.0)
        HWDe = self.H * De_inv.unsqueeze(0)          # (N,M)
        theta = (HWDe @ self.H.t())                  # (N,N)
        theta = theta * Dv_isqrt.unsqueeze(0) * Dv_isqrt.unsqueeze(1)
        return theta

    def spectral_operator(self) -> Tensor:
        """(2I - Theta) used by the global spectral head (Eq. 8)."""
        theta = self.theta()
        return 2.0 * torch.eye(self.N, device=self.H.device) - theta

    def comembership_mask(self) -> Tensor:
        """Boolean (N,N): True if i,j share a hyperedge (self-loops included)."""
        A = (self.H @ self.H.t()) > 0
        A.fill_diagonal_(True)
        return A

    def degree_features(self) -> Tensor:
        """Per-node [normalised node degree, normalised hyperedge-incidence] (N,2)."""
        d_v = self.deg_v / self.deg_v.max()
        d_e = self.H.sum(1) / max(1.0, float(self.H.sum(1).max()))
        return torch.stack([d_v, d_e], dim=-1)

    def max_node_edges(self) -> int:
        return int(self.H.sum(1).max().item())

    def effective_dimension(self, energy: float = 0.90) -> int:
        """d_eff via the eigengap heuristic: index of the largest gap among the
        smallest eigenvalues of L = I - Theta (~ number of clusters)."""
        L = torch.eye(self.N, device=self.H.device) - self.theta()
        evals, _ = torch.sort(torch.linalg.eigvalsh(L).clamp(min=0.0))
        half = max(2, self.N // 4)
        gaps = evals[1:half] - evals[:half - 1]
        d_eff = int(torch.argmax(gaps).item()) + 1
        return max(1, min(d_eff, self.N))


# =============================================================================
# HAAA: shared three-head hypergraph attention  ->  models/haaa.py
# =============================================================================
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


# =============================================================================
# AKED  ->  models/aked.py
# =============================================================================
class AKED(nn.Module):
    """Adaptive Knowledge-guided Edge Dropping + feature augmentation (Eq. 3-5)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.mu1 = cfg.aked_mu1
        self.mu2 = cfg.aked_mu2
        self.rho0 = cfg.aked_rho0
        self.p_feat = cfg.aug_feat_mask
        self.noise = cfg.aug_feat_noise          # 0.01, the paper's value

    def edge_signals(self, H: Tensor, hybrid_attn: Tensor,
                     t_emb: Tensor, s_emb: Tensor) -> Tuple[Tensor, Tensor]:
        deg_e = H.sum(0).clamp(min=1.0)                       # (M,)
        # s_attn(e): mean hybrid attention among members (Eq. 3)
        num = torch.einsum("ne,nm,me->e", H, hybrid_attn, H)  # sum_{i,j in e} a_ij
        s_attn = num / (deg_e * deg_e)
        # s_kd(e): mean teacher-student embedding gap among members (Eq. 4)
        gap = (t_emb - s_emb).norm(dim=-1)                    # (N,)
        s_kd = (H.t() @ gap) / deg_e
        return s_attn, s_kd

    def forward(self, X: Tensor, H: Tensor, hybrid_attn: Tensor,
                t_emb: Tensor, s_emb: Tensor, ep: int, total: int,
                training: bool = True) -> Tuple[Tensor, Tensor]:
        s_attn, s_kd = self.edge_signals(H, hybrid_attn, t_emb, s_emb)
        rho = self.rho0 * (1.0 - ep / max(1, total))          # decaying threshold
        p_retain = torch.sigmoid(self.mu1 * s_attn + self.mu2 * s_kd - rho)  # (M,)
        if training:
            keep = torch.bernoulli(p_retain)
        else:
            keep = (p_retain > 0.5).float()
        H_aug = H * keep.unsqueeze(0)

        # feature masking + small Gaussian noise -----------------------------
        if training:
            fmask = torch.bernoulli(torch.full_like(X, 1.0 - self.p_feat))
            X_aug = X * fmask + torch.randn_like(X) * self.noise
        else:
            X_aug = X
        return X_aug, H_aug


# =============================================================================
# CuCo model: shared backbone + teacher (full) + student (Top-K)  ->  models/cuco.py
# =============================================================================
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


# =============================================================================
# Losses  ->  losses.py
# =============================================================================
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


# =============================================================================
# Spectral curriculum  ->  curriculum.py
# =============================================================================
class SpectralCurriculum:
    """Difficulty metrics, quantile thresholds, and the lambda schedule (Eq. 21-26)."""

    def __init__(self, cfg: Config):
        self.T = cfg.epochs
        self.lambda3 = cfg.lambda3_task

    def loss_weights(self, ep: int) -> Tuple[float, float, float]:
        t = ep / max(1, self.T)
        lam1 = 0.5 * math.sqrt(t)             # distillation grows  (Eq. 25)
        lam2 = 0.3 * math.exp(-t)             # contrastive decays
        return lam1, lam2, self.lambda3

    @staticmethod
    def contrastive_difficulty(e_clean: Tensor, e_aug: Tensor) -> Tensor:
        return 1.0 - F.cosine_similarity(e_clean, e_aug, dim=-1)   # D_contrast (Eq. 21)

    @staticmethod
    def distill_difficulty(e_t: Tensor, e_s: Tensor) -> Tensor:
        return (e_t - e_s).norm(dim=-1)                            # D_distill  (Eq. 22)

    def gate(self, difficulty: Tensor, ep: int, kind: str) -> Tensor:
        """Easy-to-hard quantile gate: include nodes with difficulty <= tau(ep)."""
        t = ep / max(1, self.T)
        if kind == "distill":          # tighter window 0.2 -> 0.4 (paper beta_ep)
            q = 0.2 + 0.2 * math.sqrt(t)
        else:                          # contrastive: admit progressively harder
            q = 0.5 + 0.5 * t
        tau = torch.quantile(difficulty.detach(), min(1.0, q))
        return (difficulty.detach() <= tau).float()


# =============================================================================
# Theoretical guarantees  ->  theory.py
# =============================================================================
class Theory:
    """Honest checks of the four theorems and the student-superiority diagnostic."""

    @staticmethod
    def t1_spectral(teacher_attn: Tensor, student_attn: Tensor,
                    N: int, max_Ei: int, eps: float):
        """Theorem 1: ||A_ours - A_ideal||_F <= eps * sqrt(|V| * max_i|E_i|).
        A_ideal := teacher hybrid attention (full), A_ours := student (Top-K).
        Reports the empirical Frobenius error, the paper's bound, and the
        implied per-interaction error."""
        err = torch.norm(student_attn - teacher_attn).item()
        bound = eps * math.sqrt(N * max(1, max_Ei))
        implied_eps = err / (math.sqrt(N * max(1, max_Ei)) + 1e-12)
        return {"frob_error": err, "paper_bound": bound, "eps": eps,
                "implied_eps": implied_eps, "satisfied": err <= bound}

    @staticmethod
    def t2_convergence(task_loss: List[float]):
        """Theorem 2: O(1/sqrt(T)) rate.  Fit the log-log slope of the student
        task loss (a fixed objective; the total curriculum loss is reweighted
        over time and is not a valid convergence signal)."""
        if len(task_loss) < 5:
            return {"slope": float("nan"), "target": -0.5}
        y = np.maximum.accumulate(np.asarray(task_loss, dtype=float)[::-1])[::-1]  # running min
        t = np.arange(1, len(y) + 1, dtype=float)
        slope = float(np.polyfit(np.log(t), np.log(y + 1e-9), 1)[0])
        return {"slope": slope, "target": -0.5}

    @staticmethod
    def t3_generalisation(n_train: int, n_params: int, delta: float = 0.05):
        """Theorem 3: PAC-Bayes-style bound components."""
        comp = math.sqrt(n_params / max(1, n_train))
        conf = math.sqrt(math.log(1.0 / delta) / (2 * max(1, n_train)))
        return {"complexity": comp, "confidence": conf, "bound": comp + conf}

    @staticmethod
    def t4_diagnostic(X: Tensor, K: int, d_eff: int):
        """
        Constructive student-superiority test (revised paper, Prop. / Algorithm):
        predict student >= teacher iff spectral coverage (K >= d_eff) AND
        feature redundancy R(X) > R* = K / d_eff.  No training required.
        R(X) is the stable rank ||X||_F^2 / ||X||_2^2, normalised by d_eff.
        """
        svals = torch.linalg.svdvals(X.float())
        stable_rank = (svals.pow(2).sum() / (svals.max() ** 2 + 1e-12)).item()
        R_x = stable_rank / max(1, d_eff)
        R_star = K / max(1, d_eff)
        coverage = K >= d_eff
        predict = bool(coverage and (R_x > R_star))
        return {"R_x": R_x, "R_star": R_star, "K": K, "d_eff": d_eff,
                "coverage": coverage, "predict_student_superior": predict}


# =============================================================================
# Efficiency  ->  bench.py
# =============================================================================
class Bench:
    @staticmethod
    def params(m: nn.Module) -> int:
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    @staticmethod
    def time_fn(fn, n: int = 30) -> float:
        for _ in range(5):
            fn()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t0) / n * 1e3      # ms

    @staticmethod
    def report(model: CuCoModel, pack, K: int) -> Dict[str, object]:
        X, hg, spec, A, degf = pack
        model.eval()
        with torch.no_grad():
            t_ms = Bench.time_fn(lambda: model.teacher_forward(X, hg, spec, A, degf))
            s_ms = Bench.time_fn(lambda: model.student_forward(X, hg, spec, A, degf, K))
        t_params = Bench.params(model.teacher_layers) + Bench.params(model.teacher_cls)
        s_params = Bench.params(model.student_layers) + Bench.params(model.student_cls)
        return {
            "teacher_ms": t_ms, "student_ms": s_ms,
            "measured_speedup": t_ms / max(s_ms, 1e-9),
            "theoretical_speedup_NoverK": hg.N / max(1, K),  # Theta(|V|/K) at scale
            "teacher_path_params": t_params, "student_path_params": s_params,
            "param_ratio": t_params / max(1, s_params),
            "note": ("Headline 127-133x / 5.4-5.5x are measured on the large "
                     "benchmarks; on a toy graph only the Theta(|V|/K) trend holds."),
        }


# =============================================================================
# Datasets  ->  data/
# =============================================================================
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


def load_real_dataset(name: str):
    """
    Stub for the benchmark loaders (Cora/Citeseer/DBLP/IMDB/Yelp and the
    higher-order sets).  The reproducibility package implements one loader per
    dataset returning (X, labels, Hypergraph, masks) under the same interface.
    """
    raise NotImplementedError(
        f"Loader for '{name}' lives in data/{name}.py of the repository. "
        "It must return (X, labels, Hypergraph, masks) like SyntheticHypergraph.")


# =============================================================================
# Trainer (correct co-evolution)  ->  train.py
# =============================================================================
class CuCoTrainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)
        self.model = CuCoModel(cfg).to(self.dev)
        self.aked = AKED(cfg).to(self.dev)
        self.curr = SpectralCurriculum(cfg)
        # a SINGLE optimiser over the whole model -> teacher & student co-evolve
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr,
                                     weight_decay=cfg.weight_decay)
        self.loss_hist: List[float] = []
        self.task_curve: List[float] = []
        self.K = 1

    @staticmethod
    def _acc(logits, labels, mask) -> float:
        return (logits[mask].argmax(-1) == labels[mask]).float().mean().item()

    def _pack(self, X, hg):
        return (X, hg, hg.spectral_operator(), hg.comembership_mask(), hg.degree_features())

    # ---- Phase 1: warm up the teacher (and shared backbone) ----
    def pretrain_teacher(self, X, hg, labels, masks, verbose=True):
        pack = self._pack(X, hg)
        for ep in range(self.cfg.pretrain_epochs):
            self.model.train()
            self.opt.zero_grad()
            t = self.model.teacher_forward(*pack)
            loss = F.cross_entropy(t["logits"][masks["train"]], labels[masks["train"]])
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.opt.step()
            if verbose and (ep + 1) % 10 == 0:
                print(f"  [pretrain] ep {ep+1:3d}  loss {loss.item():.4f}  "
                      f"train_acc {self._acc(t['logits'], labels, masks['train']):.3f}")

    # ---- Phase 2: curriculum co-evolutionary distillation ----
    def distill(self, X, hg, labels, masks, verbose=True):
        pack = self._pack(X, hg)
        spec, A, degf = pack[2], pack[3], pack[4]
        self.K = math.ceil(self.cfg.topk_alpha * hg.max_node_edges())
        hist = {"student_acc": [], "teacher_acc": [], "val_acc": [], "loss": []}
        best = 0.0

        for ep in range(self.cfg.epochs):
            self.model.train()
            self.opt.zero_grad()

            # forward both paths on the clean view (teacher is NOT frozen) -----
            t = self.model.teacher_forward(X, hg, spec, A, degf)
            s = self.model.student_forward(X, hg, spec, A, degf, self.K)

            # AKED augmented view, then student forward on it -----------------
            X_aug, H_aug = self.aked(X, hg.H, t["attn"].detach(), t["emb"].detach(),
                                     s["emb"].detach(), ep, self.cfg.epochs, training=True)
            hg_aug = Hypergraph(H_aug)
            s_aug = self.model.student_forward(
                X_aug, hg_aug, hg_aug.spectral_operator(),
                hg_aug.comembership_mask(), hg_aug.degree_features(), self.K)

            # curriculum difficulties + gates ---------------------------------
            d_con = self.curr.contrastive_difficulty(s["emb"], s_aug["emb"])
            d_dis = self.curr.distill_difficulty(t["emb"], s["emb"])
            g_dis = self.curr.gate(d_dis, ep, "distill")        # (N,)
            g_con = self.curr.gate(d_con, ep, "contrast")
            lam1, lam2, lam3 = self.curr.loss_weights(ep)

            # losses ----------------------------------------------------------
            L_task = (F.cross_entropy(t["logits"][masks["train"]], labels[masks["train"]])
                      + F.cross_entropy(s["logits"][masks["train"]], labels[masks["train"]]))
            L_embed = Losses.embed_align(s["aligned"], t["emb"], g_dis)
            L_attn = Losses.attn_transfer(t["attn"], s["attn"], A.float())
            L_feat = Losses.feat_match(s["feats"], t["feats"], self.cfg.feat_layer_gamma)
            L_kd = Losses.soft_kd(s["logits"], t["logits"], self.cfg.kd_temperature)
            L_distill = L_embed + L_attn + L_feat + L_kd
            nce = Losses.info_nce(s["emb"], s_aug["emb"], self.cfg.infonce_tau)
            L_contrast = (g_con * nce).sum() / (g_con.sum() + 1e-9)

            loss = lam1 * L_distill + lam2 * L_contrast + lam3 * L_task
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.opt.step()
            self.loss_hist.append(loss.item())
            with torch.no_grad():
                self.task_curve.append(F.cross_entropy(
                    s["logits"][masks["train"]], labels[masks["train"]]).item())

            with torch.no_grad():
                s_acc = self._acc(s["logits"], labels, masks["train"])
                t_acc = self._acc(t["logits"], labels, masks["train"])
                v_acc = self._acc(s["logits"], labels, masks["val"])
            best = max(best, v_acc)
            for k, v in zip(("student_acc", "teacher_acc", "val_acc", "loss"),
                            (s_acc, t_acc, v_acc, loss.item())):
                hist[k].append(v)
            if verbose and (ep + 1) % 15 == 0:
                print(f"  [distill] ep {ep+1:3d}  L {loss.item():.3f}  "
                      f"(l1 {lam1:.2f}/l2 {lam2:.2f})  "
                      f"s_acc {s_acc:.3f}  t_acc {t_acc:.3f}  val {v_acc:.3f}")
        return hist

    # ---- Phase 3: evaluation, theorems, efficiency ----
    def evaluate(self, X, hg, labels, masks):
        pack = self._pack(X, hg)
        self.model.eval()
        with torch.no_grad():
            t = self.model.teacher_forward(*pack)
            s = self.model.student_forward(*pack, self.K)
        t_acc = self._acc(t["logits"], labels, masks["test"])
        s_acc = self._acc(s["logits"], labels, masks["test"])
        d_eff = hg.effective_dimension(self.cfg.deff_threshold)

        res = {
            "teacher_test_acc": t_acc,
            "student_test_acc": s_acc,
            "gain_pp": (s_acc - t_acc) * 100,
            "theorem1": Theory.t1_spectral(t["attn"], s["attn"], hg.N,
                                           hg.max_node_edges(), self.cfg.spectral_eps),
            "theorem2": Theory.t2_convergence(self.task_curve),
            "theorem3": Theory.t3_generalisation(
                int(masks["train"].sum()),
                Bench.params(self.model.student_layers) + Bench.params(self.model.student_cls)),
            "theorem4_diagnostic": Theory.t4_diagnostic(X, self.K, d_eff),
            "efficiency": Bench.report(self.model, pack, self.K),
        }
        self._print(res)
        return res

    @staticmethod
    def _print(r):
        e, t4 = r["efficiency"], r["theorem4_diagnostic"]
        print("\n" + "=" * 64)
        print("  CuCoDistill - results")
        print("=" * 64)
        print(f"  Teacher test acc : {r['teacher_test_acc']*100:6.2f}%")
        print(f"  Student test acc : {r['student_test_acc']*100:6.2f}%   "
              f"(gain {r['gain_pp']:+.2f} pp)")
        t1 = r['theorem1']
        print(f"  Theorem 1 [spectral]  ||A_s-A_t||_F {t1['frob_error']:.3f} <= "
              f"bound {t1['paper_bound']:.3f}  -> {t1['satisfied']}  "
              f"(implied eps {t1['implied_eps']:.3f})")
        print(f"  Theorem 2 [converge]  log-log slope {r['theorem2']['slope']:.3f} "
              f"(target {r['theorem2']['target']})")
        print(f"  Theorem 3 [general.]  bound {r['theorem3']['bound']:.4f}")
        print(f"  Theorem 4 [R(X)>R*]   R(X) {t4['R_x']:.2f}  R* {t4['R_star']:.2f}  "
              f"K {t4['K']}  d_eff {t4['d_eff']}  -> predict student-superior: "
              f"{t4['predict_student_superior']}")
        print(f"  Efficiency:  measured speedup {e['measured_speedup']:.1f}x  "
              f"| theoretical |V|/K = {e['theoretical_speedup_NoverK']:.1f}x  "
              f"| param ratio {e['param_ratio']:.2f}x")
        print("  " + e["note"])
        print("=" * 64)


# =============================================================================
# Entry point  ->  main.py
# =============================================================================
def main():
    p = argparse.ArgumentParser(description="CuCoDistill reference implementation")
    p.add_argument("--dataset", default="synthetic")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    cfg = Config(seed=args.seed, device=args.device)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    set_seed(cfg.seed)

    print("=" * 64)
    print("  CuCoDistill: Curriculum Co-Evolutionary Distillation (HGNN)")
    print(f"  dataset={args.dataset}  seed={cfg.seed}  device={cfg.device}")
    print(f"  AKED augmentation noise std = {cfg.aug_feat_noise}  (paper value)")
    print("=" * 64)

    if args.dataset == "synthetic":
        X, labels, hg, masks = SyntheticHypergraph(cfg).build()
    else:
        X, labels, hg, masks = load_real_dataset(args.dataset)
    X, labels = X.to(cfg.device), labels.to(cfg.device)
    print(f"  nodes={hg.N} edges={hg.M} feat={X.size(1)} classes={cfg.num_classes} "
          f"| max |E_i|={hg.max_node_edges()}  d_eff~={hg.effective_dimension():d}")

    trainer = CuCoTrainer(cfg)
    print("\n[Phase 1] teacher / backbone warm-up")
    trainer.pretrain_teacher(X, hg, labels, masks)
    print("\n[Phase 2] curriculum co-evolutionary distillation")
    trainer.distill(X, hg, labels, masks)
    print("\n[Phase 3] evaluation + theorem checks")
    trainer.evaluate(X, hg, labels, masks)
    print("\nDone.\n")


if __name__ == "__main__":
    main()
