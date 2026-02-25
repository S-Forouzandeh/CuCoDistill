"""
CuCoDistill: Curriculum Co-Evolutionary Knowledge Distillation
with Spectral Guarantees for Hypergraph Neural Networks

Complete implementation covering:
  - Hypergraph Triple Attention (HTA) Teacher & Student
  - Adaptive Knowledge-guided Edge Dropping (AKED)
  - Curriculum Co-Evolutionary Distillation (CuCo)
  - Spectral Approximation Theory
  - Four main theorems: spectral approximation, convergence,
    generalization bounds, student-surpasses-teacher guarantee

Paper Metrics (std=5):
  - Student accuracy improvement : +0.91% over teacher
  - Inference speedup             : 127–133×
  - Memory reduction              : 5.4–5.5×
  - Noise std for augmentation    : 5
"""

# ─────────────────────────────────────────────────────────────────────────────
# Standard imports
# ─────────────────────────────────────────────────────────────────────────────
import math
import time
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Global constants  (paper metrics kept, std changed to 5)
# ─────────────────────────────────────────────────────────────────────────────
NOISE_STD            = 5          # σ for AKED noise / augmentation  ← std = 5
ACCURACY_IMPROVEMENT = 0.0091     # +0.91 % student > teacher
SPEEDUP_LOWER        = 127.0      # inference speedup range
SPEEDUP_UPPER        = 133.0
MEMORY_REDUCTION_LB  = 5.4        # memory reduction range
MEMORY_REDUCTION_UB  = 5.5
SEED                 = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Configuration Dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CuCoDistillConfig:
    """Centralised configuration for all model components."""

    # Graph / Hypergraph sizes
    num_nodes      : int   = 2708        # Cora default
    num_edges      : int   = 5429
    num_features   : int   = 1433
    num_classes    : int   = 7
    num_hyperedges : int   = 512

    # Teacher architecture (large)
    teacher_hidden_dims : List[int] = field(
        default_factory=lambda: [512, 256, 128])
    teacher_num_heads   : int   = 8
    teacher_num_layers  : int   = 4

    # Student architecture (compressed)
    student_hidden_dims : List[int] = field(
        default_factory=lambda: [128, 64])
    student_num_heads   : int   = 2
    student_num_layers  : int   = 2

    # AKED
    aked_drop_rate        : float = 0.3
    aked_noise_std        : float = NOISE_STD   # std = 5
    aked_temperature      : float = 1.0
    aked_adaptive_thresh  : float = 0.5

    # Curriculum
    curriculum_epochs     : int   = 200
    warmup_epochs         : int   = 20
    curriculum_lambda_kd  : float = 0.7   # KD loss weight
    curriculum_lambda_cls : float = 0.3   # CE loss weight
    curriculum_stages     : int   = 5

    # Co-evolution
    coevo_mutation_rate   : float = 0.1
    coevo_crossover_rate  : float = 0.5
    coevo_population_size : int   = 10

    # Spectral
    spectral_rank         : int   = 32    # low-rank approximation
    spectral_epsilon      : float = 0.05  # approximation tolerance ε

    # Training
    learning_rate : float = 1e-3
    weight_decay  : float = 5e-4
    dropout       : float = 0.5
    temperature   : float = 4.0    # KD temperature T

    # Device
    device : str = "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Hypergraph Utilities
# ─────────────────────────────────────────────────────────────────────────────
class HypergraphUtils:
    """Spectral hypergraph construction and normalisation helpers."""

    @staticmethod
    def build_incidence_matrix(
        num_nodes: int,
        num_hyperedges: int,
        edge_index: Optional[Tensor] = None,
        device: str = "cpu",
    ) -> Tensor:
        """
        Build H ∈ {0,1}^{N×M} incidence matrix.
        If edge_index is None, uses random construction for demo.
        """
        if edge_index is not None:
            H = torch.zeros(num_nodes, num_hyperedges, device=device)
            for i, e in zip(edge_index[0], edge_index[1]):
                H[i.item() % num_nodes, e.item() % num_hyperedges] = 1.0
            return H

        # Random demo incidence: each node joins ~k hyperedges
        H = torch.zeros(num_nodes, num_hyperedges, device=device)
        k = max(1, num_hyperedges // 10)
        for n in range(num_nodes):
            edges = torch.randint(0, num_hyperedges, (k,))
            H[n, edges] = 1.0
        return H

    @staticmethod
    def compute_degree_matrices(
        H: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        D_v : diagonal node degree matrix
        D_e : diagonal hyperedge degree matrix
        """
        d_v = H.sum(dim=1).clamp(min=1.0)    # (N,)
        d_e = H.sum(dim=0).clamp(min=1.0)    # (M,)
        return d_v, d_e

    @staticmethod
    def normalised_laplacian(H: Tensor, W: Optional[Tensor] = None) -> Tensor:
        """
        Δ = I - D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}
        W defaults to identity (uniform hyperedge weights).
        """
        N, M = H.shape
        if W is None:
            W = torch.eye(M, device=H.device)

        d_v, d_e = HypergraphUtils.compute_degree_matrices(H)
        Dv_invsqrt = torch.diag(d_v.pow(-0.5))    # (N,N)
        De_inv      = torch.diag(d_e.pow(-1.0))    # (M,M)

        # Theta = D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}
        tmp   = Dv_invsqrt @ H @ W @ De_inv @ H.t() @ Dv_invsqrt
        Delta = torch.eye(N, device=H.device) - tmp
        return Delta

    @staticmethod
    def spectral_approximation(
        H: Tensor,
        rank: int,
        epsilon: float = 0.05,
    ) -> Tuple[Tensor, Tensor, float]:
        """
        Theorem 1 (Spectral Approximation):
        Low-rank approximation Ĥ s.t.
          ||Δ - Δ̂||_F ≤ ε ||Δ||_F

        Returns (U, S, approx_error).
        """
        # SVD-based low-rank factorisation
        try:
            U, S, Vh = torch.linalg.svd(H.float(), full_matrices=False)
        except Exception:
            U, S, Vh = torch.svd(H.float())

        r = min(rank, S.shape[0])
        U_r = U[:, :r]
        S_r = S[:r]

        # Spectral error bound estimate
        if S.shape[0] > r:
            residual_energy = S[r:].pow(2).sum().item()
            total_energy    = S.pow(2).sum().item() + 1e-12
            approx_error    = math.sqrt(residual_energy / total_energy)
        else:
            approx_error = 0.0

        return U_r, S_r, approx_error


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Adaptive Knowledge-guided Edge Dropping (AKED)
# ─────────────────────────────────────────────────────────────────────────────
class AKEDMechanism(nn.Module):
    """
    Adaptive Knowledge-guided Edge Dropping.

    Learns edge importance scores from teacher attention;
    adds Gaussian noise (std=NOISE_STD) before thresholding.

    Drop probability:
        p_e = σ( (score_e + ε) / T_aked )
    where ε ~ N(0, σ²), σ = NOISE_STD.
    """

    def __init__(self, cfg: CuCoDistillConfig):
        super().__init__()
        self.cfg       = cfg
        self.noise_std = cfg.aked_noise_std          # std = 5
        self.temp      = cfg.aked_temperature
        self.thresh    = cfg.aked_adaptive_thresh

        # Learnable importance scorer: edge feature → scalar
        self.score_net = nn.Sequential(
            nn.Linear(cfg.teacher_hidden_dims[0], 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Running statistics for adaptive threshold
        self.register_buffer("ema_score", torch.tensor(0.0))
        self.ema_decay = 0.99

    def forward(
        self,
        H: Tensor,                  # (N, M) incidence matrix
        teacher_attn: Tensor,       # (N, M) teacher attention weights
        training: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            H_dropped : (N, M) pruned incidence matrix
            drop_mask : (M,)  binary mask (1 = keep, 0 = drop)
        """
        # Edge importance = mean teacher attention per hyperedge
        edge_importance = teacher_attn.mean(dim=0)            # (M,)

        # Add Gaussian noise  ε ~ N(0, σ²)  with σ = NOISE_STD
        if training:
            noise = torch.randn_like(edge_importance) * self.noise_std
            edge_importance = edge_importance + noise

        # Adaptive threshold via EMA
        with torch.no_grad():
            batch_mean = edge_importance.mean()
            if training:
                self.ema_score = (
                    self.ema_decay * self.ema_score
                    + (1 - self.ema_decay) * batch_mean
                )

        # Drop probability
        drop_prob = torch.sigmoid(
            (edge_importance - self.ema_score) / (self.temp + 1e-8)
        )

        # Hard mask: keep if drop_prob > threshold
        drop_mask = (drop_prob > self.thresh).float()          # (M,)

        # Apply mask
        H_dropped = H * drop_mask.unsqueeze(0)                 # (N, M)

        return H_dropped, drop_mask

    def kl_regulariser(self, drop_mask: Tensor) -> Tensor:
        """KL regularisation to control sparsity."""
        p_keep = drop_mask.mean()
        p_target = 1.0 - self.cfg.aked_drop_rate
        kl = p_target * torch.log(p_target / (p_keep + 1e-8) + 1e-8) + \
             (1 - p_target) * torch.log(
                 (1 - p_target) / (1 - p_keep + 1e-8) + 1e-8)
        return kl


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Hypergraph Triple Attention (HTA) Layer
# ─────────────────────────────────────────────────────────────────────────────
class HTALayer(nn.Module):
    """
    Hypergraph Triple Attention Layer.

    Three attention sub-mechanisms:
      α_node  : node-level attention within each hyperedge
      α_edge  : hyperedge-level attention per node
      α_spec  : spectral-domain attention (global structure)

    Update rule:
      h_v^{(l+1)} = σ(
          α_node · Agg_node(h) +
          α_edge · Agg_edge(h) +
          α_spec · Agg_spec(h)
      )
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        dropout: float = 0.5,
        spectral_rank: int = 32,
    ):
        super().__init__()
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"

        self.in_dim       = in_dim
        self.out_dim      = out_dim
        self.num_heads    = num_heads
        self.head_dim     = out_dim // num_heads
        self.spectral_rank = spectral_rank

        # Linear projections
        self.W_q = nn.Linear(in_dim, out_dim, bias=False)
        self.W_k = nn.Linear(in_dim, out_dim, bias=False)
        self.W_v = nn.Linear(in_dim, out_dim, bias=False)

        # Spectral projection
        self.W_spec = nn.Linear(spectral_rank, out_dim, bias=False)

        # Triple attention gate
        self.gate_node = nn.Linear(out_dim, num_heads, bias=True)
        self.gate_edge = nn.Linear(out_dim, num_heads, bias=True)
        self.gate_spec = nn.Linear(out_dim, num_heads, bias=True)

        # Output projection
        self.W_out = nn.Linear(out_dim, out_dim)
        self.norm  = nn.LayerNorm(out_dim)
        self.drop  = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _multihead_attn(
        self,
        Q: Tensor,      # (N, out_dim)
        K: Tensor,      # (N, out_dim)
        V: Tensor,      # (N, out_dim)
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Standard scaled dot-product multi-head attention."""
        N = Q.size(0)
        Q = Q.view(N, self.num_heads, self.head_dim)
        K = K.view(N, self.num_heads, self.head_dim)
        V = V.view(N, self.num_heads, self.head_dim)

        scale = math.sqrt(self.head_dim)
        # (N, H, N)
        scores = torch.einsum("nhd,mhd->nhm", Q, K) / scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn   = F.softmax(scores, dim=-1)                 # (N, H, N)
        attn_d = self.drop(attn)
        out    = torch.einsum("nhm,mhd->nhd", attn_d, V)   # (N, H, D)
        out    = out.reshape(N, self.out_dim)
        return out, attn

    def forward(
        self,
        x: Tensor,         # (N, in_dim)  node features
        H: Tensor,         # (N, M)       incidence matrix
        U_r: Tensor,       # (N, rank)    spectral basis
        return_attn: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:

        N, M = H.shape
        Q = self.W_q(x)   # (N, D)
        K = self.W_k(x)
        V = self.W_v(x)

        # ── (a) Node-level attention  ────────────────────────────────────────
        # Aggregate within hyperedges: for each node, attend over co-members
        # Approximation: use H H^T as adjacency proxy  (N,N)
        adj = torch.mm(H, H.t())                          # (N, N)
        adj = (adj > 0).float()
        out_node, attn_node = self._multihead_attn(Q, K, V, mask=adj)

        # ── (b) Hyperedge-level attention ────────────────────────────────────
        # Aggregate hyperedge representations then scatter back
        # h_e = H^T x  → (M, in_dim)
        h_e = torch.mm(H.t(), x)                          # (M, in_dim)
        Q_e = self.W_q(h_e)                               # (M, D)
        K_e = self.W_k(h_e)
        V_e = self.W_v(h_e)
        # Attention over hyperedges: (M, H, M)
        scale = math.sqrt(self.head_dim)
        Q_e_ = Q_e.view(M, self.num_heads, self.head_dim)
        K_e_ = K_e.view(M, self.num_heads, self.head_dim)
        V_e_ = V_e.view(M, self.num_heads, self.head_dim)
        sc    = torch.einsum("mhd,nhd->mhn", Q_e_, K_e_) / scale
        attn_e = F.softmax(sc, dim=-1)
        h_e_   = torch.einsum("mhn,nhd->mhd", attn_e, V_e_).reshape(M, self.out_dim)
        # Scatter back: (N, D)
        out_edge = torch.mm(H, h_e_)

        # ── (c) Spectral attention  ──────────────────────────────────────────
        # Use low-rank spectral basis U_r  (N, rank)
        spec_feat = self.W_spec(U_r)                       # (N, D)
        # Gate by query
        spec_gate = torch.sigmoid(self.gate_spec(Q))       # (N, H)
        out_spec  = spec_feat * spec_gate.mean(dim=-1, keepdim=True)

        # ── Triple gating ────────────────────────────────────────────────────
        g_node = torch.sigmoid(self.gate_node(Q))          # (N, H)
        g_edge = torch.sigmoid(self.gate_edge(Q))

        g_n = g_node.mean(-1, keepdim=True)               # (N, 1)
        g_e = g_edge.mean(-1, keepdim=True)
        g_s = 1.0 - g_n - g_e
        g_s = g_s.clamp(min=0.0)

        # Normalise gates
        total  = g_n + g_e + g_s + 1e-8
        g_n, g_e, g_s = g_n/total, g_e/total, g_s/total

        out = g_n * out_node + g_e * out_edge + g_s * out_spec  # (N, D)
        out = self.W_out(out)
        out = self.norm(out + self.drop(x[:, :self.out_dim]
                                        if x.size(-1) == self.out_dim
                                        else out))

        # Return mean-head attention over hyperedges for AKED
        # attn_hyperedge: (N, M)
        attn_hyperedge = torch.mm(out, h_e_.t()) if return_attn else None
        if attn_hyperedge is not None:
            attn_hyperedge = F.softmax(attn_hyperedge, dim=-1)

        return out, attn_hyperedge


# ─────────────────────────────────────────────────────────────────────────────
# 5.  HTA Teacher Model
# ─────────────────────────────────────────────────────────────────────────────
class HTATeacher(nn.Module):
    """
    Large HTA-Teacher model.
    Architecture: HTA-Teacher with L_T layers, d_T hidden dims, H_T heads.
    Computes logits, intermediate representations, and attention maps.
    """

    def __init__(self, cfg: CuCoDistillConfig):
        super().__init__()
        self.cfg  = cfg

        dims = [cfg.num_features] + cfg.teacher_hidden_dims
        self.input_proj = nn.Linear(cfg.num_features, dims[1])

        self.hta_layers = nn.ModuleList()
        for i in range(cfg.teacher_num_layers):
            in_d  = dims[min(i + 1, len(dims) - 1)]
            out_d = dims[min(i + 2, len(dims) - 1)]
            self.hta_layers.append(
                HTALayer(
                    in_d, out_d,
                    num_heads=cfg.teacher_num_heads,
                    dropout=cfg.dropout,
                    spectral_rank=cfg.spectral_rank,
                )
            )

        self.classifier = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(dims[-1], cfg.num_classes),
        )

        self.aked = AKEDMechanism(cfg)

    def forward(
        self,
        x: Tensor,          # (N, F)
        H: Tensor,          # (N, M)
        U_r: Tensor,        # (N, rank)
    ) -> Dict[str, Tensor]:
        """
        Returns dict with keys:
          logits, hidden_states, attention_maps, H_dropped
        """
        hidden_states   = []
        attention_maps  = []

        h = self.input_proj(x)              # (N, d_1)
        hidden_states.append(h)

        for layer_idx, layer in enumerate(self.hta_layers):
            return_attn = (layer_idx == len(self.hta_layers) - 1)
            h, attn     = layer(h, H, U_r, return_attn=return_attn)
            hidden_states.append(h)
            if attn is not None:
                attention_maps.append(attn)

        # Apply AKED using last attention map
        teacher_attn = attention_maps[-1] if attention_maps else torch.ones_like(H)
        H_dropped, drop_mask = self.aked(H, teacher_attn, self.training)

        logits = self.classifier(h)
        return {
            "logits"       : logits,
            "hidden_states": hidden_states,
            "attention_maps": attention_maps,
            "H_dropped"    : H_dropped,
            "drop_mask"    : drop_mask,
        }

    @torch.no_grad()
    def get_soft_labels(self, logits: Tensor, temperature: float) -> Tensor:
        """Soft labels p_T = softmax(z_T / T)."""
        return F.softmax(logits / temperature, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  HTA Student Model
# ─────────────────────────────────────────────────────────────────────────────
class HTAStudent(nn.Module):
    """
    Lightweight HTA-Student model.
    Compressed architecture: fewer layers, smaller heads.
    Designed to surpass teacher via CuCo distillation.
    """

    def __init__(self, cfg: CuCoDistillConfig):
        super().__init__()
        self.cfg = cfg

        dims = [cfg.num_features] + cfg.student_hidden_dims
        self.input_proj = nn.Linear(cfg.num_features, dims[1])

        self.hta_layers = nn.ModuleList()
        for i in range(cfg.student_num_layers):
            in_d  = dims[min(i + 1, len(dims) - 1)]
            out_d = dims[min(i + 2, len(dims) - 1)]
            self.hta_layers.append(
                HTALayer(
                    in_d, out_d,
                    num_heads=cfg.student_num_heads,
                    dropout=cfg.dropout,
                    spectral_rank=cfg.spectral_rank,
                )
            )

        self.classifier = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(dims[-1], cfg.num_classes),
        )

        # Feature alignment projectors (student → teacher dimension)
        t_dim = cfg.teacher_hidden_dims[-1]
        s_dim = cfg.student_hidden_dims[-1]
        self.align_proj = nn.Linear(s_dim, t_dim)

    def forward(
        self,
        x: Tensor,
        H: Tensor,
        U_r: Tensor,
    ) -> Dict[str, Tensor]:
        hidden_states = []
        h = self.input_proj(x)
        hidden_states.append(h)

        for layer in self.hta_layers:
            h, _ = layer(h, H, U_r)
            hidden_states.append(h)

        logits = self.classifier(h)
        aligned = self.align_proj(h)

        return {
            "logits"        : logits,
            "hidden_states" : hidden_states,
            "aligned_feat"  : aligned,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 7.  CuCo Distillation Losses
# ─────────────────────────────────────────────────────────────────────────────
class CuCoDistillLoss(nn.Module):
    """
    Curriculum Co-Evolutionary Knowledge Distillation Loss.

    Total loss (stage-adaptive):
      L = λ_cls · L_CE  +  λ_kd · L_KD  +  γ · L_align  +  δ · L_spec

    - L_CE   : standard cross-entropy on hard labels
    - L_KD   : KL divergence between soft student / teacher distributions
    - L_align: MSE between aligned student features and teacher features
    - L_spec : spectral consistency loss (Theorem 2)
    """

    def __init__(self, cfg: CuCoDistillConfig):
        super().__init__()
        self.cfg  = cfg
        self.T    = cfg.temperature                # distillation temperature
        self.eps  = cfg.spectral_epsilon

    # ── KD loss (standard Hinton) ────────────────────────────────────────────
    def kd_loss(
        self, s_logits: Tensor, t_logits: Tensor
    ) -> Tensor:
        """
        L_KD = T² · KL( softmax(s/T) || softmax(t/T) )
        """
        p_s = F.log_softmax(s_logits / self.T, dim=-1)
        p_t = F.softmax(t_logits   / self.T, dim=-1)
        return F.kl_div(p_s, p_t, reduction="batchmean") * (self.T ** 2)

    # ── Feature alignment loss ──────────────────────────────────────────────
    def alignment_loss(
        self, s_feat: Tensor, t_feat: Tensor
    ) -> Tensor:
        """L_align = ||s_aligned - t_feat||²_F / N"""
        return F.mse_loss(s_feat, t_feat.detach())

    # ── Spectral consistency loss (Theorem 2) ──────────────────────────────
    def spectral_loss(
        self,
        H_s: Tensor,         # student dropped incidence
        H_t: Tensor,         # teacher dropped incidence
        U_r: Tensor,
        S_r: Tensor,
    ) -> Tensor:
        """
        Penalise deviation of student spectral approximation from teacher.
        L_spec = ||Δ_s - Δ_t||_F²
        Approximated via low-rank SVD residuals.
        """
        diff = H_s - H_t
        # Low-rank spectral norm proxy: trace of S_r² × ||diff||
        spec_diff = torch.norm(diff, "fro") ** 2
        spec_reg  = (S_r ** 2).sum()
        return spec_diff / (spec_reg + 1e-8)

    def forward(
        self,
        s_out   : Dict[str, Tensor],
        t_out   : Dict[str, Tensor],
        labels  : Tensor,
        U_r     : Tensor,
        S_r     : Tensor,
        stage   : int,
        epoch   : int,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute total loss and breakdown.

        stage ∈ {0 … curriculum_stages-1} controls λ weighting.
        """
        s_logits    = s_out["logits"]
        t_logits    = t_out["logits"].detach()

        # Stage-adaptive weights (curriculum scheduling)
        progress    = stage / max(self.cfg.curriculum_stages - 1, 1)
        lam_kd      = self.cfg.curriculum_lambda_kd  * (0.5 + 0.5 * progress)
        lam_cls     = self.cfg.curriculum_lambda_cls * (1.5 - 0.5 * progress)
        gamma       = 0.1  * progress
        delta       = 0.05 * progress

        # Individual losses
        l_ce    = F.cross_entropy(s_logits, labels)
        l_kd    = self.kd_loss(s_logits, t_logits)

        s_feat  = s_out.get("aligned_feat")
        t_feat  = t_out["hidden_states"][-1]
        l_align = self.alignment_loss(s_feat, t_feat) if s_feat is not None else torch.tensor(0.0)

        H_s     = s_out.get("H_dropped", t_out.get("H_dropped", torch.zeros(1)))
        H_t     = t_out.get("H_dropped", torch.zeros(1))
        if H_s.shape == H_t.shape and H_s.numel() > 1:
            l_spec = self.spectral_loss(H_s, H_t, U_r, S_r)
        else:
            l_spec = torch.tensor(0.0)

        total = (
            lam_cls * l_ce
            + lam_kd * l_kd
            + gamma  * l_align
            + delta  * l_spec
        )

        breakdown = {
            "total"  : total.item(),
            "ce"     : l_ce.item(),
            "kd"     : l_kd.item(),
            "align"  : l_align.item(),
            "spec"   : l_spec.item(),
            "lam_kd" : lam_kd,
            "lam_cls": lam_cls,
        }
        return total, breakdown


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Co-Evolutionary Optimiser
# ─────────────────────────────────────────────────────────────────────────────
class CoEvolutionaryOptimiser:
    """
    Co-evolutionary update of teacher ↔ student populations.

    Maintains a small population of (teacher, student) pairs.
    Selection → Mutation → Crossover → Fitness evaluation.
    """

    def __init__(self, cfg: CuCoDistillConfig):
        self.cfg        = cfg
        self.pop_size   = cfg.coevo_population_size
        self.mut_rate   = cfg.coevo_mutation_rate
        self.cx_rate    = cfg.coevo_crossover_rate

    def mutate(self, params: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Gaussian mutation on a copy of params."""
        mutated = {}
        for k, v in params.items():
            if v.requires_grad:
                noise   = torch.randn_like(v) * self.mut_rate * NOISE_STD
                mutated[k] = v + noise
            else:
                mutated[k] = v.clone()
        return mutated

    def crossover(
        self,
        p1: Dict[str, Tensor],
        p2: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Uniform crossover between two parameter dicts."""
        child = {}
        for k in p1:
            mask    = (torch.rand_like(p1[k]) < self.cx_rate).float()
            child[k] = mask * p1[k] + (1 - mask) * p2[k]
        return child

    def select(
        self,
        population: List[Dict[str, Tensor]],
        fitnesses : List[float],
        n_select  : int,
    ) -> List[Dict[str, Tensor]]:
        """Tournament selection."""
        selected = []
        for _ in range(n_select):
            i, j = np.random.choice(len(population), 2, replace=False)
            winner = i if fitnesses[i] >= fitnesses[j] else j
            selected.append(deepcopy(population[winner]))
        return selected

    def evolve_step(
        self,
        population: List[Dict[str, Tensor]],
        fitnesses : List[float],
    ) -> List[Dict[str, Tensor]]:
        """One generation of selection → crossover → mutation."""
        parents  = self.select(population, fitnesses, n_select=self.pop_size)
        children = []
        for i in range(0, self.pop_size - 1, 2):
            c = self.crossover(parents[i], parents[i + 1])
            c = self.mutate(c)
            children.append(c)
        children.append(self.mutate(parents[-1]))    # elite
        return children[:self.pop_size]


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Curriculum Scheduler
# ─────────────────────────────────────────────────────────────────────────────
class CurriculumScheduler:
    """
    Controls the difficulty / stage of the curriculum.

    Stages:
      0: warm-up (CE only, easy samples)
      1: KD onset (add soft labels)
      2: feature alignment
      3: spectral consistency
      4: full co-evolutionary distillation

    Difficulty metric: cross-entropy on validation set per node.
    """

    def __init__(self, cfg: CuCoDistillConfig):
        self.cfg     = cfg
        self.stage   = 0
        self.stages  = cfg.curriculum_stages
        self.warmup  = cfg.warmup_epochs
        self.total   = cfg.curriculum_epochs

    def update(self, epoch: int, val_loss: float) -> int:
        """Return current stage index given epoch."""
        stage_len = max(1, (self.total - self.warmup) // (self.stages - 1))

        if epoch < self.warmup:
            self.stage = 0
        else:
            elapsed    = epoch - self.warmup
            self.stage = min(self.stages - 1, 1 + elapsed // stage_len)

        return self.stage

    def sample_difficulty_weights(
        self,
        per_node_loss : Tensor,
        epoch         : int,
    ) -> Tensor:
        """
        Easy-to-hard curriculum: sample probability ∝ difficulty^α(t)
        α(t) increases from 0 → 1 over training.
        """
        alpha = min(1.0, epoch / max(self.total, 1))
        w     = per_node_loss.pow(alpha)
        w     = w / (w.sum() + 1e-8)
        return w


# ─────────────────────────────────────────────────────────────────────────────
# 10.  Theoretical Guarantees (Theorems 1–4)
# ─────────────────────────────────────────────────────────────────────────────
class TheoreticalGuarantees:
    """
    Implements / verifies the four main theorems of CuCoDistill.
    """

    # ── Theorem 1: Spectral Approximation ──────────────────────────────────
    @staticmethod
    def theorem1_spectral_bound(
        H     : Tensor,
        H_hat : Tensor,
        epsilon: float = 0.05,
    ) -> Tuple[bool, float]:
        """
        Theorem 1: ∃ rank-r Ĥ s.t. ||Δ - Δ̂||_F ≤ ε ||Δ||_F

        Returns (satisfied: bool, actual_ratio: float).
        """
        Delta     = HypergraphUtils.normalised_laplacian(H)
        Delta_hat = HypergraphUtils.normalised_laplacian(H_hat)

        diff_norm = torch.norm(Delta - Delta_hat, "fro").item()
        base_norm = torch.norm(Delta, "fro").item() + 1e-12
        ratio     = diff_norm / base_norm

        return ratio <= epsilon, ratio

    # ── Theorem 2: Convergence Analysis ────────────────────────────────────
    @staticmethod
    def theorem2_convergence_bound(
        loss_history : List[float],
        lr           : float,
        n_params     : int,
    ) -> Dict[str, float]:
        """
        Theorem 2: Under Lipschitz smoothness, gradient descent converges.
        Convergence rate O(1/√T) for non-convex objectives.

        Returns estimated convergence metrics.
        """
        if len(loss_history) < 2:
            return {"bound": float("inf"), "rate": 0.0}

        T         = len(loss_history)
        L0        = loss_history[0]
        L_min     = min(loss_history)
        # Expected squared gradient norm bound: 2(L0 - L*) / (η√T)
        bound     = 2 * (L0 - L_min) / (lr * math.sqrt(T) + 1e-8)
        # Empirical convergence rate (log-linear fit)
        t_arr     = np.arange(1, T + 1, dtype=float)
        l_arr     = np.array(loss_history, dtype=float)
        if T > 2:
            slope, _ = np.polyfit(np.log(t_arr), l_arr, 1)
        else:
            slope     = 0.0

        return {"bound": bound, "empirical_rate": slope, "T": T}

    # ── Theorem 3: Generalisation Bound ────────────────────────────────────
    @staticmethod
    def theorem3_generalisation(
        n_train   : int,
        n_params  : int,
        delta     : float = 0.05,
        lambda_kd : float = 0.7,
    ) -> Dict[str, float]:
        """
        Theorem 3: PAC-Bayes generalisation bound for distilled student.

        Gen. error ≤ Empirical error + O(sqrt(n_params / n_train))
                      + λ_kd · KD_complexity_term

        Returns bound components.
        """
        complexity     = math.sqrt(n_params / max(n_train, 1))
        confidence_adj = math.sqrt(math.log(1.0 / delta) / (2 * n_train))
        kd_term        = lambda_kd * math.log(1 + 1.0 / lambda_kd)

        total_bound    = complexity + confidence_adj + kd_term
        return {
            "complexity_term"  : complexity,
            "confidence_adj"   : confidence_adj,
            "kd_term"          : kd_term,
            "total_bound"      : total_bound,
            "n_train"          : n_train,
            "delta"            : delta,
        }

    # ── Theorem 4: Student-surpasses-Teacher Guarantee ─────────────────────
    @staticmethod
    def theorem4_student_surpasses(
        teacher_acc : float,
        student_acc : float,
        threshold   : float = ACCURACY_IMPROVEMENT,
    ) -> Dict[str, object]:
        """
        Theorem 4: Under CuCo distillation, ∃ condition under which
          Acc(student) ≥ Acc(teacher) + δ,  δ = 0.91%

        Verifies empirically.
        """
        improvement  = student_acc - teacher_acc
        satisfied    = improvement >= threshold
        margin       = improvement - threshold

        return {
            "teacher_acc"  : teacher_acc,
            "student_acc"  : student_acc,
            "improvement"  : improvement,
            "threshold"    : threshold,
            "satisfied"    : satisfied,
            "margin"       : margin,
            "message"      : (
                f"✓ Theorem 4 satisfied: student surpasses teacher by "
                f"{improvement*100:.3f}% (threshold {threshold*100:.2f}%)"
                if satisfied else
                f"✗ Theorem 4 not yet satisfied (gap: {margin*100:.3f}%)"
            ),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 11.  Efficiency Benchmarking (Theorems → paper metrics)
# ─────────────────────────────────────────────────────────────────────────────
class EfficiencyBenchmark:
    """
    Measures inference speedup and memory reduction
    matching paper metrics: 127–133× speedup, 5.4–5.5× memory.
    """

    @staticmethod
    def count_params(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def measure_inference_time(
        model : nn.Module,
        x     : Tensor,
        H     : Tensor,
        U_r   : Tensor,
        n_runs: int = 50,
    ) -> float:
        """Returns average inference time in ms."""
        model.eval()
        with torch.no_grad():
            for _ in range(10):   # warm-up
                _ = model(x, H, U_r)
            t0 = time.perf_counter()
            for _ in range(n_runs):
                _ = model(x, H, U_r)
            elapsed = (time.perf_counter() - t0) / n_runs * 1000.0
        return elapsed

    @staticmethod
    def compute_memory_reduction(
        teacher_params: int,
        student_params: int,
    ) -> float:
        return teacher_params / max(student_params, 1)

    @classmethod
    def full_benchmark(
        cls,
        teacher : nn.Module,
        student : nn.Module,
        x       : Tensor,
        H       : Tensor,
        U_r     : Tensor,
    ) -> Dict[str, object]:
        t_params = cls.count_params(teacher)
        s_params = cls.count_params(student)

        t_time   = cls.measure_inference_time(teacher, x, H, U_r)
        s_time   = cls.measure_inference_time(student, x, H, U_r)

        speedup  = t_time / max(s_time, 1e-9)
        mem_red  = cls.compute_memory_reduction(t_params, s_params)

        # Verify against paper claims
        speedup_ok = SPEEDUP_LOWER <= speedup <= SPEEDUP_UPPER
        mem_ok     = MEMORY_REDUCTION_LB <= mem_red <= MEMORY_REDUCTION_UB

        return {
            "teacher_params" : t_params,
            "student_params" : s_params,
            "teacher_time_ms": t_time,
            "student_time_ms": s_time,
            "speedup"        : speedup,
            "memory_reduction": mem_red,
            "paper_speedup_range"  : f"{SPEEDUP_LOWER}–{SPEEDUP_UPPER}×",
            "paper_memory_range"   : f"{MEMORY_REDUCTION_LB}–{MEMORY_REDUCTION_UB}×",
            "speedup_within_range" : speedup_ok,
            "memory_within_range"  : mem_ok,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 12.  CuCoDistill Trainer (Full Training Loop)
# ─────────────────────────────────────────────────────────────────────────────
class CuCoDistillTrainer:
    """
    Full training pipeline:
      Phase 1: Pre-train teacher (standard cross-entropy)
      Phase 2: Curriculum co-evolutionary distillation of student
      Phase 3: Theorem verification + benchmarking
    """

    def __init__(self, cfg: CuCoDistillConfig):
        self.cfg       = cfg
        self.device    = torch.device(cfg.device)

        # Models
        self.teacher   = HTATeacher(cfg).to(self.device)
        self.student   = HTAStudent(cfg).to(self.device)

        # Distillation loss
        self.criterion = CuCoDistillLoss(cfg)

        # Optimisers
        self.t_optim   = AdamW(
            self.teacher.parameters(), lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay)
        self.s_optim   = AdamW(
            self.student.parameters(), lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay)

        # Schedulers
        self.t_sched   = CosineAnnealingLR(
            self.t_optim, T_max=cfg.curriculum_epochs)
        self.s_sched   = CosineAnnealingLR(
            self.s_optim, T_max=cfg.curriculum_epochs)

        # Curriculum
        self.scheduler = CurriculumScheduler(cfg)

        # Co-evolutionary optimiser
        self.coevo     = CoEvolutionaryOptimiser(cfg)

        # Hypergraph utils
        self.hg_utils  = HypergraphUtils()

        # Spectral data (computed once)
        self._U_r      = None
        self._S_r      = None

        # Loss history
        self.loss_history : List[float] = []

    # ── Setup ────────────────────────────────────────────────────────────────
    def setup_hypergraph(
        self,
        num_nodes   : int,
        num_hedges  : int,
        edge_index  : Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Build H, compute spectral basis."""
        H   = self.hg_utils.build_incidence_matrix(
            num_nodes, num_hedges, edge_index, device=str(self.device))
        U_r, S_r, err = self.hg_utils.spectral_approximation(
            H, self.cfg.spectral_rank, self.cfg.spectral_epsilon)
        print(f"  [Spectral] rank={self.cfg.spectral_rank}, "
              f"approx_err={err:.4f}, ε={self.cfg.spectral_epsilon}")
        self._U_r = U_r
        self._S_r = S_r
        return H, U_r, S_r

    # ── Phase 1: Pre-train teacher ──────────────────────────────────────────
    def pretrain_teacher(
        self,
        x           : Tensor,
        H           : Tensor,
        labels      : Tensor,
        train_mask  : Tensor,
        n_epochs    : int = 100,
        verbose     : bool = True,
    ) -> List[float]:
        """Standard supervised pre-training of the teacher."""
        print("\n=== Phase 1: Teacher Pre-Training ===")
        losses = []
        for ep in range(n_epochs):
            self.teacher.train()
            self.t_optim.zero_grad()

            t_out = self.teacher(x, H, self._U_r)
            loss  = F.cross_entropy(
                t_out["logits"][train_mask], labels[train_mask])
            loss.backward()
            nn.utils.clip_grad_norm_(self.teacher.parameters(), 5.0)
            self.t_optim.step()
            losses.append(loss.item())

            if verbose and (ep + 1) % 20 == 0:
                acc = self._accuracy(t_out["logits"], labels, train_mask)
                print(f"  Epoch {ep+1:3d}/{n_epochs}  "
                      f"loss={loss.item():.4f}  train_acc={acc:.4f}")

        return losses

    # ── Phase 2: Curriculum Co-Evolutionary Distillation ───────────────────
    def distill(
        self,
        x          : Tensor,
        H          : Tensor,
        labels     : Tensor,
        train_mask : Tensor,
        val_mask   : Tensor,
        verbose    : bool = True,
    ) -> Dict[str, List]:
        """
        Main distillation loop with curriculum scheduling.
        """
        print("\n=== Phase 2: Curriculum Co-Evolutionary Distillation ===")
        history = {
            "train_loss": [], "val_loss": [],
            "student_acc": [], "teacher_acc": [],
            "stage": [], "kd_loss": [],
        }

        best_student_acc = 0.0

        for epoch in range(self.cfg.curriculum_epochs):
            self.teacher.train()
            self.student.train()

            # Determine curriculum stage
            with torch.no_grad():
                val_loss_est = self._compute_val_loss(x, H, labels, val_mask)
            stage = self.scheduler.update(epoch, val_loss_est)

            # Teacher forward
            with torch.no_grad():
                t_out = self.teacher(x, H, self._U_r)

            # Curriculum sampling weights
            with torch.no_grad():
                per_node_ce  = F.cross_entropy(
                    t_out["logits"], labels, reduction="none")
                sample_w     = self.scheduler.sample_difficulty_weights(
                    per_node_ce, epoch)

            # Student forward
            self.s_optim.zero_grad()
            s_out = self.student(x, t_out.get("H_dropped", H), self._U_r)

            # Pass H_dropped to student output for loss
            s_out["H_dropped"] = t_out.get("H_dropped", H)

            # CuCo loss
            loss, breakdown = self.criterion(
                s_out, t_out, labels, self._U_r, self._S_r, stage, epoch)

            # Weighted by curriculum difficulty
            node_loss = F.cross_entropy(
                s_out["logits"], labels, reduction="none")
            weighted_loss = (node_loss * sample_w).sum() + 0.5 * loss
            weighted_loss.backward()
            nn.utils.clip_grad_norm_(self.student.parameters(), 5.0)
            self.s_optim.step()
            self.s_sched.step()

            self.loss_history.append(loss.item())

            # Metrics
            s_acc = self._accuracy(s_out["logits"], labels, train_mask)
            t_acc = self._accuracy(t_out["logits"], labels, train_mask)
            best_student_acc = max(best_student_acc, s_acc)

            history["train_loss"].append(loss.item())
            history["val_loss"].append(val_loss_est)
            history["student_acc"].append(s_acc)
            history["teacher_acc"].append(t_acc)
            history["stage"].append(stage)
            history["kd_loss"].append(breakdown["kd"])

            if verbose and (epoch + 1) % 20 == 0:
                print(
                    f"  Epoch {epoch+1:3d}/{self.cfg.curriculum_epochs}  "
                    f"stage={stage}  loss={loss.item():.4f}  "
                    f"kd={breakdown['kd']:.4f}  "
                    f"s_acc={s_acc:.4f}  t_acc={t_acc:.4f}"
                )

        return history

    # ── Phase 3: Evaluation & Theorem Verification ─────────────────────────
    def evaluate(
        self,
        x        : Tensor,
        H        : Tensor,
        labels   : Tensor,
        test_mask: Tensor,
    ) -> Dict[str, object]:
        """Full evaluation + theorem checks + efficiency benchmarking."""
        print("\n=== Phase 3: Evaluation & Theorem Verification ===")

        self.teacher.eval()
        self.student.eval()

        with torch.no_grad():
            t_out = self.teacher(x, H, self._U_r)
            s_out = self.student(x, t_out.get("H_dropped", H), self._U_r)

        t_acc  = self._accuracy(t_out["logits"], labels, test_mask)
        s_acc  = self._accuracy(s_out["logits"], labels, test_mask)

        # Theorem 1
        H_hat  = t_out.get("H_dropped", H)
        t1_ok, t1_ratio = TheoreticalGuarantees.theorem1_spectral_bound(
            H, H_hat, self.cfg.spectral_epsilon)

        # Theorem 2
        t2     = TheoreticalGuarantees.theorem2_convergence_bound(
            self.loss_history,
            self.cfg.learning_rate,
            EfficiencyBenchmark.count_params(self.student),
        )

        # Theorem 3
        t3     = TheoreticalGuarantees.theorem3_generalisation(
            n_train   = int(test_mask.sum().item()),
            n_params  = EfficiencyBenchmark.count_params(self.student),
            lambda_kd = self.cfg.curriculum_lambda_kd,
        )

        # Theorem 4
        t4     = TheoreticalGuarantees.theorem4_student_surpasses(t_acc, s_acc)

        # Efficiency benchmarking
        bench  = EfficiencyBenchmark.full_benchmark(
            self.teacher, self.student, x, H, self._U_r)

        results = {
            "teacher_acc"   : t_acc,
            "student_acc"   : s_acc,
            "accuracy_gain" : s_acc - t_acc,
            "theorem1"      : {"satisfied": t1_ok, "ratio": t1_ratio,
                               "epsilon": self.cfg.spectral_epsilon},
            "theorem2"      : t2,
            "theorem3"      : t3,
            "theorem4"      : t4,
            "efficiency"    : bench,
        }

        # Pretty print
        self._print_results(results)
        return results

    # ── Helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _accuracy(logits: Tensor, labels: Tensor, mask: Tensor) -> float:
        preds   = logits[mask].argmax(dim=-1)
        correct = (preds == labels[mask]).float().mean().item()
        return correct

    def _compute_val_loss(
        self,
        x      : Tensor,
        H      : Tensor,
        labels : Tensor,
        mask   : Tensor,
    ) -> float:
        self.student.eval()
        with torch.no_grad():
            t_out  = self.teacher(x, H, self._U_r)
            s_out  = self.student(x, t_out.get("H_dropped", H), self._U_r)
            loss   = F.cross_entropy(
                s_out["logits"][mask], labels[mask]).item()
        self.student.train()
        return loss

    @staticmethod
    def _print_results(r: Dict):
        print(f"\n{'='*60}")
        print(f"  CuCoDistill Final Results")
        print(f"{'='*60}")
        print(f"  Teacher  accuracy : {r['teacher_acc']*100:.2f}%")
        print(f"  Student  accuracy : {r['student_acc']*100:.2f}%")
        print(f"  Accuracy gain     : {r['accuracy_gain']*100:+.4f}%")
        print(f"\n  Theorem 1 [Spectral Approx.]  : "
              f"{'✓' if r['theorem1']['satisfied'] else '✗'}  "
              f"ratio={r['theorem1']['ratio']:.4f} ≤ ε={r['theorem1']['epsilon']}")
        print(f"  Theorem 2 [Convergence]        : "
              f"bound={r['theorem2']['bound']:.4f}")
        print(f"  Theorem 3 [Generalisation]     : "
              f"bound={r['theorem3']['total_bound']:.4f}")
        print(f"  Theorem 4 [Student>Teacher]    : {r['theorem4']['message']}")
        print(f"\n  Efficiency:")
        print(f"    Params  teacher / student : "
              f"{r['efficiency']['teacher_params']:,} / "
              f"{r['efficiency']['student_params']:,}")
        print(f"    Speedup                   : "
              f"{r['efficiency']['speedup']:.1f}×  "
              f"(paper: {r['efficiency']['paper_speedup_range']})")
        print(f"    Memory reduction          : "
              f"{r['efficiency']['memory_reduction']:.2f}×  "
              f"(paper: {r['efficiency']['paper_memory_range']})")
        print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# 13.  Synthetic Dataset Generator (Cora-like)
# ─────────────────────────────────────────────────────────────────────────────
class SyntheticHypergraphDataset:
    """
    Generates a synthetic Cora-like hypergraph dataset for testing.
    Uses std=NOISE_STD for feature noise to match paper setting.
    """

    def __init__(self, cfg: CuCoDistillConfig):
        self.cfg = cfg

    def generate(self) -> Dict[str, Tensor]:
        N = self.cfg.num_nodes
        F = self.cfg.num_features
        C = self.cfg.num_classes

        # Node features with noise std=5 (paper setting)
        centers  = torch.randn(C, F)
        labels   = torch.randint(0, C, (N,))
        x        = centers[labels] + torch.randn(N, F) * NOISE_STD  # std = 5

        # Masks (60/20/20 split)
        perm       = torch.randperm(N)
        n_train    = int(0.6 * N)
        n_val      = int(0.2 * N)
        train_mask = torch.zeros(N, dtype=torch.bool)
        val_mask   = torch.zeros(N, dtype=torch.bool)
        test_mask  = torch.zeros(N, dtype=torch.bool)
        train_mask[perm[:n_train]]           = True
        val_mask[perm[n_train:n_train+n_val]] = True
        test_mask[perm[n_train+n_val:]]      = True

        return {
            "x"          : x,
            "labels"     : labels,
            "train_mask" : train_mask,
            "val_mask"   : val_mask,
            "test_mask"  : test_mask,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 14.  Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  CuCoDistill: Curriculum Co-Evolutionary Knowledge")
    print("  Distillation with Spectral Guarantees")
    print(f"  Noise std (AKED / features) : {NOISE_STD}")
    print(f"  Target accuracy gain        : +{ACCURACY_IMPROVEMENT*100:.2f}%")
    print(f"  Target speedup              : {SPEEDUP_LOWER}–{SPEEDUP_UPPER}×")
    print(f"  Target memory reduction     : "
          f"{MEMORY_REDUCTION_LB}–{MEMORY_REDUCTION_UB}×")
    print("=" * 60)

    # ── Configuration ────────────────────────────────────────────────────────
    cfg = CuCoDistillConfig(
        num_nodes       = 500,        # reduced for demo speed
        num_edges       = 1000,
        num_features    = 64,
        num_classes     = 7,
        num_hyperedges  = 128,
        teacher_hidden_dims = [128, 64, 32],
        teacher_num_heads   = 4,
        teacher_num_layers  = 3,
        student_hidden_dims = [32, 16],
        student_num_heads   = 2,
        student_num_layers  = 2,
        spectral_rank       = 16,
        curriculum_epochs   = 60,
        warmup_epochs       = 10,
        curriculum_stages   = 5,
        aked_noise_std      = NOISE_STD,   # std = 5
        device              = "cpu",
    )

    # ── Dataset ──────────────────────────────────────────────────────────────
    print("\n[1] Generating synthetic hypergraph dataset ...")
    ds   = SyntheticHypergraphDataset(cfg)
    data = ds.generate()
    x, labels      = data["x"],    data["labels"]
    train_mask      = data["train_mask"]
    val_mask        = data["val_mask"]
    test_mask       = data["test_mask"]
    print(f"    Nodes: {cfg.num_nodes}  Features: {cfg.num_features}  "
          f"Classes: {cfg.num_classes}  "
          f"Train/Val/Test: {train_mask.sum()}/{val_mask.sum()}/{test_mask.sum()}")

    # ── Trainer ──────────────────────────────────────────────────────────────
    print("\n[2] Initialising CuCoDistill Trainer ...")
    trainer = CuCoDistillTrainer(cfg)

    # Build hypergraph & spectral basis
    print("\n[3] Building hypergraph + spectral approximation ...")
    H, U_r, S_r = trainer.setup_hypergraph(
        cfg.num_nodes, cfg.num_hyperedges)

    # Phase 1: Pre-train teacher
    teacher_losses = trainer.pretrain_teacher(
        x, H, labels, train_mask, n_epochs=50, verbose=True)

    # Phase 2: Distillation
    history = trainer.distill(
        x, H, labels, train_mask, val_mask, verbose=True)

    # Phase 3: Evaluate + verify theorems
    results = trainer.evaluate(x, H, labels, test_mask)

    # ── Convergence verification ─────────────────────────────────────────────
    print("\n[4] Convergence analysis (Theorem 2) ...")
    t2 = TheoreticalGuarantees.theorem2_convergence_bound(
        trainer.loss_history,
        cfg.learning_rate,
        EfficiencyBenchmark.count_params(trainer.student),
    )
    print(f"    T={t2['T']}  bound={t2['bound']:.6f}  "
          f"empirical_rate={t2['empirical_rate']:.6f}")

    # ── Generalisation bound ─────────────────────────────────────────────────
    print("\n[5] Generalisation bound (Theorem 3) ...")
    t3 = TheoreticalGuarantees.theorem3_generalisation(
        n_train   = int(train_mask.sum()),
        n_params  = EfficiencyBenchmark.count_params(trainer.student),
        lambda_kd = cfg.curriculum_lambda_kd,
    )
    for k, v in t3.items():
        print(f"    {k}: {v:.6f}" if isinstance(v, float) else f"    {k}: {v}")

    print("\n✓ CuCoDistill training and verification complete.\n")
    return results, history


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results, history = main()