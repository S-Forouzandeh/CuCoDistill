"""Hypergraph: incidence, Laplacian, effective dimension, masks."""
from __future__ import annotations

import torch
from torch import Tensor


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
