"""Honest checks of Theorems 1-4 and the student-superiority diagnostic."""
from __future__ import annotations

import math
import numpy as np
import torch
from typing import List
from torch import Tensor


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
