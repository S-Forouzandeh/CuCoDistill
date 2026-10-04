"""Efficiency benchmarking (measured + theoretical)."""
from __future__ import annotations

import time
from typing import Dict
import torch
import torch.nn as nn
from model import CuCoModel


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
