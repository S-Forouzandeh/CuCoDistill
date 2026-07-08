"""Efficiency benchmark: inference latency, parameter ratio, and the Theta(|V|/K) speedup.

Wraps the package Bench module on a trained synthetic model and prints the
measured teacher-vs-student latency and parameter ratio, plus a scalability
sweep of the theoretical Theta(|V|/K) speedup as |V| grows.

Note: the headline 127-133x / 5.4-5.5x figures in the paper are measured on the
large benchmarks (V100, full-graph batches). On a small CPU graph only the
Theta(|V|/K) *trend* is meaningful; this script demonstrates the mechanism, not
the absolute datacentre numbers.

Usage
-----
    python scripts/bench.py
    python scripts/bench.py --epochs 40 --scaling
"""
from __future__ import annotations

import os
import sys
import argparse
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from config import Config
from seed import set_seed
from datasets import load_dataset
from trainer import CuCoTrainer
from bench import Bench


def _train(cfg, seed):
    set_seed(seed)
    X, labels, hg, masks = load_dataset("synthetic", cfg, seed=seed)
    X, labels = X.to(cfg.device), labels.to(cfg.device)
    tr = CuCoTrainer(cfg)
    tr.pretrain_teacher(X, hg, labels, masks, verbose=False)
    tr.distill(X, hg, labels, masks, verbose=False)
    return tr, X, hg


def main():
    ap = argparse.ArgumentParser(description="Efficiency benchmark")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--base-seed", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--scaling", action="store_true",
                    help="also sweep the theoretical Theta(|V|/K) speedup over |V|")
    args = ap.parse_args()

    cfg = Config(device=args.device, seed=args.base_seed, epochs=args.epochs)
    tr, X, hg = _train(cfg, args.base_seed)
    pack = tr._pack(X, hg)
    rep = Bench.report(tr.model, pack, tr.K)

    print("=" * 66)
    print("  Efficiency benchmark (bundled synthetic graph)")
    print("=" * 66)
    print(f"  |V| = {hg.N}   |E| = {hg.M}   K (top-K) = {tr.K}")
    print("-" * 66)
    print(f"  teacher inference   : {rep['teacher_ms']:.3f} ms")
    print(f"  student inference   : {rep['student_ms']:.3f} ms")
    print(f"  measured speedup    : {rep['measured_speedup']:.1f}x")
    print(f"  theoretical |V|/K   : {rep['theoretical_speedup_NoverK']:.1f}x")
    print(f"  teacher-path params : {rep['teacher_path_params']:,}")
    print(f"  student-path params : {rep['student_path_params']:,}")
    print(f"  parameter ratio     : {rep['param_ratio']:.2f}x")
    print("-" * 66)
    print(f"  {rep['note']}")
    print("=" * 66)

    if args.scaling:
        # The paper's premise (Sec. 4.7 / Corollary): the maximum hyperedge size
        # is bounded, so K = ceil(alpha * max_i|E_i|) is ~constant as |V| grows,
        # giving an asymptotic Theta(|V|/K) speedup that increases with |V|. The
        # measured 127-133x on the benchmarks reflects dataset-specific K and
        # hardware constants; this table shows the asymptotic mechanism only.
        K_fixed = max(1, math.ceil(cfg.topk_alpha * 12))   # bounded max|E_i| ~ 12
        print(f"\n  Asymptotic Theta(|V|/K) speedup with K fixed at {K_fixed} "
              "(bounded hyperedge size):")
        print(f"  {'|V|':>9} {'K':>5} {'|V|/K (asymptotic)':>20}")
        for N in [1_000, 5_000, 10_000, 50_000, 100_000]:
            print(f"  {N:>9,} {K_fixed:>5} {N / K_fixed:>17.0f}x")
        print("  (K stays ~constant while |V| grows, so the speedup rises with scale;")
        print("   the measured benchmark range is 127-133x with dataset-specific K.)")


if __name__ == "__main__":
    main()
