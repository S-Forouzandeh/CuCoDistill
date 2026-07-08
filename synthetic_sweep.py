"""Controlled H-SBM-RF sweeps across all axes (paper's synthetic-validation table).

Runs the no-training redundancy diagnostic (R(X) vs R* = K/d_eff, spectral
coverage, predicted student-superiority) over every controllable axis -
redundancy, spectral coverage (Top-K alpha), and hyperedge cardinality - and,
with ``--train``, also measures the empirical teacher/student gap from a short
co-evolutionary run.

Usage
-----
    python scripts/synthetic_sweep.py                 # all axes, diagnostic only (fast)
    python scripts/synthetic_sweep.py --train         # also measure empirical gaps
    python scripts/synthetic_sweep.py --train --epochs 150 --seeds 5
"""
from __future__ import annotations

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hsbmrf
import run_sweep   # reuse the axis definitions + diagnose() + empirical_gap()
from config import Config
from seed import seed_list
from runlog import log_run


def main():
    ap = argparse.ArgumentParser(description="H-SBM-RF sweep over all axes")
    ap.add_argument("--train", action="store_true",
                    help="also measure the empirical teacher/student gap")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=1,
                    help="seeds per point when --train is set")
    ap.add_argument("--topk-alpha", type=float, default=0.5)
    ap.add_argument("--base-seed", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    base = hsbmrf.HSBMRFParams()
    print("#" * 92)
    print(f"  H-SBM-RF controlled sweeps  (base seed {args.base_seed}, "
          f"train={args.train})")
    print("  Diagnostic predicts: student-superior iff coverage (K >= d_eff) AND R(X) > R*.")
    print("#" * 92)

    for axis_name, axis_fn in run_sweep.AXES.items():
        label, points = axis_fn(base, args.topk_alpha)
        print(f"\n== axis: {axis_name} -- {label} ==")
        hdr = (f"  {'value':<16} {'R(X)':>6} {'R*':>6} {'K':>4} {'d_eff':>6} "
               f"{'cover':>6} {'predict':>9}")
        if args.train:
            hdr += f" {'emp.gap(pp)':>13}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for value, params, alpha in points:
            _, _, _, _, d = run_sweep.diagnose(params, args.base_seed, alpha)
            row = (f"  {value:<16} {d['R_x']:>6.2f} {d['R_star']:>6.2f} {d['K']:>4d} "
                   f"{d['d_eff']:>6d} {str(d['coverage']):>6} "
                   f"{str(d['predict_student_superior']):>9}")
            if args.train:
                mean, std = run_sweep.empirical_gap(
                    params, args.device, args.base_seed, alpha, args.epochs, args.seeds)
                row += f" {mean:>+8.2f}+/-{std:0.2f}"
            print(row)

    print("\n" + "#" * 92)
    print("  The redundancy axis crosses zero exactly as R(X) passes R*; collapsing the")
    print("  spectral gap (low homophily) voids coverage and restores the teacher; scale-free")
    print("  cardinality stresses the fixed Top-K constraint. See paper's synthetic section.")
    print("#" * 92)

    # provenance for the synthetic-validation table -> runs/synth_manifest.json
    log_run("synth_manifest", table="synthetic_sweep", dataset="hsbmrf",
            cfg=Config(topk_alpha=args.topk_alpha, epochs=args.epochs),
            seeds=seed_list(args.seeds, args.base_seed),
            extra={"axes": list(run_sweep.AXES.keys()),
                   "topk_alpha": args.topk_alpha,
                   "trained": bool(args.train)})


if __name__ == "__main__":
    main()
