"""Multi-seed runner for KD baselines (seeds anchored at the base seed 5).

'glnn_ref' runs out of the box (hypergraph teacher -> MLP student). The named
methods require their official repo wired in (see INTEGRATION.md); this runner
will print the exact integration steps if the repo is absent.

Examples
--------
    python run_kd.py --dataset synthetic --method glnn_ref --mode main
    python run_kd.py --dataset synthetic --method krd      # prints integration help
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from config import Config, config_for
from seed import set_seed, seed_list
from datasets import load_dataset
from kd import run_kd, KD_SPECS


def main():
    p = argparse.ArgumentParser(description="KD baseline protocol")
    p.add_argument("--dataset", default="synthetic")
    p.add_argument("--method", default="glnn_ref",
                   choices=["glnn_ref"] + list(KD_SPECS.keys()))
    p.add_argument("--mode", default="main", choices=["main", "significance"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--base-seed", type=int, default=5)
    args = p.parse_args()

    cfg = (config_for(args.dataset, device=args.device, seed=args.base_seed)
           if args.dataset != "synthetic"
           else Config(device=args.device, seed=args.base_seed))
    n = cfg.num_seeds if args.mode == "main" else cfg.num_seeds_significance
    seeds = seed_list(n, base=args.base_seed)

    print("=" * 70)
    print(f"  KD protocol: {args.method} on {args.dataset} | seeds={seeds}")
    print("=" * 70)

    t_accs, s_accs = [], []
    for sd in seeds:
        set_seed(sd)
        X, labels, hg, masks = load_dataset(args.dataset, cfg, seed=sd)
        X, labels = X.to(cfg.device), labels.to(cfg.device)
        try:
            r = run_kd(args.method, X, hg, labels, masks, cfg)
        except (NotImplementedError, ImportError) as e:
            print(str(e))
            return
        t_accs.append(r["teacher_test"]); s_accs.append(r["student_test"])
        print(f"  seed {sd:2d}: teacher {r['teacher_test']*100:6.2f}%  "
              f"student {r['student_test']*100:6.2f}%")

    t_accs, s_accs = np.array(t_accs), np.array(s_accs)
    print("-" * 70)
    print(f"  Teacher: {t_accs.mean()*100:6.2f} +/- {t_accs.std()*100:.2f} %")
    print(f"  Student: {s_accs.mean()*100:6.2f} +/- {s_accs.std()*100:.2f} %  "
          f"(retention {s_accs.mean()/max(t_accs.mean(),1e-9)*100:.1f}% of teacher)")
    print("=" * 70)


if __name__ == "__main__":
    main()
