"""Multi-seed protocol: reproduce mean +/- std and the paired significance test.

Seeds are anchored at the base seed (default 5):
    main mode         -> 5 seeds  [5, 6, 7, 8, 9]
    significance mode -> 10 seeds [5, 6, ..., 14]

Examples
--------
    python run_protocol.py --dataset synthetic --method cuco --mode main
    python run_protocol.py --dataset synthetic --method cuco --mode significance
    python run_protocol.py --dataset dblp --method hgnn --mode main
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from config import Config, config_for
from seed import set_seed, seed_list
from datasets import load_dataset
from trainer import CuCoTrainer
from baselines import BASELINES, train_baseline, backend_label, HAVE_DHG

try:
    from scipy import stats as _stats
    _HAVE_SCIPY = True
except Exception:                      # pragma: no cover
    _HAVE_SCIPY = False


def _summ(name, arr):
    return f"{name}: {np.mean(arr)*100:6.2f} +/- {np.std(arr)*100:.2f} %"


def run_cuco(dataset, cfg, seed, epochs):
    set_seed(seed)
    X, labels, hg, masks = load_dataset(dataset, cfg, seed=seed)
    X, labels = X.to(cfg.device), labels.to(cfg.device)
    if epochs is not None:
        cfg.epochs = epochs
    tr = CuCoTrainer(cfg)
    tr.pretrain_teacher(X, hg, labels, masks, verbose=False)
    tr.distill(X, hg, labels, masks, verbose=False)
    tr.model.eval()  # quiet evaluation (the trainer's evaluate() prints a report)
    with torch.no_grad():
        pack = tr._pack(X, hg)
        t = tr.model.teacher_forward(*pack)
        s = tr.model.student_forward(*pack, tr.K)
        t_acc = tr._acc(t["logits"], labels, masks["test"])
        s_acc = tr._acc(s["logits"], labels, masks["test"])
    return t_acc, s_acc


def run_baseline_seed(dataset, method, cfg, seed, epochs):
    set_seed(seed)
    X, labels, hg, masks = load_dataset(dataset, cfg, seed=seed)
    X, labels = X.to(cfg.device), labels.to(cfg.device)
    out = train_baseline(method, X, hg, labels, masks, cfg,
                         epochs=epochs or 200, verbose=False)
    return out["test_acc"]


def main():
    p = argparse.ArgumentParser(description="CuCoDistill multi-seed protocol")
    p.add_argument("--dataset", default="synthetic")
    p.add_argument("--method", default="cuco",
                   choices=["cuco"] + list(BASELINES.keys()))
    p.add_argument("--mode", default="main", choices=["main", "significance"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--base-seed", type=int, default=5)
    args = p.parse_args()

    cfg = (config_for(args.dataset, device=args.device, seed=args.base_seed)
           if args.dataset != "synthetic"
           else Config(device=args.device, seed=args.base_seed))
    n = cfg.num_seeds if args.mode == "main" else cfg.num_seeds_significance
    seeds = seed_list(n, base=args.base_seed)

    print("=" * 70)
    print(f"  Protocol: {args.method} on {args.dataset} | mode={args.mode} "
          f"| seeds={seeds}")
    print("=" * 70)

    if args.method == "cuco":
        t_accs, s_accs = [], []
        for sd in seeds:
            t_acc, s_acc = run_cuco(args.dataset, cfg, sd, args.epochs)
            t_accs.append(t_acc)
            s_accs.append(s_acc)
            print(f"  seed {sd:2d}: teacher {t_acc*100:6.2f}%  student {s_acc*100:6.2f}%"
                  f"  (gain {(s_acc-t_acc)*100:+.2f} pp)")
        t_accs, s_accs = np.array(t_accs), np.array(s_accs)
        print("-" * 70)
        print("  " + _summ("Teacher", t_accs))
        print("  " + _summ("Student", s_accs))
        gain = (s_accs - t_accs) * 100
        print(f"  Mean gain: {gain.mean():+.2f} pp")

        if len(seeds) >= 2:
            d = gain.mean() / (gain.std(ddof=1) + 1e-9)         # Cohen's d (paired)
            ci = 1.96 * gain.std(ddof=1) / np.sqrt(len(gain))
            print(f"  Cohen's d: {d:.2f}   95% CI of gain: "
                  f"[{gain.mean()-ci:+.2f}, {gain.mean()+ci:+.2f}] pp")
            if _HAVE_SCIPY:
                tstat, pval = _stats.ttest_rel(s_accs, t_accs)
                print(f"  Paired t-test (student vs teacher): "
                      f"t={tstat:.2f}, p={pval:.4f}")
            else:
                print("  (install scipy for the exact paired-t p-value)")
    else:
        print(f"  backend: {backend_label(args.method)}"
              + ("" if HAVE_DHG else "   (install 'dhg' for official implementations)"))
        accs = []
        for sd in seeds:
            a = run_baseline_seed(args.dataset, args.method, cfg, sd, args.epochs)
            accs.append(a)
            print(f"  seed {sd:2d}: {args.method} test {a*100:6.2f}%")
        print("-" * 70)
        print("  " + _summ(args.method, np.array(accs)))
    print("=" * 70)


if __name__ == "__main__":
    main()
