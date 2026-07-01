"""Single training run of CuCoDistill on one dataset/seed.

Examples
--------
    python run.py --dataset synthetic --seed 5 --epochs 120
    python run.py --dataset dblp --seed 5          # requires data_files/dblp/
"""
from __future__ import annotations

import argparse

import torch

from config import Config, config_for
from seed import set_seed
from datasets import load_dataset
from trainer import CuCoTrainer


def main():
    p = argparse.ArgumentParser(description="CuCoDistill single run")
    p.add_argument("--dataset", default="synthetic")
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    cfg = config_for(args.dataset, device=args.device, seed=args.seed) \
        if args.dataset != "synthetic" else Config(device=args.device, seed=args.seed)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    set_seed(cfg.seed)

    print("=" * 64)
    print(f"  CuCoDistill | dataset={args.dataset} seed={cfg.seed} device={cfg.device}")
    print(f"  AKED augmentation noise std = {cfg.aug_feat_noise}")
    print("=" * 64)

    X, labels, hg, masks = load_dataset(args.dataset, cfg, seed=cfg.seed)
    X, labels = X.to(cfg.device), labels.to(cfg.device)
    print(f"  nodes={hg.N} edges={hg.M} feat={X.size(1)} classes={cfg.num_classes} "
          f"| max|E_i|={hg.max_node_edges()} d_eff~={hg.effective_dimension()}")

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
