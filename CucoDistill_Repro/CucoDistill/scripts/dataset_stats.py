"""Hyperedge-cardinality statistics for a hypergraph dataset.

Reproduces the structural columns referenced in the paper's dataset tables:
|V|, |E|, mean / median / max hyperedge size, and the fraction of genuinely
higher-order edges (|e| > 2).

Usage
-----
    python scripts/dataset_stats.py                      # bundled synthetic
    python scripts/dataset_stats.py --dataset synthetic
    python scripts/dataset_stats.py --dataset dblp       # needs data_files/dblp/
    python scripts/dataset_stats.py --dataset house_bills

Real datasets must be present under ``data_files/<name>/`` (see README + datasets.load_real).
"""
from __future__ import annotations

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config import Config, config_for
from seed import set_seed
from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser(description="Hyperedge-cardinality statistics")
    ap.add_argument("--dataset", default="synthetic")
    ap.add_argument("--base-seed", type=int, default=5)
    args = ap.parse_args()

    cfg = (config_for(args.dataset, seed=args.base_seed)
           if args.dataset != "synthetic" else Config(seed=args.base_seed))
    set_seed(args.base_seed)

    try:
        X, labels, hg, masks = load_dataset(args.dataset, cfg, seed=args.base_seed)
    except Exception as e:
        print(f"  Could not load '{args.dataset}': {e}")
        print(f"  Real datasets must live under data_files/{args.dataset}/ "
              f"(see README 'Data').")
        sys.exit(1)

    sizes = hg.deg_e.cpu().numpy()                # per-hyperedge cardinalities
    print("=" * 60)
    print(f"  Hyperedge-cardinality statistics : {args.dataset}")
    print("=" * 60)
    print(f"  |V| (nodes)            : {hg.N}")
    print(f"  |E| (hyperedges)       : {hg.M}")
    print(f"  #features              : {X.size(1)}")
    print(f"  #classes               : {int(labels.max().item()) + 1}")
    print("-" * 60)
    print(f"  mean |e|               : {sizes.mean():.2f}")
    print(f"  median |e|             : {np.median(sizes):.1f}")
    print(f"  min / max |e|          : {int(sizes.min())} / {int(sizes.max())}")
    print(f"  fraction |e| > 2       : {100.0 * (sizes > 2).mean():.1f} %")
    print(f"  mean node degree       : {hg.deg_v.cpu().numpy().mean():.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
