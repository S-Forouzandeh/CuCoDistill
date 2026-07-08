#!/usr/bin/env bash
# Reproduce the main accuracy table (mean +/- std over 5 seeds, anchored at 5).
# Replace the dataset list with the benchmarks you have placed under data_files/.
set -e
DATASETS="${1:-synthetic}"
for ds in $DATASETS; do
  echo "### $ds : CuCoDistill (teacher + student)"
  python run_protocol.py --dataset "$ds" --method cuco --mode main
  for base in hgnn hgnnp hypergcn hnhn unigat hypergat mlp; do
    echo "### $ds : baseline $base"
    python run_protocol.py --dataset "$ds" --method "$base" --mode main
  done
done
