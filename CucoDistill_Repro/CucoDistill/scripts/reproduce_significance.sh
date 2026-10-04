#!/usr/bin/env bash
# Reproduce the paired student-vs-teacher significance test (10 seeds, anchored at 5).
set -e
DATASETS="${1:-synthetic}"
for ds in $DATASETS; do
  echo "### $ds : significance (10 seeds)"
  python run_protocol.py --dataset "$ds" --method cuco --mode significance
done
