# CuCoDistill — Reproducibility Package

Curriculum Co-Evolutionary Knowledge Distillation with Spectral Guarantees for
Hypergraph Neural Networks.

This repository is the modular, reproducible implementation requested in review
(R1.5 / R1.6). It contains the full method, baseline models, a multi-seed
protocol with significance testing, per-dataset configs, and one-command table
reproduction. It runs **out of the box on synthetic data**; the real benchmarks
are loaded from a documented on-disk format (see *Datasets* below) because the
third-party data cannot be redistributed here.

## Layout

```
config.py         Config dataclass + per-dataset overrides (DATASET_CONFIGS)
seed.py           set_seed (one value -> all RNGs) + seed_list (anchored at 5)
hypergraph.py     incidence, normalised Laplacian, effective spectral dimension
haaa.py           HAAA: shared three-head attention (local / set / global)
aked.py           Adaptive Knowledge-guided Edge Dropping (+ feature aug, noise 0.01)
model.py          CuCoModel: shared backbone + full teacher + Top-K student
losses.py         L_embed, L_attn, L_feat, soft-KD, InfoNCE contrastive
curriculum.py     spectral curriculum: D_contrast / D_distill, quantile gates, lambdas
theory.py         Theorems 1-4 checks + constructive R* = K/d_eff diagnostic
bench.py          measured + theoretical efficiency
baselines.py      HGNN, HyperGCN, HyperGAT, MLP + train_baseline
official.py       official DHG-backed baselines (HGNN/HGNNP/HyperGCN/HNHN/Uni-*)
adapters.py       plug external baseline repos (Hyper-SAGNN/CHGNN/HyGCL-AdT) behind the API
kd.py             KD baselines: runnable glnn_ref + official-repo pointers
trainer.py        CuCoTrainer (teacher is NOT frozen -> real co-evolution)
datasets.py       synthetic generator + real-benchmark loader (standard format)
hsbmrf.py         H-SBM-RF controlled generator (redundancy / coverage / cardinality)
run.py            single run
run_protocol.py   multi-seed mean +/- std + paired significance test
run_kd.py         multi-seed KD-baseline runner
run_sweep.py      controlled H-SBM-RF sweeps (diagnostic + optional training)
scripts/run_table.py       one command per paper table (--table accuracy|f1_auc|
                           significance|distillation|theorems|hparam|ablation|higher_order)
scripts/dataset_stats.py   hyperedge-cardinality statistics for a dataset
scripts/synthetic_sweep.py H-SBM-RF sweep over all axes (wraps run_sweep)
scripts/bench.py           efficiency benchmark (latency / params / Theta(|V|/K))
scripts/reproduce_main.sh, scripts/reproduce_significance.sh   batch helpers
INTEGRATION.md    how to wire in external baseline repos and KD methods
```

## Install

```bash
pip install -r requirements.txt    # torch, numpy, scipy
```

## Quick start (synthetic, no data needed)

```bash
python run.py --dataset synthetic --seed 5 --epochs 120
python run_protocol.py --dataset synthetic --method cuco --mode main          # 5 seeds
python run_protocol.py --dataset synthetic --method cuco --mode significance  # 10 seeds
python run_protocol.py --dataset synthetic --method hgnn --mode main          # a baseline
```

## Seeds

A single base seed (default **5**) is applied uniformly to every RNG via
`set_seed`. Multi-run protocols are anchored at that base seed:

* main results — `num_seeds = 5` → seeds `[5, 6, 7, 8, 9]`
* significance — `num_seeds_significance = 10` → seeds `[5, 6, …, 14]`

Each run uses one distinct seed; averaging over runs is what produces the
reported mean ± std and the paired *t*-test (student vs teacher).

## Datasets

`--dataset synthetic` uses the built-in generator. Any other name is loaded by
`datasets.load_real` from:

```
data_files/<name>/
    features.npy        # float (N, F)
    labels.npy          # int   (N,)
    hyperedges.txt      # one hyperedge per line: space-separated node indices
    splits/<seed>.npz   # optional: arrays 'train','val','test' (bool, len N)
```

If `splits/<seed>.npz` is absent, a deterministic 60/20/20 split is generated
from the seed. Per-dataset hyperparameters (classes, Top-K α, learning rate)
live in `DATASET_CONFIGS` in `config.py`.

**Where to obtain each benchmark** (convert to the format above):

| Dataset(s) | Source |
|---|---|
| CC-Cora, CC-Citeseer | Planetoid splits (Yang et al., 2016) |
| DBLP, DBLP-paper/term/Conf, IMDB, IMDB-AW | the heterogeneous-graph / HGB releases used by the cited HGNN works |
| Yelp | Yelp Open Dataset (review hypergraph construction in the paper) |
| senate/house/congress-bills, contact-primary-school, email-Enron | Benson et al. hypergraph collection |
| ModelNet40, NTU2012 | the 3D-shape hypergraph releases used by HGNN (Feng et al., 2019) |

Place the converted files under `data_files/<name>/` and the loaders work
unchanged.

## Baselines

Run any baseline through the same protocol with `--method`:

```bash
python run_protocol.py --dataset synthetic --method hgnn --mode main
```

When the **DHG (DeepHypergraph)** library is installed, the package uses the
authors' *official* implementations and reports `backend: official (DHG)`:

| `--method` | Backend | Source |
|---|---|---|
| `hgnn` | official | DHG — Feng et al., 2019 |
| `hgnnp` | official | DHG — HGNN⁺ |
| `hypergcn` | official | DHG — Yadati et al., 2019 (mediator) |
| `hnhn` | official | DHG — Dong et al., 2020 |
| `unigcn`, `unigat`, `unisage`, `unigin` | official | DHG — Huang & Yang, 2021 |
| `hgnn_ref`, `hypergcn_ref` | reference | in-repo (for comparison) |
| `hypergat` | reference | in-repo — Bai et al., 2021 |
| `mlp` | reference | in-repo (GLNN-style) |

Install the official backends with:

```bash
pip install dhg                 # pulls optuna + scikit-learn (used by dhg)
```

DHG conservatively pins `numpy<2` / `torch<2`, but the model classes run on
newer versions (verified on torch 2.x). **If DHG is not installed, the package
falls back to the in-repo reference implementations automatically** — `hgnn` and
`hypergcn` then resolve to the reference versions and `backend: reference` is
reported. The reference and official versions are intentionally kept side by
side (`hgnn` vs `hgnn_ref`) so you can confirm parity.

The remaining published baselines (Hyper-SAGNN, CHGNN, HyGCL-AdT, and the KD
methods GLNN/KRD/LightHGNN/DistillHGNN/SSGNN/LAD-GNN) come from third-party
repositories that cannot be bundled here. They are wired in through
`adapters.py` (model-type) and `kd.py` (KD-type) behind the same
`forward(X, hg) -> logits` interface; **see `INTEGRATION.md` for step-by-step
instructions.** Until a repo is wired in, the adapters print the exact clone /
PYTHONPATH / builder steps.

### Knowledge-distillation baselines

A runnable, faithful **GLNN-style** KD baseline ships out of the box — a
hypergraph teacher (the official DHG model when available) distilled into an MLP
student:

```bash
python run_kd.py --dataset synthetic --method glnn_ref --mode main   # teacher + student, 5 seeds
python run_kd.py --dataset synthetic --method krd                    # prints integration steps
```

The named KD methods (`glnn`, `krd`, `lighthgnn`, `distillhgnn`, `ssgnn`,
`lad_gnn`) run once their official repo is wired in per `INTEGRATION.md`.

## Controlled synthetic sweeps (H-SBM-RF)

`run_sweep.py` reproduces the revised paper's controlled experiments using the
H-SBM-RF generator (`hsbmrf.py`), validating the closed-form student-superiority
condition `R(X) > R* = K/d_eff` with the **no-training diagnostic** and,
optionally, a short training run:

```bash
python run_sweep.py --axis redundancy     # R(X) crosses R* -> prediction flips
python run_sweep.py --axis coverage       # K >= d_eff (Top-K alpha) coverage transition
python run_sweep.py --axis cardinality    # fixed vs scale-free hyperedge sizes
python run_sweep.py --axis redundancy --train --epochs 150 --seeds 5   # + empirical gap
```

The diagnostic is the clean theoretical check (it computes R(X), d_eff, K, R\*
and the predicted gap sign exactly). `--train` adds the empirical
teacher–student gap from a finite-budget run; the student must converge (use
enough epochs) before the predicted advantage appears, so the diagnostic is the
primary signal.

## Reproducing the tables

Each table in the paper maps to **one command**, matching the paper's
reproducibility appendix. These run out of the box on the bundled synthetic
hypergraph; pass `--dataset <name>` to target a real benchmark you have placed
under `data_files/` (see *Datasets*).

```bash
python scripts/dataset_stats.py                      # hyperedge-cardinality statistics
python scripts/run_table.py --table accuracy         # node-classification accuracy (5 seeds)
python scripts/run_table.py --table f1_auc           # Macro-F1 and AUC-ROC
python scripts/run_table.py --table significance     # paired t-tests (10 seeds, df=9)
python scripts/run_table.py --table distillation     # CKA / DKR / KL
python scripts/run_table.py --table higher_order     # higher-order benchmarks (or scale-free proxy)
python scripts/synthetic_sweep.py                    # H-SBM-RF controlled sweeps
python scripts/run_table.py --table theorems         # Thm 1-4 empirical checks
python scripts/bench.py                              # latency, params, Theta(|V|/K), scalability
python scripts/run_table.py --table ablation         # Top-K / spectral-regularisation slice
python scripts/run_table.py --table hparam           # Top-K alpha sensitivity
```

> **Epoch budget.** The per-table commands use the default training budget
> (`Config.epochs = 120`), at which the synthetic Top-K student converges and
> matches the teacher. A small `--epochs` value (e.g. 10) is fast but *undertrains*
> the student, so the student accuracy will look artificially low — use the
> default for a fair comparison. The same caveat applies to `run_sweep.py --train`.

The batch helpers below loop the multi-seed protocol over several datasets via
`run_protocol.py` (equivalent to `--table accuracy` / `--table significance`):

```bash
bash scripts/reproduce_main.sh "dblp imdb yelp cc_cora cc_citeseer"
bash scripts/reproduce_significance.sh "dblp imdb dblp_term yelp"
```

Every command logs the seed list, per-seed numbers, and the aggregate
statistics, so each reported figure is traceable to its seeds and config.

## Notes on fidelity

* The teacher is trained **simultaneously** with the student through a shared
  backbone — this is the co-evolution, not sequential KD.
* The student uses **Top-K** neighbour selection (`N_i^K`), the source of the
  O(K|V|d) inference cost and the spectral regularisation behind Theorem 2.
* AKED feature-augmentation noise is **0.01** (paper value), not 5.
* `theory.py` reports the theorem quantities **honestly**: on a small synthetic
  graph the Theorem-1 Frobenius error can exceed the bound and the speedup is
  only a few ×; the 127–133× speedup and the bound hold at benchmark scale (the
  efficiency report prints the theoretical Θ(|V|/K) factor alongside the
  measured one).
* Core baselines (`hgnn`, `hgnnp`, `hypergcn`, `hnhn`, `unigcn/unigat/unisage/unigin`)
  use the **official DHG implementations** when DHG is installed, with in-repo
  reference versions as a fallback (and as `*_ref` for parity checks). See the
  Baselines section for the published methods not covered by DHG.
