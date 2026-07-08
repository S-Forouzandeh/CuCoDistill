"""One-command-per-table reproduction entry point (matches the paper's reproducibility appendix).

Usage
-----
    python scripts/run_table.py --table accuracy
    python scripts/run_table.py --table f1_auc
    python scripts/run_table.py --table significance
    python scripts/run_table.py --table distillation
    python scripts/run_table.py --table theorems
    python scripts/run_table.py --table hparam
    python scripts/run_table.py --table ablation
    python scripts/run_table.py --table higher_order

Notes
-----
* The default dataset is the bundled synthetic hypergraph, so every table runs
  out of the box with no external data.
* Pass ``--dataset <name>`` (e.g. ``dblp``) to target a real benchmark; that
  requires the data under ``data_files/<name>/`` (see README + ``datasets.py``).
* Seeds are anchored at the base seed (default 5): the standard protocol uses
  ``cfg.num_seeds`` (5) seeds {5..9}; the significance table uses
  ``cfg.num_seeds_significance`` (10) seeds {5..14}.
* Tables that intrinsically need the full nine benchmarks or external baselines
  print an explicit pointer rather than fabricating numbers.
"""
from __future__ import annotations

import os
import sys
import argparse

# --- make the package modules (one level up) importable -----------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from config import Config, config_for
from seed import set_seed, seed_list
from datasets import load_dataset
from trainer import CuCoTrainer
from theory import Theory
from runlog import log_run


# ----------------------------------------------------------------------------- #
# Helpers                                                                        #
# ----------------------------------------------------------------------------- #
def _cfg(dataset: str, device: str, base_seed: int, epochs):
    cfg = (config_for(dataset, device=device, seed=base_seed)
           if dataset != "synthetic" else Config(device=device, seed=base_seed))
    if epochs is not None:
        cfg.epochs = epochs
    return cfg


def _train_once(dataset, cfg, seed):
    """Train one teacher+student pair and return everything downstream code needs."""
    set_seed(seed)
    X, labels, hg, masks = load_dataset(dataset, cfg, seed=seed)
    X, labels = X.to(cfg.device), labels.to(cfg.device)
    tr = CuCoTrainer(cfg)
    tr.pretrain_teacher(X, hg, labels, masks, verbose=False)
    tr.distill(X, hg, labels, masks, verbose=False)
    tr.model.eval()
    with torch.no_grad():
        pack = tr._pack(X, hg)
        t = tr.model.teacher_forward(*pack)
        s = tr.model.student_forward(*pack, tr.K)
    return tr, X, labels, hg, masks, t, s


def _bar(title):
    print("=" * 74)
    print(f"  {title}")
    print("=" * 74)


def _data_note(dataset):
    if dataset == "synthetic":
        return
    print(f"\n  [data] '{dataset}' must be present under data_files/{dataset}/ "
          f"(see README 'Data' and datasets.load_real).")


# ----------------------------------------------------------------------------- #
# Metric helpers                                                                 #
# ----------------------------------------------------------------------------- #
def _linear_cka(A: torch.Tensor, B: torch.Tensor) -> float:
    A = A - A.mean(0, keepdim=True)
    B = B - B.mean(0, keepdim=True)
    num = (A.t() @ B).pow(2).sum()
    den = (A.t() @ A).norm() * (B.t() @ B).norm() + 1e-12
    return float((num / den).clamp(0, 1))


def _jsd_log2(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    m = 0.5 * (p + q)
    def _kl(a, b):
        return (a * (a.clamp_min(1e-9).log() - b.clamp_min(1e-9).log())).sum(-1)
    return (0.5 * _kl(p, m) + 0.5 * _kl(q, m)) / np.log(2.0)   # in [0,1]


def _dark_knowledge_retention(s_logits, t_logits, labels) -> float:
    ps, pt = F.softmax(s_logits, 1).clone(), F.softmax(t_logits, 1).clone()
    idx = torch.arange(labels.numel())
    ps[idx, labels] = 0.0
    pt[idx, labels] = 0.0
    ps = ps / ps.sum(1, keepdim=True).clamp_min(1e-9)
    pt = pt / pt.sum(1, keepdim=True).clamp_min(1e-9)
    return float(1.0 - _jsd_log2(pt, ps).mean())


def _soft_kl(s_logits, t_logits, T=4.0) -> float:
    pt = F.softmax(t_logits / T, 1)
    ps = F.softmax(s_logits / T, 1)
    return float((pt * (pt.clamp_min(1e-9).log() - ps.clamp_min(1e-9).log())).sum(1).mean())


# ----------------------------------------------------------------------------- #
# Tables                                                                         #
# ----------------------------------------------------------------------------- #
def table_accuracy(dataset, cfg, base_seed, significance=False):
    n = cfg.num_seeds_significance if significance else cfg.num_seeds
    seeds = seed_list(n, base=base_seed)
    kind = "significance (10-seed)" if significance else "main (5-seed)"
    _bar(f"Node-classification accuracy | {dataset} | {kind} protocol | seeds={seeds}")
    _data_note(dataset)
    t_accs, s_accs = [], []
    for sd in seeds:
        tr, X, labels, hg, masks, t, s = _train_once(dataset, cfg, sd)
        ta = tr._acc(t["logits"], labels, masks["test"])
        sa = tr._acc(s["logits"], labels, masks["test"])
        t_accs.append(ta); s_accs.append(sa)
        print(f"  seed {sd:2d}: teacher {ta*100:6.2f}%   student {sa*100:6.2f}%   "
              f"gain {(sa-ta)*100:+.2f} pp")
    t_accs, s_accs = np.array(t_accs), np.array(s_accs)
    gain = (s_accs - t_accs) * 100
    print("-" * 74)
    print(f"  Teacher: {t_accs.mean()*100:6.2f} +/- {t_accs.std()*100:.2f} %")
    print(f"  Student: {s_accs.mean()*100:6.2f} +/- {s_accs.std()*100:.2f} %")
    print(f"  Mean gain (student - teacher): {gain.mean():+.2f} pp")
    if significance and len(seeds) >= 2:
        d = gain.mean() / (gain.std(ddof=1) + 1e-9)
        ci = 1.96 * gain.std(ddof=1) / np.sqrt(len(gain))
        print(f"  Cohen's d: {d:.2f}   95% CI of gain: "
              f"[{gain.mean()-ci:+.2f}, {gain.mean()+ci:+.2f}] pp")
        try:
            from scipy import stats
            tstat, pval = stats.ttest_rel(s_accs, t_accs)
            print(f"  Paired t-test (df={len(seeds)-1}): t={tstat:.2f}, p={pval:.4f}")
        except Exception:
            print("  (install scipy for the exact paired-t p-value)")
    print("=" * 74)


def table_f1_auc(dataset, cfg, base_seed):
    try:
        from sklearn.metrics import f1_score, roc_auc_score
    except Exception:
        print("  scikit-learn is required for Macro-F1 / AUC-ROC "
              "(pip install scikit-learn --break-system-packages).")
        return
    seeds = seed_list(cfg.num_seeds, base=base_seed)
    _bar(f"Macro-F1 and AUC-ROC (macro-OvR) | {dataset} | 5-seed protocol | seeds={seeds}")
    _data_note(dataset)
    rows = {"teacher": {"f1": [], "auc": []}, "student": {"f1": [], "auc": []}}
    for sd in seeds:
        tr, X, labels, hg, masks, t, s = _train_once(dataset, cfg, sd)
        m = masks["test"].cpu().numpy()
        y = labels.cpu().numpy()[m]
        for who, out in (("teacher", t), ("student", s)):
            prob = F.softmax(out["logits"], 1).cpu().numpy()[m]
            pred = prob.argmax(1)
            rows[who]["f1"].append(f1_score(y, pred, average="macro"))
            try:
                rows[who]["auc"].append(
                    roc_auc_score(y, prob, multi_class="ovr", average="macro"))
            except Exception:
                rows[who]["auc"].append(float("nan"))
    print(f"  {'model':<9}{'Macro-F1 (%)':>16}{'AUC-ROC (%)':>16}")
    for who in ("teacher", "student"):
        f1 = np.array(rows[who]["f1"]) * 100
        au = np.array(rows[who]["auc"]) * 100
        print(f"  {who:<9}{f1.mean():>10.2f} +/-{f1.std():<4.2f}"
              f"{au.mean():>10.2f} +/-{au.std():<4.2f}")
    print("=" * 74)


def table_distillation(dataset, cfg, base_seed):
    seeds = seed_list(cfg.num_seeds, base=base_seed)
    _bar(f"Distillation quality (CKA / DKR / KL@T=4) | {dataset} | seeds={seeds}")
    _data_note(dataset)
    cka, dkr, kl = [], [], []
    for sd in seeds:
        tr, X, labels, hg, masks, t, s = _train_once(dataset, cfg, sd)
        cka.append(_linear_cka(t["emb"], s["emb"]))
        dkr.append(_dark_knowledge_retention(s["logits"], t["logits"], labels))
        kl.append(_soft_kl(s["logits"], t["logits"], cfg.kd_temperature))
    cka, dkr, kl = np.array(cka), np.array(dkr), np.array(kl)
    print(f"  CKA similarity (higher better): {cka.mean():.3f} +/- {cka.std():.3f}")
    print(f"  Dark-knowledge retention      : {dkr.mean():.3f} +/- {dkr.std():.3f}")
    print(f"  Soft-label KL (lower better)  : {kl.mean():.3f} +/- {kl.std():.3f}")
    print("=" * 74)


def table_theorems(dataset, cfg, base_seed):
    _bar(f"Empirical theorem checks (Thm 1-4) | {dataset} | seed={base_seed}")
    _data_note(dataset)
    tr, X, labels, hg, masks, t, s = _train_once(dataset, cfg, base_seed)
    N, max_Ei = hg.N, hg.max_node_edges()
    d_eff = hg.effective_dimension(cfg.deff_threshold)
    n_train = int(masks["train"].sum().item())
    n_params = sum(p.numel() for p in tr.model.parameters())

    t1 = Theory.t1_spectral(t["attn"], s["attn"], N, max_Ei, cfg.spectral_eps)
    t2 = Theory.t2_convergence(getattr(tr, "task_curve", []))
    t3 = Theory.t3_generalisation(n_train, n_params)
    t4 = Theory.t4_diagnostic(X, tr.K, d_eff)

    print(f"  Thm 1 (spectral approx): Frobenius err {t1['frob_error']:.4f} "
          f"<= bound {t1['paper_bound']:.4f}  -> {'OK' if t1['satisfied'] else 'CHECK'}")
    print(f"          implied per-interaction eps = {t1['implied_eps']:.4f} "
          f"(paper eps = {t1['eps']})")
    print(f"  Thm 2 (convergence)    : log-log task-loss slope {t2['slope']:.3f} "
          f"(target {t2['target']:.2f}, O(1/sqrt(T)))")
    print(f"  Thm 3 (generalisation) : bound {t3['bound']:.3f}  "
          f"(complexity {t3['complexity']:.3f} + confidence {t3['confidence']:.3f})")
    print(f"  Thm 4 (student-superior diagnostic): R(X)={t4['R_x']:.2f}  "
          f"R*={t4['R_star']:.2f}  coverage(K>=d_eff)={t4['coverage']}  "
          f"-> predict student-superior = {t4['predict_student_superior']}")
    print("=" * 74)


def table_hparam(dataset, cfg, base_seed):
    grid = [0.30, 0.40, 0.45, 0.50, 0.60, 0.70]   # Top-K alpha (paper Table 13, Group A)
    _bar(f"Hyperparameter sensitivity: Top-K alpha | {dataset} | seed={base_seed}")
    _data_note(dataset)
    accs = []
    for a in grid:
        c = _cfg(dataset, cfg.device, base_seed, cfg.epochs)
        c.topk_alpha = a
        tr, X, labels, hg, masks, t, s = _train_once(dataset, c, base_seed)
        sa = tr._acc(s["logits"], labels, masks["test"]) * 100
        accs.append(sa)
        print(f"  alpha={a:.2f}:  student {sa:6.2f}%   (K={tr.K})")
    accs = np.array(accs)
    print("-" * 74)
    print(f"  Delta = max - min = {accs.max()-accs.min():.2f} pp   "
          f"(alpha is the paper's single High-sensitivity knob)")
    print("=" * 74)


def table_ablation(dataset, cfg, base_seed):
    _bar(f"Ablation (Top-K spectral-regularisation slice) | {dataset} | seed={base_seed}")
    _data_note(dataset)
    # Full model (compressed Top-K student) vs a near-teacher student that keeps
    # almost the full neighbourhood -> isolates the Top-K regularisation pillar.
    variants = [("Full CuCoDistill (alpha=0.40)", 0.40),
                ("w/o Top-K compression (alpha=0.95)", 0.95)]
    res = []
    for name, a in variants:
        c = _cfg(dataset, cfg.device, base_seed, cfg.epochs)
        c.topk_alpha = a
        tr, X, labels, hg, masks, t, s = _train_once(dataset, c, base_seed)
        sa = tr._acc(s["logits"], labels, masks["test"]) * 100
        res.append((name, sa))
        print(f"  {name:<36} student {sa:6.2f}%")
    if len(res) == 2:
        print("-" * 74)
        print(f"  Delta (Full - w/o compression) = {res[0][1]-res[1][1]:+.2f} pp")
    print("\n  Note: this reproduces the Top-K / spectral-regularisation pillar on the")
    print("  bundled data. The full 23-config ablation (co-evolution, AKED, multi-level")
    print("  transfer, curriculum) toggles the components described in paper Sec. 3 and")
    print("  INTEGRATION.md; enable those flags / baselines to regenerate the full table.")
    print("=" * 74)


def table_higher_order(dataset, cfg, base_seed):
    _bar("Higher-order benchmarks (large-cardinality stress test)")
    real = ["senate_bills", "house_bills", "contact_primary", "modelnet40", "ntu2012"]
    # Try the real higher-order sets first; fall back to a scale-free synthetic proxy.
    ran_real = False
    for name in real:
        try:
            c = config_for(name, device=cfg.device, seed=base_seed, epochs=cfg.epochs)
            tr, X, labels, hg, masks, t, s = _train_once(name, c, base_seed)
            ta = tr._acc(t["logits"], labels, masks["test"]) * 100
            sa = tr._acc(s["logits"], labels, masks["test"]) * 100
            print(f"  {name:<18} teacher {ta:6.2f}%  student {sa:6.2f}%  "
                  f"delta {sa-ta:+.2f} pp  (mean|e|~{hg.deg_e.mean().item():.1f})")
            ran_real = True
        except Exception:
            continue
    if not ran_real:
        print("  Real higher-order datasets not found under data_files/.")
        print("  Running a scale-free synthetic proxy (large hyperedges) instead:\n")
        import hsbmrf
        p = hsbmrf.HSBMRFParams(card_mode="scalefree", card_min=4, card_max=40,
                                card_alpha=2.0, noise_dim=96)
        set_seed(base_seed)
        X, labels, hg, masks = hsbmrf.generate(p, seed=base_seed)
        X, labels = X.to(cfg.device), labels.to(cfg.device)
        c = Config(num_features=X.size(1), num_classes=p.n_classes,
                   num_nodes=p.n_nodes, topk_alpha=0.50, device=cfg.device,
                   seed=base_seed, epochs=cfg.epochs)
        tr = CuCoTrainer(c)
        tr.pretrain_teacher(X, hg, labels, masks, verbose=False)
        tr.distill(X, hg, labels, masks, verbose=False)
        tr.model.eval()
        with torch.no_grad():
            pack = tr._pack(X, hg)
            ta = tr._acc(tr.model.teacher_forward(*pack)["logits"], labels, masks["test"]) * 100
            sa = tr._acc(tr.model.student_forward(*pack, tr.K)["logits"], labels, masks["test"]) * 100
        print(f"  scale-free proxy   teacher {ta:6.2f}%  student {sa:6.2f}%  "
              f"delta {sa-ta:+.2f} pp  (max|e|={int(hg.deg_e.max().item())})")
        print("\n  For the paper's exact higher-order numbers, place senate-bills / house-bills /")
        print("  contact-primary-school / ModelNet40 / NTU2012 under data_files/ (sources in README).")
    print("=" * 74)


TABLES = {
    "accuracy":     lambda ds, cfg, bs: table_accuracy(ds, cfg, bs, significance=False),
    "significance": lambda ds, cfg, bs: table_accuracy(ds, cfg, bs, significance=True),
    "f1_auc":       table_f1_auc,
    "distillation": table_distillation,
    "theorems":     table_theorems,
    "hparam":       table_hparam,
    "ablation":     table_ablation,
    "higher_order": table_higher_order,
}


def main():
    ap = argparse.ArgumentParser(description="Per-table reproduction dispatcher")
    ap.add_argument("--table", required=True, choices=list(TABLES.keys()))
    ap.add_argument("--dataset", default="synthetic")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--base-seed", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    cfg = _cfg(args.dataset, args.device, args.base_seed, args.epochs)
    TABLES[args.table](args.dataset, cfg, args.base_seed)
    # provenance: seeds + config hash + commit -> runs/<table>_<dataset>.json
    n_seeds = cfg.num_seeds_significance if args.table == "significance" else cfg.num_seeds
    log_run(f"{args.table}_{args.dataset}", table=args.table, dataset=args.dataset,
            cfg=cfg, seeds=seed_list(n_seeds, args.base_seed))


if __name__ == "__main__":
    main()
