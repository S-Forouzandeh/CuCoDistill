# Integrating external baselines

This package ships the official **DHG** baselines (`hgnn`, `hgnnp`, `hypergcn`,
`hnhn`, `unigcn/unigat/unisage/unigin`) and a runnable GLNN-style KD reference
(`glnn_ref`). The remaining published baselines come from third-party
repositories that **cannot be redistributed here**. This guide shows how to plug
them in behind the package's `forward(X, hg) -> logits` interface.

> Always obtain the URLs and code from each method's own paper/repo, and cite
> them. The links below are starting points — verify against the publication.

## Model-type baselines (Hyper-SAGNN, CHGNN, HyGCL-AdT)

These are wired through `adapters.py`.

| `--method` | Paper | Official repo (verify) | Graph input |
|---|---|---|---|
| `hyper_sagnn` | Zhang et al., 2019 | github.com/ma-compbio/Hyper-SAGNN | `edge_index` |
| `chgnn` | Song et al., 2024 (TKDE) | repo linked in the paper | `incidence` |
| `hygcl_adt` | Qian et al., 2024 (WWW) | repo linked in the paper | `incidence` |

Steps:

1. **Clone** the repo into `external/<name>/` (or anywhere on `PYTHONPATH`):
   ```bash
   git clone <repo-url> external/hyper_sagnn
   export PYTHONPATH=$PYTHONPATH:external/hyper_sagnn
   ```
2. **Point the spec at the repo's model** in `adapters.py`:
   ```python
   EXTERNAL_SPECS["hyper_sagnn"].module = "models.hyper_sagnn"   # repo's module path
   EXTERNAL_SPECS["hyper_sagnn"].cls    = "HyperSAGNN"           # repo's class name
   EXTERNAL_SPECS["hyper_sagnn"].graph  = "edge_index"          # or 'incidence' / 'clique_adj' / 'dhg'
   ```
3. **Match the constructor** by editing `.builder` (signatures differ per repo):
   ```python
   def _build(cfg, hidden):
       from models.hyper_sagnn import HyperSAGNN
       return HyperSAGNN(in_dim=cfg.num_features, hid=hidden,
                         n_classes=cfg.num_classes)        # adapt to the real signature
   EXTERNAL_SPECS["hyper_sagnn"].builder = _build
   ```
   If the model's `forward` is not `net(X, graph)`, also set `.call`:
   ```python
   EXTERNAL_SPECS["hyper_sagnn"].call = lambda net, X, g: net(X, g, return_logits=True)
   ```
4. **Run** it via the standard interface:
   ```python
   from adapters import build_external
   model = build_external("hyper_sagnn", cfg)   # nn.Module with forward(X, hg)
   ```
   `build_external` raises a step-by-step error if the repo isn't importable yet.

The converters available for `graph` are in `adapters.CONVERTERS`:
`incidence` (H), `clique_adj` (normalised A), `edge_index` (2×E), `dhg`
(`dhg.Hypergraph`).

## KD-type baselines (GLNN, KRD, LightHGNN, DistillHGNN, SSGNN, LAD-GNN)

These are distillation **procedures**, handled in `kd.py`.

| `--method` | Paper | Official repo (verify) |
|---|---|---|
| `glnn_ref` | — (runnable here) | hypergraph GLNN-style reference shipped in `kd.py` |
| `glnn` | Tian et al., 2022 | github.com/snap-stanford/graphless-neural-networks |
| `krd` | Wu et al., 2023 | github.com/LirongWu/RKD |
| `lighthgnn` | Feng et al., 2024 | github.com/iMoonLab/LightHGNN |
| `distillhgnn` | Forouzandeh et al., 2025 | authors' repo |
| `ssgnn` | Wu et al., 2024 | authors' repo |
| `lad_gnn` | Hong et al., 2024 | authors' repo |

To wire one in, clone its repo (as above) and implement a hook in `kd.run_kd`
that runs the authors' train/distill and returns the test accuracies:

```python
def run_kd(name, X, hg, labels, masks, cfg):
    if name in ("glnn_ref", "softlabel"):
        return glnn_reference(X, hg, labels, masks, cfg)
    if name == "lighthgnn":
        from lighthgnn import train_lighthgnn          # repo entry point
        out = train_lighthgnn(X, hg, labels, masks, cfg)
        return {"teacher_test": out["teacher_acc"], "student_test": out["student_acc"]}
    ...
```

Until then, `run_kd`/`run_kd.py` prints the repo URL and these steps. The
runnable `glnn_ref` (a hypergraph teacher distilled into an MLP student) gives a
working KD point of comparison out of the box.

## Adding any baseline to the sweep / protocol

Once a model exposes `forward(X, hg) -> logits`, register it so the multi-seed
protocol can run it:

```python
# in baselines.py
from adapters import build_external
BASELINES["hyper_sagnn"] = lambda cfg: build_external("hyper_sagnn", cfg)
```

It is then available via `python run_protocol.py --method hyper_sagnn ...` with
the same mean ± std + significance machinery as every other baseline.
