"""Run provenance logging.

Every table / sweep command records, under ``runs/``, the exact seed list, a
hash of the resolved configuration, and the producing git commit, so that each
reported number is traceable to the seeds and configuration that produced it
(as described in the paper's reproducibility appendix). Per-seed metrics are
emitted to stdout by the individual commands; this module records the
provenance that ties those numbers to a specific configuration and commit.

The logger has no heavy dependencies (no torch is required to write a manifest),
so it runs even in a minimal environment.
"""
from __future__ import annotations

import os
import sys
import json
import hashlib
import platform
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

RUNS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "runs"
)


def _config_dict(cfg: Any) -> Dict[str, Any]:
    """Best-effort JSON-serialisable view of a Config (dataclass or object)."""
    if is_dataclass(cfg):
        return asdict(cfg)
    if hasattr(cfg, "__dict__"):
        return {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
    return {"repr": repr(cfg)}


def config_hash(cfg: Any) -> str:
    """Stable 12-char hash of the resolved configuration."""
    payload = json.dumps(_config_dict(cfg), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def git_commit() -> str:
    """Short git commit of the working tree, or 'unknown' outside a repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip() or "unknown"
    except Exception:
        return "unknown"


def _torch_version() -> str:
    try:
        import torch  # local import: manifests must be writable without torch
        return torch.__version__
    except Exception:
        return "not-installed"


def write_manifest(
    name: str,
    *,
    table: Optional[str] = None,
    dataset: Optional[str] = None,
    cfg: Any = None,
    seeds: Optional[List[int]] = None,
    extra: Optional[Dict[str, Any]] = None,
    results: Optional[Dict[str, Any]] = None,
    runs_dir: str = RUNS_DIR,
) -> str:
    """Write ``runs/<name>.json`` recording seeds, config hash, and commit.

    Returns the path written. ``results`` is optional structured output; the
    canonical per-seed numbers are printed to stdout by the calling command.
    """
    os.makedirs(runs_dir, exist_ok=True)
    manifest: Dict[str, Any] = {
        "name": name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "table": table,
        "dataset": dataset,
        "seeds": list(seeds) if seeds is not None else None,
        "num_seeds": len(seeds) if seeds is not None else None,
        "config_hash": config_hash(cfg) if cfg is not None else None,
        "git_commit": git_commit(),
        "environment": {
            "python": platform.python_version(),
            "torch": _torch_version(),
            "platform": platform.platform(),
        },
        "config": _config_dict(cfg) if cfg is not None else None,
    }
    if extra:
        manifest.update(extra)
    if results is not None:
        manifest["results"] = results
    path = os.path.join(runs_dir, f"{name}.json")
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    return path


def log_run(
    name: str,
    *,
    table: Optional[str] = None,
    dataset: Optional[str] = None,
    cfg: Any = None,
    seeds: Optional[List[int]] = None,
    extra: Optional[Dict[str, Any]] = None,
    results: Optional[Dict[str, Any]] = None,
) -> str:
    """Write a manifest and print a one-line pointer to it."""
    path = write_manifest(
        name, table=table, dataset=dataset, cfg=cfg,
        seeds=seeds, extra=extra, results=results,
    )
    ch = config_hash(cfg) if cfg is not None else "n/a"
    print(f"  [runlog] provenance -> {os.path.relpath(path)} "
          f"(config {ch}, commit {git_commit()})")
    return path
