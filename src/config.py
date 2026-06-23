"""Configuration loading and global seeding.

Central place that reads ``config.yaml`` (§6 reproducibility requirement) and
seeds every source of randomness (Python ``random``, NumPy, and -- lazily --
PyTorch) from the single global seed.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Repository root = parent of the ``src`` package directory.
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML configuration as a plain dict.

    Relative paths in the ``paths`` section are resolved against the repo root
    and returned as absolute :class:`pathlib.Path` objects under ``cfg['_paths']``.
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(path, "r") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)

    cfg["_root"] = ROOT
    cfg["_paths"] = {key: (ROOT / value) for key, value in cfg["paths"].items()}
    return cfg


def ensure_dirs(cfg: dict[str, Any]) -> None:
    """Create the results/figures/tables directories if missing."""
    for key in ("results_dir", "figures_dir", "tables_dir"):
        cfg["_paths"][key].mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and (if importable) PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is optional at import time for non-PNN code paths
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:  # pragma: no cover - torch is a hard dependency in practice
        pass
