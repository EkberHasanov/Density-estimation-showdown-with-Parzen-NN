"""End-to-end reproducible run: data -> experiments -> tables -> figures (§8.7).

Usage:
    python scripts/run_all.py [--config config.yaml]

Regenerates every CSV in ``results/tables/`` and every figure in
``results/figures/`` from scratch, deterministically (single global seed).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make the repo root importable when run as a plain script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ensure_dirs, load_config, seed_everything  # noqa: E402
from src.experiments import run_all_experiments  # noqa: E402
from src.plotting import make_all_figures  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Density Estimation Showdown runner")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    seed_everything(cfg["seed"])

    print(f"=== Density Estimation Showdown (seed={cfg['seed']}) ===", flush=True)
    t0 = time.perf_counter()

    results = run_all_experiments(cfg)
    make_all_figures(cfg, results)

    dt = time.perf_counter() - t0
    tables = sorted(cfg["_paths"]["tables_dir"].glob("*.csv"))
    figures = sorted(cfg["_paths"]["figures_dir"].glob("*.png"))
    print(f"\n=== done in {dt:.1f}s ===")
    print(f"  {len(tables)} tables  -> {cfg['_paths']['tables_dir']}")
    print(f"  {len(figures)} figures -> {cfg['_paths']['figures_dir']}")


if __name__ == "__main__":
    main()
