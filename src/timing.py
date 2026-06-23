"""Query-time / scalability benchmark (§4.3) -- the PNN punchline.

Measures the wall-clock time to evaluate the estimated density at ``M`` query
points as a function of the training-set size ``N``. Parzen and k-NN cost grows
with ``N`` (each query sums/searches over all training points, O(N)); the PNN is
~constant per query after training (a single forward pass). The plot of this
table is the headline scalability result.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Callable

import numpy as np
import pandas as pd

from .datasets import make_synthetic_2d
from .estimators import GMMDensity, KNNDensity, ParzenNeuralNetwork, ParzenWindow


def _timing_estimators(cfg: dict[str, Any], seed: int) -> dict[str, Callable[[], Any]]:
    """Factories for the four methods, tuned for the timing benchmark.

    The PNN uses fewer epochs here -- training quality is irrelevant to the
    *query-time* measurement, only the architecture (forward-pass cost) matters.
    """
    return {
        "Parzen": lambda: ParzenWindow(bandwidth="silverman"),
        "k-NN": lambda: KNNDensity(k=cfg["knn"]["k"], renormalize=False),
        "GMM": lambda: GMMDensity(
            n_components=cfg["gmm"]["n_components_synthetic"], random_state=seed
        ),
        "PNN": lambda: ParzenNeuralNetwork(
            hidden=cfg["pnn"]["hidden"],
            activation=cfg["pnn"]["activation"],
            output_head=cfg["pnn"]["output_head"],
            epochs=150,
            n_grid_inputs=cfg["pnn"]["n_grid_inputs"],
            integral_penalty=cfg["pnn"]["integral_penalty"],
            integral_grid_res=cfg["pnn"]["integral_grid_res"],
            random_state=seed,
        ),
    }


def run_timing(cfg: dict[str, Any]) -> pd.DataFrame:
    """Benchmark query time vs N for every method.

    Returns a tidy DataFrame with columns
    ``[method, n_train, n_query, fit_time_s, query_time_s]`` (median over repeats).
    """
    tcfg = cfg["experiments"]["timing"]
    train_sizes = tcfg["train_sizes"]
    n_query = tcfg["n_query"]
    repeats = tcfg["repeats"]
    seed = cfg["seed"]

    mix = make_synthetic_2d(cfg)
    rng = np.random.default_rng(seed)
    # Fixed query set (independent of N) so timings are comparable across sizes.
    query = mix.sample(n_query, rng)

    rows = []
    for n in train_sizes:
        X = mix.sample(n, np.random.default_rng(seed + n))
        for name, factory in _timing_estimators(cfg, seed).items():
            est = factory()
            t0 = perf_counter()
            est.fit(X)
            fit_time = perf_counter() - t0

            est.score_samples(query[:16])  # warm-up (lazy init, caches)
            times = []
            for _ in range(repeats):
                t0 = perf_counter()
                est.score_samples(query)
                times.append(perf_counter() - t0)
            rows.append(
                {
                    "method": name,
                    "n_train": n,
                    "n_query": n_query,
                    "fit_time_s": fit_time,
                    "query_time_s": float(np.median(times)),
                }
            )
    return pd.DataFrame(rows)
