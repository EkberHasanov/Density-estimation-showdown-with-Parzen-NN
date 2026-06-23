"""Sanity checks on the synthetic generators: the true density integrates to 1."""

from __future__ import annotations

import numpy as np

from src.config import load_config
from src.datasets import (
    make_synthetic_1d,
    make_synthetic_2d,
    synthetic_1d_grid,
    synthetic_2d_grid,
)
from src.metrics import integrate_on_grid


def test_p_true_2d_integrates_to_one():
    cfg = load_config()
    mix = make_synthetic_2d(cfg)
    grid = synthetic_2d_grid(cfg)
    mass = integrate_on_grid(mix.p_true(grid.points), grid)
    assert abs(mass - 1.0) < 1e-2, f"2D p_true integrates to {mass}"


def test_p_true_1d_integrates_to_one():
    cfg = load_config()
    mix = make_synthetic_1d(cfg)
    grid = synthetic_1d_grid(cfg)
    mass = integrate_on_grid(mix.p_true(grid.points), grid)
    assert abs(mass - 1.0) < 1e-3, f"1D p_true integrates to {mass}"


def test_sample_shapes_and_reproducibility():
    cfg = load_config()
    mix = make_synthetic_2d(cfg)
    a = mix.sample(500, np.random.default_rng(0))
    b = mix.sample(500, np.random.default_rng(0))
    assert a.shape == (500, 2)
    np.testing.assert_allclose(a, b)  # same seed -> same draw


def test_weights_must_sum_to_one():
    import pytest

    from src.datasets import GaussianMixture1D

    with pytest.raises(ValueError):
        GaussianMixture1D(
            weights=np.array([0.5, 0.4]),
            means=np.array([0.0, 1.0]),
            stds=np.array([1.0, 1.0]),
        )
