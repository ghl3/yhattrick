"""Tests for the held-out harness's pure scoring helpers (the full harness is data-gated and runs
via its CLI: python -m yhattrick.models.generative_holdout)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

from yhattrick.models import generative_holdout as H


def test_poisson_deviance_prefers_truth():
    rng = np.random.default_rng(0)
    mu_true = np.exp(rng.normal(-2.0, 0.5, 20000))
    N = rng.poisson(mu_true).astype(float)
    d_true = H.poisson_deviance(N, mu_true)
    assert d_true < H.poisson_deviance(N, mu_true * 1.3)          # biased-up rates score worse
    assert d_true < H.poisson_deviance(N, np.full_like(mu_true, float(mu_true.mean())))  # flat too


def test_weighted_stats():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    w = np.ones(4)
    assert abs(H.wpearson(x, 2 * x + 1, w) - 1.0) < 1e-12         # affine ⇒ corr 1
    assert abs(H.wmae(x, x + 0.5, w) - 0.5) < 1e-12
    w2 = np.array([0.0, 0.0, 1.0, 1.0])                           # zero-weight rows ignored
    assert abs(H.wmae(x, np.array([9.0, 9.0, 3.5, 4.5]), w2) - 0.5) < 1e-12


def test_calibration_slope_recovers_truth():
    """γ̂ recovers the true out-of-sample calibration of a parameter block: face-value effects
    give γ≈1; effects fitted 2× too wide give γ≈0.5; pure noise gives γ≈0."""
    import numpy as np
    from yhattrick.models.generative_holdout import calibration_slope
    rng = np.random.default_rng(3)
    n = 150_000
    term = rng.normal(0.0, 0.35, n)                          # the block's fitted contribution
    base = np.full(n, 0.4)
    offs = np.full(n, np.log(40 * 45.0 / 3600.0))            # exposure ~40 aggregated stints
    for true_g in (1.0, 0.5, 0.0):
        N = rng.poisson(np.exp(base + true_g * term + offs))
        g = calibration_slope(N, offs, base, term)
        assert abs(g - true_g) < 0.05
