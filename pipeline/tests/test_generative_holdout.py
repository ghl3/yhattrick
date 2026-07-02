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
