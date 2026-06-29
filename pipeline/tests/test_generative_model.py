"""Tests for the experimental generative (Poisson) model. Requires JAX (the `experimental` group),
so the whole module skips when JAX isn't installed:  uv run --group experimental pytest.

Synthetic tests are self-contained; a data-gated test exercises the real loaders/fit when a
processed/ tree is present."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")          # skip entirely unless the experimental deps are installed

from yhattrick import config as C
from yhattrick.models import generative_model as G


# ── synthetic builders ──────────────────────────────────────────────────────────────────────────

def _synth_rate(P=80, n=40000, dur=45.0, seed=0, r_disp=None):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 0.30, P)
    d = rng.normal(0, 0.30, P)
    mu0 = np.log(30.0)                                   # baseline ~30 shots/60
    off = np.array([rng.choice(P, 5, replace=False) for _ in range(n)])
    dff = np.array([rng.choice(P, 5, replace=False) for _ in range(n)])
    Xctx = rng.integers(0, 2, (n, 1)).astype(float)     # one varying context col (home), true coef 0
    eta = mu0 + a[off].sum(1) + d[dff].sum(1)
    lam = np.exp(eta) * dur / 3600.0
    if r_disp is None:
        count = rng.poisson(lam).astype(float)
    else:                                               # overdispersed: Poisson-Gamma (=NB)
        count = rng.poisson(lam * rng.gamma(r_disp, 1.0 / r_disp, size=n)).astype(float)
    R = {"players": list(range(P)), "idx": {p: p for p in range(P)},
         "off_idx": off, "def_idx": dff, "Xctx": Xctx, "count": count,
         "offset": np.full(n, np.log(dur / 3600.0)), "dur": np.full(n, dur),
         "def_goalie": np.zeros(n, dtype=object)}
    return R, a, d


def _synth_quality(P=60, n=60000, seed=1):
    rng = np.random.default_rng(seed)
    u = rng.normal(0, 0.30, P)
    w = rng.normal(0, 0.30, P)
    mu0 = np.log(0.07 / 0.93)                            # baseline mean xG ~0.07
    off = np.array([rng.choice(P, 5, replace=False) for _ in range(n)])
    dff = np.array([rng.choice(P, 5, replace=False) for _ in range(n)])
    Xctx = rng.integers(0, 2, (n, 1)).astype(float)     # one varying context col (home), true coef 0
    p = 1.0 / (1.0 + np.exp(-(mu0 + u[off].sum(1) + w[dff].sum(1))))
    s = 12.0
    xg = rng.beta(s * p, s * (1 - p))
    y = np.log(np.clip(xg, 1e-6, 1 - 1e-6) / (1 - np.clip(xg, 1e-6, 1 - 1e-6)))
    Q = {"off_idx": off, "def_idx": dff, "Xctx": Xctx, "y": y, "goal": rng.binomial(1, np.clip(xg, 0, 1))}
    return Q, u, w


# ── rate layer ────────────────────────────────────────────────────────────────────────────────

def test_rate_recovers_known_effects():
    R, a, d = _synth_rate()
    fit = G.fit_rate(R)
    assert fit["converged"]
    assert np.corrcoef(fit["a"], a)[0, 1] > 0.8       # offense effects recovered
    assert np.corrcoef(fit["d"], d)[0, 1] > 0.8       # defense effects recovered


def test_rate_reconciles_total_shots():
    """A Poisson fit with a free intercept reproduces the total count (Σμ ≈ Σcount)."""
    R, _, _ = _synth_rate()
    fit = G.fit_rate(R)
    eta = (fit["intercept"] + fit["a"][R["off_idx"]].sum(1) + fit["d"][R["def_idx"]].sum(1)
           + R["Xctx"] @ fit["beta"])
    mu = np.exp(eta + R["offset"])
    assert abs(mu.sum() - R["count"].sum()) / R["count"].sum() < 0.01


def test_rate_hessian_se_positive():
    R, _, _ = _synth_rate()
    fit = G.fit_rate(R)
    assert np.all(fit["se_a"] > 0) and np.all(np.isfinite(fit["se_a"]))


def test_nb_recovers_and_estimates_dispersion():
    """On overdispersed (Poisson-Gamma) counts, NB recovers the effects and a finite positive r."""
    R, a, d = _synth_rate(r_disp=2.0, seed=7)
    fit = G.fit_rate(R, count_model="nb")
    assert fit["converged"]
    assert fit["r"] is not None and np.isfinite(fit["r"]) and fit["r"] > 0
    assert np.corrcoef(fit["a"], a)[0, 1] > 0.8
    assert np.all(fit["se_a"] > 0)


# ── quality layer ───────────────────────────────────────────────────────────────────────────────

def test_quality_recovers_and_is_mean_calibrated():
    Q, u, w = _synth_quality()
    fit = G.fit_quality(Q, P=len(u))
    assert fit["converged"]
    assert np.corrcoef(fit["u"], u)[0, 1] > 0.8
    # fractional-logit is mean-calibrated: mean σ(η̂) ≈ mean xG
    eta = (fit["intercept"] + fit["u"][Q["off_idx"]].sum(1) + fit["w"][Q["def_idx"]].sum(1)
           + Q["Xctx"] @ fit["beta"])
    xg = 1.0 / (1.0 + np.exp(-Q["y"]))
    assert abs(float(G._sigmoid(eta).mean()) - float(xg.mean())) < 1e-3


# ── simulation ────────────────────────────────────────────────────────────────────────────────

def test_simulate_count_matches_intensity():
    """E[N] over many draws ≈ λ·t/3600 for a known intensity."""
    rng = np.random.default_rng(3)
    lam, t = 30.0, 60.0
    draws = rng.poisson(lam * t / 3600.0, size=200000)
    assert abs(draws.mean() - lam * t / 3600.0) < 0.01


# ── data-gated real fit ─────────────────────────────────────────────────────────────────────────

def _has_data():
    d = C.PROCESSED / "stints"
    return d.exists() and any(d.glob("*.parquet"))


@pytest.mark.skipif(not _has_data(), reason="no processed/ tree")
def test_real_fit_smoke():
    seasons = G.shooting_model.available_seasons()[-1:]
    R = G.rate_rows(seasons)
    assert R is not None and len(R["count"]) > 0
    fit = G.fit_rate(R)
    assert fit["converged"] and np.all(np.isfinite(fit["a"]))
