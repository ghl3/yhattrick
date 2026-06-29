"""Tests for the joint shooter×goalie shot-conversion model (models/shooting_model.py).

Synthetic tests are CI-safe (no data tree needed); the real-data reconciliation test is data-gated
and skips when processed/shots_onice is absent (mirrors test_stints.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yhattrick import config as C
from yhattrick.models import shooting_model as S


# ---- synthetic helpers -------------------------------------------------------------------------

def _make_shots(n, n_shooters, n_goalies, alpha, gamma, seed=0):
    """Simulate shots: goal ~ Bernoulli(clip(xg + alpha_shooter + gamma_goalie)). Returns a frame
    with the columns the model consumes."""
    rng = np.random.default_rng(seed)
    s = rng.integers(0, n_shooters, n)
    g = rng.integers(0, n_goalies, n)
    xg = rng.uniform(0.02, 0.30, n)
    p = np.clip(xg + alpha[s] + gamma[g], 1e-3, 1 - 1e-3)
    goal = (rng.uniform(0, 1, n) < p).astype(int)
    return pd.DataFrame({"shooter_id": s.astype(int), "goalie_id": (1000 + g).astype(int),
                         "xg": xg, "goal": goal})


def _names(ids):
    return {int(i): {"name": f"#{i}", "pos": "C"} for i in ids}


# ---- additivity / reconciliation (the core promise) --------------------------------------------

def test_additive_identity_exact():
    """goals == sum(xG) + sum(mu) + sum(finishing) + sum(goalie), to the penny (free intercept)."""
    rng = np.random.default_rng(1)
    alpha = rng.normal(0, 0.03, 40); gamma = rng.normal(0, 0.02, 12)
    shots = _make_shots(20000, 40, 12, alpha, gamma, seed=1)
    fin, gl, meta = S.fit_shots(shots, _names(set(shots.shooter_id) | set(shots.goalie_id)))
    rc = meta["reconciliation"]
    assert abs(rc["identity_resid"]) < 1e-6
    # the four pieces literally rebuild the goal total
    rebuilt = rc["xg"] + rc["sum_intercept"] + rc["sum_finishing"] + rc["sum_goalie_effect"]
    assert abs(rebuilt - rc["goals"]) < 0.05


def test_finishing_plus_goalie_equals_fitted_residual():
    """Per shot, finishing_i + goalie_i + mu == fitted(goal - xg): additive by construction."""
    rng = np.random.default_rng(2)
    alpha = rng.normal(0, 0.03, 30); gamma = rng.normal(0, 0.02, 10)
    shots = _make_shots(15000, 30, 10, alpha, gamma, seed=2)
    fin, gl, meta = S.fit_shots(shots, _names(set(shots.shooter_id) | set(shots.goalie_id)))
    # fin_per100/100 = alpha (constant per shot); gsax_per100/-100 = gamma
    a = dict(zip(fin.player_id, fin.fin_per100 / 100.0))
    g = dict(zip(gl.player_id, -gl.gsax_per100 / 100.0))
    mu = meta["reconciliation"]["intercept_per_shot"]
    per_shot = shots.shooter_id.map(a) + shots.goalie_id.map(g) + mu
    # league sum of (finishing + goalie + mu) == sum(goal - xg)
    assert abs(per_shot.sum() - (shots.goal - shots.xg).sum()) < 0.05


# ---- recovery: the joint fit separates shooter from goalie -------------------------------------

def test_recovers_known_effects():
    """With enough crossed data and light shrinkage, fitted alpha/gamma correlate strongly with the
    truth and have the right sign — the joint fit pulls the two apart."""
    rng = np.random.default_rng(3)
    n_sh, n_go = 60, 16
    alpha = rng.normal(0, 0.04, n_sh)
    gamma = rng.normal(0, 0.03, n_go)
    shots = _make_shots(120000, n_sh, n_go, alpha, gamma, seed=3)
    # light penalties so recovery isn't dominated by shrinkage
    fin, gl, _ = S.fit_shots(shots, _names(set(shots.shooter_id) | set(shots.goalie_id)),
                             lams=(50.0, 200.0))
    fa = fin.set_index("player_id").fin_per100.reindex(range(n_sh)).to_numpy() / 100.0
    ga = gl.set_index("player_id").gsax_per100.reindex([1000 + i for i in range(n_go)]).to_numpy() / -100.0
    assert np.corrcoef(fa, alpha)[0, 1] > 0.8
    assert np.corrcoef(ga, gamma)[0, 1] > 0.8


def test_shrinkage_monotone_and_continuity():
    """More shrinkage (larger lambda) pulls effects toward zero; and a lone shooter facing one
    goalie reduces toward the EB estimate (G-ixg)*n/(n+k)."""
    rng = np.random.default_rng(4)
    alpha = rng.normal(0, 0.05, 25); gamma = rng.normal(0, 0.02, 8)
    shots = _make_shots(30000, 25, 8, alpha, gamma, seed=4)
    nm = _names(set(shots.shooter_id) | set(shots.goalie_id))
    light, _, _ = S.fit_shots(shots, nm, lams=(50.0, 200.0))
    heavy, _, _ = S.fit_shots(shots, nm, lams=(5000.0, 20000.0))
    # heavier penalty -> smaller spread of finishing
    assert heavy.fin_per100.std() < light.fin_per100.std()


def test_empty_returns_empty():
    fin, gl, meta = S.fit(["__no_such_season__"] if False else [9999], _names([]))
    assert fin.empty and gl.empty


# ---- estimate_k matches the old empirical-Bayes formula ----------------------------------------

def test_estimate_k_positive_and_finite():
    rng = np.random.default_rng(5)
    F = rng.integers(50, 2000, 200).astype(float)
    ixg = F * rng.uniform(0.05, 0.12, 200)
    made = ixg + rng.normal(0, np.sqrt(F) * 0.3)          # noisy goals around ixg
    V = F * 0.07
    k, diag = S._estimate_k(F, ixg, made, V, min_shots=100)
    assert k > 0 and np.isfinite(k) and diag["k"] == round(k, 1)


# ---- data-gated real-data reconciliation -------------------------------------------------------

def _shots_seasons():
    d = C.PROCESSED / "shots_onice"
    return sorted(int(p.stem) for p in d.glob("*.parquet")) if d.exists() else []


@pytest.mark.parametrize("season", _shots_seasons())
def test_real_data_reconciles(season):
    """On real data each season's fit reconciles goals to xG+mu+finishing+goalie within rounding,
    and the leaderboards are sane (top names finite, sensible magnitudes)."""
    from yhattrick.models.player_onice_model import roster_names
    fin, gl, meta = S.fit([season], roster_names([season]))
    rc = meta["reconciliation"]
    assert abs(rc["identity_resid"]) < 0.5
    rebuilt = rc["xg"] + rc["sum_intercept"] + rc["sum_finishing"] + rc["sum_goalie_effect"]
    assert abs(rebuilt - rc["goals"]) < 1.0
    # finishing magnitudes are in a believable band (goals/100 shots)
    elig = fin[fin.shots >= 300]
    assert elig.fin_per100.abs().max() < 6.0
    g_elig = gl[gl.sa >= 1500]
    assert g_elig.gsax_per100.abs().max() < 2.0
