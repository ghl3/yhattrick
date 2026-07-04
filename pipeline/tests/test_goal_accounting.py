"""Tests for the additive reconciliation arithmetic (yhattrick.models.goal_accounting).

`decompose` attaches the per-shot identity goal ≈ xg + μ + finishing + goalie. It's the exact
arithmetic the league reconciliation and the validator both rely on, so we check it directly here
(the real fit's reconciliation is data-gated in test_shooting_model)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yhattrick.models.goal_accounting import decompose


def _shots():
    return pd.DataFrame(
        {
            "shooter_id": [1, 1, 2, 3],
            "goalie_id": [10, 11, 10, 11],
            "xg": [0.10, 0.20, 0.30, 0.05],
            "goal": [0, 1, 0, 0],
        }
    )


def test_decompose_is_elementwise_additive():
    alpha = {1: 0.05, 2: -0.02, 3: 0.0}
    gamma = {10: -0.01, 11: 0.02}
    mu = -0.001
    s = decompose(_shots(), mu, alpha, gamma)
    assert np.allclose(s.fin, [0.05, 0.05, -0.02, 0.0])
    assert np.allclose(s.gol, [-0.01, 0.02, -0.01, 0.02])
    assert np.allclose(s.recon, s.xg + s.mu + s.fin + s.gol)  # recon is exactly the sum of pieces


def test_decompose_does_not_mutate_input():
    shots = _shots()
    decompose(shots, 0.0, {1: 0.0, 2: 0.0, 3: 0.0}, {10: 0.0, 11: 0.0})
    assert "recon" not in shots.columns  # operates on a copy


def test_free_intercept_closes_the_league_identity():
    # the shooting model's free intercept forces Σ(goal − xg − fin − goalie − μ) = 0; emulate that by
    # setting μ to the mean residual, and the reconstruction then sums to actual goals exactly.
    shots = _shots()
    alpha = {1: 0.05, 2: -0.02, 3: 0.01}
    gamma = {10: -0.01, 11: 0.02}
    fin = shots.shooter_id.map(alpha)
    gol = shots.goalie_id.map(gamma)
    mu = float((shots.goal - shots.xg - fin - gol).mean())
    s = decompose(shots, mu, alpha, gamma)
    assert s.recon.sum() == pytest.approx(s.goal.sum())  # books balance to the penny


def test_team_rollup_components_sum_to_reconstruction():
    # a team's reconstructed goals = Σxg + Σμ + Σfin + Σgoalie over its shots (what reconcile() rolls up)
    s = decompose(_shots(), -0.001, {1: 0.05, 2: -0.02, 3: 0.0}, {10: -0.01, 11: 0.02})
    assert s.recon.sum() == pytest.approx(s.xg.sum() + s.mu.sum() + s.fin.sum() + s.gol.sum())
