"""Tests for the expected-goals feature build + metric helpers (yhattrick.models.expected_goal_model).

`build_features` turns the raw event stream into one modeled-shot row per unblocked shot, orienting
geometry to the attacking net and dropping the situations xG must never score (empty net, <3 skaters,
shootout, blocked shots). The metric helpers are pure summaries of (y, p)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yhattrick.models import expected_goal_model as X


def _event(gid, idx, etype, x, y, sit, *, is_home=True, period=1,
           defside="left", pid=10, shot_type="wrist", t=100):
    """One pbp event row with every column build_features reads."""
    return {
        "nhl_game_id": gid, "event_idx": idx, "time_g": t, "period": period,
        "type": etype, "is_home": is_home, "x": x, "y": y, "zone": "O",
        "situation_code": sit, "home_defending_side": defside,
        "primary_player_id": pid, "shot_type": shot_type,
    }


def test_build_features_geometry_and_label():
    # home attacks +89 (defends "left"); a goal from (70,0) is 19 ft straight on
    ev = pd.DataFrame([_event(2024020001, 5, "goal", 70, 0, "1551")])
    out = X.build_features(ev, hand={})
    assert len(out) == 1
    r = out.iloc[0]
    assert r.distance == pytest.approx(19.0)
    assert r.abs_angle == pytest.approx(0.0)
    assert r.goal == 1
    assert r.strength_diff == 0          # 5 skaters a side
    assert r.shooter_id == 10


def test_build_features_drops_unmodeled_situations():
    rows = [
        _event(2024020001, 1, "goal", 70, 0, "1551"),          # KEEP: clean 5v5 goal
        _event(2024020002, 1, "shot-on-goal", 70, 0, "0551"),  # empty net (away goalie pulled) -> drop
        _event(2024020003, 1, "shot-on-goal", 70, 0, "1521"),  # home shooter on a 2-skater side -> drop
        _event(2024020004, 1, "shot-on-goal", 70, 0, "1551", period=5),  # shootout -> drop
        _event(2024020005, 1, "blocked-shot", 70, 0, "1551"),  # not a Fenwick shot -> drop
        _event(2024030001, 1, "goal", 70, 0, "1551"),          # playoffs -> drop
    ]
    out = X.build_features(pd.DataFrame(rows), hand={})
    assert list(out.nhl_game_id) == [2024020001]   # only the clean regular-season shot survives


def test_build_features_powerplay_strength_diff():
    # situationCode digits = away_g, away_sk, home_sk, home_g. "1451" = home 5 vs away 4 (home PP)
    ev = pd.DataFrame([_event(2024020001, 5, "shot-on-goal", 70, 0, "1451")])
    out = X.build_features(ev, hand={})
    assert out.iloc[0].strength_diff == 1        # home shooter (5) − away defenders (4)


def test_metrics_basic():
    y = np.array([0, 1, 1, 0])
    p = np.array([0.1, 0.9, 0.8, 0.2])
    m = X._metrics(y, p)
    assert m["total_goals"] == 2
    assert m["total_xg"] == pytest.approx(2.0)
    assert m["n"] == 4
    assert m["auc"] == 1.0               # perfectly separable
    assert 0 < m["logloss"] < 1


def test_reliability_bins_are_calibrated_on_perfect_input():
    # 100 shots in one probability band, observed rate matches predicted
    p = np.full(100, 0.10)
    y = np.array([1] * 10 + [0] * 90)    # 10% actually score
    rows = X._reliability(y, p)
    assert len(rows) == 1
    assert rows[0]["pred"] == pytest.approx(0.10)
    assert rows[0]["obs"] == pytest.approx(0.10)
    assert rows[0]["n"] == 100
