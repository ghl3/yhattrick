"""Tests for the isolated-impact (RAPM) model (yhattrick.models.player_onice_model).

`build_design` turns stints into the penalized-regression design (off/def indicators per on-ice
player + context covariates, response = xG/60); `_solve_pen` is the penalized normal-equation solve;
`fit` ties them together and should recover a planted offensive signal while the free intercept
absorbs the league baseline."""
from __future__ import annotations

import numpy as np
import pandas as pd

from yhattrick.models import player_onice_model as M


def _stint(gid, home, away, hxgf, axgf, dur=1200, season=2024):
    return {
        "nhl_game_id": gid, "season": season, "start_g": 0, "duration_s": dur,
        "home_skaters": home, "away_skaters": away, "home_xgf": hxgf, "away_xgf": axgf,
        "home_n": len(home), "away_n": len(away),
        "start_type": "fly", "start_zone": None, "home_lead": 0,
    }


# --- build_design: the design-matrix contract --------------------------------------------------
def test_build_design_dual_rows_and_response():
    st = pd.DataFrame([_stint(1, [1, 2], [3, 4], hxgf=1.0, axgf=0.5)])
    X, y, w, players, games, off_toi, def_toi, cov = M.build_design(st, dual=True)

    assert players == [1, 2, 3, 4]
    assert X.shape[0] == 2                       # EV emits both attacking perspectives
    # response is xG per 60 min: 1.0 over a 1200s (1/3 hr) stint -> 3.0; away 0.5 -> 1.5
    assert np.allclose(y, [3.0, 1.5])
    assert list(w) == [1200, 1200]               # raw weight = stint seconds (fit normalizes later)
    assert list(games) == [1, 1]

    P = len(players)
    idx = {p: i for i, p in enumerate(players)}
    D = X.toarray()
    # row 0: home {1,2} attack, away {3,4} defend, + the HOME-ice covariate
    assert D[0, idx[1]] == 1 and D[0, idx[2]] == 1
    assert D[0, P + idx[3]] == 1 and D[0, P + idx[4]] == 1
    assert D[0, cov["home"]] == 1
    # row 1: away {3,4} attack, home {1,2} defend, NOT home ice
    assert D[1, idx[3]] == 1 and D[1, idx[4]] == 1
    assert D[1, P + idx[1]] == 1 and D[1, P + idx[2]] == 1
    assert D[1, cov["home"]] == 0


def test_build_design_accumulates_role_toi():
    st = pd.DataFrame([_stint(1, [1, 2], [3, 4], 1.0, 0.5, dur=600)])
    *_, off_toi, def_toi, _ = M.build_design(st, dual=True)
    # each player is on offence once and defence once (the dual rows), 600s each
    assert off_toi == {1: 600, 2: 600, 3: 600, 4: 600}
    assert def_toi == {1: 600, 2: 600, 3: 600, 4: 600}


def test_build_design_special_teams_only_powerplay_attacks():
    # 5v4: only the team with more skaters attacks (one row, no dual)
    st = pd.DataFrame([_stint(1, [1, 2, 3], [4, 5], hxgf=2.0, axgf=0.0)])
    X, y, *_ = M.build_design(st, dual=False)
    assert X.shape[0] == 1
    assert np.allclose(y, [6.0])                  # 2.0 over 1200s -> per-60 = 6.0


# --- _solve_pen: penalized normal equations ----------------------------------------------------
def test_solve_pen_unpenalized_is_exact():
    B = np.array([[1.0, 0.0], [0.0, 1.0]])
    c = np.array([2.0, 4.0])
    beta, A, _ = M._solve_pen(B, c, pen=np.zeros(2))
    assert np.allclose(beta, [2.0, 4.0])


def test_solve_pen_shrinks_toward_zero():
    B = np.array([[1.0, 0.0], [0.0, 1.0]])
    c = np.array([2.0, 4.0])
    beta, _, _ = M._solve_pen(B, c, pen=np.array([9.0, 9.0]))   # (1+9)β = c -> β = c/10
    assert np.allclose(beta, [0.2, 0.4])
    assert np.all(np.abs(beta) < np.abs([2.0, 4.0]))            # strictly smaller than unpenalized


# --- fit: recover a planted offensive signal, intercept absorbs the baseline --------------------
def _synthetic_season(rng, n_games=30, stints_per_game=5, mu0=2.5, boost=1.5):
    """Six players shuffled into 3-on-3 stints; player 1 adds `boost` xG/60 to whichever side he's on.
    The fit should rank him the top offensive player and leave the rest near zero."""
    rows = []
    for g in range(n_games):
        for s in range(stints_per_game):
            order = list(rng.permutation([1, 2, 3, 4, 5, 6]))
            home, away = sorted(order[:3]), sorted(order[3:])
            hr = mu0 + (boost if 1 in home else 0.0)        # home xG/60
            ar = mu0 + (boost if 1 in away else 0.0)        # away xG/60
            dur = 1200
            rows.append(_stint(2024020000 + g, home, away,
                               hxgf=hr * dur / 3600.0, axgf=ar * dur / 3600.0, dur=dur))
    return pd.DataFrame(rows)


def test_fit_recovers_top_offensive_player_and_baseline():
    rng = np.random.default_rng(0)
    st = _synthetic_season(rng)
    names = {p: {"name": f"P{p}", "pos": "C"} for p in range(1, 7)}
    coef, meta = M.fit(st, names, M.SPECS["ev"])

    ev = coef.set_index("player_id")
    assert ev.ev_off.idxmax() == 1                      # planted offensive star is identified
    assert ev.loc[1].ev_off > 0
    # the other five hover near zero (well below the star)
    others = ev.drop(index=1).ev_off.abs().max()
    assert ev.loc[1].ev_off > others
    # the free intercept absorbs the league baseline (≈ mu0 + half the star's spread)
    assert 2.0 < meta["baseline_xgf60"] < 4.0
    assert meta["n_players"] == 6
