"""Tests for the player value synthesis (yhattrick.export.export_players).

`value_table` turns the model coefficients into goals-attributed: absolute created/allowed shares,
the scoring/playmaking partition, finishing, penalties, and the net. The invariants we lock down:
the ledger (g_net = created + fin − allowed + pen), the positive partition (scoring + playmaking =
offense + finishing), and the 5v5 baseline cancellation in the net. Plus JSON sanitization and the
penalty-value derivation."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from yhattrick.export import export_players as E
from yhattrick import config as C


def _pooled(**over):
    base = dict(
        player_id=1,
        ev_off=0.3,
        ev_def=-0.2,
        ev_off_toi=600.0,
        pp_off=0.0,
        pk_def=0.0,
        pp_off_toi=0.0,
        pk_def_toi=0.0,
        ev_off_base=2.5,
        pp_off_base=6.0,
        fin_per100=2.0,
        fin_goals=1.5,
        gp=10,
        toi_all=120.0,
        pen_drawn60=0.0,
        pen_taken60=0.0,
        ev_xgf60=3.0,
        pp_xgf60=8.0,
    )
    base.update(over)
    return pd.DataFrame([base])


def _shots(**over):
    base = dict(player_id=1, shots5=50.0, shotspp=0.0, ixg5=1.5, ixgpp=0.0)
    base.update(over)
    return pd.DataFrame([base])


def test_value_ledger_g_net():
    out = E.value_table(_pooled(), _shots(), pen_v=0.14).iloc[0]
    # created = (2.5/5+0.3)*10 = 8.0 ; allowed = (2.5/5−0.2)*10 = 3.0 ; fin = 1.5 ; pen = 0
    assert out.g_created == pytest.approx(8.0)
    assert out.g_allowed == pytest.approx(3.0)
    assert out.g_fin == pytest.approx(1.5)
    assert out.g_net == pytest.approx(out.g_created + out.g_fin - out.g_allowed + out.g_pen)
    assert out.g_net == pytest.approx(6.5)


def test_scoring_plus_playmaking_equals_offense_plus_finishing():
    out = E.value_table(_pooled(), _shots(), pen_v=0.14).iloc[0]
    create60 = 2.5 / 5 + 0.3  # 0.8
    fin5_60 = (2.0 / 100) * 50 / (600 / 60)  # alpha·shots5 / blocks5 = 0.1
    assert out.scoring60 + out.playmaking60 == pytest.approx(create60 + fin5_60)
    assert out.scoring60 >= 0 and out.playmaking60 >= 0  # positive partition


def test_phi_is_clipped_to_unit_interval():
    # huge own ixG vs tiny on-ice xGF would push φ>1; it must clip so playmaking stays ≥ 0
    out = E.value_table(_pooled(ev_xgf60=0.1), _shots(ixg5=99.0), pen_v=0.14).iloc[0]
    assert out.playmaking60 >= 0


def test_5v5_baseline_cancels_in_the_net():
    # with no PP/PK/penalties/finishing, the net is the pure marginal differential (ev_off − ev_def)·blocks5
    out = E.value_table(_pooled(fin_goals=0.0), _shots(), pen_v=0.14).iloc[0]
    blocks5 = 600 / 60
    assert out.g_net == pytest.approx((0.3 - (-0.2)) * blocks5)


def test_dump_sanitizes_nan_and_numpy():
    s = E._dump({"a": float("nan"), "b": np.int64(3), "c": np.float64(1.5), "d": [np.bool_(True)]})
    assert '"a":null' in s
    assert '"b":3' in s and '"c":1.5' in s and '"d":[true]' in s
    assert "NaN" not in s  # strict JSON, no bare NaN token


def test_penalty_value_from_data(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "PROCESSED", tmp_path)
    d = tmp_path / "shots_onice"
    d.mkdir()
    # two power-play goals (home 5v4 shooter), no shorthanded goals
    shots = pd.DataFrame(
        {
            "is_home": [1, 1, 0],
            "strength": ["5v4", "5v4", "5v5"],
            "goal": [1, 1, 1],
        }
    )
    shots.to_parquet(d / "2024.parquet")
    allbox = pd.DataFrame({"pen_drawn": [6, 4]})  # 10 drawn penalties league-wide
    v = E.penalty_value([2024], allbox)
    assert v == pytest.approx((2 - 0) / 10)  # (PP GF − SH GF) / drawn = 0.2


def test_career_totals_primary_assists_and_faceoffs():
    box = pd.DataFrame(
        [
            dict(
                player_id=1,
                toi_s=600.0,
                g=2,
                a1=3,
                a2=1,
                pen_drawn=1,
                pen_taken=0,
                fo_won=10,
                fo_lost=6,
            ),
            dict(
                player_id=1,
                toi_s=600.0,
                g=0,
                a1=1,
                a2=2,
                pen_drawn=0,
                pen_taken=1,
                fo_won=4,
                fo_lost=4,
            ),
        ]
    )
    out = E.career_totals(box).set_index("player_id").loc[1]
    assert out.c_a1 == 4  # primary assists summed (3 + 1)
    assert out.c_a == 7  # total assists (a1 + a2)
    assert out.c_fo == 24  # faceoffs taken (10 + 6 + 4 + 4)
    assert out.fo_won == 14


def test_individual_rates_primary_assists_and_faceoff_win():
    career = pd.DataFrame(
        [
            dict(
                player_id=1,
                toi_all=60.0,
                c_g=0.0,
                c_a=0.0,
                c_a1=2.0,
                c_pd=0.0,
                c_pt=0.0,
                fo_won=11.0,
                c_fo=20.0,
            )
        ]
    )
    fin = pd.DataFrame(
        [dict(player_id=1, shots=0.0, ixg=0.0, fin_per100=0.0, fin_per100_se=0.0, fin_goals=0.0)]
    )
    out = E.individual_table(fin, career).set_index("player_id").loc[1]
    assert out.a1_60 == pytest.approx(2.0)  # 2 primary assists in 60 min -> 2 / 60min
    assert out.fo_win == pytest.approx(11 / 20)  # 11 of 20 draws won


def test_onice_zone_start_share():
    row = {c: 0.0 for c in E.ONICE_COLS}
    row["ev_oz_starts"], row["ev_dz_starts"] = 6.0, 4.0
    out = E.onice_table(pd.DataFrame([{"player_id": 1, **row}])).set_index("player_id").loc[1]
    assert out.n_zs == pytest.approx(10.0)  # O + D faceoff starts
    assert out.ozs == pytest.approx(0.6)  # O / (O + D)


# --- current team (trades) ----------------------------------------------------
def test_current_team_is_most_recent_game_not_season_majority():
    """A deadline trade: 60 games for the old club, then games for the new one. The game log
    arrives most-recent-first, so the newest game's team wins regardless of the majority."""
    glog = [{"team": "NEW"}] + [{"team": "OLD"}] * 60
    assert E.current_team_of(glog, fallback="OLD") == "NEW"


def test_current_team_falls_back_without_game_log():
    assert E.current_team_of([], fallback="TOR") == "TOR"
    assert E.current_team_of([], fallback="") == ""


def test_team_season_splits_partitions_a_trade():
    """A deadline trade in 2025 plus a full 2024: each (team, season) cell holds only the games
    played for that team in that season, so the team-season roster pages read one cell."""
    glog = (
        [{"team": "NEW", "season": 2025, "g": 1, "a": 0, "p": 1, "toi_s": 900}] * 2
        + [{"team": "OLD", "season": 2025, "g": 0, "a": 2, "p": 2, "toi_s": 1000}] * 3
        + [{"team": "OLD", "season": 2024, "g": 1, "a": 1, "p": 2, "toi_s": 1100}] * 4
    )
    out = E.team_season_splits(glog)
    assert out["NEW"] == {"2025": {"gp": 2, "g": 2, "a": 0, "p": 2, "toi_s": 1800}}
    assert out["OLD"]["2025"] == {"gp": 3, "g": 0, "a": 6, "p": 6, "toi_s": 3000}
    assert out["OLD"]["2024"] == {"gp": 4, "g": 4, "a": 4, "p": 8, "toi_s": 4400}
    assert E.team_season_splits([]) == {}
