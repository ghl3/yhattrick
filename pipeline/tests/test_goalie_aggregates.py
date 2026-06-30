"""Tests for the goalie split helpers (yhattrick.data.goalie_aggregates).

`_split_wide` pivots a goalie's Fenwick SA / xGA / GA (+ SOG) into per-level columns; `_shottype_long`
does the same in long form. The split totals must reconcile to the goalie's overall totals so the
danger / situation / shot-type GSAx all roll up to the headline GSAx."""
from __future__ import annotations

import pandas as pd

from yhattrick.data import goalie_aggregates as GA


def _facing():
    # one goalie (1) facing four shots across danger buckets; goalie 2 faces one
    return pd.DataFrame({
        "goalie_id": [1, 1, 1, 1, 2],
        "bucket": ["ld", "md", "hd", "hd", "ld"],
        "event": ["shot-on-goal", "missed-shot", "goal", "shot-on-goal", "shot-on-goal"],
        "xg": [0.02, 0.08, 0.20, 0.18, 0.03],
        "goal": [0, 0, 1, 0, 0],
        "is_sog": [True, False, True, True, True],
        "shot_type": ["wrist", "snap", "slap", "wrist", "wrist"],
    })


def test_split_wide_levels_and_reconciliation():
    out = GA._split_wide(_facing(), "bucket", GA.DANGER).set_index("player_id")
    g1 = out.loc[1]
    # Fenwick SA splits sum to the goalie's total shots faced
    assert g1.ld_sa + g1.md_sa + g1.hd_sa == 4
    # high-danger: 2 shots faced, 1 goal, 1 missed-shot is NOT a SOG -> sog counts only the on-goal ones
    assert g1.hd_sa == 2 and g1.hd_ga == 1 and g1.hd_sog == 2
    assert g1.md_sog == 0                       # the only md shot was a missed-shot
    # xGA splits sum to total xGA faced
    assert round(g1.ld_xga + g1.md_xga + g1.hd_xga, 4) == round(0.02 + 0.08 + 0.20 + 0.18, 4)


def test_split_wide_situation_levels():
    df = _facing().assign(sit=["ev", "ev", "pp", "pk", "ev"])
    out = GA._split_wide(df, "sit", GA.SITUATIONS).set_index("player_id")
    g1 = out.loc[1]
    assert g1.ev_sa + g1.pp_sa + g1.pk_sa == 4
    assert g1.pp_ga == 1                          # the goal was filed under the pp split


def test_shottype_long_sums_to_total():
    long = GA._shottype_long(_facing(), season=2024)
    g1 = long[long.player_id == 1]
    assert g1.sa.sum() == 4                       # every shot-type row for goalie 1 sums to his SA
    assert g1.ga.sum() == 1
    assert set(long.columns) == {"player_id", "season", "shot_type", "sa", "xga", "ga", "sog"}
    assert (long.season == 2024).all()
