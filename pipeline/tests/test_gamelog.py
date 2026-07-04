"""Tests for the per-player game-log counts (yhattrick.data.gamelog).

`_counts` tallies goals / assists / shots-on-goal / penalties per (game, player) from the events
table, excluding the shootout (period ≥ 5) so it never inflates flow-play totals."""

from __future__ import annotations

import pandas as pd

from yhattrick.data import gamelog


def _events(rows):
    cols = [
        "nhl_game_id",
        "period",
        "type",
        "primary_player_id",
        "assist1_player_id",
        "assist2_player_id",
    ]
    return pd.DataFrame(rows, columns=cols)


def test_counts_goals_assists_sog_penalties():
    ev = _events(
        [
            (100, 1, "goal", 10, 11, 12),
            (100, 1, "shot-on-goal", 10, None, None),
            (100, 1, "missed-shot", 10, None, None),  # not a shot ON goal
            (100, 1, "penalty", 14, None, None),
            (100, 5, "goal", 10, None, None),  # shootout — excluded
        ]
    )
    out = gamelog._counts(ev).set_index(["nhl_game_id", "player_id"])
    r10 = out.loc[(100, 10)]
    assert r10.g == 1
    assert r10.sog == 2  # goal + shot-on-goal (missed shot is not SOG, shootout excluded)
    assert out.loc[(100, 11)].a1 == 1
    assert out.loc[(100, 12)].a2 == 1
    assert out.loc[(100, 14)].pen == 1


def test_counts_shootout_excluded_entirely():
    ev = _events([(100, 5, "goal", 10, None, None)])  # only a shootout goal
    out = gamelog._counts(ev)
    assert len(out) == 0  # nothing flows through


def test_counts_separate_games():
    ev = _events(
        [
            (100, 1, "goal", 10, None, None),
            (101, 1, "goal", 10, None, None),
        ]
    )
    out = gamelog._counts(ev).set_index(["nhl_game_id", "player_id"])
    assert out.loc[(100, 10)].g == 1 and out.loc[(101, 10)].g == 1  # kept per-game, not merged
