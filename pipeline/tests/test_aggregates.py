"""Tests for the descriptive box score (yhattrick.data.aggregates).

`event_counts` tallies per-player counting stats from raw pbp JSON (shootout kept separate);
`ice_and_team` derives GP / TOI / primary team from the shift table. Both feed the player page."""
from __future__ import annotations

import json

import pandas as pd

from yhattrick.data import aggregates as A
from yhattrick import config as C


def _write_pbp(tmp_path, gid, plays):
    (tmp_path / f"{gid}.json").write_text(json.dumps({"plays": plays}))


def _play(t, details, period_type="REG"):
    return {"typeDescKey": t, "periodDescriptor": {"periodType": period_type}, "details": details}


def test_event_counts_tallies_and_points(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "RAW_PBP", tmp_path)
    plays = [
        _play("goal", {"scoringPlayerId": 10, "assist1PlayerId": 11, "assist2PlayerId": 12}),
        _play("shot-on-goal", {"shootingPlayerId": 10}),
        _play("missed-shot", {"shootingPlayerId": 10}),
        _play("blocked-shot", {"blockingPlayerId": 20, "shootingPlayerId": 13}),
        _play("hit", {"hittingPlayerId": 14}),
        _play("faceoff", {"winningPlayerId": 10, "losingPlayerId": 20}),
        _play("penalty", {"committedByPlayerId": 14, "drawnByPlayerId": 11}),
        _play("giveaway", {"playerId": 12}),
        _play("takeaway", {"playerId": 13}),
        _play("goal", {"scoringPlayerId": 10}, period_type="SO"),   # shootout — separate bucket
    ]
    _write_pbp(tmp_path, 2024020001, plays)
    box = A.event_counts([2024020001]).set_index("player_id")

    p10 = box.loc[10]
    assert p10.g == 1 and p10.sog == 2 and p10.icf == 3   # goal: g+sog+icf; sog: sog+icf; missed: icf
    assert p10.fo_won == 1
    assert p10.so_g == 1 and p10.so_att == 1              # shootout goal never mixes into flow `g`
    assert p10.points == 1                                # g + a1 + a2

    assert box.loc[11].a1 == 1 and box.loc[11].pen_drawn == 1 and box.loc[11].points == 1
    assert box.loc[12].a2 == 1 and box.loc[12].giveaways == 1
    assert box.loc[13].icf == 1 and box.loc[13].takeaways == 1   # blocked-shot shooter gets a Corsi-for
    assert box.loc[20].blocks == 1 and box.loc[20].fo_lost == 1
    assert box.loc[14].hits == 1 and box.loc[14].pen_taken == 1


def test_event_counts_points_is_goals_plus_both_assists(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "RAW_PBP", tmp_path)
    _write_pbp(tmp_path, 2024020002, [
        _play("goal", {"scoringPlayerId": 1, "assist1PlayerId": 2, "assist2PlayerId": 3}),
        _play("goal", {"scoringPlayerId": 2, "assist1PlayerId": 1}),
    ])
    box = A.event_counts([2024020002]).set_index("player_id")
    assert box.loc[1].points == box.loc[1].g + box.loc[1].a1 + box.loc[1].a2   # 1 goal + 1 a1 = 2
    assert box.loc[1].points == 2


def test_ice_and_team_gp_toi_and_primary():
    # player 1 plays 3 games (2 for AAA, 1 for BBB); player 2 a single game
    sh = pd.DataFrame({
        "player_id": [1, 1, 1, 2],
        "nhl_game_id": [100, 101, 102, 100],
        "team": ["AAA", "AAA", "BBB", "AAA"],
        "duration_s": [600, 600, 100, 300],
    })
    out = A.ice_and_team(sh).set_index("player_id")
    assert out.loc[1].gp == 3
    assert out.loc[1].toi_s == 1300
    assert out.loc[1].team == "AAA"                  # primary = most games
    assert set(out.loc[1].teams) == {"AAA", "BBB"}
    assert out.loc[2].gp == 1 and out.loc[2].toi_s == 300
