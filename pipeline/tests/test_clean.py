"""Tests for the raw->interim parse helpers (yhattrick.data.clean).

Covers the pure pieces: the period-elapsed clock conversion, the season decoder, the duplicate/
overlapping-shift merge (the fix that took illegal >6-skater stints from ~0.8% to ~0.01%), and the
roster name builder."""
from __future__ import annotations

import pytest

from yhattrick.data import clean


# --- game clock --------------------------------------------------------------------------------
def test_mmss_to_sec():
    assert clean._mmss_to_sec("00:00") == 0
    assert clean._mmss_to_sec("01:30") == 90
    assert clean._mmss_to_sec("20:00") == 1200


@pytest.mark.parametrize("period,mmss,expected", [
    (1, "00:00", 0),
    (1, "00:30", 30),
    (2, "00:00", 1200),       # start of P2 == one full period elapsed
    (3, "10:00", 3000),       # 2*1200 + 600
])
def test_game_sec(period, mmss, expected):
    assert clean.game_sec(period, mmss) == expected


def test_season_of():
    assert clean.season_of(2024020001) == 2024
    assert clean.season_of(2021030411) == 2021


# --- merging a player's overlapping / duplicate shift intervals --------------------------------
def test_merge_collapses_overlap():
    # the NHL "shift #21 and #22, same times" case + a genuine overlap
    rows = [(0, 10, 1), (5, 15, 1)]
    assert clean._merge_player_intervals(rows) == [(0, 15, 1)]


def test_merge_exact_duplicate():
    assert clean._merge_player_intervals([(58, 60, 1), (58, 60, 1)]) == [(58, 60, 1)]


def test_merge_touching_intervals_stay_separate():
    # next.start == current.end is NOT an overlap (half-open): two back-to-back shifts stay distinct
    assert clean._merge_player_intervals([(0, 10, 1), (10, 20, 1)]) == [(0, 10, 1), (10, 20, 1)]


def test_merge_unsorted_input_and_chain():
    # out-of-order, with a chain of three overlapping intervals collapsing to one
    rows = [(20, 30, 1), (0, 10, 1), (8, 25, 1)]
    assert clean._merge_player_intervals(rows) == [(0, 30, 1)]


def test_merge_disjoint_preserved():
    rows = [(0, 5, 1), (10, 15, 1)]
    assert clean._merge_player_intervals(rows) == [(0, 5, 1), (10, 15, 1)]


# --- roster name -------------------------------------------------------------------------------
def test_name_from_rosterspot():
    rs = {"firstName": {"default": "Sidney"}, "lastName": {"default": "Crosby"}}
    assert clean._name(rs) == "Sidney Crosby"


def test_name_missing_parts_are_tolerated():
    assert clean._name({"lastName": {"default": "Ovechkin"}}) == "Ovechkin"
    assert clean._name({}) == ""
