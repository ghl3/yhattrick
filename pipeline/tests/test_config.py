"""Tests for config helpers: season labels, the NHL 8-digit season id, and game-id decoding."""
from __future__ import annotations

import pytest

from yhattrick import config as C


def test_season_label():
    assert C.season_label(2024) == "2024-25"
    assert C.season_label(2021) == "2021-22"
    assert C.season_label(1999) == "1999-00"      # century rollover keeps the 2-digit suffix


def test_nhl_season8():
    assert C.nhl_season8(2024) == "20242025"
    assert C.nhl_season8(2021) == "20212022"


@pytest.mark.parametrize("gid,expected", [
    (2024020001, True),    # ...02... regular season
    (2024020500, True),
    (2024030001, False),   # ...03... playoffs
    (2024010001, False),   # ...01... preseason
])
def test_is_regular_season(gid, expected):
    assert C.is_regular_season(gid) is expected


def test_game6():
    assert C.game6(2024020500) == "020500"   # drop the 4-digit season prefix
    assert C.game6(2021020001) == "020001"
    assert len(C.game6(2024030415)) == 6
