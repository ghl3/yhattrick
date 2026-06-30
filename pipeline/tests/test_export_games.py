"""Tests for the per-game JSON builder helpers (yhattrick.export.export_games).

`_sanitize` makes the timeline strict JSON (numpy scalars unwrapped, NaN/inf -> None); `_clock`
formats game-seconds back to a period clock; `_count_by` tallies a player-id series."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from yhattrick.export import export_games as G


def test_sanitize_unwraps_numpy_and_nulls_nonfinite():
    obj = {"i": np.int64(5), "f": np.float64(1.5), "b": np.bool_(True),
           "nan": float("nan"), "inf": float("inf"), "list": [np.int32(2), float("nan")]}
    out = G._sanitize(obj)
    assert out == {"i": 5, "f": 1.5, "b": True, "nan": None, "inf": None, "list": [2, None]}
    # round-trips through strict JSON
    assert json.loads(G._dump(obj))["list"] == [2, None]


def test_dump_is_strict_json():
    s = G._dump({"x": float("nan")})
    assert "NaN" not in s and '"x":null' in s


def test_clock_formats_period_and_remaining():
    assert G._clock(0) == "P1 00:00"
    assert G._clock(30) == "P1 00:30"
    assert G._clock(1200) == "P2 00:00"        # start of the 2nd period
    assert G._clock(3000) == "P3 10:00"        # 2*1200 + 600


def test_count_by_tallies_and_drops_na():
    s = pd.Series([10, 10, 11, None, 12, 10])
    assert G._count_by(s) == {10: 3, 11: 1, 12: 1}
