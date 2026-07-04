"""Tests for shot geometry + pre-shot context (yhattrick.data.shot_geom).

These pure functions orient raw NHL coordinates so the attacking net is always at +89, then derive
distance/angle and the rebound/rush flags. Both the shot table and the xG features call them, so
pinning them down keeps the two from drifting (the module's whole reason for existing)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yhattrick.data import shot_geom as G


# --- attack_sign: which way is the attacking net? ----------------------------------------------
def test_attack_sign_home_and_away():
    # home defends "right" -> home attacks left (-1); away attacks the opposite (+1)
    sign = G.attack_sign(is_home=[True, False], home_defending_side=["right", "right"])
    assert list(sign) == [-1.0, 1.0]
    # home defends "left" -> home attacks right (+1)
    sign = G.attack_sign(is_home=[True, False], home_defending_side=["left", "left"])
    assert list(sign) == [1.0, -1.0]


def test_oriented_flips_with_sign():
    x, y = G.oriented([50, 50], [10, 10], np.array([1.0, -1.0]))
    assert list(x) == [50.0, -50.0]
    assert list(y) == [10.0, -10.0]


# --- distance_angle: measured from the net at +89 ----------------------------------------------
def test_distance_angle_on_goal_line_and_straight_on():
    # straight on, 25 ft out (x_adj = 64, y = 0)
    dist, ang = G.distance_angle(np.array([64.0]), np.array([0.0]))
    assert dist[0] == pytest.approx(25.0)
    assert ang[0] == pytest.approx(0.0)


def test_distance_angle_right_on_the_goal_line():
    # level with the net (x_adj = 89), 10 ft to the side -> 10 ft away, 90° angle
    dist, ang = G.distance_angle(np.array([89.0]), np.array([10.0]))
    assert dist[0] == pytest.approx(10.0)
    assert ang[0] == pytest.approx(90.0)


def test_angle_is_absolute_value_symmetric():
    _, pos = G.distance_angle(np.array([70.0]), np.array([15.0]))
    _, neg = G.distance_angle(np.array([70.0]), np.array([-15.0]))
    assert pos[0] == pytest.approx(neg[0])  # |y| -> mirror shots share an angle


# --- rebound / rush flags ----------------------------------------------------------------------
def test_rebound_flag_only_after_a_recent_shot():
    prev = ["shot-on-goal", "shot-on-goal", "faceoff", "goal"]
    tsl = [2.0, 5.0, 1.0, 3.0]  # within 3s, too slow, not-a-shot, exactly 3s
    assert list(G.rebound_flag(prev, tsl)) == [1, 0, 0, 1]


def test_rush_flag_only_from_neutral_or_def_zone():
    prev_zone = ["N", "D", "O", "N"]
    tsl = [3.0, 4.0, 1.0, 5.0]  # ok, exactly 4s, wrong zone, too slow
    assert list(G.rush_flag(prev_zone, tsl)) == [1, 1, 0, 0]


# --- decode_situation: the 4-digit situationCode -----------------------------------------------
def test_decode_situation_even_strength_and_powerplay():
    out = G.decode_situation(pd.Series(["1551", "1541"]))
    assert list(out.away_goalie) == [1, 1]
    assert list(out.away_skaters) == [5, 5]
    assert list(out.home_skaters) == [5, 4]
    assert list(out.home_goalie) == [1, 1]


def test_decode_situation_pads_and_nulls_bad_codes():
    out = G.decode_situation(pd.Series(["551", None]))  # 3-char zfills to "0551"; None -> NaN row
    assert out.iloc[0].tolist() == [0, 5, 5, 1]
    assert out.iloc[1].isna().all()
