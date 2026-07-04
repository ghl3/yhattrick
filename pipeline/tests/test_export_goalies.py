"""Tests for the goalie rate derivations (yhattrick.export.export_goalies._rates).

The career box sums (saves, shots, GA, GSAx, TOI, danger/start splits) turn into the displayed rates;
these are simple ratios but feed the goalie page headline, so we pin them down."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yhattrick.export import export_goalies as Gx


def _career(**over):
    base = dict(
        player_id=1,
        saves=920,
        sog_against=1000,
        ga=80,
        toi_s=180000,
        gsax=10.0,
        hd_sog=200,
        hd_ga=40,
        qs=30,
        starts=50,
    )
    base.update(over)
    return pd.DataFrame([base])


def test_rates_save_pct_and_gaa():
    r = Gx._rates(_career()).iloc[0]
    assert r.sv_pct == pytest.approx(0.920)  # saves / shots-on-goal
    assert r.gaa == pytest.approx(80 * 3600.0 / 180000)  # GA per 60 (TOI 50 hrs -> 1.6)
    assert r.gaa == pytest.approx(1.6)


def test_rates_gsax60_and_hd_savepct_and_qspct():
    r = Gx._rates(_career()).iloc[0]
    assert r.gsax60 == pytest.approx(10.0 * 3600.0 / 180000)  # GSAx per 60
    assert r.hd_sv_pct == pytest.approx((200 - 40) / 200)  # high-danger save %
    assert r.qs_pct == pytest.approx(30 / 50)  # quality-start rate


def test_rates_zero_denominator_is_nan_not_error():
    r = Gx._rates(_career(sog_against=0, saves=0, toi_s=0, starts=0)).iloc[0]
    assert np.isnan(r.sv_pct) and np.isnan(r.gaa) and np.isnan(r.qs_pct)
