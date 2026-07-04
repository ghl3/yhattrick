"""Tests for the validation stage (yhattrick.validate).

Two layers: the pure result constructors / reporter in `core`, and the stage checks in `checks` run
against tiny synthetic artifacts — each check must PASS on clean data and FAIL (EXACT) or WARN
(APPROX) on a deliberate perturbation. That's what makes the validator trustworthy as a gate."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from yhattrick import config as C
from yhattrick.validate import core
from yhattrick.validate import checks


# --- core: result constructors + reporter ------------------------------------------------------
def test_exact_pass_and_fail():
    assert core.exact("x", "s", True).status == "pass"
    bad = core.exact("x", "s", False, observed=3, expected=0)
    assert bad.status == "fail" and bad.kind == "EXACT"


def test_approx_is_warn_only_outside_band():
    assert core.approx("x", "s", 0.5, 0.0, 1.0).status == "pass"
    out = core.approx("x", "s", 2.0, None, 1.0)
    assert out.status == "warn" and out.kind == "APPROX"  # APPROX never 'fail'


def test_json_clean_detects_leaks():
    assert core.json_clean('{"a":1.0,"b":null}')
    assert not core.json_clean('{"a":NaN}')
    assert not core.json_clean('{"a":Infinity}')


def test_report_counts_exact_failures():
    cs = [
        core.exact("a", "s", True),
        core.exact("b", "s", False),
        core.approx("c", "s", 9.0, None, 1.0),
        core.skip("d", "s", "EXACT", "n/a"),
    ]
    res = core.report(cs)
    assert res["tally"] == {"pass": 1, "warn": 1, "fail": 1, "skip": 1}
    assert res["exact_fail"] == 1


# --- stage checks on synthetic artifacts -------------------------------------------------------
def _write_stints(tmp_path, stints_rows, shots_rows):
    (tmp_path / "stints").mkdir(parents=True, exist_ok=True)
    (tmp_path / "shots_onice").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(stints_rows).to_parquet(tmp_path / "stints" / "2024.parquet")
    pd.DataFrame(shots_rows).to_parquet(tmp_path / "shots_onice" / "2024.parquet")


def _clean_stints():
    rows = [
        dict(
            nhl_game_id=1,
            stint_idx=0,
            start_g=0,
            end_g=100,
            duration_s=100,
            home_skaters=[1, 2, 3, 4, 5],
            away_skaters=[6, 7, 8, 9, 10],
            home_n=5,
            away_n=5,
            strength="5v5",
            home_xgf=0.6,
            away_xgf=0.0,
            overload=False,
        ),
        dict(
            nhl_game_id=1,
            stint_idx=1,
            start_g=100,
            end_g=200,
            duration_s=100,
            home_skaters=[1, 2, 3, 4, 5],
            away_skaters=[6, 7, 8, 9, 10],
            home_n=5,
            away_n=5,
            strength="5v5",
            home_xgf=0.4,
            away_xgf=0.2,
            overload=False,
        ),
    ]
    shots = [
        dict(
            nhl_game_id=1,
            stint_idx=0,
            xg=0.6,
            goal=1,
            onice_match="exact",
            event="goal",
            strength="5v5",
            sit_home_n=5,
            sit_away_n=5,
        ),
        dict(
            nhl_game_id=1,
            stint_idx=1,
            xg=0.4,
            goal=0,
            onice_match="exact",
            event="shot-on-goal",
            strength="5v5",
            sit_home_n=5,
            sit_away_n=5,
        ),
        dict(
            nhl_game_id=1,
            stint_idx=1,
            xg=0.2,
            goal=0,
            onice_match="within1",
            event="missed-shot",
            strength="5v5",
            sit_home_n=5,
            sit_away_n=5,
        ),
    ]
    return rows, shots


def _status(check_iter, name_contains):
    for c in check_iter:
        if name_contains in c.name:
            return c.status
    raise AssertionError(f"no check matching {name_contains!r}")


def test_check_stints_clean_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "PROCESSED", tmp_path)
    _write_stints(tmp_path, *_clean_stints())
    out = list(checks.check_stints([2024]))
    assert all(c.status != "fail" for c in out)  # no EXACT failures on clean data
    assert _status(out, "Σ stint xGF == Σ shot xg") == "pass"


def test_check_stints_catches_strength_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "PROCESSED", tmp_path)
    rows, shots = _clean_stints()
    rows[0]["strength"] = "9v9"  # contradicts home_n/away_n = 5
    _write_stints(tmp_path, rows, shots)
    out = list(checks.check_stints([2024]))
    assert _status(out, "strength == 'home_n v away_n'") == "fail"


def test_check_stints_catches_partition_gap(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "PROCESSED", tmp_path)
    rows, shots = _clean_stints()
    rows[1]["start_g"] = 150  # leaves a gap after stint 0 (ends at 100)
    _write_stints(tmp_path, rows, shots)
    out = list(checks.check_stints([2024]))
    assert _status(out, "stints partition the game") == "fail"


def test_check_stints_catches_xgf_leak(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "PROCESSED", tmp_path)
    rows, shots = _clean_stints()
    rows[0]["home_xgf"] = 5.0  # stint xGF no longer matches the shots
    _write_stints(tmp_path, rows, shots)
    out = list(checks.check_stints([2024]))
    assert _status(out, "Σ stint xGF == Σ shot xg") == "fail"


# --- export ledger -----------------------------------------------------------------------------
def _write_players(site_dir, rows):
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "players.json").write_text(json.dumps(rows))


def _player(**over):
    base = dict(
        id=1,
        g_created=8.0,
        g_fin=1.5,
        g_allowed=3.0,
        g_pen=0.0,
        g_net=6.5,
        scoring60=0.14,
        playmaking60=0.76,
        allow60=0.3,
        gnet_pg_pct=90.0,
    )
    base.update(over)
    return base


def test_check_export_clean_ledger_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "SITE_JSON", tmp_path)
    monkeypatch.setattr(C, "MODELS", tmp_path / "nomodels")
    _write_players(tmp_path, [_player()])
    out = list(checks.check_export([1999]))  # season with no model files -> finishing skip
    assert _status(out, "g_net == created + fin − allowed + pen") == "pass"
    assert _status(out, "all percentiles ∈ [0,100]") == "pass"


def test_check_export_catches_broken_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "SITE_JSON", tmp_path)
    monkeypatch.setattr(C, "MODELS", tmp_path / "nomodels")
    _write_players(tmp_path, [_player(g_net=99.0)])  # net no longer equals the components
    out = list(checks.check_export([1999]))
    assert _status(out, "g_net == created + fin − allowed + pen") == "fail"


# --- RAPM creation/suppression reconciliation --------------------------------------------------
def _write_ev_and_stints(tmp_models, tmp_proc, ev_rows, stint_rows):
    tmp_models.mkdir(parents=True, exist_ok=True)
    (tmp_proc / "stints").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ev_rows).to_parquet(tmp_models / "ev_2024.parquet")
    pd.DataFrame(stint_rows).to_parquet(tmp_proc / "stints" / "2024.parquet")


def _ev_row(**over):
    base = dict(
        player_id=1, ev_off=0.0, ev_def=0.0, ev_off_base=2.5, ev_off_toi=600.0, ev_def_toi=600.0
    )  # toi in minutes
    base.update(over)
    return base


def _xgf_stint(hxgf, axgf, **over):
    base = dict(
        nhl_game_id=2024020001,
        strength="5v5",
        overload=False,
        duration_s=100,
        home_xgf=hxgf,
        away_xgf=axgf,
    )
    base.update(over)
    return base


def test_ev_creation_reconciles_when_consistent(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "MODELS", tmp_path / "models")
    monkeypatch.setattr(C, "PROCESSED", tmp_path / "proc")
    # created = (2.5/5+0)*(600/60) = 5.0 ; target 5v5 xGF = 3.0+2.0 = 5.0 -> ratio 1.0
    _write_ev_and_stints(
        tmp_path / "models", tmp_path / "proc", [_ev_row()], [_xgf_stint(3.0, 2.0)]
    )
    out = list(checks._ev_creation_reconciliation([2024], "2024", "models"))
    assert _status(out, "Σ created shares ≈ league 5v5 ΣxGF") == "pass"
    assert _status(out, "Σ allowed shares ≈ league 5v5 ΣxGF") == "pass"


def test_ev_creation_warns_when_toi_accounting_breaks(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "MODELS", tmp_path / "models")
    monkeypatch.setattr(C, "PROCESSED", tmp_path / "proc")
    # 10x the TOI but the same xGF target -> created badly overshoots -> APPROX warn
    _write_ev_and_stints(
        tmp_path / "models", tmp_path / "proc", [_ev_row(ev_off_toi=6000.0)], [_xgf_stint(3.0, 2.0)]
    )
    out = list(checks._ev_creation_reconciliation([2024], "2024", "models"))
    assert _status(out, "Σ created shares ≈ league 5v5 ΣxGF") == "warn"


def test_league_5v5_xgf_filters_the_model_stint_universe(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "PROCESSED", tmp_path)
    (tmp_path / "stints").mkdir(parents=True)
    rows = [
        _xgf_stint(1.0, 0.0),  # counted
        _xgf_stint(9.0, 9.0, strength="5v4"),  # special teams -> excluded
        _xgf_stint(9.0, 9.0, overload=True),  # overloaded -> excluded
        _xgf_stint(9.0, 9.0, duration_s=5),  # sub-10s sliver -> excluded
        _xgf_stint(9.0, 9.0, nhl_game_id=2024030001),  # playoffs -> excluded
    ]
    pd.DataFrame(rows).to_parquet(tmp_path / "stints" / "2024.parquet")
    assert checks._league_5v5_xgf([2024]) == 1.0  # only the clean 5v5 stint survives


def test_check_export_flags_nan_leak(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "SITE_JSON", tmp_path)
    monkeypatch.setattr(C, "MODELS", tmp_path / "nomodels")
    (tmp_path).mkdir(parents=True, exist_ok=True)
    (tmp_path / "players.json").write_text('[{"id":1,"g_net":NaN}]')  # a NaN token leaked
    out = list(checks.check_export([1999]))
    assert _status(out, "players.json has no NaN/Infinity") == "fail"
