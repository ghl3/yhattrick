"""Tests for the one-shot in-season updater (yhattrick.update).

The updater is stateless: the plan comes from the NHL schedule plus what is on disk. These
tests exercise the planning (fetch plan, staleness, season resolution), the stage runner's
fail-fast behavior, the lock, and the main() short-circuits — with all downloads and stage
bodies faked."""

from __future__ import annotations

import datetime as dt
import json
import os

import pytest

from yhattrick import config as C
from yhattrick import update as U
from yhattrick.data import download as D


# --- season resolution --------------------------------------------------------
def test_resolve_season_defaults_to_current(monkeypatch):
    assert U.resolve_season(None, today=dt.date(2026, 1, 15)) == 2025  # in config.SEASONS
    assert U.resolve_season(2024) == 2024


def test_resolve_season_rejects_unconfigured_season():
    with pytest.raises(SystemExit, match="config.SEASONS"):
        U.resolve_season(2030)


# --- fetch plan ---------------------------------------------------------------
def test_fetch_plan_classifies_missing_stale_and_pending(tmp_path, monkeypatch):
    (tmp_path / "pbp").mkdir()
    (tmp_path / "sc").mkdir()
    monkeypatch.setattr(C, "RAW_PBP", tmp_path / "pbp")
    monkeypatch.setattr(C, "RAW_SHIFTS", tmp_path / "sc")
    good, stale, missing, noshift = 2025020001, 2025020002, 2025020003, 2025020004
    future = 2025020900
    (tmp_path / "pbp" / f"{good}.json").write_text(json.dumps({"plays": [{"e": 1}]}))
    (tmp_path / "pbp" / f"{stale}.json").write_text(json.dumps({"plays": []}))
    (tmp_path / "pbp" / f"{noshift}.json").write_text(json.dumps({"plays": [{"e": 1}]}))
    for g in (good, stale):
        (tmp_path / "sc" / f"{g}.json").write_text('{"data":[]}')
    sched = [
        {"id": g, "date": "d", "state": s}
        for g, s in (
            (good, "OFF"),
            (stale, "OFF"),
            (missing, "FINAL"),
            (noshift, "OFF"),
            (future, "FUT"),
        )
    ]
    plan = U.fetch_plan(2025, sched)
    assert plan["missing"] == [missing, noshift]  # no pbp at all, and pbp without a shiftchart
    assert plan["stale"] == [stale]
    assert plan["pending"] == 1
    assert plan["states"] == {"OFF": 3, "FINAL": 1, "FUT": 1}


# --- interim staleness --------------------------------------------------------
_INTERIM_TABLES = ("shifts", "events", "roster", "shots")


def test_interim_stale_tracks_raw_vs_parquet(tmp_path, monkeypatch):
    raw_pbp = tmp_path / "pbp"
    raw_sc = tmp_path / "sc"
    raw_html = tmp_path / "html"
    interim = tmp_path / "interim"
    for d in (raw_pbp, raw_sc, raw_html):
        d.mkdir(parents=True)
    for t in _INTERIM_TABLES:
        (interim / t).mkdir(parents=True)
    monkeypatch.setattr(C, "RAW_PBP", raw_pbp)
    monkeypatch.setattr(C, "RAW_SHIFTS", raw_sc)
    monkeypatch.setattr(C, "RAW_HTMLSHIFTS", raw_html)
    monkeypatch.setattr(C, "INTERIM", interim)

    assert not U.interim_stale(2025)  # no raw data at all

    pbp = raw_pbp / "2025020001.json"
    pbp.write_text("{}")
    assert U.interim_stale(2025)  # raw exists, parquets missing

    outs = [interim / t / "2025.parquet" for t in _INTERIM_TABLES]
    for out in outs:
        out.write_text("parquet")
        os.utime(out, (2_000, 2_000))
    os.utime(pbp, (1_000, 1_000))
    assert not U.interim_stale(2025)  # every table newer than raw

    os.utime(outs[1], (500, 500))  # clean died mid-run: events older than the raw fetch
    assert U.interim_stale(2025)

    os.utime(outs[1], (2_000, 2_000))
    os.utime(pbp, (3_000, 3_000))
    assert U.interim_stale(2025)  # raw newer -> a crashed run left work behind


# --- env loading --------------------------------------------------------------
def test_load_env_file_sets_missing_keys_only(tmp_path, monkeypatch):
    envfile = tmp_path / ".env"
    envfile.write_text('# comment\nR2_BUCKET="b"\nR2_ENDPOINT=e\nPATH=clobbered\n\nbad line\n')
    monkeypatch.delenv("R2_BUCKET", raising=False)
    monkeypatch.delenv("R2_ENDPOINT", raising=False)
    n = U.load_env_file(envfile)
    assert os.environ["R2_BUCKET"] == "b" and os.environ["R2_ENDPOINT"] == "e"
    assert os.environ["PATH"] != "clobbered"  # existing environment wins
    assert n == 2
    monkeypatch.delenv("R2_BUCKET")
    monkeypatch.delenv("R2_ENDPOINT")


def test_load_env_file_missing_is_noop(tmp_path):
    assert U.load_env_file(tmp_path / "nope.env") == 0


# --- stage plumbing -----------------------------------------------------------
def test_build_stages_order_and_toggles():
    names = [n for n, _ in U.build_stages(2025, refit_xg=False, publish=True, deploy=True)]
    assert names == [
        "clean",
        "handedness",
        "xg-score",
        "stints",
        "box",
        "model",
        "shooting",
        "goal-accounting",
        "games",
        "gamelog",
        "players",
        "goalie-gamelog",
        "goalie-box",
        "goalies",
        "validate",
        "publish",
        "deploy",
    ]
    lean = [n for n, _ in U.build_stages(2025, refit_xg=True, publish=False, deploy=False)]
    assert lean[2] == "xg-refit" and "publish" not in lean and "deploy" not in lean


def test_run_stages_stops_at_first_failure(capsys):
    ran = []

    def ok(name):
        return lambda: ran.append(name)

    def boom():
        raise RuntimeError("kaput")

    with pytest.raises(SystemExit, match="FAILED at b: kaput"):
        U.run_stages([("a", ok("a")), ("b", boom), ("c", ok("c"))])
    assert ran == ["a"]  # c never ran


def test_run_stages_treats_clean_systemexit_as_success():
    def exits_zero():
        raise SystemExit(0)

    results = U.run_stages([("a", exits_zero)])
    assert results[0]["ok"] is True


# --- lock ---------------------------------------------------------------------
def test_update_lock_excludes_second_holder(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "LOGS", tmp_path)
    with U.update_lock():
        with pytest.raises(SystemExit, match="already running"):
            with U.update_lock():
                pass
    with U.update_lock():  # released cleanly, can re-acquire
        pass


# --- main() short-circuits ----------------------------------------------------
@pytest.fixture
def wired(tmp_path, monkeypatch):
    """main() with schedule/downloads faked and all real processing forbidden."""
    monkeypatch.setattr(C, "LOGS", tmp_path / "logs")
    monkeypatch.setattr(C, "RAW_PBP", tmp_path / "pbp")
    monkeypatch.setattr(C, "RAW_SHIFTS", tmp_path / "sc")
    monkeypatch.setattr(C, "RAW_HTMLSHIFTS", tmp_path / "html")
    monkeypatch.setattr(C, "INTERIM", tmp_path / "interim")
    for d in ("pbp", "sc", "html", "logs"):
        (tmp_path / d).mkdir(parents=True)
    for t in _INTERIM_TABLES:
        (tmp_path / "interim" / t).mkdir(parents=True)

    state = {"stages_ran": [], "downloads": 0}
    sched = [{"id": 2025020001, "date": "d", "state": "OFF"}]
    (tmp_path / "pbp" / "2025020001.json").write_text(json.dumps({"plays": [{"e": 1}]}))
    (tmp_path / "sc" / "2025020001.json").write_text('{"data":[]}')
    for t in _INTERIM_TABLES:
        (tmp_path / "interim" / t / "2025.parquet").write_text("parquet")

    monkeypatch.setattr(D, "schedule_for_season", lambda season, sess=None: sched)

    def fake_games(season, limit=None, only_final=True, schedule=None):
        state["downloads"] += 1
        return {
            "fetched": 0,
            "repaired": 0,
            "cached": 1,
            "missing": 0,
            "not_ready": 0,
            "pending": 0,
        }

    monkeypatch.setattr(D, "download_games", fake_games)
    monkeypatch.setattr(
        D, "download_html_shifts", lambda season: {"fetched": 0, "cached": 0, "missing": 0}
    )

    def fake_build(season, **kw):
        return [("s1", lambda: state["stages_ran"].append("s1"))]

    monkeypatch.setattr(U, "build_stages", fake_build)
    return state, tmp_path


def test_main_up_to_date_short_circuit(wired, capsys):
    state, tmp = wired
    U.main(["--season", "2025"])
    assert "up to date" in capsys.readouterr().out
    assert state["stages_ran"] == [] and state["downloads"] == 1
    runs = (tmp / "logs" / "update_runs.jsonl").read_text().splitlines()
    assert json.loads(runs[-1])["up_to_date"] is True


def test_main_dry_run_never_downloads(wired, capsys):
    state, _ = wired
    U.main(["--season", "2025", "--dry-run"])
    assert "dry run" in capsys.readouterr().out
    assert state["downloads"] == 0 and state["stages_ran"] == []


def test_main_force_runs_stages(wired, capsys):
    state, _ = wired
    U.main(["--season", "2025", "--force"])
    assert state["stages_ran"] == ["s1"]
    assert "stage summary" in capsys.readouterr().out


def test_main_runs_stages_when_new_games_arrive(wired, monkeypatch):
    state, _ = wired

    def fake_games(season, limit=None, only_final=True, schedule=None):
        state["downloads"] += 1
        return {
            "fetched": 2,
            "repaired": 0,
            "cached": 0,
            "missing": 0,
            "not_ready": 0,
            "pending": 3,
        }

    monkeypatch.setattr(D, "download_games", fake_games)
    U.main(["--season", "2025"])
    assert state["stages_ran"] == ["s1"]


def test_main_fetch_only_stops_before_stages(wired, monkeypatch, capsys):
    state, _ = wired
    U.main(["--season", "2025", "--fetch-only"])
    assert state["downloads"] == 1 and state["stages_ran"] == []
    assert "fetch-only" in capsys.readouterr().out
