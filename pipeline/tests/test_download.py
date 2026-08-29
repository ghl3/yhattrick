"""Tests for the raw-data fetch stage (yhattrick.data.download).

The downloader is schedule-aware: only finished games (state OFF/FINAL) are fetched, because
the NHL pbp endpoint answers 200 with zero plays for unplayed games and a cached empty file
would poison the resumable exists-check. These tests fake `_get` so no network is touched.
"""

from __future__ import annotations

import json

import pytest

from yhattrick import config as C
from yhattrick.data import download as D


def _pbp_body(gid, n_plays=3):
    return json.dumps({"id": gid, "plays": [{"eventId": i} for i in range(n_plays)]})


def _sched(*games):
    """[(gid, state), ...] -> schedule rows."""
    return [{"id": gid, "date": "2026-10-08", "state": state} for gid, state in games]


@pytest.fixture
def rawdirs(tmp_path, monkeypatch):
    """Point every raw dir at tmp and silence throttling/dir creation."""
    for name in ("RAW_PBP", "RAW_SHIFTS", "RAW_HTMLSHIFTS", "RAW_PLAYERS"):
        d = tmp_path / name.lower()
        d.mkdir()
        monkeypatch.setattr(C, name, d)
    monkeypatch.setattr(C, "ensure_dirs", lambda: None)
    monkeypatch.setattr(D.time, "sleep", lambda s: None)
    return tmp_path


def _serve(monkeypatch, responses):
    """Fake _get: first substring match in `responses` wins; None mimics a clean 404."""
    calls = []

    def fake_get(sess, url, *, binary):
        calls.append(url)
        for frag, body in responses.items():
            if frag in url:
                return body
        raise AssertionError(f"unexpected URL fetched: {url}")

    monkeypatch.setattr(D, "_get", fake_get)
    return calls


# --- gating on schedule state -------------------------------------------------
def test_download_games_fetches_only_finished_games(rawdirs, monkeypatch):
    gid_off, gid_fut = 2026020001, 2026020900
    calls = _serve(
        monkeypatch,
        {
            "play-by-play": _pbp_body(gid_off),
            "shiftcharts": '{"data":[{"x":1}]}',
        },
    )
    n = D.download_games(2026, schedule=_sched((gid_off, "OFF"), (gid_fut, "FUT")))
    assert n["fetched"] == 2 and n["pending"] == 1
    assert (C.RAW_PBP / f"{gid_off}.json").exists()
    assert not (C.RAW_PBP / f"{gid_fut}.json").exists()
    assert all(str(gid_fut) not in u for u in calls)


def test_download_games_include_unfinished_fetches_everything_with_plays_guard(
    rawdirs, monkeypatch
):
    gid = 2026020900
    _serve(monkeypatch, {"play-by-play": json.dumps({"plays": []}), "shiftcharts": '{"data":[]}'})
    n = D.download_games(2026, only_final=False, schedule=_sched((gid, "FUT")))
    # even unguarded, an empty-plays pbp is never cached — the content check is the backstop
    assert n["not_ready"] == 1
    assert not (C.RAW_PBP / f"{gid}.json").exists()


# --- repair of stale caches ---------------------------------------------------
def test_download_games_repairs_cached_empty_pbp(rawdirs, monkeypatch):
    gid = 2026020002
    (C.RAW_PBP / f"{gid}.json").write_text(json.dumps({"plays": []}))  # fetched pre-game
    (C.RAW_SHIFTS / f"{gid}.json").write_text('{"data":[]}')
    _serve(monkeypatch, {"play-by-play": _pbp_body(gid)})
    n = D.download_games(2026, schedule=_sched((gid, "OFF")))
    assert n["repaired"] == 1 and n["cached"] == 0
    assert len(json.loads((C.RAW_PBP / f"{gid}.json").read_text())["plays"]) == 3


def test_download_games_keeps_valid_cached_files(rawdirs, monkeypatch):
    gid = 2026020003
    (C.RAW_PBP / f"{gid}.json").write_text(_pbp_body(gid))  # small but has plays
    (C.RAW_SHIFTS / f"{gid}.json").write_text('{"data":[]}')
    _serve(monkeypatch, {})  # any fetch would raise
    n = D.download_games(2026, schedule=_sched((gid, "OFF")))
    assert n["cached"] == 1 and n["fetched"] == 0


def test_download_games_records_empty_shiftchart_on_404(rawdirs, monkeypatch):
    gid = 2026020004
    _serve(monkeypatch, {"play-by-play": _pbp_body(gid), "shiftcharts": None})
    n = D.download_games(2026, schedule=_sched((gid, "OFF")))
    # the game still reaches clean (both files exist); HTML TOI reports carry the shifts
    assert json.loads((C.RAW_SHIFTS / f"{gid}.json").read_text()) == {"data": []}
    assert n["missing"] == 1 and n["fetched"] == 1


def test_pbp_file_stale_only_for_small_playless_files(rawdirs):
    p = C.RAW_PBP / "x.json"
    p.write_text(json.dumps({"plays": []}))
    assert D._pbp_file_stale(p)
    p.write_text(_pbp_body(1))
    assert not D._pbp_file_stale(p)


# --- team list / schedule -----------------------------------------------------
def test_teams_for_season_falls_back_when_standings_empty(monkeypatch):
    """Before mid-January the mid-season standings date is in the future and the endpoint
    answers an empty list; the previous season's standings supply the club set."""
    import datetime as dt

    calls = []
    today = dt.date.today().isoformat()

    def fake_get(sess, url, *, binary):
        calls.append(url)
        if "2027-01-15" in url or today in url:  # future mid-season date + preseason today
            return '{"standings": []}'
        return json.dumps({"standings": [{"teamAbbrev": {"default": t}} for t in ("TOR", "BOS")]})

    monkeypatch.setattr(D, "_get", fake_get)
    assert D._teams_for_season(D._session(), 2026) == ["BOS", "TOR"]
    assert len(calls) >= 2  # tried at least one empty date before falling back


def test_schedule_for_season_keeps_state_and_dedupes(monkeypatch):
    def fake_get(sess, url, *, binary):
        if "standings" in url:
            return json.dumps(
                {"standings": [{"teamAbbrev": {"default": t}} for t in ("TOR", "BOS")]}
            )
        # both clubs list the same head-to-head game; playoffs (03) must be dropped
        return json.dumps(
            {
                "games": [
                    {"id": 2026020001, "gameType": 2, "gameDate": "2026-10-08", "gameState": "OFF"},
                    {"id": 2026030001, "gameType": 3, "gameDate": "2027-04-20", "gameState": "FUT"},
                ]
            }
        )

    monkeypatch.setattr(D, "_get", fake_get)
    monkeypatch.setattr(D.time, "sleep", lambda s: None)
    sched = D.schedule_for_season(2026)
    assert sched == [{"id": 2026020001, "date": "2026-10-08", "state": "OFF"}]


# --- HTML TOI fallback --------------------------------------------------------
def test_download_html_shifts_targets_empty_json_games(rawdirs, monkeypatch):
    gid_empty, gid_full = 2026020005, 2026020006
    for gid, sc in ((gid_empty, '{"data":[]}'), (gid_full, '{"data":[{"x":1}]}')):
        (C.RAW_PBP / f"{gid}.json").write_text(_pbp_body(gid))
        (C.RAW_SHIFTS / f"{gid}.json").write_text(sc)
    calls = _serve(monkeypatch, {"htmlreports": "<html>" + "x" * 2000 + "</html>"})
    n = D.download_html_shifts(2026)
    assert n["fetched"] == 2  # TH + TV for the empty-JSON game only
    assert all(C.game6(gid_full) not in u for u in calls)
    assert (C.RAW_HTMLSHIFTS / f"TH{C.game6(gid_empty)}.HTM").exists()
