"""Stage 3: processed/ + interim/ -> web/public/data/ (site-facing JSON).

Writes the data the inspection website reads:
  web/public/data/games.json            index row per game (teams, score, health flags)
  web/public/data/game/<gid>.json       full ordered timeline: stints, each with on-ice
                                        players + the events that occurred during it

The per-game timeline is the manual-inspection tool: stints in order, and within each stint
the play-by-play events (faceoffs, shots w/ xG + on-ice-match flag, goals, penalties,
hits) so the on-ice sets, strength, and shot attribution can be eyeballed against reality.

Usage:
  uv run python -m yhattrick.build_games                 # all processed seasons
  uv run python -m yhattrick.build_games --season 2021 --limit 50
"""

from __future__ import annotations

import argparse
import json
import math
import shutil

import numpy as np
import pandas as pd

from .. import config as C
from ..data.stints import shot_stint_index, other_stint_index, _SHOTLIKE


def _sanitize(o):
    """Recursively convert numpy scalars and replace NaN/inf with None so the result is
    strictly-valid JSON (json.dumps would otherwise emit a bare `NaN` token)."""
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _dump(obj) -> str:
    return json.dumps(_sanitize(obj), separators=(",", ":"), allow_nan=False)


# pbp event types we surface in the timeline (others are dropped as noise).
_KEEP_EVENTS = {
    "faceoff",
    "shot-on-goal",
    "missed-shot",
    "blocked-shot",
    "goal",
    "penalty",
    "hit",
    "giveaway",
    "takeaway",
    "stoppage",
}


def _clock(game_seconds: int) -> str:
    """Game-seconds -> 'P<period> MM:SS' elapsed within the period."""
    period = game_seconds // C.PERIOD_SECONDS + 1
    within = game_seconds % C.PERIOD_SECONDS
    return f"P{period} {within // 60:02d}:{within % 60:02d}"


def _roster_lookup(season: int) -> dict[int, dict]:
    r = pd.read_parquet(C.INTERIM / "roster" / f"{season}.parquet")
    return {
        int(row.player_id): {
            "id": int(row.player_id),
            "name": row.player_name,
            "pos": row.position,
            "number": row.number,
        }
        for row in r.itertuples()
    }


def _players(ids, lookup) -> list[dict]:
    out = []
    for pid in ids:
        if pid is None or pd.isna(pid):
            continue
        pid = int(pid)
        out.append(lookup.get(pid, {"id": pid, "name": f"#{pid}", "pos": None, "number": None}))
    return out


def _ids(xs) -> list[int]:
    """Stint on-ice personnel as bare player ids (names resolved client-side from players[])."""
    return [int(x) for x in xs if not (x is None or pd.isna(x))]


def _goalie_id(x) -> int | None:
    return None if x is None or pd.isna(x) else int(x)


def _game_meta(gid: int) -> dict:
    pbp = json.loads((C.RAW_PBP / f"{gid}.json").read_text())
    return {
        "date": pbp.get("gameDate"),
        "home": pbp["homeTeam"]["abbrev"],
        "away": pbp["awayTeam"]["abbrev"],
        "home_score": pbp["homeTeam"].get("score"),
        "away_score": pbp["awayTeam"].get("score"),
    }


def _count_by(series) -> dict:
    s = series.dropna()
    return s.astype("int64").value_counts().to_dict()


def _game_aggregates(gid, stints, shifts_g, shots_g, events_g, meta, lookup) -> tuple[dict, list]:
    home, away = meta["home"], meta["away"]
    totals = {
        "home_xgf": round(sum(s["home_xgf"] for s in stints), 3),
        "away_xgf": round(sum(s["away_xgf"] for s in stints), 3),
        "home_shots": int((shots_g.is_home == 1).sum()),
        "away_shots": int((shots_g.is_home == 0).sum()),
        "home_score": meta["home_score"],
        "away_score": meta["away_score"],
    }
    # per-player scoring from pbp goal events (primary = scorer, a1/a2 = assists)
    goals_ev = events_g[events_g.type == "goal"]
    goals = _count_by(goals_ev.primary_player_id)
    a1 = _count_by(goals_ev.assist1_player_id)
    a2 = _count_by(goals_ev.assist2_player_id)
    # per-player ice time from shifts, shots + xG from shots_onice
    sh = shifts_g.groupby("player_id").agg(
        team=("team", "first"), shifts=("shift_number", "count"), toi_s=("duration_s", "sum")
    )
    sc = shots_g.groupby("shooter_id").agg(shots=("event_idx", "count"), xg=("xg", "sum"))
    players = []
    for pid, row in sh.iterrows():
        pid = int(pid)
        info = lookup.get(pid, {"name": f"#{pid}", "pos": None, "number": None})
        s = sc.loc[pid] if pid in sc.index else None
        g, p1, p2 = goals.get(pid, 0), a1.get(pid, 0), a2.get(pid, 0)
        players.append(
            {
                "id": pid,
                "name": info["name"],
                "pos": info["pos"],
                "number": info["number"],
                "team": row.team,
                "side": "home" if row.team == home else "away",
                "shifts": int(row.shifts),
                "toi_s": int(row.toi_s),
                "g": g,
                "a1": p1,
                "a2": p2,
                "pts": g + p1 + p2,
                "shots": int(s.shots) if s is not None else 0,
                "xg": round(float(s.xg), 3) if s is not None else 0.0,
            }
        )
    players.sort(key=lambda p: -p["toi_s"])
    return totals, players


def build_game(gid: int, stints_g, events_g, shots_g, shifts_g, lookup) -> dict:
    meta = _game_meta(gid)
    stints = stints_g.sort_values("stint_idx").to_dict("records")
    starts = [s["start_g"] for s in stints]

    # index shot rows by (shooter, time) for attaching xG + features to pbp shot events
    shot_by_key = {}
    for s in shots_g.itertuples():
        shot_by_key.setdefault((s.shooter_id, s.game_seconds), s)

    # bucket events into their stint
    buckets: dict[int, list] = {i: [] for i in range(len(stints))}
    for e in events_g.itertuples():
        if e.type not in _KEEP_EVENTS:
            continue
        # a shot/goal on a boundary belongs to the stint ENDING there (the personnel who took it);
        # other events (faceoff, penalty, …) open the next stint. Reuses stints.py's single source of
        # truth so the timeline and the per-stint xGF never disagree.
        j = (
            shot_stint_index(starts, e.time_g)
            if e.type in _SHOTLIKE
            else other_stint_index(starts, e.time_g)
        )
        if j < 0:
            continue
        ev = {
            "t": int(e.time_g),
            "clock": _clock(int(e.time_g)),
            "type": e.type,
            "team": None if pd.isna(e.team) else e.team,
            "x": None if pd.isna(e.x) else int(e.x),
            "y": None if pd.isna(e.y) else int(e.y),
            "zone": None if pd.isna(e.zone) else e.zone,
        }
        if e.primary_player_id is not None and not pd.isna(e.primary_player_id):
            ev["player"] = lookup.get(int(e.primary_player_id), {}).get("name")
        if e.detail_key and not pd.isna(e.detail_key):
            ev["detail"] = e.detail_key
        if e.type in ("shot-on-goal", "missed-shot", "goal"):
            sk = (
                shot_by_key.get((int(e.primary_player_id), int(e.time_g)))
                if e.primary_player_id and not pd.isna(e.primary_player_id)
                else None
            )
            if sk is not None:
                if pd.notna(sk.xg):
                    ev["xg"] = round(float(sk.xg), 3)
                ev["onice_match"] = sk.onice_match
                ev["shot_type"] = sk.shot_type
                ev["distance"] = sk.distance
                ev["angle"] = sk.angle
                if sk.rebound:
                    ev["rebound"] = True
                if sk.rush:
                    ev["rush"] = True
        buckets[j].append(ev)

    out_stints = []
    for i, s in enumerate(stints):
        out_stints.append(
            {
                "idx": s["stint_idx"],
                "start": s["start_g"],
                "end": s["end_g"],
                "clock_start": _clock(s["start_g"]),
                "clock_end": _clock(s["end_g"]),
                "duration_s": s["duration_s"],
                "strength": s["strength"],
                "overload": bool(s["overload"]),
                "home_skaters": _ids(s["home_skaters"]),
                "away_skaters": _ids(s["away_skaters"]),
                "home_goalie": _goalie_id(s["home_goalie"]),
                "away_goalie": _goalie_id(s["away_goalie"]),
                "home_xgf": s["home_xgf"],
                "away_xgf": s["away_xgf"],
                "events": buckets[i],
            }
        )
    totals, players = _game_aggregates(gid, stints, shifts_g, shots_g, events_g, meta, lookup)
    return {"game_id": gid, **meta, "totals": totals, "players": players, "stints": out_stints}


def build_season(season: int, limit: int | None = None) -> list[dict]:
    sp = C.PROCESSED / "stints" / f"{season}.parquet"
    if not sp.exists():
        print(f"[games] {season}: no processed stints, skipping")
        return []
    stints = pd.read_parquet(sp)
    shots = pd.read_parquet(C.PROCESSED / "shots_onice" / f"{season}.parquet")
    events = pd.read_parquet(C.INTERIM / "events" / f"{season}.parquet")
    shifts = pd.read_parquet(C.INTERIM / "shifts" / f"{season}.parquet")
    lookup = _roster_lookup(season)

    gids = sorted(stints.nhl_game_id.unique())
    if limit:
        gids = gids[:limit]
    out_dir = C.SITE_JSON / "game"
    out_dir.mkdir(parents=True, exist_ok=True)

    index = []
    for gid in gids:
        st_g = stints[stints.nhl_game_id == gid]
        ev_g = events[events.nhl_game_id == gid]
        sh_g = shots[shots.nhl_game_id == gid]
        shf_g = shifts[shifts.nhl_game_id == gid]
        game = build_game(gid, st_g, ev_g, sh_g, shf_g, lookup)
        (out_dir / f"{gid}.json").write_text(_dump(game))

        n_shots = len(sh_g)
        index.append(
            {
                "game_id": int(gid),
                "season": season,
                "date": game["date"],
                "home": game["home"],
                "away": game["away"],
                "home_score": game["home_score"],
                "away_score": game["away_score"],
                # per-team xG + shots and an overtime flag, for team-season summaries
                "home_xgf": game["totals"]["home_xgf"],
                "away_xgf": game["totals"]["away_xgf"],
                "home_shots": game["totals"]["home_shots"],
                "away_shots": game["totals"]["away_shots"],
                "ot": bool((ev_g.period >= 4).any()),
                "n_stints": len(st_g),
                "n_shots": n_shots,
                "n_events": sum(len(s["events"]) for s in game["stints"]),
                "onice_exact": round((sh_g.onice_match == "exact").mean(), 4) if n_shots else None,
                "large_mismatch": int((sh_g.onice_match == "large").sum()),
                "overload_stints": int(st_g.overload.sum()),
            }
        )
    print(f"[games] {C.season_label(season)}: wrote {len(gids)} game files -> {out_dir}")
    return index


def sync_to_web() -> None:
    """Copy the canonical site JSON (data/site) into the web app's static dir (web/public/data)."""
    dst = C.WEB_DATA
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "game").mkdir(exist_ok=True)
    shutil.copy2(C.SITE_JSON / "games.json", dst / "games.json")
    n = 0
    for f in (C.SITE_JSON / "game").glob("*.json"):
        shutil.copy2(f, dst / "game" / f.name)
        n += 1
    print(f"[games] synced index + {n} game files -> {dst}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Build site JSON (games index + per-game timelines)")
    p.add_argument("--season", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-sync", action="store_true", help="write data/site only, skip copy to web/")
    args = p.parse_args(argv)
    C.SITE_JSON.mkdir(parents=True, exist_ok=True)

    seasons = [args.season] if args.season else C.SEASONS
    index = []
    for season in seasons:
        index.extend(build_season(season, args.limit))
    # a per-season run refreshes only that season's index rows; the rest carry over from the
    # existing games.json so the site index always spans every exported season
    if args.season and (C.SITE_JSON / "games.json").exists():
        rebuilt = set(seasons)
        prior = json.loads((C.SITE_JSON / "games.json").read_text())
        index.extend(r for r in prior if r["game_id"] // 1_000_000 not in rebuilt)
    index.sort(key=lambda r: (r["date"] or "", r["game_id"]), reverse=True)  # reverse chronological
    (C.SITE_JSON / "games.json").write_text(_dump(index))
    print(f"[games] index: {len(index)} games -> {C.SITE_JSON / 'games.json'}")
    if not args.no_sync:
        sync_to_web()


if __name__ == "__main__":
    main()
