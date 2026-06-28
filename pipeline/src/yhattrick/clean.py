"""Stage 1: raw/ -> interim/ (parse + type, one tidy table per source, per season).

Outputs (parquet, partitioned by season):
  interim/shots/<season>.parquet    unblocked shots from NHL pbp, with geometry + on-ice strength
  interim/shifts/<season>.parquet   shiftcharts -> tidy shift intervals in game-seconds
  interim/events/<season>.parquet   NHL pbp -> tidy event rows in game-seconds
  interim/roster/<season>.parquet   playerId -> name/position/team/number (from pbp rosterSpots)

No joining happens here (that's stints.py). raw/ is never modified.

Usage:
  uv run python -m yhattrick.clean                 # all configured seasons
  uv run python -m yhattrick.clean --season 2024
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import config as C
from . import html_shifts
from . import shot_geom as G


def _mmss_to_sec(s: str) -> int:
    m, sec = s.split(":")
    return int(m) * 60 + int(sec)


def game_sec(period: int, mmss: str) -> int:
    """Period-elapsed MM:SS -> seconds elapsed since the game's opening faceoff."""
    return (int(period) - 1) * C.PERIOD_SECONDS + _mmss_to_sec(mmss)


def season_of(nhl_game_id: int) -> int:
    return int(str(nhl_game_id)[:4])


def _downloaded_game_ids(season: int) -> list[int]:
    """NHL gameIds for `season` that have BOTH a shiftchart and a pbp file on disk."""
    sc = {int(p.stem) for p in C.RAW_SHIFTS.glob("*.json")}
    pbp = {int(p.stem) for p in C.RAW_PBP.glob("*.json")}
    return sorted(g for g in (sc & pbp) if season_of(g) == season)


# --- shots (unblocked, from NHL pbp events) ----------------------------------
def build_shots(season: int, names: dict[int, str]) -> int:
    """interim/events -> interim/shots: one row per regular-season unblocked (Fenwick) shot, with
    oriented geometry, rebound/rush, and on-ice strength decoded from situationCode. `event_idx` is
    the per-shot id and the join key for the xG model. Model-free (no xG attached here)."""
    ep = C.INTERIM / "events" / f"{season}.parquet"
    if not ep.exists():
        print(f"[shots] {season}: no events, skipping")
        return 0
    ev = pd.read_parquet(ep).sort_values(["nhl_game_id", "time_g", "event_idx"]).reset_index(drop=True)
    g = ev.groupby("nhl_game_id", sort=False)
    tsl = ev.time_g - g.time_g.shift(1)
    sit = G.decode_situation(ev.situation_code)
    ev = ev.assign(prev_type=g.type.shift(1), prev_zone=g.zone.shift(1), time_since_last=tsl, **sit)
    home_team_by = ev[ev.is_home].groupby("nhl_game_id").team.first()

    df = ev[ev.type.isin(G.FENWICK)].copy()
    df = df[df.nhl_game_id.map(C.is_regular_season)]
    df = df[(df.period < 5) & df.x.notna() & df.y.notna() & df.primary_player_id.notna()]

    is_home = df.is_home.to_numpy(bool)
    x_adj, y_adj = G.oriented(df.x, df.y, G.attack_sign(is_home, df.home_defending_side.to_numpy()))
    dist, ang = G.distance_angle(x_adj, y_adj)
    def_goalie = np.where(is_home, df.away_goalie, df.home_goalie)
    sid = df.primary_player_id.astype("int64")

    out = pd.DataFrame({
        "nhl_game_id": df.nhl_game_id.to_numpy(), "event_idx": df.event_idx.to_numpy(),
        "time_g": df.time_g.astype("int32").to_numpy(), "period": df.period.astype("int16").to_numpy(),
        "is_home": is_home.astype(int), "home_team": df.nhl_game_id.map(home_team_by).to_numpy(),
        "shooter_id": sid.to_numpy(), "shooter": sid.map(names).to_numpy(),
        "goal": (df.type == "goal").astype(int).to_numpy(), "event": df.type.to_numpy(),
        "x": df.x.astype("int16").to_numpy(), "y": df.y.astype("int16").to_numpy(),
        "shot_type": df.shot_type.to_numpy(),
        "distance": np.round(dist, 1), "angle": np.round(ang, 1),
        "rebound": G.rebound_flag(df.prev_type, df.time_since_last),
        "rush": G.rush_flag(df.prev_zone, df.time_since_last),
        "home_n": df.home_skaters.astype("Int64").to_numpy(), "away_n": df.away_skaters.astype("Int64").to_numpy(),
        "empty_net": (def_goalie == 0).astype(int),
    })
    dup = int(out.duplicated(["nhl_game_id", "event_idx"]).sum())
    assert dup == 0, f"{season}: {dup} duplicate (game, event_idx) shot keys — not a valid join key"
    op = C.INTERIM / "shots" / f"{season}.parquet"
    op.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(op, index=False)
    print(f"[shots] {season}: {len(out):,} unblocked shots across {out.nhl_game_id.nunique()} games -> {op.name}")
    return len(out)


# --- shifts ------------------------------------------------------------------
def _merge_player_intervals(rows: list[tuple]) -> list[tuple]:
    """Merge a single player-game's overlapping/duplicate shift intervals.

    NHL shiftcharts sometimes emit duplicate or overlapping shift rows for the same player
    (e.g. shift #21 and #22 both 58:46-60:00), which would double-count the player on the ice
    and inflate TOI. We collapse any intervals that overlap (next.start < current.end) into one;
    intervals that merely touch (next.start == current.end) stay separate.
    Each row is (start_g, end_g, period). Returns merged (start_g, end_g, period).
    """
    rows = sorted(rows)
    merged: list[list] = []
    for start, end, period in rows:
        if merged and start < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end, period])
    return [(s, e, p) for s, e, p in merged]


def clean_shifts(season: int) -> int:
    gids = _downloaded_game_ids(season)
    raw: dict[tuple, list[tuple]] = {}   # (gid, player_id) -> [(start,end,period), ...]
    meta: dict[tuple, tuple] = {}        # (gid, player_id) -> (name, team, team_id)
    html_games = 0
    for gid in gids:
        data = json.loads((C.RAW_SHIFTS / f"{gid}.json").read_text()).get("data", [])
        if data:
            for s in data:
                if not s.get("duration") or not s.get("startTime") or not s.get("endTime"):
                    continue  # goal-marker / malformed rows carry no interval
                p = s["period"]
                start, end = game_sec(p, s["startTime"]), game_sec(p, s["endTime"])
                if end <= start:
                    continue
                key = (gid, s["playerId"])
                raw.setdefault(key, []).append((start, end, int(p)))
                meta.setdefault(key, (f"{s['firstName']} {s['lastName']}", s["teamAbbrev"], s["teamId"]))
        else:
            # the JSON shift feed is empty (dead for late-2024-25 on) -> parse the HTML TOI reports
            hrows = html_shifts.shifts_for_game(gid)
            if hrows:
                html_games += 1
            for pid, name, team, team_id, period, start, end in hrows:
                key = (gid, pid)
                raw.setdefault(key, []).append((start, end, period))
                meta.setdefault(key, (name, team, team_id))
    if not raw:
        print(f"[shifts] {season}: no games on disk yet")
        return 0
    rows = []
    for (gid, pid), intervals in raw.items():
        name, team, team_id = meta[(gid, pid)]
        for n, (start, end, period) in enumerate(_merge_player_intervals(intervals), 1):
            rows.append((gid, pid, name, team, team_id, period, start, end, n))
    df = pd.DataFrame(rows, columns=[
        "nhl_game_id", "player_id", "player_name", "team", "team_id",
        "period", "start_g", "end_g", "shift_number"])
    df.sort_values(["nhl_game_id", "player_id", "start_g"], inplace=True)
    df["duration_s"] = (df.end_g - df.start_g).astype("int32")
    out = C.INTERIM / "shifts" / f"{season}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    via = f" ({html_games} via HTML TOI reports)" if html_games else ""
    print(f"[shifts] {season}: {len(df):,} shifts across {df.nhl_game_id.nunique()} games{via} -> {out.name}")
    return len(df)


# --- events + roster (both come from pbp) ------------------------------------
def _name(rs: dict) -> str:
    return f"{rs.get('firstName', {}).get('default', '')} {rs.get('lastName', {}).get('default', '')}".strip()


def clean_events_and_roster(season: int) -> tuple[int, pd.DataFrame]:
    gids = _downloaded_game_ids(season)
    ev_rows, roster = [], {}
    for gid in gids:
        pbp = json.loads((C.RAW_PBP / f"{gid}.json").read_text())
        home_id = pbp["homeTeam"]["id"]
        team_abbr = {pbp["homeTeam"]["id"]: pbp["homeTeam"]["abbrev"],
                     pbp["awayTeam"]["id"]: pbp["awayTeam"]["abbrev"]}
        pid_name = {}
        for rs in pbp.get("rosterSpots", []):
            pid = rs["playerId"]
            pid_name[pid] = _name(rs)
            roster[(pid, season)] = (pid, season, _name(rs), rs.get("positionCode"),
                                     rs.get("sweaterNumber"), team_abbr.get(rs.get("teamId")))
        for pl in pbp.get("plays", []):
            pd_ = pl.get("periodDescriptor", {})
            period = pd_.get("number")
            tip = pl.get("timeInPeriod")
            if period is None or not tip:
                continue
            d = pl.get("details", {})
            owner = d.get("eventOwnerTeamId")
            ev_rows.append((
                gid, pl.get("sortOrder", pl.get("eventId")), int(period), game_sec(period, tip),
                pl.get("typeDescKey"), team_abbr.get(owner), owner == home_id,
                d.get("xCoord"), d.get("yCoord"), d.get("zoneCode"),
                pl.get("situationCode"), pl.get("homeTeamDefendingSide"),
                d.get("scoringPlayerId") or d.get("shootingPlayerId") or d.get("hittingPlayerId")
                or d.get("committedByPlayerId") or d.get("winningPlayerId"),
                d.get("assist1PlayerId"), d.get("assist2PlayerId"),  # populated on goal events
                d.get("shotType"), d.get("descKey"),  # descKey = penalty type when present
            ))
    if not ev_rows:
        print(f"[events] {season}: no games on disk yet")
        return 0, pd.DataFrame()
    ev = pd.DataFrame(ev_rows, columns=[
        "nhl_game_id", "event_idx", "period", "time_g", "type", "team", "is_home",
        "x", "y", "zone", "situation_code", "home_defending_side", "primary_player_id",
        "assist1_player_id", "assist2_player_id", "shot_type", "detail_key"])
    ev.sort_values(["nhl_game_id", "time_g", "event_idx"], inplace=True)
    out = C.INTERIM / "events" / f"{season}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    ev.to_parquet(out, index=False)
    rdf = pd.DataFrame(roster.values(),
                       columns=["player_id", "season", "player_name", "position", "number", "team"])
    print(f"[events] {season}: {len(ev):,} events, {len(rdf)} roster players "
          f"across {ev.nhl_game_id.nunique()} games -> {out.name}")
    return len(ev), rdf


def clean_season(season: int) -> None:
    print(f"\n=== clean {C.season_label(season)} ===")
    clean_shifts(season)
    _, roster = clean_events_and_roster(season)
    if len(roster):
        out = C.INTERIM / "roster" / f"{season}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        roster.to_parquet(out, index=False)
        print(f"[roster] {season}: {len(roster)} players -> {out.name}")
    # shots are derived from the just-written events; names resolve shooter ids
    names = dict(zip(roster.player_id, roster.player_name)) if len(roster) else {}
    build_shots(season, names)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Parse raw/ into typed interim/ tables")
    p.add_argument("--season", type=int, default=None, help="one season (default: all configured)")
    args = p.parse_args(argv)
    C.ensure_dirs()
    for season in ([args.season] if args.season else C.SEASONS):
        clean_season(season)


if __name__ == "__main__":
    main()
