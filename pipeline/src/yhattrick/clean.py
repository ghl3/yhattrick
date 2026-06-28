"""Stage 1: raw/ -> interim/ (parse + type, one tidy table per source, per season).

Outputs (parquet, partitioned by season):
  interim/shots/<season>.parquet    MoneyPuck shots: selected/typed cols, NHL game_id resolved
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
import zipfile

import pandas as pd

from . import config as C

# MoneyPuck columns we keep (focused subset of the 137).
_SHOT_COLS = [
    "shotID", "game_id", "season", "period", "time", "timeLeft",
    "event", "goal", "isHomeTeam", "teamCode", "homeTeamCode", "awayTeamCode",
    "shooterPlayerId", "shooterName", "goalieIdForShot", "goalieNameForShot",
    "xCord", "yCord", "xCordAdjusted", "yCordAdjusted",
    "shotDistance", "shotAngle", "shotType",
    "homeSkatersOnIce", "awaySkatersOnIce", "homeEmptyNet", "awayEmptyNet",
    "homeTeamGoals", "awayTeamGoals", "shotRebound", "shotRush", "xGoal",
]


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


# --- shots -------------------------------------------------------------------
def clean_shots(season: int) -> int:
    zpath = C.RAW_MONEYPUCK / f"shots_{season}.zip"
    if not zpath.exists():
        print(f"[shots] {season}: no zip, skipping")
        return 0
    zf = zipfile.ZipFile(zpath)
    with zf.open(zf.namelist()[0]) as fh:
        df = pd.read_csv(fh, usecols=lambda c: c in _SHOT_COLS)
    df = df[[c for c in _SHOT_COLS if c in df.columns]].copy()
    df.rename(columns={"game_id": "mp_game_id", "time": "game_seconds"}, inplace=True)
    df["nhl_game_id"] = [C.mp_to_nhl_game_id(g, season) for g in df.mp_game_id]
    df["game_seconds"] = df.game_seconds.astype("int32")
    df["period"] = df.period.astype("int16")
    out = C.INTERIM / "shots" / f"{season}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"[shots] {season}: {len(df):,} shots across {df.nhl_game_id.nunique()} games -> {out.name}")
    return len(df)


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
    for gid in gids:
        data = json.loads((C.RAW_SHIFTS / f"{gid}.json").read_text())["data"]
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
    print(f"[shifts] {season}: {len(df):,} shifts across {df.nhl_game_id.nunique()} games -> {out.name}")
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
    clean_shots(season)
    clean_shifts(season)
    _, roster = clean_events_and_roster(season)
    if len(roster):
        out = C.INTERIM / "roster" / f"{season}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        roster.to_parquet(out, index=False)
        print(f"[roster] {season}: {len(roster)} players -> {out.name}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Parse raw/ into typed interim/ tables")
    p.add_argument("--season", type=int, default=None, help="one season (default: all configured)")
    args = p.parse_args(argv)
    C.ensure_dirs()
    for season in ([args.season] if args.season else C.SEASONS):
        clean_season(season)


if __name__ == "__main__":
    main()
