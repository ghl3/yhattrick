"""Stage 0: pull raw data into data/raw/ (immutable). Resumable + throttled. All NHL sources.

Raw artifacts:
  - NHL shiftcharts/<gameId>.json + pbp/<gameId>.json  (on-ice identities + events)
  - NHL players/<playerId>.json                         (handedness, for off-wing)

The game list comes from the NHL schedule: the season's teams (standings) -> each club's
schedule -> regular-season gameIds with their gameState. Only FINISHED games (state OFF/FINAL)
are fetched — an unplayed game's pbp endpoint answers 200 with zero plays, and caching that
would poison the resumable skip. Every fetch checks for an existing file first, so re-running
only pulls what's missing (resumable), and a cached pbp with no plays for a now-finished game
is re-fetched automatically (repair).

Usage:
  uv run python -m yhattrick.download games --season 2024  # shiftcharts+pbp for that season
  uv run python -m yhattrick.download games --season 2024 --limit 25
  uv run python -m yhattrick.download handedness           # player handedness json
  uv run python -m yhattrick.download all                  # everything, all seasons
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import pandas as pd
import requests

from .. import config as C


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": C.USER_AGENT})
    return s


def _get(sess: requests.Session, url: str, *, binary: bool) -> bytes | str | None:
    """GET with retries. Returns content, or None on a clean 404 (missing resource)."""
    for attempt in range(1, C.MAX_RETRIES + 1):
        try:
            r = sess.get(url, timeout=C.REQUEST_TIMEOUT)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content if binary else r.text
        except requests.RequestException as e:
            if attempt == C.MAX_RETRIES:
                print(f"    ! failed after {attempt} tries: {url}\n      {e}", file=sys.stderr)
                raise
            time.sleep(C.THROTTLE_SECONDS * 2 * attempt)
    return None


# --- NHL schedule (the game list) --------------------------------------------
def _teams_for_season(sess: requests.Session, season: int) -> list[str]:
    """Team abbreviations active in `season`, from the standings.

    Using the standings (not a hardcoded list) auto-tracks relocations/expansion per season.
    The standings endpoint answers `{"standings": []}` for any date it has no rows for (e.g. a
    mid-season date that is still in the future), so try progressively earlier dates: mid-season,
    today, then mid-PREVIOUS-season (the club set carries over between seasons outside an
    expansion year, and the schedule is published months before the standings exist)."""
    import datetime as dt

    candidates = [f"{season + 1}-01-15", dt.date.today().isoformat(), f"{season}-01-15"]
    for date in candidates:
        txt = _get(sess, C.NHL_STANDINGS_URL.format(date=date), binary=False)
        rows = json.loads(txt).get("standings", []) if txt else []
        if rows:
            return sorted({r["teamAbbrev"]["default"] for r in rows})
    raise RuntimeError(f"NHL standings empty for all of {candidates}")


def schedule_for_season(season: int, sess: requests.Session | None = None) -> list[dict]:
    """The season's regular-season schedule, unioned across each club's full schedule.

    One dict per game: {"id", "date", "state"} where `state` is the NHL gameState (OFF/FINAL =
    finished, FUT/PRE = not started, LIVE/CRIT = underway). Sorted by id."""
    sess = sess or _session()
    games: dict[int, dict] = {}
    for team in _teams_for_season(sess, season):
        txt = _get(
            sess,
            C.NHL_CLUB_SCHEDULE_URL.format(team=team, season8=C.nhl_season8(season)),
            binary=False,
        )
        if txt:
            for g in json.loads(txt).get("games", []):
                if g.get("gameType") == 2 and C.is_regular_season(int(g["id"])):
                    games[int(g["id"])] = {
                        "id": int(g["id"]),
                        "date": g.get("gameDate"),
                        "state": g.get("gameState", ""),
                    }
        time.sleep(C.THROTTLE_SECONDS)
    return [games[k] for k in sorted(games)]


def game_ids_for_season(season: int) -> list[int]:
    """Regular-season NHL gameIds for a season (every schedule state)."""
    return [g["id"] for g in schedule_for_season(season)]


# --- NHL per-game (shiftcharts + pbp) ----------------------------------------
# A cached pbp smaller than this is suspect (an empty-plays stub is ~1 KB, a real game ~130 KB);
# only files under the threshold are parsed for the plays check, so the repair scan stays cheap.
_PBP_STUB_BYTES = 20_000


def _pbp_has_plays(txt: str) -> bool:
    try:
        return len(json.loads(txt).get("plays", [])) > 0
    except json.JSONDecodeError:
        return False


def _pbp_file_stale(path) -> bool:
    """True for a cached pbp with no plays — fetched while the game was still FUT/LIVE."""
    return path.stat().st_size < _PBP_STUB_BYTES and not _pbp_has_plays(path.read_text())


def download_games(
    season: int,
    limit: int | None = None,
    only_final: bool = True,
    schedule: list[dict] | None = None,
) -> dict:
    """Ensure raw shiftcharts+pbp for the season's games. Resumable and self-repairing.

    Only games the schedule marks finished (OFF/FINAL) are fetched — the pbp endpoint answers
    200 with zero plays for unplayed games, and a cached empty file would otherwise never be
    retried. A cached pbp with no plays for a now-finished game is re-fetched (repair). A
    shiftchart 404 is recorded as `{"data": []}` so the game still reaches clean, which falls
    back to the HTML TOI reports. Returns the counts it prints."""
    C.ensure_dirs()
    sess = _session()
    games = schedule if schedule is not None else schedule_for_season(season, sess)
    if limit:
        games = games[:limit]
    if only_final:
        todo = [g for g in games if g["state"] in C.FINAL_GAME_STATES]
        pending = [g for g in games if g["state"] not in C.FINAL_GAME_STATES]
    else:
        todo, pending = list(games), []
    print(
        f"[games] {C.season_label(season)}: {len(games)} scheduled, "
        f"{len(todo)} finished, {len(pending)} not final yet"
    )

    n = {"fetched": 0, "repaired": 0, "cached": 0, "missing": 0, "not_ready": 0}
    for i, g in enumerate(todo, 1):
        gid = g["id"]
        sc_path = C.RAW_SHIFTS / f"{gid}.json"
        pbp_path = C.RAW_PBP / f"{gid}.json"
        need = []
        if not sc_path.exists():
            need.append(("shift", C.NHL_SHIFTCHARTS_URL.format(game_id=gid), sc_path))
        if not pbp_path.exists():
            need.append(("pbp", C.NHL_PBP_URL.format(game_id=gid), pbp_path))
        elif _pbp_file_stale(pbp_path):
            need.append(("pbp-repair", C.NHL_PBP_URL.format(game_id=gid), pbp_path))
        if not need:
            n["cached"] += 1
            continue
        for kind, url, path in need:
            txt = _get(sess, url, binary=False)
            if txt is None:
                if kind == "shift":
                    # dead feed: record the emptiness so clean sees the game and uses HTML shifts
                    path.write_text('{"data":[]}')
                    print(f"    ! shift 404 for {gid} -> recorded empty (HTML fallback)")
                else:
                    print(f"    ! {kind} 404 for {gid}")
                n["missing"] += 1
                continue
            if kind.startswith("pbp") and not _pbp_has_plays(txt):
                n["not_ready"] += 1
                print(f"    ! pbp for {gid} has no plays yet (state {g['state']}) — will retry")
                continue
            path.write_text(txt)
            n["repaired" if kind == "pbp-repair" else "fetched"] += 1
            time.sleep(C.THROTTLE_SECONDS)
        if i % 50 == 0:
            print(f"    {i}/{len(todo)}  ({', '.join(f'{k} {v}' for k, v in n.items() if v)})")
    n["pending"] = len(pending)
    print(
        f"[games] {C.season_label(season)} done: fetched {n['fetched']}, repaired {n['repaired']}, "
        f"cached {n['cached']}, missing {n['missing']}, not_ready {n['not_ready']}, "
        f"pending {n['pending']}"
    )
    return n


# --- NHL per-player landing (handedness) -------------------------------------
def _shooter_ids() -> list[int]:
    """All skaters who took a shot, from the cleaned roster tables."""
    ids: set[int] = set()
    for p in sorted((C.INTERIM / "roster").glob("*.parquet")):
        ids.update(int(x) for x in pd.read_parquet(p, columns=["player_id"]).player_id.dropna())
    return sorted(ids)


def download_handedness() -> dict:
    """Fetch each player's landing json (for shootsCatches). Resumable; needs cleaned rosters.
    Returns the counts it prints."""
    C.ensure_dirs()
    sess = _session()
    ids = _shooter_ids()
    if not ids:
        raise FileNotFoundError("no interim rosters -- run `make clean-data` first")
    print(f"[players] ensuring landing json for {len(ids)} players")
    fetched = skipped = missing = 0
    for i, pid in enumerate(ids, 1):
        path = C.RAW_PLAYERS / f"{pid}.json"
        if path.exists():
            skipped += 1
            continue
        txt = _get(sess, C.NHL_PLAYER_URL.format(player_id=pid), binary=False)
        if txt is None:
            missing += 1
        else:
            path.write_text(txt)
            fetched += 1
        time.sleep(C.THROTTLE_SECONDS)
        if i % 100 == 0:
            print(f"    {i}/{len(ids)}  (fetched {fetched}, cached {skipped}, missing {missing})")
    print(f"[players] done: fetched {fetched}, cached {skipped}, missing {missing}")
    return {"fetched": fetched, "cached": skipped, "missing": missing}


# --- HTML TOI reports (shift fallback when the JSON feed is empty) ------------
def _json_shifts_empty(gid: int) -> bool:
    """True if the JSON shiftchart for a game is missing or carries no shift rows."""
    p = C.RAW_SHIFTS / f"{gid}.json"
    if not p.exists():
        return True
    try:
        return len(json.loads(p.read_text()).get("data", [])) == 0
    except (json.JSONDecodeError, OSError):
        return True


def download_html_shifts(season: int) -> dict:
    """Fetch TH/TV HTML TOI reports for games whose JSON shiftchart feed is empty (the legacy
    feed stopped updating ~spring 2025). Resumable: skips reports already on disk. Returns the
    counts it prints."""
    C.ensure_dirs()
    sess = _session()
    gids = sorted(
        int(p.stem)
        for p in C.RAW_PBP.glob("*.json")
        if int(str(p.stem)[:4]) == season and _json_shifts_empty(int(p.stem))
    )
    print(f"[htmlshifts] {C.season_label(season)}: {len(gids)} games need HTML TOI reports")
    fetched = skipped = missing = 0
    for i, gid in enumerate(gids, 1):
        for hv in ("H", "V"):
            path = C.RAW_HTMLSHIFTS / f"T{hv}{C.game6(gid)}.HTM"
            if path.exists() and path.stat().st_size > 1000:
                skipped += 1
                continue
            txt = _get(
                sess,
                C.NHL_HTML_SHIFTS_URL.format(
                    season8=C.nhl_season8(season), hv=hv, game6=C.game6(gid)
                ),
                binary=False,
            )
            if txt is None:
                missing += 1
                continue
            path.write_text(txt)
            fetched += 1
            time.sleep(C.THROTTLE_SECONDS)
        if i % 50 == 0:
            print(f"    {i}/{len(gids)}  (fetched {fetched}, cached {skipped}, missing {missing})")
    print(
        f"[htmlshifts] {C.season_label(season)} done: fetched {fetched}, cached {skipped}, missing {missing}"
    )
    return {"fetched": fetched, "cached": skipped, "missing": missing}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Download raw NHL data -> data/raw/")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("games", help="download NHL shiftcharts+pbp for a season")
    g.add_argument("--season", type=int, required=True)
    g.add_argument("--limit", type=int, default=None)
    g.add_argument(
        "--include-unfinished",
        action="store_true",
        help="fetch every scheduled game, not just finished (OFF/FINAL) ones",
    )
    sub.add_parser("handedness", help="download NHL player landing json (handedness)")
    hs = sub.add_parser(
        "htmlshifts", help="download HTML TOI reports for games with an empty JSON shift feed"
    )
    hs.add_argument("--season", type=int, default=None)
    sub.add_parser("all", help="games + HTML shift fallback + handedness, every configured season")

    args = p.parse_args(argv)
    if args.cmd == "games":
        download_games(args.season, args.limit, only_final=not args.include_unfinished)
    elif args.cmd == "handedness":
        download_handedness()
    elif args.cmd == "htmlshifts":
        for season in [args.season] if args.season else C.SEASONS:
            download_html_shifts(season)
    elif args.cmd == "all":
        for season in C.SEASONS:
            download_games(season)
            download_html_shifts(season)
        download_handedness()


if __name__ == "__main__":
    main()
