"""Stage 0: pull raw data into data/raw/ (immutable). Resumable + throttled.

Three kinds of raw artifact:
  - MoneyPuck shots_<season>.zip  (per-shot rows + borrowed xGoal; also the game list)
  - MoneyPuck skaters_<season>.csv (season box-score, validation totals)
  - NHL shiftcharts/<gameId>.json + pbp/<gameId>.json  (on-ice identities + events)

The shots zip's unique game_ids ARE the authoritative game list, so we derive which NHL
games to fetch from it -- no schedule API needed. Every fetch checks for an existing file
first, so re-running only pulls what's missing (resumable).

Usage:
  uv run python -m yhattrick.download moneypuck            # all season zips + skaters
  uv run python -m yhattrick.download games --season 2024  # shiftcharts+pbp for that season
  uv run python -m yhattrick.download games --season 2024 --limit 25
  uv run python -m yhattrick.download all                  # everything, all seasons
"""
from __future__ import annotations

import argparse
import io
import sys
import time
import zipfile

import pandas as pd
import requests

from . import config as C


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


# --- MoneyPuck ---------------------------------------------------------------
def download_moneypuck(seasons=C.SEASONS) -> None:
    C.ensure_dirs()
    sess = _session()
    for season in seasons:
        dest = C.RAW_MONEYPUCK / f"shots_{season}.zip"
        if dest.exists():
            print(f"[shots] {season}: cached ({dest.stat().st_size // 1024} KB)")
        else:
            url = C.MONEYPUCK_SHOTS_URL.format(season=season)
            print(f"[shots] {season}: downloading {url}")
            data = _get(sess, url, binary=True)
            if data is None:
                print(f"    ! 404 for shots_{season} -- skipping")
                continue
            dest.write_bytes(data)
            print(f"    saved {len(data) // 1024} KB")

        skaters = C.RAW_MONEYPUCK / f"skaters_{season}.csv"
        if skaters.exists():
            print(f"[skaters] {season}: cached")
        else:
            url = C.MONEYPUCK_SKATERS_URL.format(season=season)
            txt = _get(sess, url, binary=False)
            if txt is None:
                print(f"    ! skaters_{season} not available (404)")
            else:
                skaters.write_text(txt)
                print(f"[skaters] {season}: saved {len(txt) // 1024} KB")
        time.sleep(C.THROTTLE_SECONDS)


def game_ids_for_season(season: int) -> list[int]:
    """Unique NHL gameIds present in the MoneyPuck shots zip for a season."""
    zpath = C.RAW_MONEYPUCK / f"shots_{season}.zip"
    if not zpath.exists():
        raise FileNotFoundError(f"{zpath} missing -- run `download moneypuck` first")
    zf = zipfile.ZipFile(zpath)
    with zf.open(zf.namelist()[0]) as fh:
        mp_ids = pd.read_csv(fh, usecols=["game_id"]).game_id.unique()
    return sorted(C.mp_to_nhl_game_id(int(m), season) for m in mp_ids)


# --- NHL per-game (shiftcharts + pbp) ----------------------------------------
def download_games(season: int, limit: int | None = None) -> None:
    C.ensure_dirs()
    sess = _session()
    game_ids = game_ids_for_season(season)
    if limit:
        game_ids = game_ids[:limit]
    print(f"[games] {C.season_label(season)}: {len(game_ids)} games to ensure")

    fetched = skipped = missing = 0
    for i, gid in enumerate(game_ids, 1):
        sc_path = C.RAW_SHIFTS / f"{gid}.json"
        pbp_path = C.RAW_PBP / f"{gid}.json"
        need = []
        if not sc_path.exists():
            need.append(("shift", C.NHL_SHIFTCHARTS_URL.format(game_id=gid), sc_path))
        if not pbp_path.exists():
            need.append(("pbp", C.NHL_PBP_URL.format(game_id=gid), pbp_path))
        if not need:
            skipped += 1
            continue
        for kind, url, path in need:
            txt = _get(sess, url, binary=False)
            if txt is None:
                missing += 1
                print(f"    ! {kind} 404 for {gid}")
                continue
            path.write_text(txt)
            fetched += 1
            time.sleep(C.THROTTLE_SECONDS)
        if i % 50 == 0:
            print(f"    {i}/{len(game_ids)}  (fetched {fetched}, cached {skipped}, missing {missing})")
    print(f"[games] {C.season_label(season)} done: fetched {fetched}, cached {skipped}, missing {missing}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Download raw hockey data -> data/raw/")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("moneypuck", help="download MoneyPuck shots zips + skaters csvs")
    g = sub.add_parser("games", help="download NHL shiftcharts+pbp for a season")
    g.add_argument("--season", type=int, required=True)
    g.add_argument("--limit", type=int, default=None)
    sub.add_parser("all", help="moneypuck + games for every configured season")

    args = p.parse_args(argv)
    if args.cmd == "moneypuck":
        download_moneypuck()
    elif args.cmd == "games":
        download_games(args.season, args.limit)
    elif args.cmd == "all":
        download_moneypuck()
        for season in C.SEASONS:
            download_games(season)


if __name__ == "__main__":
    main()
