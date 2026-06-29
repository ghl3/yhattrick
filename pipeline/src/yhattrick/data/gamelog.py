"""Per-player game log: one row per (player, game) with that game's headline box-score line.

Built from the typed interim tables (events for scoring/penalties, shifts for ice time and team)
joined to the games index for date / opponent / score — no raw pbp re-read. Feeds the player page's
"Game log" panel via export_players (which embeds each player's rows, most-recent-first).

Output: processed/gamelog/<season>.parquet — columns:
  player_id, game_id, season, date, team, opp, home, gf, ga, result, toi_s, g, a, p, sog, pen

Usage:
  uv run python -m yhattrick.gamelog                # all seasons with data
  uv run python -m yhattrick.gamelog --season 2024
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from .. import config as C

# events whose primary_player_id is a "shot on goal" for the shooter (goals count as shots on goal)
_SOG_TYPES = ("shot-on-goal", "goal")


def _games_meta() -> dict[int, dict]:
    """game_id -> {date, home, away, hs, as} from the already-built games index."""
    path = C.SITE_JSON / "games.json"
    if not path.exists():
        raise SystemExit(f"{path} not found — run `make games` first")
    out = {}
    for g in json.loads(path.read_text()):
        out[int(g["game_id"])] = {
            "date": g.get("date"), "home": g["home"], "away": g["away"],
            "hs": g.get("home_score"), "as": g.get("away_score"),
        }
    return out


def _counts(ev: pd.DataFrame) -> pd.DataFrame:
    """Per-(game, player) goals / assists / shots-on-goal / penalties from the events table.
    Shootout (period >= 5) is excluded so it never inflates flow-play totals."""
    ev = ev[ev.period < 5]
    goals = ev[ev.type == "goal"]

    def by(df, col, name):
        s = df.dropna(subset=[col]).groupby(["nhl_game_id", col]).size()
        s.index = s.index.set_names(["nhl_game_id", "player_id"])
        return s.rename(name)

    g = by(goals, "primary_player_id", "g")
    a1 = by(goals, "assist1_player_id", "a1")
    a2 = by(goals, "assist2_player_id", "a2")
    sog = by(ev[ev.type.isin(_SOG_TYPES)], "primary_player_id", "sog")
    pen = by(ev[ev.type == "penalty"], "primary_player_id", "pen")
    out = pd.concat([g, a1, a2, sog, pen], axis=1).fillna(0).reset_index()
    out["player_id"] = out.player_id.astype("int64")
    for c in ("g", "a1", "a2", "sog", "pen"):
        out[c] = out[c].astype("int64")
    return out


def build_season(season: int, meta: dict[int, dict]) -> int:
    ev_p = C.INTERIM / "events" / f"{season}.parquet"
    sh_p = C.INTERIM / "shifts" / f"{season}.parquet"
    if not (ev_p.exists() and sh_p.exists()):
        print(f"[gamelog] {season}: missing events/shifts, skipping")
        return 0
    ev = pd.read_parquet(ev_p)
    sh = pd.read_parquet(sh_p)

    # one row per (game, player) for everyone who took a shift; counts left-joined (0 where none)
    base = (sh.groupby(["nhl_game_id", "player_id"])
              .agg(team=("team", "first"), toi_s=("duration_s", "sum")).reset_index())
    base["player_id"] = base.player_id.astype("int64")
    df = base.merge(_counts(ev), on=["nhl_game_id", "player_id"], how="left")
    for c in ("g", "a1", "a2", "sog", "pen"):
        df[c] = df[c].fillna(0).astype("int64")
    df["a"] = df.a1 + df.a2
    df["p"] = df.g + df.a

    # games that reached overtime (any event in period >= 4) -> a regulation-tie loss is an OT loss
    ot_games = set(ev[ev.period >= 4].nhl_game_id.unique())

    rows = []
    for r in df.itertuples():
        m = meta.get(int(r.nhl_game_id))
        if not m:
            continue
        is_home = r.team == m["home"]
        gf = m["hs"] if is_home else m["as"]
        ga = m["as"] if is_home else m["hs"]
        if gf is None or ga is None:
            result = None
        elif gf > ga:
            result = "W"
        elif gf < ga:
            result = "OTL" if int(r.nhl_game_id) in ot_games else "L"
        else:
            result = "T"
        rows.append({
            "player_id": int(r.player_id), "game_id": int(r.nhl_game_id), "season": season,
            "date": m["date"], "team": r.team, "opp": m["away"] if is_home else m["home"],
            "home": bool(is_home), "gf": gf, "ga": ga, "result": result,
            "toi_s": int(r.toi_s), "g": int(r.g), "a": int(r.a), "p": int(r.p),
            "sog": int(r.sog), "pen": int(r.pen),
        })
    out = C.PROCESSED / "gamelog" / f"{season}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"[gamelog] {C.season_label(season)}: {len(rows):,} player-games -> {out.name}")
    return len(rows)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Per-player game log from events + shifts")
    p.add_argument("--season", type=int, default=None)
    args = p.parse_args(argv)
    meta = _games_meta()
    for s in ([args.season] if args.season else C.SEASONS):
        if (C.INTERIM / "events" / f"{s}.parquet").exists():
            build_season(s, meta)


if __name__ == "__main__":
    main()
