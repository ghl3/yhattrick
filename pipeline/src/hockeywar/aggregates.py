"""Per-player, per-season descriptive box score, derived entirely from our own NHL data
(raw play-by-play for event counts; shifts for time-on-ice, games played, and team).

No MoneyPuck here — this is the non-borrowed descriptive layer. Each NHL pbp event exposes the
role-specific player ids we need (scorer, assisters, shooter, blocker, hitter, faceoff winner,
penalty committer/drawer, give/takeaway player), so the counting stats are attributed correctly.

Output: interim/box/<season>.parquet — one row per player-season.

Usage:
  uv run python -m hockeywar.aggregates                 # all seasons with data
  uv run python -m hockeywar.aggregates --season 2021
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import pandas as pd

from . import config as C
from .clean import _downloaded_game_ids

# counting stats tallied from pbp (points is derived). so_* are shootout-only and kept separate
# from the flow-play totals (shootout is a tiebreaker, not real-play offence).
STAT_KEYS = ["g", "a1", "a2", "sog", "icf", "blocks", "hits", "takeaways", "giveaways",
             "fo_won", "fo_lost", "pen_taken", "pen_drawn", "so_g", "so_att"]


def _tally(box: dict, pid, key: str) -> None:
    if pid is not None:
        box[int(pid)][key] += 1


def event_counts(gids: list[int]) -> pd.DataFrame:
    box: dict[int, Counter] = defaultdict(Counter)
    for gid in gids:
        plays = json.loads((C.RAW_PBP / f"{gid}.json").read_text()).get("plays", [])
        for pl in plays:
            t, d = pl.get("typeDescKey"), pl.get("details", {})
            if pl.get("periodDescriptor", {}).get("periodType") == "SO":
                # shootout: tracked as its own stat, never mixed into flow-play totals
                if t == "goal":
                    _tally(box, d.get("scoringPlayerId"), "so_g")
                    _tally(box, d.get("scoringPlayerId"), "so_att")
                elif t in ("shot-on-goal", "missed-shot"):
                    _tally(box, d.get("shootingPlayerId"), "so_att")
                continue
            if t == "goal":
                _tally(box, d.get("scoringPlayerId"), "g")
                _tally(box, d.get("scoringPlayerId"), "sog")
                _tally(box, d.get("scoringPlayerId"), "icf")
                _tally(box, d.get("assist1PlayerId"), "a1")
                _tally(box, d.get("assist2PlayerId"), "a2")
            elif t == "shot-on-goal":
                _tally(box, d.get("shootingPlayerId"), "sog")
                _tally(box, d.get("shootingPlayerId"), "icf")
            elif t == "missed-shot":
                _tally(box, d.get("shootingPlayerId"), "icf")
            elif t == "blocked-shot":
                _tally(box, d.get("blockingPlayerId"), "blocks")
                _tally(box, d.get("shootingPlayerId"), "icf")
            elif t == "hit":
                _tally(box, d.get("hittingPlayerId"), "hits")
            elif t == "faceoff":
                _tally(box, d.get("winningPlayerId"), "fo_won")
                _tally(box, d.get("losingPlayerId"), "fo_lost")
            elif t == "penalty":
                _tally(box, d.get("committedByPlayerId"), "pen_taken")
                _tally(box, d.get("drawnByPlayerId"), "pen_drawn")
            elif t == "giveaway":
                _tally(box, d.get("playerId"), "giveaways")
            elif t == "takeaway":
                _tally(box, d.get("playerId"), "takeaways")
    rows = []
    for pid, c in box.items():
        row = {"player_id": pid, **{k: int(c.get(k, 0)) for k in STAT_KEYS}}
        row["points"] = row["g"] + row["a1"] + row["a2"]
        rows.append(row)
    return pd.DataFrame(rows)


def ice_and_team(sh: pd.DataFrame) -> pd.DataFrame:
    """GP, TOI and team(s) per player from a (game-type-filtered) shift table."""
    g = sh.groupby("player_id").agg(gp=("nhl_game_id", "nunique"), toi_s=("duration_s", "sum"))
    by_team = (sh.groupby(["player_id", "team"]).nhl_game_id.nunique()
               .reset_index().sort_values("nhl_game_id", ascending=False))
    primary = by_team.groupby("player_id").team.first()
    teams = by_team.groupby("player_id").team.apply(list)
    return pd.DataFrame({"gp": g.gp, "toi_s": g.toi_s, "team": primary, "teams": teams}).reset_index()


def _box_for(gids, sh, roster, season, game_type) -> pd.DataFrame:
    counts = event_counts(gids)
    box = ice_and_team(sh).merge(counts, on="player_id", how="left").merge(roster, on="player_id", how="left")
    for k in STAT_KEYS + ["points"]:
        box[k] = box[k].fillna(0).astype("int32")
    box["season"] = season
    box["game_type"] = game_type     # regular | playoff -- kept separate, never mixed
    box["toi_min"] = (box.toi_s / 60.0).round(1)
    box.rename(columns={"player_name": "name", "position": "pos"}, inplace=True)
    return box


def build_season(season: int) -> int:
    gids = _downloaded_game_ids(season)
    if not gids:
        print(f"[box] {season}: no games on disk")
        return 0
    sh = pd.read_parquet(C.INTERIM / "shifts" / f"{season}.parquet")
    roster = pd.read_parquet(C.INTERIM / "roster" / f"{season}.parquet")[["player_id", "player_name", "position"]]
    reg = [g for g in gids if C.is_regular_season(g)]
    post = [g for g in gids if not C.is_regular_season(g)]

    frames = []
    for game_type, gg in (("regular", reg), ("playoff", post)):
        if not gg:
            continue
        frames.append(_box_for(gg, sh[sh.nhl_game_id.isin(gg)], roster, season, game_type))
    box = pd.concat(frames, ignore_index=True)

    out = C.INTERIM / "box" / f"{season}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    box.to_parquet(out, index=False)
    for gt, sub in box.groupby("game_type"):
        print(f"[box] {C.season_label(season)} {gt}: {len(sub)} players, {sub.gp.sum()} player-games")
    return len(box)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Per-player-season box score from our NHL data")
    p.add_argument("--season", type=int, default=None)
    args = p.parse_args(argv)
    seasons = [args.season] if args.season else C.SEASONS
    for s in seasons:
        if (C.INTERIM / "shifts" / f"{s}.parquet").exists():
            build_season(s)


if __name__ == "__main__":
    main()
