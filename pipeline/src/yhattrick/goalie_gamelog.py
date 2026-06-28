"""Per-goalie game log: one row per (goalie, game) with that game's goaltending line.

Built from the shot-level on-ice table (each Fenwick shot carries its facing goalie + xG) joined to
shifts (ice time, team, who started) and the games index (date / opponent / score / OT). The skater
game log (gamelog.py) tallies G/A/SOG/PEN for the shooter; this is its goalie counterpart: shots
faced, saves, goals against, xGA, GSAx, the decision, and the quality-start flags.

Decisions / conventions (documented; see plan):
  - start    = the goalie with the earliest first shift for his team in that game
  - decision = the team's W/L/OTL credited to the starter only (relief goalies get null) — the true
               goalie-of-record rule needs per-stint goal timing, out of scope for v1
  - shutout  = a start where the goalie was his team's only goalie and the team allowed 0 goals
  - QS / RBS = Hockey-Reference quality / really-bad start (on the game's actual save %), starters only
  - empty-net / penalty / playoff shots carry no xG and no facing goalie, so they never attribute

Output: processed/goalie_gamelog/<season>.parquet — columns:
  player_id, game_id, season, date, team, opp, home, decision, toi_s,
  sog_against, saves, ga, xga, gsax, started, shutout, qs, rbs

Usage:
  uv run python -m yhattrick.goalie_gamelog                # all seasons with data
  uv run python -m yhattrick.goalie_gamelog --season 2024
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import config as C

_SOG = ("shot-on-goal", "goal")
# Hockey-Reference quality-start thresholds (on the game's actual save percentage)
QS_SVPCT = 0.917
QS_LOWVOL_SHOTS = 20
QS_LOWVOL_GA = 2
RBS_SVPCT = 0.850


def _games_meta() -> dict[int, dict]:
    """game_id -> {date, home, away, hs, as, ot} from the already-built games index."""
    path = C.SITE_JSON / "games.json"
    if not path.exists():
        raise SystemExit(f"{path} not found — run `make games` first")
    out = {}
    for g in json.loads(path.read_text()):
        out[int(g["game_id"])] = {
            "date": g.get("date"), "home": g["home"], "away": g["away"],
            "hs": g.get("home_score"), "as": g.get("away_score"), "ot": bool(g.get("ot")),
        }
    return out


def _shot_lines(season: int, goalie_ids: set[int]) -> pd.DataFrame:
    """Per (game, facing goalie) shot aggregates over the regular-season modeled set."""
    p = C.PROCESSED / "shots_onice" / f"{season}.parquet"
    cols = ["nhl_game_id", "is_home", "home_goalie", "away_goalie", "xg", "goal", "event"]
    df = pd.read_parquet(p, columns=cols)
    df = df[df.nhl_game_id.map(C.is_regular_season)]
    df["goalie_id"] = np.where(df.is_home == 1, df.away_goalie, df.home_goalie)
    df = df[df.goalie_id.notna() & df.xg.notna()].copy()
    df["goalie_id"] = df.goalie_id.astype("int64")
    df["is_sog"] = df.event.isin(_SOG)
    g = df.groupby(["nhl_game_id", "goalie_id"]).agg(
        sa=("event", "size"), xga=("xg", "sum"), ga=("goal", "sum"),
        sog_against=("is_sog", "sum")).reset_index()
    g = g.rename(columns={"goalie_id": "player_id"})
    g["saves"] = g.sog_against - g.ga
    g["gsax"] = g.xga - g.ga
    return g


def build_season(season: int, meta: dict[int, dict]) -> int:
    sh_p = C.INTERIM / "shifts" / f"{season}.parquet"
    ro_p = C.INTERIM / "roster" / f"{season}.parquet"
    so_p = C.PROCESSED / "shots_onice" / f"{season}.parquet"
    if not (sh_p.exists() and ro_p.exists() and so_p.exists()):
        print(f"[goalie-gamelog] {season}: missing inputs, skipping")
        return 0
    roster = pd.read_parquet(ro_p)
    goalie_ids = set(int(p) for p in roster.loc[roster.position == "G", "player_id"])
    sh = pd.read_parquet(sh_p)
    sh = sh[sh.player_id.isin(goalie_ids) & sh.nhl_game_id.map(C.is_regular_season)]

    # goalie-games from shifts: ice time, team, first shift (for start detection)
    base = (sh.groupby(["nhl_game_id", "player_id"])
              .agg(team=("team", "first"), toi_s=("duration_s", "sum"),
                   first_start=("start_g", "min")).reset_index())
    base["player_id"] = base.player_id.astype("int64")
    # starter per (game, team) = earliest first shift; count goalies per team for shutout eligibility
    base = base.sort_values(["nhl_game_id", "team", "first_start"])
    base["started"] = ~base.duplicated(["nhl_game_id", "team"], keep="first")
    n_goalies = base.groupby(["nhl_game_id", "team"]).player_id.transform("size")
    base["sole_goalie"] = n_goalies == 1

    lines = _shot_lines(season, goalie_ids)
    df = base.merge(lines, on=["nhl_game_id", "player_id"], how="left")
    for c in ("sa", "xga", "ga", "sog_against", "saves", "gsax"):
        df[c] = df[c].fillna(0.0)

    rows = []
    for r in df.itertuples():
        m = meta.get(int(r.nhl_game_id))
        if not m:
            continue
        is_home = r.team == m["home"]
        gf = m["hs"] if is_home else m["as"]
        team_ga = m["as"] if is_home else m["hs"]            # goals the team allowed all game
        if gf is None or team_ga is None:
            result = None
        elif gf > team_ga:
            result = "W"
        elif gf < team_ga:
            result = "OTL" if m["ot"] else "L"
        else:
            result = "T"
        sog, ga, saves = int(r.sog_against), int(r.ga), int(r.saves)
        game_sv = saves / sog if sog > 0 else None
        started = bool(r.started)
        qs = bool(started and game_sv is not None and (
            game_sv >= QS_SVPCT or (sog < QS_LOWVOL_SHOTS and ga <= QS_LOWVOL_GA)))
        rbs = bool(started and game_sv is not None and game_sv < RBS_SVPCT)
        shutout = bool(started and r.sole_goalie and team_ga == 0)
        rows.append({
            "player_id": int(r.player_id), "game_id": int(r.nhl_game_id), "season": season,
            "date": m["date"], "team": r.team, "opp": m["away"] if is_home else m["home"],
            "home": bool(is_home), "decision": result if started else None,
            "toi_s": int(r.toi_s), "sa": int(r.sa), "sog_against": sog, "saves": saves, "ga": ga,
            "xga": round(float(r.xga), 3), "gsax": round(float(r.gsax), 3),
            "started": started, "shutout": shutout, "qs": qs, "rbs": rbs,
        })
    out = C.PROCESSED / "goalie_gamelog" / f"{season}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"[goalie-gamelog] {C.season_label(season)}: {len(rows):,} goalie-games -> {out.name}")
    return len(rows)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Per-goalie game log from shots_onice + shifts")
    p.add_argument("--season", type=int, default=None)
    args = p.parse_args(argv)
    meta = _games_meta()
    for s in ([args.season] if args.season else C.SEASONS):
        if (C.PROCESSED / "shots_onice" / f"{s}.parquet").exists():
            build_season(s, meta)


if __name__ == "__main__":
    main()
