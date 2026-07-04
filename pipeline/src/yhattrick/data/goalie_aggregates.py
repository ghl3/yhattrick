"""Per-goalie, per-season descriptive box + splits — the goalie counterpart of aggregates.py.

GP / starts / shutouts / QS / RBS / TOI come from the per-game goalie log; the shot-derived totals
and the splits (by danger, by situation EV/PK/PP, by shot type) are accumulated from the shot-level
on-ice table as raw additive sums, so they pool cleanly across seasons in export (rates are formed
there, like the skater on-ice sums).

Universes / conventions:
  - GSAx is xGA − GA over the modeled **Fenwick** set (unblocked shots faced) — the only universe on
    which the xG model is calibrated (Σ xg ≈ Σ goals), so GSAx is league-neutral. Every split's GSAx
    is Fenwick too, so the danger / situation / shot-type GSAx all reconcile to the headline.
  - Save % / saves / shots-against (for display) use **shots on goal** (event ∈ {shot-on-goal, goal}).
  - Situation comes from each shot's own play-by-play situationCode skater counts (`sit_home_n` /
    `sit_away_n`), NOT the stint strength — a power-play goal sits on the stint boundary where the
    post-goal even-strength stint begins, which would misfile it as EV.
  - Danger buckets (xG): low < 0.05, med [0.05, 0.15), high ≥ 0.15.
  - Empty-net / penalty / playoff shots carry no xG / facing goalie and never attribute.

Outputs (regular season):
  processed/goalie_box/<season>.parquet       one row per goalie-season (overall + danger + situation)
  processed/goalie_shottype/<season>.parquet  long: (player_id, season, shot_type, sa, xga, ga, sog)

Usage:
  uv run python -m yhattrick.goalie_aggregates                 # all seasons with data
  uv run python -m yhattrick.goalie_aggregates --season 2024
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .. import config as C

_SOG = ("shot-on-goal", "goal")
HD_THRESH = 0.15
MD_THRESH = 0.05
DANGER = ("ld", "md", "hd")
SITUATIONS = ("ev", "pk", "pp")


def _facing_shots(season: int, goalie_ids: set[int]) -> pd.DataFrame:
    """Regular-season modeled shots with facing goalie, danger bucket and situation attached."""
    p = C.PROCESSED / "shots_onice" / f"{season}.parquet"
    cols = [
        "nhl_game_id",
        "is_home",
        "home_goalie",
        "away_goalie",
        "xg",
        "goal",
        "event",
        "shot_type",
        "sit_home_n",
        "sit_away_n",
    ]
    df = pd.read_parquet(p, columns=cols)
    df = df[df.nhl_game_id.map(C.is_regular_season)]
    df["goalie_id"] = np.where(df.is_home == 1, df.away_goalie, df.home_goalie)
    df = df[df.goalie_id.notna() & df.xg.notna()].copy()
    df["goalie_id"] = df.goalie_id.astype("int64")
    df["is_sog"] = df.event.isin(_SOG)
    df["bucket"] = np.where(df.xg < MD_THRESH, "ld", np.where(df.xg < HD_THRESH, "md", "hd"))
    # situation from the shot's own situationCode skater counts (goalie's team vs the shooting team)
    dh, da = df.sit_home_n, df.sit_away_n
    def_sk = np.where(df.is_home == 1, da, dh)  # goalie (defending) team skaters
    opp_sk = np.where(df.is_home == 1, dh, da)  # shooting team skaters
    sit = np.where(def_sk == opp_sk, "ev", np.where(def_sk < opp_sk, "pk", "pp"))
    valid = dh.notna().values & da.notna().values
    df["sit"] = np.where(valid, sit, "ev")  # rare unverifiable strength -> even
    return df


def _split_wide(df: pd.DataFrame, key: str, levels) -> pd.DataFrame:
    """Per goalie: Fenwick SA / xGA / GA and the SOG count within each level of `key`, wide."""
    fen = (
        df.groupby(["goalie_id", key])
        .agg(sa=("event", "size"), xga=("xg", "sum"), ga=("goal", "sum"))
        .reset_index()
    )
    sog = df[df.is_sog].groupby(["goalie_id", key]).size().rename("sog").reset_index()
    out = pd.DataFrame({"player_id": sorted(df.goalie_id.unique())}).set_index("player_id")
    for lv in levels:
        f = fen[fen[key] == lv].set_index("goalie_id")
        so = sog[sog[key] == lv].set_index("goalie_id")
        out[f"{lv}_sa"] = f.sa
        out[f"{lv}_xga"] = f.xga
        out[f"{lv}_ga"] = f.ga
        out[f"{lv}_sog"] = so.sog
    return out.fillna(0.0).reset_index()


def _shottype_long(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Long per (goalie, shot_type): Fenwick SA / xGA / GA and SOG count."""
    sub = df.dropna(subset=["shot_type"])
    fen = (
        sub.groupby(["goalie_id", "shot_type"])
        .agg(sa=("event", "size"), xga=("xg", "sum"), ga=("goal", "sum"))
        .reset_index()
    )
    sog = sub[sub.is_sog].groupby(["goalie_id", "shot_type"]).size().rename("sog").reset_index()
    g = fen.merge(sog, on=["goalie_id", "shot_type"], how="left").fillna({"sog": 0})
    g = g.rename(columns={"goalie_id": "player_id"})
    g["season"] = season
    return g[["player_id", "season", "shot_type", "sa", "xga", "ga", "sog"]]


def build_season(season: int) -> int:
    gl_p = C.PROCESSED / "goalie_gamelog" / f"{season}.parquet"
    so_p = C.PROCESSED / "shots_onice" / f"{season}.parquet"
    ro_p = C.INTERIM / "roster" / f"{season}.parquet"
    if not (gl_p.exists() and so_p.exists() and ro_p.exists()):
        print(f"[goalie-box] {season}: missing inputs (run goalie-gamelog first), skipping")
        return 0
    gl = pd.read_parquet(gl_p)
    if not len(gl):
        print(f"[goalie-box] {season}: empty game log, skipping")
        return 0
    roster = pd.read_parquet(ro_p)[["player_id", "player_name", "position"]]
    goalie_ids = set(int(p) for p in roster.loc[roster.position == "G", "player_id"])

    # games / starts / shutouts / QS / RBS / TOI come from the per-game log
    overall = (
        gl.groupby("player_id")
        .agg(
            gp=("game_id", "nunique"),
            starts=("started", "sum"),
            shutouts=("shutout", "sum"),
            qs=("qs", "sum"),
            rbs=("rbs", "sum"),
            toi_s=("toi_s", "sum"),
        )
        .reset_index()
    )
    by_team = (
        gl.groupby(["player_id", "team"])
        .game_id.nunique()
        .reset_index()
        .sort_values("game_id", ascending=False)
    )
    primary = by_team.groupby("player_id").team.first()
    teams = by_team.groupby("player_id").team.apply(list)
    overall = overall.merge(
        pd.DataFrame({"team": primary, "teams": teams}).reset_index(), on="player_id", how="left"
    )

    # shot-derived totals (Fenwick for xGA/GSAx, SOG for saves/Sv%) from the shot table
    df = _facing_shots(season, goalie_ids)
    sa = (
        df.groupby("goalie_id")
        .agg(sa=("event", "size"), xga=("xg", "sum"), ga=("goal", "sum"))
        .reset_index()
    )
    sog = (
        df[df.is_sog]
        .groupby("goalie_id")
        .agg(sog_against=("event", "size"), xga_sog=("xg", "sum"))
        .reset_index()
    )
    shots = sa.merge(sog, on="goalie_id", how="left").rename(columns={"goalie_id": "player_id"})
    shots["saves"] = shots.sog_against - shots.ga
    shots["gsax"] = shots.xga - shots.ga

    box = (
        overall.merge(shots, on="player_id", how="left")
        .merge(_split_wide(df, "bucket", DANGER), on="player_id", how="left")
        .merge(_split_wide(df, "sit", SITUATIONS), on="player_id", how="left")
    )
    names = roster.rename(columns={"player_name": "name", "position": "pos"})
    box = box.merge(names[["player_id", "name"]], on="player_id", how="left")
    box["season"] = season
    fill = (
        ["sa", "xga", "ga", "sog_against", "xga_sog", "saves", "gsax"]
        + [f"{b}_{m}" for b in DANGER for m in ("sa", "xga", "ga", "sog")]
        + [f"{s}_{m}" for s in SITUATIONS for m in ("sa", "xga", "ga", "sog")]
    )
    for c in fill:
        box[c] = box[c].fillna(0.0)

    out = C.PROCESSED / "goalie_box" / f"{season}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    box.to_parquet(out, index=False)
    st = _shottype_long(df, season)
    st_out = C.PROCESSED / "goalie_shottype" / f"{season}.parquet"
    st_out.parent.mkdir(parents=True, exist_ok=True)
    st.to_parquet(st_out, index=False)
    print(
        f"[goalie-box] {C.season_label(season)}: {len(box)} goalies, "
        f"{int(box.gp.sum())} goalie-games -> {out.name} (+ goalie_shottype)"
    )
    return len(box)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Per-goalie-season descriptive box + splits")
    p.add_argument("--season", type=int, default=None)
    args = p.parse_args(argv)
    for s in [args.season] if args.season else C.SEASONS:
        if (C.PROCESSED / "goalie_gamelog" / f"{s}.parquet").exists():
            build_season(s)


if __name__ == "__main__":
    main()
