"""Goal accounting: the additive-theory reconciliation. Proves the pieces sum to actual goals.

See docs/modeling.md. The additive theory says every goal is chance + conversion, with the
conversion residual split between shooter (finishing) and goalie (saving). Per shot the shooting
team's expected goal contribution is

  goal_i  ≈  xG_i  +  mu  +  finishing_i  +  goalie_i           (finishing_i = alpha_shooter,
                                                                 goalie_i   = gamma_goalie)

where mu is the shooting_model intercept (the tiny average-shot baseline that makes the books
balance, see shooting_model.py). The shooting_model fit guarantees this holds EXACTLY in aggregate
(the unpenalized intercept forces sum(residual)=0), so league-wide

  actual goals = sum(xG) + sum(mu) + sum(finishing) + sum(goalie)

This module rolls that identity up to team-season and league level and reports the reconciliation
gap (actual − reconstructed) — which, for any subset, is that subset's unmodeled residual (finishing
/ goaltending luck the season-long player effects don't capture). The chance term is *raw xG*, the
same quantity RAPM (player_onice_model) distributes among the on-ice skaters as creation /
suppression — so this closes the loop: team xGF (created by skaters) + finishing − opponent goalie
saves = goals for; team xGA (allowed by skaters) + opponent finishing − own goalie saves = goals
against.

Usage:
  uv run python -m yhattrick.models.goal_accounting           # all seasons, pooled
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd

from .. import config as C
from . import shooting_model
from .player_onice_model import roster_names


def _team_maps(seasons) -> tuple[dict, dict]:
    """nhl_game_id -> home / away team abbrev (from interim events)."""
    home, away = {}, {}
    for s in seasons:
        p = C.INTERIM / "events" / f"{s}.parquet"
        if not p.exists():
            continue
        ev = pd.read_parquet(p, columns=["nhl_game_id", "team", "is_home"])
        home.update(ev[ev.is_home].groupby("nhl_game_id").team.first().to_dict())
        away.update(ev[~ev.is_home].groupby("nhl_game_id").team.first().to_dict())
    return home, away


def load_shots_teamed(seasons) -> pd.DataFrame:
    """Modeled shots with the shooting (off) and defending (def) team attached."""
    home, away = _team_maps(seasons)
    frames = []
    for s in seasons:
        p = C.PROCESSED / "shots_onice" / f"{s}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(
            p,
            columns=[
                "nhl_game_id",
                "is_home",
                "home_goalie",
                "away_goalie",
                "shooter_id",
                "xg",
                "goal",
            ],
        )
        df["goalie_id"] = np.where(df.is_home == 1, df.away_goalie, df.home_goalie)
        df = df[df.shooter_id.notna() & df.goalie_id.notna() & df.xg.notna()].copy()
        df["season"] = s
        ht, at = df.nhl_game_id.map(home), df.nhl_game_id.map(away)
        df["off_team"] = np.where(df.is_home == 1, ht, at)
        df["def_team"] = np.where(df.is_home == 1, at, ht)
        frames.append(
            df[["season", "off_team", "def_team", "shooter_id", "goalie_id", "xg", "goal"]].astype(
                {"shooter_id": int, "goalie_id": int}
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def decompose(shots: pd.DataFrame, mu: float, alpha: dict, gamma: dict) -> pd.DataFrame:
    """Attach the per-shot additive decomposition ``goal ≈ xg + mu + finishing + goalie``.

    Pure (no I/O): `shots` needs `shooter_id`, `goalie_id`, `xg`; `alpha` maps shooter_id ->
    finishing goals/shot (α), `gamma` maps goalie_id -> goalie effect goals/shot (γ, < 0 when the
    goalie saves). Returns a copy with `mu`/`fin`/`gol`/`recon` columns added — `recon` is the
    reconstructed goal contribution whose league sum equals actual goals to rounding (free intercept).
    Shared by `reconcile()` (real fit) and the validator/tests (synthetic), so the identity is checked
    against the same arithmetic the pipeline ships."""
    s = shots.copy()
    s["mu"] = mu
    s["fin"] = s.shooter_id.map(alpha)
    s["gol"] = s.goalie_id.map(gamma)  # < 0 when the goalie saves
    s["recon"] = s.xg + s.mu + s.fin + s.gol  # reconstructed goal contribution
    return s


def reconcile(seasons, names=None) -> tuple[pd.DataFrame, dict]:
    """Build the per-shot additive decomposition and roll it up to team-season. Returns the team
    table (GF/GA reconstruction with components) and a league-level summary dict."""
    names = names or roster_names(seasons)
    fin, gl, meta = shooting_model.fit(seasons, names)
    if fin.empty:
        return pd.DataFrame(), {}
    mu = float(meta["reconciliation"]["intercept_per_shot"])
    alpha = dict(zip(fin.player_id, fin.fin_per100 / 100.0))  # finishing goals / shot
    gamma = dict(
        zip(gl.player_id, -gl.gsax_per100 / 100.0)
    )  # goalie effect goals / shot (gsax = -gamma)

    s = decompose(load_shots_teamed(seasons), mu, alpha, gamma)

    # offence: each team's shots -> goals for, expected goals for, finishing, opponent-goalie effect
    off = (
        s.groupby(["season", "off_team"])
        .agg(
            shots_for=("goal", "size"),
            gf=("goal", "sum"),
            xgf=("xg", "sum"),
            mu_for=("mu", "sum"),
            fin_for=("fin", "sum"),
            opp_goalie=("gol", "sum"),
            gf_recon=("recon", "sum"),
        )
        .reset_index()
        .rename(columns={"off_team": "team"})
    )
    # defence: shots faced -> goals against, expected goals against, opponent finishing, own goalie
    deff = (
        s.groupby(["season", "def_team"])
        .agg(
            shots_against=("goal", "size"),
            ga=("goal", "sum"),
            xga=("xg", "sum"),
            mu_against=("mu", "sum"),
            opp_fin=("fin", "sum"),
            own_goalie=("gol", "sum"),
            ga_recon=("recon", "sum"),
        )
        .reset_index()
        .rename(columns={"def_team": "team"})
    )
    team = off.merge(deff, on=["season", "team"], how="outer")
    team["gf_gap"] = (team.gf - team.gf_recon).round(2)  # unmodeled offence (finishing luck)
    team["ga_gap"] = (team.ga - team.ga_recon).round(2)  # unmodeled defence (goalie luck)
    for c in (
        "xgf",
        "mu_for",
        "fin_for",
        "opp_goalie",
        "gf_recon",
        "xga",
        "mu_against",
        "opp_fin",
        "own_goalie",
        "ga_recon",
    ):
        team[c] = team[c].round(2)
    team["gsax_against"] = (-team.own_goalie).round(2)  # own goalies' saves (+ = good)

    n = len(s)
    league = {
        "model": "goal_accounting",
        "seasons": sorted(int(x) for x in seasons),
        "n_shots": int(n),
        "goals": int(s.goal.sum()),
        "xg": round(float(s.xg.sum()), 1),
        "mu_total": round(mu * n, 1),
        "finishing_total": round(float(s.fin.sum()), 1),
        "goalie_total": round(float(s.gol.sum()), 1),
        "gsax_saved_total": round(float(-s.gol.sum()), 1),
        "reconstructed": round(float(s.recon.sum()), 1),
        "league_identity_resid": round(float(s.goal.sum() - s.recon.sum()), 4),
        "team_gf_gap_abs_mean": round(float(team.gf_gap.abs().mean()), 2),
        "team_gf_gap_pct_mean": round(float((team.gf_gap.abs() / team.gf).mean() * 100), 2),
        "team_ga_gap_abs_mean": round(float(team.ga_gap.abs().mean()), 2),
        "team_ga_gap_pct_mean": round(float((team.ga_gap.abs() / team.ga).mean() * 100), 2),
        "team_gf_gap_max": round(float(team.gf_gap.abs().max()), 2),
        "team_ga_gap_max": round(float(team.ga_gap.abs().max()), 2),
    }
    return team, league


def _write(team: pd.DataFrame, league: dict) -> None:
    label = "+".join(map(str, league["seasons"]))
    C.MODELS.mkdir(parents=True, exist_ok=True)
    team.to_parquet(C.MODELS / f"goal_accounting_{label}.parquet", index=False)
    C.LOGS_MODEL.mkdir(parents=True, exist_ok=True)
    (C.LOGS_MODEL / f"goal_accounting_{label}.meta.json").write_text(json.dumps(league, indent=2))
    with (C.LOGS / "model_fits.jsonl").open("a") as f:
        f.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"), **league}) + "\n")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Reconcile the additive goals identity (league + team)")
    p.add_argument("--season", type=int, default=None, help="one season (default: all available)")
    args = p.parse_args(argv)
    seasons = [args.season] if args.season else shooting_model.available_seasons()
    if not seasons:
        raise SystemExit("no processed shots available — run `make stints` first")
    team, league = reconcile(seasons)
    if not len(team):
        print("[goal_accounting] no shots — skipping")
        return
    label = "+".join(map(str, seasons))
    L = league
    print(f"\n=== goal accounting — seasons {label} : {L['n_shots']:,} shots ===")
    print(
        f"  LEAGUE identity:  goals={L['goals']} = xG={L['xg']} + μ={L['mu_total']:+.1f} "
        f"+ finishing={L['finishing_total']:+.1f} + goalie={L['goalie_total']:+.1f}  "
        f"(reconstructed={L['reconstructed']}, resid={L['league_identity_resid']:+.3f})"
    )
    print(
        f"  TEAM-SEASON reconstruction gap (actual − reconstructed; = unmodeled finishing/goalie luck):"
    )
    print(
        f"    GF gap: mean |{L['team_gf_gap_abs_mean']}| ({L['team_gf_gap_pct_mean']}%), max {L['team_gf_gap_max']}"
    )
    print(
        f"    GA gap: mean |{L['team_ga_gap_abs_mean']}| ({L['team_ga_gap_pct_mean']}%), max {L['team_ga_gap_max']}"
    )
    worst = team.reindex(team.gf_gap.abs().sort_values(ascending=False).index).head(5)
    print("  largest GF gaps (team finished hotter/colder than its shooters' season finishing):")
    for r in worst.itertuples():
        print(
            f"    {r.team} {int(r.season)}: GF={int(r.gf)} vs recon={r.gf_recon:.0f} "
            f"(gap {r.gf_gap:+.1f}; xGF={r.xgf:.0f}, fin={r.fin_for:+.1f}, oppG={r.opp_goalie:+.1f})"
        )
    _write(team, league)
    print(f"    -> goal_accounting_{label}.parquet (+ logs/model/)")


if __name__ == "__main__":
    main()
