"""Finishing: goals scored above the expected-goals value of a player's own shots, regressed.

For each unblocked shot a player takes, our xG model gives an expected-goals value (the
league-average probability it becomes a goal from that spot). A player's *finishing* is converting
better than that baseline — scoring more goals than the summed xG of his shots. Per player:

  F   = unblocked shots taken (Fenwick; the only attempts that carry xG and can score)
  ixG = sum of xg over those shots (individual expected goals)
  G   = goals
  raw finishing = G - ixG   (goals above expected)

Finishing is real but low-repeatability, so the raw number is dominated by luck and must be
**regressed toward zero** (zero = a league-average finisher — the natural null, no informative
prior). We use empirical-Bayes shrinkage. Under league-average finishing each shot is a coin flip,
goal ~ Bernoulli(xg), so the per-player sampling noise floor is V = Σ xg(1-xg). Estimating
the population spread of true per-shot finishing talent τ² by method of moments gives a shot-count
shrinkage constant k = σ̄²/τ² (σ̄² = mean per-shot Bernoulli variance):

  fin_per100 = (G - ixG) / (F + k) * 100      # headline: goals above expected per 100 shots (shrunk)
  fin_goals  = (G - ixG) * F / (F + k)        # shrunk total goals above expected (WAR-ready)
  SE_per100  = sqrt(V) / (F + k) * 100         # honest, wide CI

All in-play situations are pooled (more shots -> steadier estimate); empty-net and shootout shots
are excluded (they're absent from the xG model's modeled set). Regular season only.

Usage:
  uv run python -m yhattrick.finishing                 # all available seasons, pooled
  uv run python -m yhattrick.finishing --season 2024
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd

from .. import config as C
from .player_onice_model import roster_names

MIN_SHOTS_EST = 200      # min unblocked shots to enter the talent-variance (k) estimation
TAU2_FLOOR = 1e-6        # floor on per-shot talent variance (-> heavy shrinkage if finishing flat)
SNIFF_MIN_SHOTS = 300    # min shots to appear in the printed leaderboard


def available_seasons() -> list[int]:
    d = C.PROCESSED / "xg"
    return sorted(int(p.stem) for p in d.glob("*.parquet")) if d.exists() else []


def load_shots(seasons: list[int]) -> pd.DataFrame:
    """The xG model's modeled shots (regular-season, goalie-present, unblocked) with a valid shooter.

    `processed/xg` is already the right universe — empty-net, shootout and malformed-strength shots
    are excluded upstream — so finishing just reads it."""
    frames = []
    for s in seasons:
        p = C.PROCESSED / "xg" / f"{s}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p, columns=["shooter_id", "xg", "goal"])
        frames.append(df[df.shooter_id.notna() & df.xg.notna()])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def aggregate(shots: pd.DataFrame) -> pd.DataFrame:
    """Per shooter: F (shots), ixG, G, V (Bernoulli noise floor Σ xg(1-xg))."""
    shots = shots.assign(_v=shots.xg * (1.0 - shots.xg))
    g = shots.groupby("shooter_id").agg(
        shots=("xg", "size"), ixg=("xg", "sum"), goals=("goal", "sum"), v=("_v", "sum"))
    g.index = g.index.astype(int)
    g.index.name = "player_id"
    return g.reset_index()


def estimate_k(agg: pd.DataFrame) -> tuple[float, dict]:
    """Empirical-Bayes shrinkage constant k (in shots) = σ̄²/τ², from players with enough shots.

    Volume-weighted variance of the per-shot finishing rate across players = true-talent variance
    τ² + mean sampling variance; subtract the latter (method of moments) to isolate τ²."""
    e = agg[agg.shots >= MIN_SHOTS_EST]
    F, V = e.shots.to_numpy(float), e.v.to_numpy(float)
    r = (e.goals.to_numpy(float) - e.ixg.to_numpy(float)) / F        # per-shot finishing rate
    W = F / F.sum()                                                  # volume weights
    sigbar2 = float(V.sum() / F.sum())                              # mean per-shot Bernoulli var
    wmean = float(np.sum(W * r))                                     # league per-shot finishing (~0)
    wvar = float(np.sum(W * (r - wmean) ** 2))                       # total weighted spread
    mean_sampling = float(np.sum(W * (V / F ** 2)))                  # expected noise contribution
    tau2 = max(wvar - mean_sampling, TAU2_FLOOR)
    k = sigbar2 / tau2
    diag = {"k": round(k, 1), "tau2": float(tau2), "sigma2_bar": round(sigbar2, 5),
            "wvar": float(wvar), "mean_sampling_var": float(mean_sampling),
            "league_finishing_per_shot": round(wmean, 5), "n_players_est": int(len(e))}
    return k, diag


def fit(seasons: list[int], names: dict, k: float | None = None) -> tuple[pd.DataFrame, dict]:
    shots = load_shots(seasons)
    if shots.empty:
        return pd.DataFrame(), {}
    agg = aggregate(shots)
    if k is None:
        k, diag = estimate_k(agg)
    else:
        diag = {"k": round(k, 1), "k_source": "pooled"}

    F = agg.shots.to_numpy(float)
    gax = agg.goals.to_numpy(float) - agg.ixg.to_numpy(float)        # raw goals above expected
    w = F / (F + k)                                                  # shrinkage weight
    se_rate = np.sqrt(agg.v.to_numpy(float)) / (F + k)               # SE of the shrunk per-shot rate
    agg = agg.assign(
        name=[names.get(p, {}).get("name", f"#{p}") for p in agg.player_id],
        pos=[names.get(p, {}).get("pos") for p in agg.player_id],
        fin_per100=np.round(gax / (F + k) * 100.0, 3),
        fin_per100_se=np.round(se_rate * 100.0, 3),
        fin_goals=np.round(gax * w, 2),
        fin_goals_se=np.round(np.sqrt(agg.v.to_numpy(float)) * w, 2),
    )
    agg["ixg"] = agg.ixg.round(2)
    meta = {
        "model": "finishing", "seasons": sorted(int(s) for s in seasons),
        "shrinkage": diag,
        "league_goals": int(agg.goals.sum()), "league_ixg": round(float(agg.ixg.sum()), 1),
        "n_players": int(len(agg)),
        "min_shots_est": MIN_SHOTS_EST,
    }
    cols = ["player_id", "name", "pos", "shots", "ixg", "goals",
            "fin_per100", "fin_per100_se", "fin_goals", "fin_goals_se"]
    return agg[cols], meta


def _write_meta(meta: dict) -> None:
    C.LOGS_MODEL.mkdir(parents=True, exist_ok=True)
    label = "+".join(map(str, meta["seasons"]))
    (C.LOGS_MODEL / f"finishing_{label}.meta.json").write_text(json.dumps(meta, indent=2))
    with (C.LOGS / "model_fits.jsonl").open("a") as f:
        f.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"), **meta}) + "\n")


def _cache_fresh(path, seasons: list[int]) -> bool:
    pm = path.stat().st_mtime
    for s in seasons:
        sp = C.PROCESSED / "xg" / f"{s}.parquet"
        if sp.exists() and sp.stat().st_mtime > pm:
            return False
    return True


def fit_cached(seasons: list[int], names: dict | None = None) -> pd.DataFrame:
    label = "+".join(map(str, seasons))
    path = C.MODELS / f"finishing_{label}.parquet"
    if path.exists() and _cache_fresh(path, seasons):
        return pd.read_parquet(path)
    df, meta = fit(seasons, names or roster_names(seasons))
    if df.empty:
        return df
    C.MODELS.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    _write_meta(meta)
    return df


def pooled_k(seasons: list[int], names: dict) -> float:
    """The shrinkage constant from the pooled fit (so per-season fits shrink consistently)."""
    _, diag = estimate_k(aggregate(load_shots(seasons)))
    return diag["k"]


def season_finishing(season: int, names: dict, k: float) -> pd.DataFrame:
    """Per-season finishing (using the pooled k) for the trend: player_id, shots, ixg, goals,
    fin_per100."""
    df, _ = fit([season], names, k=k)
    return df[["player_id", "shots", "ixg", "goals", "fin_per100"]] if not df.empty else df


def sniff(df: pd.DataFrame, label: str) -> None:
    elig = df[df.shots >= SNIFF_MIN_SHOTS]
    top = elig.sort_values("fin_per100", ascending=False).head(12)
    print(f"\n[{label}] best finishers (goals above expected /100 shots, ±95% CI, ≥{SNIFF_MIN_SHOTS} shots):")
    for r in top.itertuples():
        print(f"    {r.fin_per100:+.2f} ±{1.96 * r.fin_per100_se:.2f}  {r.name} ({r.pos}, "
              f"{int(r.shots)} sh, {r.fin_goals:+.1f} G)")
    bot = elig.sort_values("fin_per100").head(6)
    print(f"[{label}] worst finishers:")
    for r in bot.itertuples():
        print(f"    {r.fin_per100:+.2f} ±{1.96 * r.fin_per100_se:.2f}  {r.name} ({r.pos}, {int(r.shots)} sh)")


def run(seasons: list[int], pool: bool) -> None:
    C.MODELS.mkdir(parents=True, exist_ok=True)
    names = roster_names(seasons)
    groups = [seasons] if pool else [[s] for s in seasons]
    for grp in groups:
        df, meta = fit(grp, names)
        label = "+".join(map(str, grp))
        if df.empty:
            print(f"\n[finishing {label}] no shots — skipping")
            continue
        d = meta["shrinkage"]
        print(f"\n=== finishing — seasons {label} : {meta['n_players']} shooters, "
              f"league G={meta['league_goals']} vs ixG={meta['league_ixg']}, "
              f"k={d['k']} shots (τ²={d['tau2']:.2e}, σ̄²={d['sigma2_bar']}) ===")
        out = C.MODELS / f"finishing_{label}.parquet"
        df.to_parquet(out, index=False)
        _write_meta(meta)
        print(f"    -> {out.name} (+ logs/model/finishing_{label}.meta.json)")
        sniff(df, label)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Fit the finishing metric (goals above expected, regressed)")
    p.add_argument("--season", type=int, default=None, help="one season (default: all available)")
    p.add_argument("--pool", action="store_true", help="pool all available seasons into one fit")
    args = p.parse_args(argv)
    seasons = [args.season] if args.season else available_seasons()
    if not seasons:
        raise SystemExit("no interim shots available — run `make clean-data` first")
    # default to pooled when no single season is requested (matches the model's headline)
    run(seasons, pool=args.pool or args.season is None)


if __name__ == "__main__":
    main()
