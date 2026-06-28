"""Goalie save talent: goals saved above expected (GSAx), regressed — the mirror of finishing.py.

For each unblocked shot a goalie faces, our xG model gives the league-average probability it
becomes a goal. A goalie's *save talent* is conceding fewer goals than that baseline — stopping more
than the summed xG of the shots he faces. Per goalie:

  SA       = unblocked shots faced (Fenwick; the shots that carry an xG)
  xGA      = sum of xg over those shots
  GA       = goals allowed
  raw GSAx = xGA - GA   (goals saved above expected; sign-flipped vs finishing's G - ixG)

Save talent is real but low-repeatability, so the raw number is **regressed toward zero** (zero = a
league-average goalie — the natural null) with the same empirical-Bayes shrinkage finishing uses.
Under league-average goaltending each shot is goal ~ Bernoulli(xg), so the per-goalie sampling noise
floor is V = Σ xg(1-xg). Estimating the population spread of true per-shot save talent τ² by method
of moments gives a shot-count shrinkage constant k = σ̄²/τ² (σ̄² = mean per-shot Bernoulli variance):

  gsax_per100 = (xGA - GA) / (SA + k) * 100      # headline: goals saved above expected per 100 shots
  gsax_saved  = (xGA - GA) * SA / (SA + k)        # shrunk total goals saved (WAR-ready)
  SE_per100   = sqrt(V) / (SA + k) * 100          # honest, wide CI

The facing goalie on each shot is `away_goalie` for a home shot and `home_goalie` for an away one.
Empty-net and penalty shots carry no xG (and no facing goalie) and are excluded, exactly as in
finishing. Regular season only (playoff shots carry no modeled xG).

Usage:
  uv run python -m yhattrick.goalie                 # all available seasons, pooled
  uv run python -m yhattrick.goalie --season 2024
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd

from . import config as C
from .player_onice_model import roster_names

MIN_SHOTS_EST = 1000     # min shots faced to enter the talent-variance (k) estimation
TAU2_FLOOR = 1e-6        # floor on per-shot talent variance (-> heavy shrinkage if save talent flat)
SNIFF_MIN_SHOTS = 1500   # min shots faced to appear in the printed leaderboard


def available_seasons() -> list[int]:
    d = C.PROCESSED / "shots_onice"
    return sorted(int(p.stem) for p in d.glob("*.parquet")) if d.exists() else []


def load_shots(seasons: list[int]) -> pd.DataFrame:
    """Per-shot rows with the *facing* goalie attached, restricted to the modeled set.

    `processed/shots_onice` carries every Fenwick shot with the on-ice goalies and the model xG
    (NaN for empty-net / penalty / playoff shots). The goalie facing a shot is the *defending*
    team's goalie, so home shots are faced by `away_goalie` and away shots by `home_goalie`.
    Keeping only rows with a facing goalie AND a non-null xg leaves exactly the modeled,
    goalie-present, regular-season universe — the mirror of finishing.load_shots."""
    frames = []
    for s in seasons:
        p = C.PROCESSED / "shots_onice" / f"{s}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p, columns=["is_home", "home_goalie", "away_goalie", "xg", "goal"])
        df["goalie_id"] = np.where(df.is_home == 1, df.away_goalie, df.home_goalie)
        frames.append(df[df.goalie_id.notna() & df.xg.notna()])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def aggregate(shots: pd.DataFrame) -> pd.DataFrame:
    """Per goalie: SA (shots faced), xGA, GA, V (Bernoulli noise floor Σ xg(1-xg))."""
    shots = shots.assign(_v=shots.xg * (1.0 - shots.xg))
    g = shots.groupby("goalie_id").agg(
        sa=("xg", "size"), xga=("xg", "sum"), ga=("goal", "sum"), v=("_v", "sum"))
    g.index = g.index.astype(int)
    g.index.name = "player_id"
    return g.reset_index()


def estimate_k(agg: pd.DataFrame) -> tuple[float, dict]:
    """Empirical-Bayes shrinkage constant k (in shots) = σ̄²/τ², from goalies with enough shots.

    Volume-weighted variance of the per-shot save rate across goalies = true-talent variance τ² +
    mean sampling variance; subtract the latter (method of moments) to isolate τ²."""
    e = agg[agg.sa >= MIN_SHOTS_EST]
    F, V = e.sa.to_numpy(float), e.v.to_numpy(float)
    r = (e.xga.to_numpy(float) - e.ga.to_numpy(float)) / F          # per-shot save rate (GSAx/shot)
    W = F / F.sum()                                                 # volume weights
    sigbar2 = float(V.sum() / F.sum())                             # mean per-shot Bernoulli var
    wmean = float(np.sum(W * r))                                    # league per-shot save talent (~0)
    wvar = float(np.sum(W * (r - wmean) ** 2))                      # total weighted spread
    mean_sampling = float(np.sum(W * (V / F ** 2)))                 # expected noise contribution
    tau2 = max(wvar - mean_sampling, TAU2_FLOOR)
    k = sigbar2 / tau2
    diag = {"k": round(k, 1), "tau2": float(tau2), "sigma2_bar": round(sigbar2, 5),
            "wvar": float(wvar), "mean_sampling_var": float(mean_sampling),
            "league_save_per_shot": round(wmean, 5), "n_goalies_est": int(len(e))}
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

    F = agg.sa.to_numpy(float)
    gsax = agg.xga.to_numpy(float) - agg.ga.to_numpy(float)         # raw goals saved above expected
    w = F / (F + k)                                                 # shrinkage weight
    se_rate = np.sqrt(agg.v.to_numpy(float)) / (F + k)             # SE of the shrunk per-shot rate
    agg = agg.assign(
        name=[names.get(p, {}).get("name", f"#{p}") for p in agg.player_id],
        pos=[names.get(p, {}).get("pos") for p in agg.player_id],
        gsax_per100=np.round(gsax / (F + k) * 100.0, 3),
        gsax_per100_se=np.round(se_rate * 100.0, 3),
        gsax_saved=np.round(gsax * w, 2),
        gsax_saved_se=np.round(np.sqrt(agg.v.to_numpy(float)) * w, 2),
    )
    agg["xga"] = agg.xga.round(2)
    meta = {
        "model": "goalie", "seasons": sorted(int(s) for s in seasons),
        "shrinkage": diag,
        "league_ga": int(agg.ga.sum()), "league_xga": round(float(agg.xga.sum()), 1),
        "n_goalies": int(len(agg)),
        "min_shots_est": MIN_SHOTS_EST,
    }
    cols = ["player_id", "name", "pos", "sa", "xga", "ga",
            "gsax_per100", "gsax_per100_se", "gsax_saved", "gsax_saved_se"]
    return agg[cols], meta


def _write_meta(meta: dict) -> None:
    C.LOGS_MODEL.mkdir(parents=True, exist_ok=True)
    label = "+".join(map(str, meta["seasons"]))
    (C.LOGS_MODEL / f"goalie_{label}.meta.json").write_text(json.dumps(meta, indent=2))
    with (C.LOGS / "model_fits.jsonl").open("a") as f:
        f.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"), **meta}) + "\n")


def _cache_fresh(path, seasons: list[int]) -> bool:
    pm = path.stat().st_mtime
    for s in seasons:
        sp = C.PROCESSED / "shots_onice" / f"{s}.parquet"
        if sp.exists() and sp.stat().st_mtime > pm:
            return False
    return True


def fit_cached(seasons: list[int], names: dict | None = None) -> pd.DataFrame:
    label = "+".join(map(str, seasons))
    path = C.MODELS / f"goalie_{label}.parquet"
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


def season_goalie(season: int, names: dict, k: float) -> pd.DataFrame:
    """Per-season save talent (using the pooled k) for the trend: player_id, sa, xga, ga,
    gsax_per100."""
    df, _ = fit([season], names, k=k)
    return df[["player_id", "sa", "xga", "ga", "gsax_per100"]] if not df.empty else df


def sniff(df: pd.DataFrame, label: str) -> None:
    elig = df[df.sa >= SNIFF_MIN_SHOTS]
    top = elig.sort_values("gsax_per100", ascending=False).head(12)
    print(f"\n[{label}] best goalies (goals saved above expected /100 shots, ±95% CI, ≥{SNIFF_MIN_SHOTS} sh):")
    for r in top.itertuples():
        print(f"    {r.gsax_per100:+.2f} ±{1.96 * r.gsax_per100_se:.2f}  {r.name} "
              f"({int(r.sa)} sh, {r.gsax_saved:+.1f} G saved)")
    bot = elig.sort_values("gsax_per100").head(6)
    print(f"[{label}] worst goalies:")
    for r in bot.itertuples():
        print(f"    {r.gsax_per100:+.2f} ±{1.96 * r.gsax_per100_se:.2f}  {r.name} ({int(r.sa)} sh)")


def run(seasons: list[int], pool: bool) -> None:
    C.MODELS.mkdir(parents=True, exist_ok=True)
    names = roster_names(seasons)
    groups = [seasons] if pool else [[s] for s in seasons]
    for grp in groups:
        df, meta = fit(grp, names)
        label = "+".join(map(str, grp))
        if df.empty:
            print(f"\n[goalie {label}] no shots — skipping")
            continue
        d = meta["shrinkage"]
        print(f"\n=== goalie save talent — seasons {label} : {meta['n_goalies']} goalies, "
              f"league GA={meta['league_ga']} vs xGA={meta['league_xga']}, "
              f"k={d['k']} shots (τ²={d['tau2']:.2e}, σ̄²={d['sigma2_bar']}) ===")
        out = C.MODELS / f"goalie_{label}.parquet"
        df.to_parquet(out, index=False)
        _write_meta(meta)
        print(f"    -> {out.name} (+ logs/model/goalie_{label}.meta.json)")
        sniff(df, label)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Fit goalie save talent (goals saved above expected, regressed)")
    p.add_argument("--season", type=int, default=None, help="one season (default: all available)")
    p.add_argument("--pool", action="store_true", help="pool all available seasons into one fit")
    args = p.parse_args(argv)
    seasons = [args.season] if args.season else available_seasons()
    if not seasons:
        raise SystemExit("no processed shots available — run `make stints` first")
    # default to pooled when no single season is requested (matches the model's headline)
    run(seasons, pool=args.pool or args.season is None)


if __name__ == "__main__":
    main()
