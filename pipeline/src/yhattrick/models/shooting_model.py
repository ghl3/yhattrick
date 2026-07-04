"""Shooting model: the per-shot conversion fit — STAGE 2 of the additive theory of goals.

See docs/modeling.md for the full theory. Every goal is *chance × conversion*:

  Stage 1 (creation)   — player_onice_model (RAPM): who manufactured the shot's xG.
  Stage 2 (conversion) — THIS module: given a shot worth xG, who finishes it / who stops it.

A shot is a duel between the shooter (who can finish above the league baseline) and the goalie
(who can save below it). We attribute the per-shot residual (goal − xG) to the two of them *jointly*,
in one penalized ridge regression with crossed shooter + goalie effects:

  goal_i − xG_i = mu + alpha_shooter(i) + gamma_goalie(i) + e_i        (ridge on alpha, gamma)

`alpha` is finishing (goals per shot above expected, alpha>0 = good finisher); `gamma` is the goalie
(goals per shot above expected *allowed*, gamma<0 = good goalie). Fitting them together is the whole
point: it *splits* the residual between the two actors instead of letting each claim it in full (the
old finishing.py and goalie.py each modelled (goal − xG) ignoring the other, so the same residual was
double-counted). Now finishing is adjusted for the goalie faced and vice-versa.

Why linear (not logistic). The model is **additive by construction** on the goals scale: the fitted
residual is mu + alpha + gamma, so finishing_i = alpha and goalie_i = gamma sum *exactly* to it — no
nonlinearity term, and the chance baseline stays exactly raw xG (so it ties cleanly to RAPM's
creation layer). With a free intercept the least-squares residuals sum to zero, giving the exact
reconciliation

  actual goals = sum(xG) + sum(mu) + sum(finishing) + sum(goalie)        (identity, league-wide)

Shrinkage. The ridge penalty on each effect block *is* empirical-Bayes shrinkage toward zero (a
league-average shooter / goalie — the natural null). For an unweighted dummy with n shots, ridge
lambda shrinks the effect by n/(n+lambda); setting lambda = k (the finishing/goalie EB shot-count
constant k = sigbar2/tau2) makes a player with NO crossing reduce *exactly* to the old shrunk metric
(G − ixG)·n/(n+k). So we keep the trusted shrinkage and change only the (now joint, mutually
adjusted) estimation.

The intercept (mu) — why it exists, why it isn't zero, and where it goes. The design is
rank-deficient (every shot has one shooter and one goalie, so the intercept column equals the sum of
the shooter columns equals the sum of the goalie columns); a *free* (unpenalized) intercept resolves
that and, crucially, its normal equation forces sum(goal − xG − mu − alpha − gamma) = 0, i.e. it is
what makes the league identity goals = sum(xG) + sum(mu) + sum(finishing) + sum(goalie) hold to the
penny. mu is NOT zero even though xG is well calibrated (~0.1%): regularization shrinks every alpha
and gamma toward zero, so the shrunk effects no longer sum to the raw residual, and mu absorbs
exactly that shrinkage leakage (plus a second-order selection effect — xG is calibrated to the
shot-weighted population, in which good finishers shoot more, so an average *player* sits a hair
below xG). So mu is the accounting footprint of the shrinkage, not a calibration patch: drop the
regularization and mu -> ~0, but the per-player estimates become overfit noise. mu is tiny per shot
(~ -0.0007 goals, ~ -0.07 / 100 shots) and is attributed to NO player — it is a league-baseline
line in the identity (goal_accounting.py), deliberately not split into finishing/goalie (it is not
skill, and an even split would be arbitrary and change no ranking).

Per shooter: fin_goals = alpha · shots; fin_per100 = alpha · 100.
Per goalie:  gsax_saved = −gamma · sa (sign so + = good); gsax_per100 = −gamma · 100.

This module REPLACES finishing.py and the modeled half of goalie.py: it emits both tables with the
same columns the site already reads (fin_*, gsax_*), so export is a drop-in swap.

Usage:
  uv run python -m yhattrick.models.shooting_model                 # all seasons, pooled
  uv run python -m yhattrick.models.shooting_model --season 2024
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import sparse

from .. import config as C
from .player_onice_model import roster_names, _xtwx, _xtwy, _solve_pen, _edf_and_se

MIN_SHOTS_FIN_EST = (
    200  # min shots for a shooter to enter the finishing talent-variance (k) estimate
)
MIN_SHOTS_GOALIE_EST = (
    1000  # min shots faced for a goalie to enter the save talent-variance (k) estimate
)
TAU2_FLOOR = 1e-6  # floor on per-shot talent variance (-> heavy shrinkage if flat)
SNIFF_FIN_SHOTS = 300  # min shots to appear in the finishing leaderboard
SNIFF_GOALIE_SHOTS = 1500  # min shots faced to appear in the goalie leaderboard


def available_seasons() -> list[int]:
    d = C.PROCESSED / "shots_onice"
    return sorted(int(p.stem) for p in d.glob("*.parquet")) if d.exists() else []


def load_shots(seasons: list[int]) -> pd.DataFrame:
    """Per-shot rows carrying BOTH duelists + the model xG, restricted to the modeled set.

    `processed/shots_onice` has every Fenwick shot with the shooter, the on-ice goalies and the
    calibrated xG (NaN for empty-net / penalty / playoff shots). The facing goalie is the defending
    team's: home shots are faced by `away_goalie`, away shots by `home_goalie`. Keeping rows with a
    shooter, a facing goalie, and a non-null xg leaves exactly the modeled, regular-season universe —
    the union of finishing's and goalie's old inputs."""
    frames = []
    for s in seasons:
        p = C.PROCESSED / "shots_onice" / f"{s}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(
            p, columns=["is_home", "home_goalie", "away_goalie", "shooter_id", "xg", "goal"]
        )
        df["goalie_id"] = np.where(df.is_home == 1, df.away_goalie, df.home_goalie)
        df = df[df.shooter_id.notna() & df.goalie_id.notna() & df.xg.notna()]
        frames.append(
            df[["shooter_id", "goalie_id", "xg", "goal"]].astype(
                {"shooter_id": int, "goalie_id": int, "xg": float, "goal": int}
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --- shrinkage: method-of-moments k per block (the ridge penalty) --------------------------------


def _estimate_k(F, ixg, made, V, min_shots) -> tuple[float, dict]:
    """Empirical-Bayes shrinkage constant k (in shots) = sigbar2/tau2 from high-volume entities.

    Volume-weighted variance of the per-shot residual rate = true-talent variance tau2 + mean
    sampling variance; subtract the latter (method of moments) to isolate tau2. Identical machinery
    to the old finishing.estimate_k / goalie.estimate_k (the sign of the residual is irrelevant to
    its variance, so one helper serves both blocks). Used directly as the ridge penalty: lambda = k.
    """
    m = F >= min_shots
    F, ixg, made, V = (
        F[m].astype(float),
        ixg[m].astype(float),
        made[m].astype(float),
        V[m].astype(float),
    )
    r = (made - ixg) / F  # per-shot residual rate
    W = F / F.sum()
    sigbar2 = float(V.sum() / F.sum())  # mean per-shot Bernoulli variance
    wmean = float(np.sum(W * r))
    wvar = float(np.sum(W * (r - wmean) ** 2))
    mean_sampling = float(np.sum(W * (V / F**2)))
    tau2 = max(wvar - mean_sampling, TAU2_FLOOR)
    k = sigbar2 / tau2
    return k, {
        "k": round(k, 1),
        "tau2": float(tau2),
        "sigma2_bar": round(sigbar2, 5),
        "n_entities_est": int(m.sum()),
    }


def _block_agg(shots: pd.DataFrame, key: str) -> pd.DataFrame:
    s = shots.assign(_v=shots.xg * (1.0 - shots.xg))
    return (
        s.groupby(key)
        .agg(F=("xg", "size"), ixg=("xg", "sum"), made=("goal", "sum"), V=("_v", "sum"))
        .reset_index()
    )


def lambdas(shots: pd.DataFrame) -> tuple[float, float, dict]:
    """Ridge penalties (lambda_shooter, lambda_goalie) = the EB shot-count constants k. With an
    unweighted design, lambda = k reproduces the finishing/goalie shrinkage weight n/(n+k) exactly.
    """
    sg = _block_agg(shots, "shooter_id")
    gg = _block_agg(shots, "goalie_id")
    k_s, ds = _estimate_k(
        sg.F.values, sg.ixg.values, sg.made.values, sg.V.values, MIN_SHOTS_FIN_EST
    )
    k_g, dg = _estimate_k(
        gg.F.values, gg.ixg.values, gg.made.values, gg.V.values, MIN_SHOTS_GOALIE_EST
    )
    return (
        k_s,
        k_g,
        {"shooter": {**ds, "lambda": round(k_s, 1)}, "goalie": {**dg, "lambda": round(k_g, 1)}},
    )


# --- the crossed penalized-ridge fit (linear, additive in goals) ---------------------------------


def _design(shots, shooters, goalies):
    si = {p: i for i, p in enumerate(shooters)}
    gi = {p: len(shooters) + i for i, p in enumerate(goalies)}
    n, Ps, Pg = len(shots), len(shooters), len(goalies)
    m = Ps + Pg + 1  # + intercept (last column)
    s_col = shots.shooter_id.map(si).to_numpy()
    g_col = shots.goalie_id.map(gi).to_numpy()
    rows = np.repeat(np.arange(n), 3)
    cols = np.empty(n * 3, dtype=np.int64)
    cols[0::3], cols[1::3], cols[2::3] = s_col, g_col, m - 1
    Z = sparse.csr_matrix((np.ones(n * 3), (rows, cols)), shape=(n, m))
    return Z, m, s_col, g_col


def fit(seasons, names, lams=None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Joint fit over a season set. Returns (finishing_table, goalie_table, meta). `lams` lets
    callers reuse the pooled penalties for per-season fits (mirrors the old pooled-k pattern)."""
    shots = load_shots(seasons)
    if shots.empty:
        return pd.DataFrame(), pd.DataFrame(), {}
    fin, gl, meta = fit_shots(shots, names, lams)
    meta["seasons"] = sorted(int(s) for s in seasons)
    return fin, gl, meta


def fit_shots(shots: pd.DataFrame, names, lams=None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """The core fit on an in-memory shots frame (shooter_id, goalie_id, xg, goal). Separated from
    `fit` so tests can drive it with synthetic data."""
    if lams is None:
        lam_s, lam_g, ldiag = lambdas(shots)
    else:
        lam_s, lam_g = lams
        ldiag = {
            "source": "pooled",
            "shooter": {"lambda": round(lam_s, 1)},
            "goalie": {"lambda": round(lam_g, 1)},
        }

    shooters = sorted(shots.shooter_id.unique())
    goalies = sorted(shots.goalie_id.unique())
    Z, m, s_col, g_col = _design(shots, shooters, goalies)
    Ps, Pg = len(shooters), len(goalies)

    xg = shots.xg.to_numpy(float)
    y = shots.goal.to_numpy(float)
    target = y - xg  # goals above expected, the thing we split
    w = np.ones(len(y))  # unweighted -> lambda=k reproduces EB exactly

    B = _xtwx(Z, w)
    c = _xtwy(Z, w, target)
    pen = np.zeros(m)
    pen[:Ps] = lam_s
    pen[Ps : Ps + Pg] = lam_g  # intercept (last col) stays free
    beta, A, _ = _solve_pen(B, c, pen)

    fit_resid = np.asarray(Z @ beta)
    rss = float(np.sum(w * (target - fit_resid) ** 2))
    edf, _ = _edf_and_se(B, A, 1.0)
    dispersion = rss / max(len(y) - edf, 1.0)
    _, se = _edf_and_se(B, A, dispersion)

    intercept = float(beta[-1])
    alpha = beta[:Ps]
    gamma = beta[Ps : Ps + Pg]
    se_a = se[:Ps]
    se_g = se[Ps : Ps + Pg]
    ai = {p: i for i, p in enumerate(shooters)}
    gj = {p: i for i, p in enumerate(goalies)}

    # per-shooter: finishing_i = alpha (constant per shot) -> fin_goals = alpha*shots
    sdf = pd.DataFrame({"player_id": shots.shooter_id.values, "xg": xg, "goal": y})
    fin = (
        sdf.groupby("player_id")
        .agg(shots=("goal", "size"), ixg=("xg", "sum"), goals=("goal", "sum"))
        .reset_index()
    )
    fa = fin.player_id.map(lambda p: alpha[ai[p]])
    fse = fin.player_id.map(lambda p: se_a[ai[p]])
    fin["fin_per100"] = np.round(fa * 100.0, 3)
    fin["fin_per100_se"] = np.round(fse * 100.0, 3)
    fin["fin_goals"] = np.round(fa * fin.shots, 2)
    fin["fin_goals_se"] = np.round(fse * fin.shots, 2)
    fin["ixg"] = fin.ixg.round(2)
    fin["name"] = [names.get(int(p), {}).get("name", f"#{p}") for p in fin.player_id]
    fin["pos"] = [names.get(int(p), {}).get("pos") for p in fin.player_id]
    fin_cols = [
        "player_id",
        "name",
        "pos",
        "shots",
        "ixg",
        "goals",
        "fin_per100",
        "fin_per100_se",
        "fin_goals",
        "fin_goals_se",
    ]

    # per-goalie: goalie_i = gamma -> gsax_saved = -gamma*sa (+ = good)
    gdf = pd.DataFrame({"player_id": shots.goalie_id.values, "xg": xg, "goal": y})
    gl = (
        gdf.groupby("player_id")
        .agg(sa=("goal", "size"), xga=("xg", "sum"), ga=("goal", "sum"))
        .reset_index()
    )
    gg_ = gl.player_id.map(lambda p: gamma[gj[p]])
    gse = gl.player_id.map(lambda p: se_g[gj[p]])
    gl["gsax_per100"] = np.round(-gg_ * 100.0, 3)
    gl["gsax_per100_se"] = np.round(gse * 100.0, 3)
    gl["gsax_saved"] = np.round(-gg_ * gl.sa, 2)
    gl["gsax_saved_se"] = np.round(gse * gl.sa, 2)
    gl["xga"] = gl.xga.round(2)
    gl["name"] = [names.get(int(p), {}).get("name", f"#{p}") for p in gl.player_id]
    gl["pos"] = [names.get(int(p), {}).get("pos") for p in gl.player_id]
    gl_cols = [
        "player_id",
        "name",
        "pos",
        "sa",
        "xga",
        "ga",
        "gsax_per100",
        "gsax_per100_se",
        "gsax_saved",
        "gsax_saved_se",
    ]

    # reconciliation (exact, free intercept -> residuals sum to zero):
    #   actual goals = sum(xG) + sum(mu) + sum(finishing) + sum(goalie_effect)
    sum_fin = float(beta[s_col].sum())  # s_col / g_col are global column indices into beta
    sum_gol = float(beta[g_col].sum())
    recon = {
        "n_shots": int(len(shots)),
        "n_shooters": Ps,
        "n_goalies": Pg,
        "goals": int(y.sum()),
        "xg": round(float(xg.sum()), 1),
        "residual_above_xg": round(float(target.sum()), 1),
        "sum_intercept": round(intercept * len(y), 1),
        "intercept_per_shot": round(intercept, 6),
        "sum_finishing": round(sum_fin, 1),
        "sum_goalie_effect": round(sum_gol, 1),
        "sum_gsax_saved": round(-sum_gol, 1),
        "identity_resid": round(
            float(y.sum() - xg.sum() - intercept * len(y) - sum_fin - sum_gol), 4
        ),
        "edf": round(edf, 1),
        "dispersion": round(dispersion, 5),
    }
    meta = {
        "model": "shooting_model",
        "engine": "linear_crossed_ridge",
        "seasons": [],
        "shrinkage": ldiag,
        "reconciliation": recon,
    }
    return fin[fin_cols], gl[gl_cols], meta


# --- caching + public API (drop-in for the old finishing / goalie surfaces) ----------------------


def _paths(seasons):
    label = "+".join(map(str, seasons))
    return (
        C.MODELS / f"shooting_finishing_{label}.parquet",
        C.MODELS / f"shooting_goalie_{label}.parquet",
        label,
    )


def _fresh(path, seasons) -> bool:
    pm = path.stat().st_mtime
    for s in seasons:
        sp = C.PROCESSED / "shots_onice" / f"{s}.parquet"
        if sp.exists() and sp.stat().st_mtime > pm:
            return False
    return True


def _write_meta(meta: dict) -> None:
    C.LOGS_MODEL.mkdir(parents=True, exist_ok=True)
    label = "+".join(map(str, meta["seasons"]))
    (C.LOGS_MODEL / f"shooting_model_{label}.meta.json").write_text(json.dumps(meta, indent=2))
    with (C.LOGS / "model_fits.jsonl").open("a") as f:
        f.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"), **meta}) + "\n")


def fit_cached(seasons, names=None, lams=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit (and cache) both tables for a season set. Per-season callers pass the pooled `lams` so
    season fits shrink with the pooled penalties (the cache assumes the standard pooled config)."""
    fp, gp, _ = _paths(seasons)
    if fp.exists() and gp.exists() and _fresh(fp, seasons) and _fresh(gp, seasons):
        return pd.read_parquet(fp), pd.read_parquet(gp)
    fin, gl, meta = fit(seasons, names or roster_names(seasons), lams)
    if fin.empty:
        return fin, gl
    C.MODELS.mkdir(parents=True, exist_ok=True)
    fin.to_parquet(fp, index=False)
    gl.to_parquet(gp, index=False)
    _write_meta(meta)
    return fin, gl


def pooled_lambdas(seasons, names=None) -> tuple[float, float]:
    """The (lambda_shooter, lambda_goalie) = pooled EB k's — so per-season fits use the same
    shrinkage (the analogue of the old finishing/goalie pooled_k)."""
    k_s, k_g, _ = lambdas(load_shots(seasons))
    return k_s, k_g


def season_finishing(season: int, names: dict, lams) -> pd.DataFrame:
    fin, _ = fit_cached([season], names, lams)
    return fin[["player_id", "shots", "ixg", "goals", "fin_per100"]] if not fin.empty else fin


def season_goalie(season: int, names: dict, lams) -> pd.DataFrame:
    _, gl = fit_cached([season], names, lams)
    return gl[["player_id", "sa", "xga", "ga", "gsax_per100"]] if not gl.empty else gl


def sniff(fin: pd.DataFrame, gl: pd.DataFrame, label: str) -> None:
    f = fin[fin.shots >= SNIFF_FIN_SHOTS].sort_values("fin_per100", ascending=False).head(10)
    print(
        f"\n[{label}] best finishers (goals above expected /100 shots, goalie-adjusted, "
        f">={SNIFF_FIN_SHOTS} sh):"
    )
    for r in f.itertuples():
        print(
            f"    {r.fin_per100:+.2f} ±{1.96 * r.fin_per100_se:.2f}  {r.name} ({r.pos}, "
            f"{int(r.shots)} sh, {r.fin_goals:+.1f} G)"
        )
    g = gl[gl.sa >= SNIFF_GOALIE_SHOTS].sort_values("gsax_per100", ascending=False).head(10)
    print(
        f"[{label}] best goalies (goals saved above expected /100 shots, shooter-adjusted, "
        f">={SNIFF_GOALIE_SHOTS} sh):"
    )
    for r in g.itertuples():
        print(
            f"    {r.gsax_per100:+.2f} ±{1.96 * r.gsax_per100_se:.2f}  {r.name} "
            f"({int(r.sa)} sh, {r.gsax_saved:+.1f} G saved)"
        )


def run(seasons, pool: bool) -> None:
    C.MODELS.mkdir(parents=True, exist_ok=True)
    names = roster_names(seasons)
    groups = [seasons] if pool else [[s] for s in seasons]
    for grp in groups:
        fin, gl, meta = fit(grp, names)
        label = "+".join(map(str, grp))
        if fin.empty:
            print(f"\n[shooting_model {label}] no shots — skipping")
            continue
        rc = meta["reconciliation"]
        print(
            f"\n=== shooting_model (linear) — seasons {label} : {rc['n_shots']:,} shots, "
            f"{rc['n_shooters']} shooters × {rc['n_goalies']} goalies, "
            f"λ_s={meta['shrinkage']['shooter']['lambda']}, λ_g={meta['shrinkage']['goalie']['lambda']} ==="
        )
        print(
            f"    reconcile: goals={rc['goals']} = xG={rc['xg']} + μ={rc['sum_intercept']:+.1f} "
            f"+ finishing={rc['sum_finishing']:+.1f} + goalie={rc['sum_goalie_effect']:+.1f} "
            f"→ resid={rc['identity_resid']:+.3f}"
        )
        fp, gp, _ = _paths(grp)
        fin.to_parquet(fp, index=False)
        gl.to_parquet(gp, index=False)
        _write_meta(meta)
        print(f"    -> {fp.name} + {gp.name} (+ logs/model/shooting_model_{label}.meta.json)")
        sniff(fin, gl, label)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Fit the joint shooter×goalie shot-conversion model")
    p.add_argument("--season", type=int, default=None, help="one season (default: all available)")
    p.add_argument("--pool", action="store_true", help="pool all available seasons into one fit")
    args = p.parse_args(argv)
    seasons = [args.season] if args.season else available_seasons()
    if not seasons:
        raise SystemExit("no processed shots available — run `make stints` first")
    run(seasons, pool=args.pool or args.season is None)


if __name__ == "__main__":
    main()
