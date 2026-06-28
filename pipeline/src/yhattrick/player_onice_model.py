"""Isolated-impact model: each player's per-60 effect on the expected-goal rate, adjusted for
who else is on the ice (linemates and competition), by strength state.

Method (a regularized adjusted plus-minus). A stint contributes one or two observations: the
response is a team's expected goals per 60 minutes during the stint; the predictors are indicator
columns for each on-ice player — an OFFENCE column when his team is attacking and a DEFENCE
column when defending — plus shared context covariates (home ice, zone start, score state, season,
period) and an intercept. We fit separate models per strength state:

  even strength (5v5)   -> ev_off  (xG/60 a player ADDS),       ev_def  (xG/60 he ALLOWS)
  special teams (5v4)   -> pp_off  (power-play offence added),  pk_def  (penalty-kill xG allowed)

How regularization is scaled (the important part). We solve the *penalized weighted normal
equations* directly: β = (ZᵀWZ + diag(pen))⁻¹ ZᵀWy. Two design choices fix the original
"λ pinned at the grid max / nothing shrinks" failure:
  * Weights are normalized so Σw = n (w = dur/mean(dur)), decoupling λ from absolute seconds and
    sample size. The data term then lives on a per-observation scale.
  * Only the 2·P player columns are penalized; the intercept and covariates are FREE (pen = 0), so
    deployment/zone/score/era bias is absorbed cleanly instead of leaking into player coefficients.
  * The λ grid is anchored to the data scale: λ = multiplier × median player-column curvature
    (median nonzero diagonal of ZᵀWZ). A player coefficient shrinks ~ curvature/(curvature+λ), so
    this grid spans no-shrinkage → heavy-shrinkage regardless of units.

Two response families, switchable (`--family`): a Gaussian fit on the per-60 rate (identity link,
the historical behavior, exact one-shot solve) and a Tweedie GLM (log link, compound Poisson–Gamma)
fit by IRLS — the correct likelihood for the zero-inflated, non-negative xG response, which removes
the per-60 variance blow-up of short stints. Tweedie coefficients (log-rate) are linearized to a
per-60 xGF delta so the site framing is unchanged.

Every fit logs its full provenance + diagnostics (the whole λ sweep with per-fold CV and effective
degrees of freedom, fit quality, numerical conditioning, residual summary) to logs/model/.

Built on the model xG attached to each stint (see xg.py).

Usage:
  uv run python -m yhattrick.player_onice_model                       # every model, every season
  uv run python -m yhattrick.player_onice_model --pool                # pool all seasons into one fit
  uv run python -m yhattrick.player_onice_model --model pp_pk         # just special teams (or 'ev')
  uv run python -m yhattrick.player_onice_model --family tweedie      # Tweedie GLM instead of Gaussian
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from sklearn.model_selection import GroupKFold

from . import config as C

MIN_STINT_S = 10         # drop sub-10s line-change stints (extreme per-60 rates, ~no signal)
# λ grid as multipliers of the median player-column curvature (median nonzero diag of ZᵀWZ).
# Spans ~no shrinkage (0.003×) to heavy shrinkage (100×) so the CV curve can turn over inside it.
LAMBDA_MULTS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
TWEEDIE_POWER = 1.5      # compound Poisson–Gamma; 1<p<2 fits non-negative continuous xG with a 0 mass
IRLS_MAX_ITER = 60
IRLS_TOL = 1e-7


@dataclass(frozen=True)
class Spec:
    key: str               # model id / output filename stem
    strengths: tuple       # stint strength states this model uses
    dual: bool             # EV: both teams attack; special teams: only the power-play team
    off: str               # offence coefficient name (attacking team)
    deff: str              # defence coefficient name (defending team)
    min_toi: float         # min role TOI (minutes) to appear in the sniff test


SPECS = {
    "ev": Spec("ev", ("5v5",), True, "ev_off", "ev_def", 150),
    "pp_pk": Spec("pp_pk", ("5v4", "4v5"), False, "pp_off", "pk_def", 40),
}


def available_seasons() -> list[int]:
    d = C.PROCESSED / "stints"
    return sorted(int(p.stem) for p in d.glob("*.parquet")) if d.exists() else []


def load_stints(seasons: list[int], strengths: tuple) -> pd.DataFrame:
    frames = []
    for s in seasons:
        p = C.PROCESSED / "stints" / f"{s}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if "overload" not in df.columns:   # tolerate older processed files
            df["overload"] = False
        # regular season only: rate stats shouldn't mix in the small, context-distorted playoffs
        reg = (df.nhl_game_id // 10000) % 100 == 2
        df = df[reg & df.strength.isin(strengths) & (~df.overload) & (df.duration_s >= MIN_STINT_S)].copy()
        df["season"] = s
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def roster_names(seasons: list[int]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for s in seasons:
        p = C.INTERIM / "roster" / f"{s}.parquet"
        if not p.exists():
            continue
        for r in pd.read_parquet(p).itertuples():
            out.setdefault(int(r.player_id), {"name": r.player_name, "pos": r.position})
    return out


def build_design(stints: pd.DataFrame, dual: bool):
    """Return sparse X, y, raw weights (stint seconds), the player ids per column block, per-row
    game ids (for CV), per-player offence/defence TOI (seconds), and the covariate column map.

    Columns: the 2*P player off/def indicators, then shared NON-per-player covariates oriented to
    the attacking team of each row (home ice, offensive/defensive zone start, trailing/leading
    score state, per-season indicators, 2nd/3rd period). The intercept is added by the solver, not
    here. y is xG per 60 min; the caller normalizes the weights and penalizes only player columns."""
    players = sorted(set().union(*stints.home_skaters, *stints.away_skaters))
    idx = {p: i for i, p in enumerate(players)}
    P = len(players)

    seasons_present = sorted(stints.season.unique())
    season_col = {s: i for i, s in enumerate(seasons_present[1:])}  # first season = reference

    base = 2 * P
    COV = {"home": base, "ozone": base + 1, "dzone": base + 2, "trail": base + 3, "lead": base + 4}
    s_base = base + 5
    for s, i in season_col.items():
        COV[f"season_{s}"] = s_base + i
    p_base = s_base + len(season_col)
    COV["p2"], COV["p3"] = p_base, p_base + 1
    n_cols = p_base + 2

    rows, cols, vals, y, w, games = [], [], [], [], [], []
    off_toi, def_toi = {}, {}
    r = 0

    def emit(off_ids, def_ids, xgf, dur, atk_home, s):
        nonlocal r
        for p in off_ids:
            rows.append(r); cols.append(idx[p]); vals.append(1.0)
            off_toi[p] = off_toi.get(p, 0.0) + dur
        for p in def_ids:
            rows.append(r); cols.append(P + idx[p]); vals.append(1.0)
            def_toi[p] = def_toi.get(p, 0.0) + dur

        def put(name):
            rows.append(r); cols.append(COV[name]); vals.append(1.0)

        if atk_home:
            put("home")
        # zone start (attacking perspective): home O-zone == away D-zone, so flip for away
        if s.start_type == "faceoff" and s.start_zone in ("O", "D"):
            az = s.start_zone if atk_home else ("O" if s.start_zone == "D" else "D")
            put("ozone" if az == "O" else "dzone")
        lead = s.home_lead if atk_home else -s.home_lead
        if lead < 0:
            put("trail")
        elif lead > 0:
            put("lead")
        if s.season in season_col:
            put(f"season_{s.season}")
        period = s.start_g // C.PERIOD_SECONDS + 1
        if period == 2:
            put("p2")
        elif period >= 3:
            put("p3")

        y.append(xgf * 3600.0 / dur)  # xG per 60 minutes
        w.append(dur)
        games.append(s.nhl_game_id)
        r += 1

    for s in stints.itertuples():
        if dual:
            emit(s.home_skaters, s.away_skaters, s.home_xgf, s.duration_s, True, s)
            emit(s.away_skaters, s.home_skaters, s.away_xgf, s.duration_s, False, s)
        elif s.home_n > s.away_n:  # power-play team (more skaters) attacks
            emit(s.home_skaters, s.away_skaters, s.home_xgf, s.duration_s, True, s)
        else:
            emit(s.away_skaters, s.home_skaters, s.away_xgf, s.duration_s, False, s)

    X = sparse.csr_matrix((vals, (rows, cols)), shape=(r, n_cols))
    return X, np.asarray(y), np.asarray(w), players, np.asarray(games), off_toi, def_toi, COV


# --- penalized-GLM linear algebra -------------------------------------------------------------
# Everything goes through the penalized weighted normal equations A β = ZᵀW(working response),
# with A = ZᵀW_g Z + diag(pen). Gaussian = one solve (W_g = obs weights, working response = y);
# Tweedie = IRLS reusing the same step. The same A/B give effective dof and the SE sandwich.

def _xtwx(Z, sw) -> np.ndarray:
    """Dense ZᵀWZ for diagonal weights sw (Z sparse)."""
    WZ = Z.multiply(sw[:, None]).tocsr()
    return (Z.T @ WZ).toarray()


def _xtwy(Z, sw, target) -> np.ndarray:
    return np.asarray(Z.T @ (sw * target)).ravel()


def _solve_pen(B, c, pen):
    """Solve (B + diag(pen)) β = c by Cholesky; jitter the diagonal if not quite PD."""
    A = B + np.diag(pen)
    try:
        L = cho_factor(A, lower=True, check_finite=False)
    except Exception:
        A = A + np.eye(A.shape[0]) * (1e-8 * np.trace(A) / A.shape[0])
        L = cho_factor(A, lower=True, check_finite=False)
    beta = cho_solve(L, c, check_finite=False)
    return beta, A, L


def _edf_and_se(B, A, dispersion):
    """Effective degrees of freedom trace(A⁻¹B) and the SE sandwich sqrt(disp·diag(A⁻¹BA⁻¹))."""
    Ainv = np.linalg.inv(A)
    M = Ainv @ B
    edf = float(np.trace(M))
    cov_diag = np.einsum("ik,ki->i", M, Ainv)
    se = np.sqrt(np.clip(dispersion * cov_diag, 0, None))
    return edf, se


def _tweedie_unit_dev(y, mu, p):
    y = np.clip(y, 0, None)
    mu = np.clip(mu, 1e-9, None)
    return 2.0 * (np.power(y, 2 - p) / ((1 - p) * (2 - p))
                  - y * np.power(mu, 1 - p) / (1 - p)
                  + np.power(mu, 2 - p) / (2 - p))


def _deviance(y, mu, w, family, power) -> float:
    if family == "gaussian":
        return float(np.sum(w * (y - mu) ** 2))
    return float(np.sum(w * _tweedie_unit_dev(y, mu, power)))


def _fit_glm(Z, y, w, pen, family, power, B_full=None, c_full=None, beta_init=None) -> dict:
    """Fit one penalized GLM at a fixed penalty vector. Returns coef + the matrices/diagnostics
    needed downstream (B = final Fisher ZᵀW_gZ, A = B + diag(pen), iters, convergence, deviance)."""
    n, m = Z.shape
    if family == "gaussian":
        B = _xtwx(Z, w) if B_full is None else B_full
        c = _xtwy(Z, w, y) if c_full is None else c_full
        beta, A, _ = _solve_pen(B, c, pen)
        mu = Z @ beta
        return {"beta": beta, "mu": mu, "B": B, "A": A, "n_iter": 1,
                "converged": True, "max_step": 0.0,
                "deviance": _deviance(y, mu, w, family, power)}

    # Tweedie, log link: IRLS
    if beta_init is not None:
        beta = beta_init.copy()
        eta = np.clip(Z @ beta, -30, 30)
        mu = np.exp(eta)
    else:
        ybar = float(np.average(y, weights=w))
        mu = np.full(n, max(ybar, 1e-3))
        eta = np.log(mu)
        beta = np.zeros(m)
    dev_old, converged, B, A, max_step = np.inf, False, None, None, 0.0
    n_iter = IRLS_MAX_ITER
    for it in range(1, IRLS_MAX_ITER + 1):
        irls_w = w * np.power(mu, 2 - power)            # w·(dμ/dη)²/V(μ) = w·μ^(2-p) for log link
        zwork = eta + (y - mu) / mu                      # working response
        B = _xtwx(Z, irls_w)
        c = _xtwy(Z, irls_w, zwork)
        beta_new, A, _ = _solve_pen(B, c, pen)
        max_step = float(np.max(np.abs(beta_new - beta)))
        beta = beta_new
        eta = np.clip(Z @ beta, -30, 30)
        mu = np.exp(eta)
        dev = _deviance(y, mu, w, family, power)
        if abs(dev_old - dev) <= IRLS_TOL * (abs(dev) + IRLS_TOL):
            converged, n_iter = True, it
            break
        dev_old = dev
    return {"beta": beta, "mu": mu, "B": B, "A": A, "n_iter": n_iter,
            "converged": converged, "max_step": max_step,
            "deviance": _deviance(y, mu, w, family, power)}


def _lambda_sweep(Z, y, w, games, pen_unit, lambdas, family, power,
                  B_full, c_full, null_dev, n_splits=5) -> dict:
    """For each λ: grouped-by-game CV (mean + per-fold) and full-data edf / train fit / explained
    deviance. The whole curve is logged so the λ-scale behavior is auditable after the fact."""
    n, m = Z.shape
    cv = {lam: [] for lam in lambdas}
    n_groups = len(np.unique(games))
    splits = min(n_splits, n_groups)
    if splits >= 2:
        gkf = GroupKFold(n_splits=splits)
        for tr, va in gkf.split(Z, y, games):
            Ztr, ytr, wtr, Zva, yva, wva = Z[tr], y[tr], w[tr], Z[va], y[va], w[va]
            if family == "gaussian":
                Btr, ctr = _xtwx(Ztr, wtr), _xtwy(Ztr, wtr, ytr)
                for lam in lambdas:
                    beta, _, _ = _solve_pen(Btr, ctr, pen_unit * lam)
                    pred = Zva @ beta
                    cv[lam].append(float(np.average((yva - pred) ** 2, weights=wva)))
            else:
                beta_prev = None
                for lam in lambdas:                      # ascending λ -> warm-start IRLS
                    res = _fit_glm(Ztr, ytr, wtr, pen_unit * lam, family, power, beta_init=beta_prev)
                    beta_prev = res["beta"]
                    pred = np.exp(np.clip(Zva @ res["beta"], -30, 30))
                    cv[lam].append(_deviance(yva, pred, wva, family, power) / float(wva.sum()))

    out = {}
    beta_prev = None
    for lam in lambdas:
        if family == "gaussian":
            beta, A, _ = _solve_pen(B_full, c_full, pen_unit * lam)
            mu = Z @ beta
            edf, _ = _edf_and_se(B_full, A, 1.0)
            dev = _deviance(y, mu, w, family, power)
            train = {"train_wmse": round(float(np.average((y - mu) ** 2, weights=w)), 4)}
        else:
            res = _fit_glm(Z, y, w, pen_unit * lam, family, power, beta_init=beta_prev)
            beta_prev = res["beta"]
            edf, _ = _edf_and_se(res["B"], res["A"], 1.0)
            dev = res["deviance"]
            train = {"train_deviance": round(dev, 2)}
        out[lam] = {
            "cv_mean": round(float(np.mean(cv[lam])), 4) if cv[lam] else None,
            "cv_folds": [round(x, 4) for x in cv[lam]],
            "edf": round(edf, 1),
            "explained_deviance": round(1 - dev / null_dev, 4) if null_dev > 0 else None,
            **train,
        }
    return out


def fit(stints: pd.DataFrame, names: dict, spec: Spec,
        family: str = "gaussian", power: float = TWEEDIE_POWER) -> tuple[pd.DataFrame, dict]:
    X, y, w_raw, players, games, off_toi, def_toi, cov = build_design(stints, spec.dual)
    P = len(players)
    n = len(y)
    w = w_raw * (n / w_raw.sum())                         # normalize so Σw = n (scale-free λ)
    Z = sparse.hstack([X, sparse.csr_matrix(np.ones((n, 1)))], format="csr")
    m = Z.shape[1]
    icol = m - 1
    pen_unit = np.zeros(m)
    pen_unit[:2 * P] = 1.0                                # penalize player columns only

    B0 = _xtwx(Z, w)                                      # ZᵀWZ (Gaussian Fisher; reused below)
    c0 = _xtwy(Z, w, y)
    diagB = np.diag(B0)[:2 * P]
    med_curv = float(np.median(diagB[diagB > 0]))         # data-scale anchor for λ
    lambdas = tuple(med_curv * mlt for mlt in LAMBDA_MULTS)

    ybar = float(np.average(y, weights=w))
    null_dev = _deviance(y, np.full(n, max(ybar, 1e-9)), w, family, power)

    sweep = _lambda_sweep(Z, y, w, games, pen_unit, lambdas, family, power,
                          B_full=(B0 if family == "gaussian" else None),
                          c_full=(c0 if family == "gaussian" else None), null_dev=null_dev)
    scored = {lam: v["cv_mean"] for lam, v in sweep.items() if v["cv_mean"] is not None}
    lam = min(scored, key=scored.get) if scored else lambdas[len(lambdas) // 2]
    pinned = lam in (lambdas[0], lambdas[-1])

    res = _fit_glm(Z, y, w, pen_unit * lam, family, power,
                   B_full=(B0 if family == "gaussian" else None),
                   c_full=(c0 if family == "gaussian" else None))
    beta, mu, B, A = res["beta"], res["mu"], res["B"], res["A"]
    edf, se = _edf_and_se(B, A, 1.0)
    dispersion = res["deviance"] / max(n - edf, 1.0)
    se = se * np.sqrt(dispersion)

    intercept = float(beta[icol])
    base_rate = float(np.exp(intercept)) if family == "tweedie" else intercept

    # report in site units (xGF/60 delta). Gaussian coef already is one; Tweedie linearizes the
    # log-rate against the baseline: Δ = base·(e^β − 1), SE by the delta method.
    if family == "tweedie":
        coef_d = base_rate * (np.exp(beta) - 1.0)
        se_d = base_rate * np.exp(beta) * se
    else:
        coef_d, se_d = beta, se

    rows = []
    for i, p in enumerate(players):
        info = names.get(int(p), {"name": f"#{p}", "pos": None})
        rows.append({
            "player_id": int(p), "name": info["name"], "pos": info["pos"],
            spec.off: round(float(coef_d[i]), 4), f"{spec.off}_se": round(float(se_d[i]), 4),
            f"{spec.off}_toi": round(off_toi.get(p, 0.0) / 60.0, 1),
            spec.deff: round(float(coef_d[P + i]), 4), f"{spec.deff}_se": round(float(se_d[P + i]), 4),
            f"{spec.deff}_toi": round(def_toi.get(p, 0.0) / 60.0, 1),
        })

    # covariates kept in natural-parameter units (xGF/60 for Gaussian, log-rate for Tweedie) for
    # debugging; `family` labels the interpretation.
    covariates = {name: {"coef": round(float(beta[col]), 4), "se": round(float(se[col]), 4)}
                  for name, col in cov.items()}
    resid = y - mu
    rq = {str(q): round(float(np.quantile(resid, q)), 4) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
    eig = np.linalg.eigvalsh(A)
    explained = (1 - res["deviance"] / null_dev) if null_dev > 0 else None

    meta = {
        "model": spec.key,
        "family": family,
        "tweedie_power": power if family == "tweedie" else None,
        "strengths": list(spec.strengths),
        "seasons": sorted(int(s) for s in stints.season.unique()),
        "weight_norm": "dur/mean(dur) so sum(w)=n",
        "penalty": "player columns only (intercept + covariates free)",
        "lambda": round(lam, 3),
        "lambda_mult": round(lam / med_curv, 4),
        "median_player_curvature": round(med_curv, 3),
        "lambda_grid": [round(x, 3) for x in lambdas],
        "lambda_mults": list(LAMBDA_MULTS),
        "lambda_pinned": bool(pinned),
        "lambda_sweep": {str(round(lm, 3)): sweep[lm] for lm in lambdas},
        "edf": round(edf, 1),
        "n_params": int(m),
        "intercept": round(intercept, 4),
        "baseline_xgf60": round(base_rate, 4),
        "home_ice": round(float(beta[cov["home"]]), 4),
        "covariates": covariates,
        "train_wmse": round(float(np.average((y - mu) ** 2, weights=w)), 4),
        "train_deviance": round(res["deviance"], 2),
        "null_deviance": round(null_dev, 2),
        "explained_deviance": round(explained, 4) if explained is not None else None,
        "dispersion": round(dispersion, 4),
        "residual_quantiles": rq,
        "irls_iters": int(res["n_iter"]),
        "converged": bool(res["converged"]),
        "irls_max_step": round(float(res["max_step"]), 6),
        "cond_number": round(float(eig[-1] / max(eig[0], 1e-12)), 1),
        "min_eig": round(float(eig[0]), 4),
        "max_eig": round(float(eig[-1]), 1),
        "n_obs": int(n),
        "n_stints": int(len(stints)),
        "n_players": int(P),
    }
    return pd.DataFrame(rows), meta


def _write_meta(meta: dict) -> None:
    """Record the full fit (config/provenance, the whole λ sweep, fit quality, conditioning,
    residuals, every covariate coef+SE) to the logs tree: a latest-snapshot JSON per
    (model, seasons, family), plus an appended line in the fit-history log."""
    C.LOGS_MODEL.mkdir(parents=True, exist_ok=True)
    label = "+".join(map(str, meta["seasons"]))
    suffix = "" if meta["family"] == "gaussian" else f"_{meta['family']}"
    (C.LOGS_MODEL / f"{meta['model']}_{label}{suffix}.meta.json").write_text(json.dumps(meta, indent=2))
    record = {"ts": datetime.now().isoformat(timespec="seconds"), **meta}
    with (C.LOGS / "model_fits.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")


def _cache_path(spec: Spec, seasons: list[int], family: str):
    label = "+".join(map(str, seasons))
    suffix = "" if family == "gaussian" else f"_{family}"
    return C.MODELS / f"{spec.key}_{label}{suffix}.parquet"


def _cache_fresh(path, seasons: list[int]) -> bool:
    """A cached fit is stale if any of its source stint files is newer than it."""
    pm = path.stat().st_mtime
    for s in seasons:
        sp = C.PROCESSED / "stints" / f"{s}.parquet"
        if sp.exists() and sp.stat().st_mtime > pm:
            return False
    return True


def fit_cached(seasons: list[int], spec: Spec, names: dict | None = None,
               family: str = "gaussian", power: float = TWEEDIE_POWER) -> pd.DataFrame:
    """Fit (and cache) the impact coefficients for a season set + family. Reads the saved parquet
    if present and fresh, so per-season and pooled fits are computed once and reused by export."""
    path = _cache_path(spec, seasons, family)
    if path.exists() and _cache_fresh(path, seasons):
        return pd.read_parquet(path)
    stints = load_stints(seasons, spec.strengths)
    if stints.empty:
        return pd.DataFrame()
    coef, meta = fit(stints, names or roster_names(seasons), spec, family, power)
    C.MODELS.mkdir(parents=True, exist_ok=True)
    coef.to_parquet(path, index=False)
    _write_meta(meta)
    return coef


def sniff(coef: pd.DataFrame, spec: Spec, label: str) -> None:
    off, deff = spec.off, spec.deff
    o = coef[coef[f"{off}_toi"] >= spec.min_toi].sort_values(off, ascending=False).head(10)
    print(f"\n[{label}] top {off} (xGF/60 added, ±95% CI):")
    for r in o.itertuples():
        print(f"    {getattr(r, off):+.3f} ±{1.96 * getattr(r, f'{off}_se'):.3f}  "
              f"{r.name} ({r.pos}, {getattr(r, f'{off}_toi'):.0f} min)")
    d = coef[coef[f"{deff}_toi"] >= spec.min_toi].sort_values(deff).head(10)
    print(f"[{label}] best {deff} (xGA/60 suppressed, most negative, ±95% CI):")
    for r in d.itertuples():
        print(f"    {getattr(r, deff):+.3f} ±{1.96 * getattr(r, f'{deff}_se'):.3f}  "
              f"{r.name} ({r.pos}, {getattr(r, f'{deff}_toi'):.0f} min)")


def run(seasons: list[int], pool: bool, specs: list[Spec],
        family: str = "gaussian", power: float = TWEEDIE_POWER) -> None:
    C.MODELS.mkdir(parents=True, exist_ok=True)
    names = roster_names(seasons)
    groups = [seasons] if pool else [[s] for s in seasons]
    for spec in specs:
        for grp in groups:
            stints = load_stints(grp, spec.strengths)
            label = "+".join(map(str, grp))
            if stints.empty:
                print(f"\n[{spec.key} {label}] no stints for {spec.strengths} — skipping")
                continue
            coef, meta = fit(stints, names, spec, family, power)
            print(f"\n=== {spec.key} ({'/'.join(spec.strengths)}) — {family} — seasons {label} : "
                  f"{len(stints):,} stints, {meta['n_obs']:,} obs, {meta['n_players']} players, "
                  f"λ={meta['lambda']:.1f} (×{meta['lambda_mult']} curv), edf={meta['edf']}/"
                  f"{meta['n_params']}, explained_dev={meta['explained_deviance']} ===")
            out = _cache_path(spec, grp, family)
            coef.to_parquet(out, index=False)
            _write_meta(meta)
            print(f"    -> {out.name} (+ logs/model/) cond={meta['cond_number']:.0f}"
                  f"{'  ⚠ λ pinned at grid edge' if meta['lambda_pinned'] else ''}"
                  f"{'  ⚠ IRLS not converged' if not meta['converged'] else ''}")
            sniff(coef, spec, label)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Fit isolated-impact models (EV and special teams)")
    p.add_argument("--season", type=int, default=None, help="one season (default: all available)")
    p.add_argument("--pool", action="store_true", help="pool all available seasons into one fit")
    p.add_argument("--model", choices=[*SPECS, "all"], default="all")
    p.add_argument("--family", choices=["gaussian", "tweedie"], default="gaussian",
                   help="response family: gaussian (per-60 rate) or tweedie (log-link GLM)")
    p.add_argument("--tweedie-power", type=float, default=TWEEDIE_POWER, help="Tweedie variance power (1<p<2)")
    args = p.parse_args(argv)
    seasons = [args.season] if args.season else available_seasons()
    if not seasons:
        raise SystemExit("no processed seasons available — run `make stints` first")
    specs = list(SPECS.values()) if args.model == "all" else [SPECS[args.model]]
    run(seasons, args.pool, specs, args.family, args.tweedie_power)


if __name__ == "__main__":
    main()
