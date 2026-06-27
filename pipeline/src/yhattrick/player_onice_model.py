"""Isolated-impact model: each player's per-60 effect on the expected-goal rate, adjusted for
who else is on the ice (linemates and competition), by strength state.

Method (a regularized adjusted plus-minus). A stint contributes one or two observations: the
response is a team's expected goals per 60 minutes during the stint; the predictors are indicator
columns for each on-ice player — an OFFENCE column when his team is attacking and a DEFENCE
column when defending — plus a home-ice term. A ridge (L2) penalty shrinks noisy, low-ice-time
players toward zero and untangles teammates who always play together. We fit separate models per
strength state:

  even strength (5v5)   -> ev_off  (xG/60 a player ADDS),       ev_def  (xG/60 he ALLOWS)
  special teams (5v4)   -> pp_off  (power-play offence added),  pk_def  (penalty-kill xG allowed)

EV uses both attacking perspectives per stint; special teams uses only the power-play team's
attacking perspective, so pp_off is pure PP offence and pk_def is pure PK defence. Each
coefficient carries a standard error (ridge covariance), and offence/defence time-on-ice are
tracked separately (a player's PP minutes differ from his PK minutes).

Built on the borrowed MoneyPuck xG attached to each stint; swappable for our own xG later.
Runs on whatever seasons are present in data/processed/stints (prototype-friendly).

Usage:
  uv run python -m yhattrick.player_onice_model                 # every model, every available season
  uv run python -m yhattrick.player_onice_model --season 2021
  uv run python -m yhattrick.player_onice_model --pool          # pool all available seasons into one fit
  uv run python -m yhattrick.player_onice_model --model pp_pk    # just special teams (or 'ev')
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from . import config as C

MIN_STINT_S = 10         # drop sub-10s line-change stints (extreme per-60 rates, ~no signal)
LAMBDAS = (100, 200, 400, 800, 1600, 3200, 6400, 12800)  # ridge strengths to cross-validate


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
    """Return sparse X, y, weights, the player ids per column block, per-row game ids (for CV),
    and per-player offence/defence TOI (seconds). For dual (EV) each stint emits both attacking
    perspectives; otherwise only the team with more skaters (the power play) attacks.

    Beyond the 2*P player off/def columns, the design carries shared NON-per-player covariates,
    all oriented to the attacking team of each row: home ice, offensive/defensive zone start,
    trailing/leading score state, per-season indicators (era drift), and period (2nd/3rd).
    These absorb deployment/score/era bias so player coefficients better reflect skill; they are
    fit but not reported per player."""
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
    return X, np.asarray(y), np.asarray(w), players, np.asarray(games), off_toi, def_toi


def _wmse(model, X, y, w):
    return float(np.average((y - model.predict(X)) ** 2, weights=w))


def ridge_se(X, y, w, yhat, lam):
    """Analytic ridge standard errors for every coefficient. With A = XᵀWX + λI, the estimator
    β̂ = A⁻¹XᵀWy has covariance σ²·A⁻¹(XᵀWX)A⁻¹ (TOI weights treated as inverse-variance); the
    diagonal gives a per-player SE that grows with low ice time AND with collinearity (linemates
    who never separate). σ² is the weighted residual variance over the effective dof."""
    n, m = X.shape
    Xs = sparse.csr_matrix(X).multiply(np.sqrt(w)[:, None]).tocsr()
    B = (Xs.T @ Xs).toarray()                       # XᵀWX  (m×m)
    Ainv = np.linalg.inv(B + lam * np.eye(m))
    M = Ainv @ B                                    # BLAS matmul
    edf = float(np.trace(M))                        # effective degrees of freedom
    sigma2 = float(np.sum(w * (y - yhat) ** 2)) / max(n - edf, 1.0)
    cov_diag = np.einsum("ik,ki->i", M, Ainv)       # diag(A⁻¹ B A⁻¹)
    return np.sqrt(np.clip(sigma2 * cov_diag, 0, None))


def choose_lambda(X, y, w, games, lambdas=LAMBDAS, n_splits=5):
    """Grouped (by game) CV to pick the ridge strength minimizing weighted MSE."""
    n_groups = len(np.unique(games))
    n_splits = min(n_splits, n_groups)
    if n_splits < 2:
        return lambdas[len(lambdas) // 2], {}
    gkf = GroupKFold(n_splits=n_splits)
    scores = {}
    for lam in lambdas:
        fold = []
        for tr, va in gkf.split(X, y, games):
            m = Ridge(alpha=lam, fit_intercept=True, solver="lsqr")
            m.fit(X[tr], y[tr], sample_weight=w[tr])
            fold.append(_wmse(m, X[va], y[va], w[va]))
        scores[lam] = float(np.mean(fold))
    return min(scores, key=scores.get), scores


def fit(stints: pd.DataFrame, names: dict, spec: Spec) -> tuple[pd.DataFrame, dict]:
    X, y, w, players, games, off_toi, def_toi = build_design(stints, spec.dual)
    lam, _ = choose_lambda(X, y, w, games)
    model = Ridge(alpha=lam, fit_intercept=True, solver="lsqr")
    model.fit(X, y, sample_weight=w)
    P = len(players)
    coef = model.coef_
    se = ridge_se(X, y, w, model.predict(X), lam)

    rows = []
    for i, p in enumerate(players):
        info = names.get(int(p), {"name": f"#{p}", "pos": None})
        rows.append({
            "player_id": int(p), "name": info["name"], "pos": info["pos"],
            spec.off: round(float(coef[i]), 4), f"{spec.off}_se": round(float(se[i]), 4),
            f"{spec.off}_toi": round(off_toi.get(p, 0.0) / 60.0, 1),
            spec.deff: round(float(coef[P + i]), 4), f"{spec.deff}_se": round(float(se[P + i]), 4),
            f"{spec.deff}_toi": round(def_toi.get(p, 0.0) / 60.0, 1),
        })
    meta = {"lambda": lam, "intercept": round(float(model.intercept_), 4),
            "home_ice": round(float(coef[2 * P]), 4), "n_obs": X.shape[0], "n_players": P}
    return pd.DataFrame(rows), meta


def fit_cached(seasons: list[int], spec: Spec, names: dict | None = None) -> pd.DataFrame:
    """Fit (and cache) the impact coefficients for a season set. Reads the saved parquet if
    present, so per-season and pooled fits are computed once and reused (e.g. by export)."""
    label = "+".join(map(str, seasons))
    path = C.MODELS / f"{spec.key}_{label}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    stints = load_stints(seasons, spec.strengths)
    if stints.empty:
        return pd.DataFrame()
    coef, _ = fit(stints, names or roster_names(seasons), spec)
    C.MODELS.mkdir(parents=True, exist_ok=True)
    coef.to_parquet(path, index=False)
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


def run(seasons: list[int], pool: bool, specs: list[Spec]) -> None:
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
            coef, meta = fit(stints, names, spec)
            print(f"\n=== {spec.key} ({'/'.join(spec.strengths)}) — seasons {label} : "
                  f"{len(stints):,} stints, {meta['n_obs']:,} obs, {meta['n_players']} players, "
                  f"lambda={meta['lambda']}, intercept(xGF/60)={meta['intercept']} ===")
            out = C.MODELS / f"{spec.key}_{label}.parquet"
            coef.to_parquet(out, index=False)
            print(f"    -> {out.name}")
            sniff(coef, spec, label)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Fit isolated-impact models (EV and special teams)")
    p.add_argument("--season", type=int, default=None, help="one season (default: all available)")
    p.add_argument("--pool", action="store_true", help="pool all available seasons into one fit")
    p.add_argument("--model", choices=[*SPECS, "all"], default="all")
    args = p.parse_args(argv)
    seasons = [args.season] if args.season else available_seasons()
    if not seasons:
        raise SystemExit("no processed seasons available — run `make stints` first")
    specs = list(SPECS.values()) if args.model == "all" else [SPECS[args.model]]
    run(seasons, args.pool, specs)


if __name__ == "__main__":
    main()
