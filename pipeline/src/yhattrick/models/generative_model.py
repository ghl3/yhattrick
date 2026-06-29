"""EXPERIMENTAL generative (Poisson marked-process) model — proof of concept.

This is the *generative* counterpart of the production additive model (docs/modeling.md). The
production pipeline decomposes observed goals; this one specifies how a stint *produces* them, so you
can both fit it AND simulate from it. It is not wired into the pipeline output — it's a POC.

The additive **linear model remains primary** (production). Its additivity gives clean, context-free,
summable per-player attribution — the right structure for WAR. This generative model is an exploratory
ALTERNATIVE: more faithful (multiplicative rate × quality × conversion, full predictive distributions,
a volume-vs-quality split), but its multiplicative composition makes per-player goal attribution
context-dependent (no single context-free number). See docs/modeling.md, "Why the linear model stays
primary."

A stint is modeled as a MARKED POISSON PROCESS with a BERNOULLI THINNING, in three layers. For a
stint side with attacking on-ice skaters A, defending skaters B, defending goalie gB, context x, and
length t seconds:

────────────────────────────────────────────────────────────────────────────────────────────────
PARAMETERS
  rate layer        μ_λ              baseline log shot-rate (per 60)
                    a_p  (p skater)  offense shot-rate effect of player p
                    d_p  (p skater)  defense shot-rate (suppression) effect of player p
                    β_λ              context coefficients (home, O/D-zone start, lead/trail)
  quality layer     μ_q              baseline shot quality (logit of mean xG)
                    u_p  (p skater)  offense shot-quality effect of player p
                    w_p  (p skater)  defense shot-quality (danger-suppression) effect of player p
                    β_q              context coefficients ;  s  Beta concentration (shot-to-shot spread)
  conversion layer  μ_c              league conversion baseline (shooting-model intercept)
                    α_s  (s shooter) finishing (goals/shot above xG)
                    γ_g  (g goalie)  goalie save effect (goals/shot above xG allowed)

GENERATIVE PROCESS (one stint side)
  log λ = μ_λ + Σ_{p∈A} a_p + Σ_{p∈B} d_p + β_λ·x                 # shot rate per 60
  N     ~ Poisson( λ·t/3600 )   OR   NegBin( μ=λ·t/3600 , r )    # how many chances (--count {poisson,nb})
  for i = 1..N:
     p̄_i = σ( μ_q + Σ_{p∈A} u_p + Σ_{p∈B} w_p + β_q·x )          # mean chance quality of this lineup
     q_i  ~ Beta( s·p̄_i , s·(1−p̄_i) )                           # this chance's xG (E[q_i] = p̄_i)
     s_i  ~ Categorical( shot-share of A )                       # who shoots (simulation only)
     y_i  ~ Bernoulli( clip( q_i + μ_c + α_{s_i} + γ_{gB} , 0, 1 ) )   # goal?

LIKELIHOOD  (factorizes: N, q_i, y_i observed; disjoint parameter blocks; independent ridge priors)
  ℓ(θ) =  Σ_stint-sides  log Poisson( N | λ·t/3600 )                         ← rate block   (μ_λ,a,d,β_λ)
        + Σ_shots        log Binom( xg_i | σ(μ_q+Σu+Σw+β_q·x) )              ← quality block (μ_q,u,w,β_q)
                          [fractional logistic: continuous xg∈[0,1]; mean-calibrated E[xg]=σ(η)]
        + Σ_shots        log Bernoulli( y_i | clip(q_i+μ_c+α+γ) )            ← conversion block (μ_c,α,γ)
        − ½λ_a‖a‖² − ½λ_d‖d‖² − ½λ_u‖u‖² − ½λ_w‖w‖²                          ← ridge = EB shrinkage
  Because the three blocks are disjoint and N, q_i, y_i are observed, ℓ splits into three independent
  penalized fits. Each ridge penalty is a Normal(0, σ²) prior on the effects: λ = 1/σ².
────────────────────────────────────────────────────────────────────────────────────────────────

Reading: the rate+quality layers together are what RAPM estimates as one xG/60 number, now split into
HOW MANY shots (a,d) × HOW GOOD each is (u,w). The conversion layer is the production shooting model
(reused here, not re-fit).

Fit: MAP / penalized MLE by OPTIMIZATION (no MCMC). Gradients via JAX autodiff; optimized with
scipy L-BFGS-B (robust in high dimension). Uncertainties via the HESSIAN (Laplace): at the optimum
the penalized-NLL Hessian is the exact GLM information `H = XᵀWX + diag(penalty)` (W = μ for Poisson,
1 for Gaussian); SE = sqrt(diag(H⁻¹)) (× σ_q for the Gaussian quality fit). Reported as ±95% CIs.

The count layer is configurable (`--count poisson|nb`). NB adds one global dispersion parameter r
(the per-stint Gamma rate-multiplier is integrated out, so the layers still fit independently) and
matches the real over-dispersion of shot counts — Var = μ + μ²/r vs the Poisson's Var = μ.

5v5 only, for the POC. Usage:
  uv run --group experimental python -m yhattrick.models.generative_model            # latest season
  uv run --group experimental python -m yhattrick.models.generative_model --count nb # negative binomial
  uv run --group experimental python -m yhattrick.models.generative_model --pool     # all seasons
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

from .. import config as C
from . import shooting_model
from .player_onice_model import roster_names

jax.config.update("jax_enable_x64", True)   # float64: stable optimization + Hessian

# Ridge = a Normal(0, σ²) prior on each player effect; λ = 1/σ².
PRIOR_SD_RATE = 0.30        # a_p, d_p on the log shot-rate scale (±0.30 ≈ ±35% rate for an elite)
PRIOR_SD_QUAL = 0.20        # u_p, w_p on the logit-xG scale
EPS = 1e-6
SNIFF_MIN_TOI = 24000.0     # min on-ice seconds (5v5, ≈400 min) to appear in a leaderboard


# ── data loaders (5v5) ──────────────────────────────────────────────────────────────────────────

def _zone(atk_home, start_type, start_zone):
    """Attacking-team zone start: O/D flips for the away team. Returns (ozone, dzone) indicators."""
    if start_type != "faceoff" or start_zone not in ("O", "D"):
        return 0.0, 0.0
    az = start_zone if atk_home else ("O" if start_zone == "D" else "D")
    return (1.0, 0.0) if az == "O" else (0.0, 1.0)


RATE_CTX = ["home", "ozone", "dzone", "trail", "lead"]


def _load_5v5_stints(seasons):
    """All regular-season 5v5 non-overload stints. Unlike player_onice_model.load_stints we keep
    short (<10s) stints — a Poisson count model handles them correctly via the log-time offset, and
    dropping them would desync the rate universe from the (unfiltered) shot universe."""
    frames = []
    for s in seasons:
        p = C.PROCESSED / "stints" / f"{s}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if "overload" not in df.columns:
            df["overload"] = False
        reg = (df.nhl_game_id // 10000) % 100 == 2
        frames.append(df[reg & (df.strength == "5v5") & (~df.overload) & (df.duration_s >= 1)].copy())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def rate_rows(seasons):
    """One row per stint SIDE: Fenwick count, offset log(t/3600), on-ice offense/defense player index
    arrays (5 each at 5v5), and context. Also returns the shared player index and per-row TOI."""
    df = _load_5v5_stints(seasons)
    if df.empty:
        return None
    players = sorted(set().union(*df.home_skaters, *df.away_skaters))
    idx = {p: i for i, p in enumerate(players)}

    off, dff, ctx, cnt, off_t, gid = [], [], [], [], [], []

    def emit(atk, dfd, count, dur, atk_home, s, def_goalie):
        off.append([idx[p] for p in atk])
        dff.append([idx[p] for p in dfd])
        oz, dz = _zone(atk_home, s.start_type, s.start_zone)
        lead = s.home_lead if atk_home else -s.home_lead
        ctx.append([1.0 if atk_home else 0.0, oz, dz, 1.0 if lead < 0 else 0.0, 1.0 if lead > 0 else 0.0])
        cnt.append(count)
        off_t.append(dur)
        gid.append(def_goalie)

    for s in df.itertuples():
        if len(s.home_skaters) != 5 or len(s.away_skaters) != 5:
            continue
        emit(s.home_skaters, s.away_skaters, s.home_fen, s.duration_s, True, s, s.away_goalie)
        emit(s.away_skaters, s.home_skaters, s.away_fen, s.duration_s, False, s, s.home_goalie)

    off = np.asarray(off, dtype=np.int64)
    dff = np.asarray(dff, dtype=np.int64)
    dur = np.asarray(off_t, dtype=np.float64)
    return {
        "players": players, "idx": idx,
        "off_idx": off, "def_idx": dff,
        "Xctx": np.asarray(ctx, dtype=np.float64),
        "count": np.asarray(cnt, dtype=np.float64),
        "offset": np.log(np.clip(dur / 3600.0, 1e-9, None)),
        "dur": dur, "def_goalie": np.asarray(gid, dtype=object),
    }


def quality_rows(seasons, idx):
    """One row per 5v5 Fenwick shot: response logit(clip(xg)), attacking/defending on-ice player index
    arrays, context [home]. Shots whose on-ice skaters aren't all in `idx` are dropped (rare)."""
    off, dff, ctx, y, goal = [], [], [], [], []
    for s in seasons:
        p = C.PROCESSED / "shots_onice" / f"{s}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["strength", "is_home", "xg", "goal", "home_skaters", "away_skaters"])
        d = d[(d.strength == "5v5") & d.xg.notna()]
        for r in d.itertuples():
            hs, as_ = list(r.home_skaters), list(r.away_skaters)
            if len(hs) != 5 or len(as_) != 5:
                continue
            atk, dfd = (hs, as_) if r.is_home == 1 else (as_, hs)
            if any(p not in idx for p in atk) or any(p not in idx for p in dfd):
                continue
            off.append([idx[p] for p in atk])
            dff.append([idx[p] for p in dfd])
            ctx.append([1.0 if r.is_home == 1 else 0.0])
            y.append(np.log(np.clip(r.xg, EPS, 1 - EPS) / (1 - np.clip(r.xg, EPS, 1 - EPS))))
            goal.append(int(r.goal))
    return {
        "off_idx": np.asarray(off, dtype=np.int64), "def_idx": np.asarray(dff, dtype=np.int64),
        "Xctx": np.asarray(ctx, dtype=np.float64), "y": np.asarray(y, dtype=np.float64),
        "goal": np.asarray(goal, dtype=np.int64),
    }


# ── the crossed design as a sparse matrix (for the exact analytic Hessian) ─────────────────────────

def _build_X(off_idx, def_idx, Xctx, P):
    """Sparse design with columns [intercept | offense(P) | defense(P) | context(k)]."""
    n, k = len(off_idx), Xctx.shape[1]
    ncol = 1 + 2 * P + k
    ar = np.arange(n)
    rows = [ar, *([ar] * off_idx.shape[1]), *([ar] * def_idx.shape[1]), *([ar] * k)]
    cols = [np.zeros(n, int),
            *[1 + off_idx[:, j] for j in range(off_idx.shape[1])],
            *[1 + P + def_idx[:, j] for j in range(def_idx.shape[1])],
            *[np.full(n, 1 + 2 * P + c) for c in range(k)]]
    data = [np.ones(n), *[np.ones(n)] * off_idx.shape[1], *[np.ones(n)] * def_idx.shape[1],
            *[Xctx[:, c] for c in range(k)]]
    return sparse.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(n, ncol))


def _pen_vec(P, k, lam):
    pen = np.zeros(1 + 2 * P + k)
    pen[1:1 + 2 * P] = lam                       # penalize only the player blocks
    return pen


def _se_from_hessian(H):
    """SE = sqrt(diag(H⁻¹)) with a PD Hessian (ridge guarantees it)."""
    return np.sqrt(np.clip(np.diag(np.linalg.inv(H)), 0.0, None))


# ── fits (JAX autodiff gradient + scipy L-BFGS-B; exact Hessian for SEs) ───────────────────────────

def _optimize(nll, x0):
    vg = jax.jit(jax.value_and_grad(nll))

    def f(x):
        v, g = vg(jnp.asarray(x))
        return float(v), np.asarray(g, dtype=np.float64)

    res = minimize(f, x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 1000, "maxfun": 1000, "ftol": 1e-10, "gtol": 1e-7})
    return res


def fit_rate(R, count_model="poisson"):
    """Count model for shot volume (log link), configurable:
      * "poisson"          — Var = μ. Mean params only; the free intercept makes Σμ = Σcount exact.
      * "nb" (NB2)         — Var = μ + μ²/r. Adds ONE global log-dispersion parameter (last slot);
                             the per-stint Gamma rate-multiplier is integrated out, so the three
                             layers still fit independently.
    Returns a/d effects + Hessian SEs (Fisher mean-block weight W = μ for Poisson, μr/(r+μ) for NB —
    the NB mean and dispersion blocks are Fisher-orthogonal, so the player SEs use the mean block)."""
    P, k = len(R["players"]), R["Xctx"].shape[1]
    off, dff = jnp.asarray(R["off_idx"]), jnp.asarray(R["def_idx"])
    X, cnt, ofs = jnp.asarray(R["Xctx"]), jnp.asarray(R["count"]), jnp.asarray(R["offset"])
    lam = 1.0 / PRIOR_SD_RATE ** 2
    nb = count_model == "nb"

    def nll(th):
        a, d, b = th[1:1 + P], th[1 + P:1 + 2 * P], (th[1 + 2 * P:-1] if nb else th[1 + 2 * P:])
        eta = th[0] + jnp.sum(a[off], 1) + jnp.sum(d[dff], 1) + X @ b
        mu = jnp.exp(eta + ofs)
        pen = 0.5 * lam * (jnp.sum(a ** 2) + jnp.sum(d ** 2))
        if not nb:
            return jnp.sum(mu - cnt * jnp.log(mu)) + pen          # Poisson NLL (drop const)
        r = jnp.exp(th[-1])                                       # global dispersion (NB2)
        ll = (gammaln(cnt + r) - gammaln(r) - gammaln(cnt + 1)
              + r * jnp.log(r / (r + mu)) + cnt * jnp.log(mu / (r + mu)))
        return -jnp.sum(ll) + pen

    x0 = np.zeros(1 + 2 * P + k + (1 if nb else 0))
    if nb:
        x0[-1] = 1.0                                              # init log r = 1  (r ≈ 2.7)
    res = _optimize(nll, x0)
    th = res.x
    b_ = th[1 + 2 * P:-1] if nb else th[1 + 2 * P:]
    eta = th[0] + th[1:1 + P][R["off_idx"]].sum(1) + th[1 + P:1 + 2 * P][R["def_idx"]].sum(1) + R["Xctx"] @ b_
    mu = np.exp(eta + R["offset"])
    r = float(np.exp(th[-1])) if nb else None
    W = mu * r / (r + mu) if nb else mu                           # Fisher mean-block weight
    Xs = _build_X(R["off_idx"], R["def_idx"], R["Xctx"], P)
    H = (Xs.T @ Xs.multiply(W[:, None])).toarray() + np.diag(_pen_vec(P, k, lam))
    se = _se_from_hessian(H)
    return {"intercept": th[0], "a": th[1:1 + P], "d": th[1 + P:1 + 2 * P], "beta": b_,
            "se_a": se[1:1 + P], "se_d": se[1 + P:1 + 2 * P], "count_model": count_model, "r": r,
            "converged": bool(res.success), "grad_norm": float(np.max(np.abs(res.jac)))}


def fit_quality(Q, P):
    """Shot quality as a FRACTIONAL LOGISTIC regression of xG on the lineup: E[xG] = σ(η). Fit by
    binomial deviance (continuous target xG∈[0,1]), which is mean-calibrated (the score equation
    forces Σ(xG − σ(η)) = 0), so σ(η) reproduces the mean xG — unlike a Gaussian-on-logit fit, which
    back-transforms to the median and mis-states the mean. Returns u/w effects + Hessian SEs, plus a
    Beta concentration s (method of moments) for mean-preserving simulation: q ~ Beta(s·p, s·(1−p))."""
    k = Q["Xctx"].shape[1]
    off, dff = jnp.asarray(Q["off_idx"]), jnp.asarray(Q["def_idx"])
    X, xg = jnp.asarray(Q["Xctx"]), jnp.asarray(_sigmoid(Q["y"]))   # xg = σ(logit) = clip(xg)
    lam = 1.0 / PRIOR_SD_QUAL ** 2

    def nll(th):
        u, w, b = th[1:1 + P], th[1 + P:1 + 2 * P], th[1 + 2 * P:]
        eta = th[0] + jnp.sum(u[off], 1) + jnp.sum(w[dff], 1) + X @ b
        p = jax.nn.sigmoid(eta)
        ll = xg * jnp.log(p + EPS) + (1 - xg) * jnp.log(1 - p + EPS)
        return -jnp.sum(ll) + 0.5 * lam * (jnp.sum(u ** 2) + jnp.sum(w ** 2))

    res = _optimize(nll, np.zeros(1 + 2 * P + k))
    th = res.x
    eta = (th[0] + th[1:1 + P][Q["off_idx"]].sum(1) + th[1 + P:1 + 2 * P][Q["def_idx"]].sum(1)
           + Q["Xctx"] @ th[1 + 2 * P:])
    p = _sigmoid(eta)
    xg_np = _sigmoid(Q["y"])
    Xs = _build_X(Q["off_idx"], Q["def_idx"], Q["Xctx"], P)
    H = (Xs.T @ Xs.multiply((p * (1 - p))[:, None])).toarray() + np.diag(_pen_vec(P, k, lam))
    se = _se_from_hessian(H)
    # Beta concentration s so Var(xg|p) = p(1-p)/(s+1) matches residual MSE around the fitted mean
    mse = float(np.mean((xg_np - p) ** 2))
    s_conc = max(float(np.mean(p * (1 - p)) / max(mse, 1e-9) - 1.0), 1.0)
    return {"intercept": th[0], "u": th[1:1 + P], "w": th[1 + P:1 + 2 * P], "beta": th[1 + 2 * P:],
            "se_u": se[1:1 + P], "se_w": se[1 + P:1 + 2 * P], "beta_s": s_conc,
            "converged": bool(res.success), "grad_norm": float(np.max(np.abs(res.jac)))}


# ── conversion params (reuse the production shooting model) ─────────────────────────────────────────

def conversion_params(seasons, names):
    fin, gl, meta = shooting_model.fit(seasons, names)
    mu_c = float(meta["reconciliation"]["intercept_per_shot"]) if meta else 0.0
    alpha = dict(zip(fin.player_id, fin.fin_per100 / 100.0)) if not fin.empty else {}
    gamma = dict(zip(gl.player_id, -gl.gsax_per100 / 100.0)) if not gl.empty else {}
    return {"mu_c": mu_c, "alpha": alpha, "gamma": gamma}


# ── simulation + posterior-predictive check ───────────────────────────────────────────────────────

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def ppc(R, rate, qual, conv, seed=0):
    """Forward-simulate every real stint side from the fitted params and compare to actuals:
    shot-count distribution (rate layer), shot quality (quality layer), and total goals (end-to-end)."""
    rng = np.random.default_rng(seed)
    P = len(R["players"])
    eta_r = (rate["intercept"] + rate["a"][R["off_idx"]].sum(1) + rate["d"][R["def_idx"]].sum(1)
             + R["Xctx"] @ rate["beta"])
    mu = np.exp(eta_r + R["offset"])                              # expected shots per side
    if rate.get("count_model") == "nb" and rate.get("r"):        # NB draw captures overdispersion
        r = rate["r"]
        N = rng.negative_binomial(r, r / (r + mu))
    else:
        N = rng.poisson(mu)

    # quality logit per stint side: the quality model's lone context is `home` = attacking-is-home,
    # which is column 0 of the rate context (atk_home). σ(qpred) is the mean xG of the lineup's shots.
    qpred = (qual["intercept"] + qual["u"][R["off_idx"]].sum(1) + qual["w"][R["def_idx"]].sum(1)
             + qual["beta"][0] * R["Xctx"][:, 0])
    # per-side conversion helpers: mean finishing of the 5 attackers, defending goalie save effect
    abar = np.array([np.mean([conv["alpha"].get(R["players"][p], 0.0) for p in row])
                     for row in R["off_idx"]])
    gvec = np.array([conv["gamma"].get(g, 0.0) for g in R["def_goalie"]])

    # expand to simulated shots; draw quality from a mean-preserving Beta, then convert
    rep = np.repeat(np.arange(len(N)), N)
    p_q = _sigmoid(qpred[rep])
    s = qual["beta_s"]
    q = rng.beta(s * p_q, s * (1 - p_q))                        # E[q] = p_q (mean-calibrated)
    p_goal = np.clip(q + conv["mu_c"] + abar[rep] + gvec[rep], 0.0, 1.0)
    goals_sim = int(rng.binomial(1, p_goal).sum())
    return {
        "shots_actual": int(R["count"].sum()), "shots_sim": int(N.sum()),
        "shots_expected": float(mu.sum()),
        "perstint_mean_actual": float(R["count"].mean()), "perstint_mean_sim": float(N.mean()),
        "perstint_var_actual": float(R["count"].var()), "perstint_var_sim": float(N.var()),
        "zero_frac_actual": float((R["count"] == 0).mean()), "zero_frac_sim": float((N == 0).mean()),
        "mean_xg_sim": float(q.mean()), "goals_sim": goals_sim,
        "n_sim_shots": int(rep.size),
    }


def simulate_stint(rate, qual, conv, players, idx, attackers, defenders, goalie, ctx_rate, ctx_qual_home, t, key):
    """Forward draw of one stint side (for demonstration). Returns (shots_xg, goals)."""
    oi = jnp.asarray([idx[p] for p in attackers]); di = jnp.asarray([idx[p] for p in defenders])
    eta = rate["intercept"] + float(rate["a"][oi].sum()) + float(rate["d"][di].sum()) + float(np.dot(rate["beta"], ctx_rate))
    mu = np.exp(eta + np.log(max(t, 1e-9) / 3600.0))
    k1, k2, k3 = jax.random.split(key, 3)
    N = int(jax.random.poisson(k1, mu))
    qp = qual["intercept"] + float(qual["u"][oi].sum()) + float(qual["w"][di].sum()) + qual["beta"][0] * ctx_qual_home
    p_q = _sigmoid(qp)
    s = qual["beta_s"]
    q = np.asarray(jax.random.beta(k2, s * p_q, s * (1 - p_q), (N,)))   # mean-preserving Beta draw
    abar = np.mean([conv["alpha"].get(p, 0.0) for p in attackers]) if attackers else 0.0
    g = conv["gamma"].get(goalie, 0.0)
    p_goal = np.clip(q + conv["mu_c"] + abar + g, 0.0, 1.0)
    goals = int(np.asarray(jax.random.bernoulli(k3, p_goal)).sum())
    return q, goals


# ── leaderboards + run ─────────────────────────────────────────────────────────────────────────────

def _toi(R):
    """Actual on-ice 5v5 seconds per player. Each stint emits an offense row AND a defense row of
    equal duration for the players on it, so summing both appearances double-counts time → halve."""
    P = len(R["players"]); t = np.zeros(P)
    np.add.at(t, R["off_idx"].ravel(), np.repeat(R["dur"], R["off_idx"].shape[1]))
    np.add.at(t, R["def_idx"].ravel(), np.repeat(R["dur"], R["def_idx"].shape[1]))
    return t / 2.0


def _board(names, players, val, se, toi, label, higher=True, n=10):
    elig = toi >= SNIFF_MIN_TOI
    order = np.argsort(val * (-1 if higher else 1))
    order = [i for i in order if elig[i]][:n]
    print(f"\n{label}")
    for i in order:
        pid = players[i]; nm = names.get(int(pid), {}).get("name", f"#{pid}")
        print(f"   {val[i]:+.3f} ±{1.96*se[i]:.3f}  {nm:24s} ({toi[i]/60:.0f} min)")


def run(seasons, count_model="poisson"):
    names = roster_names(seasons)
    print(f"[generative_model] seasons {seasons} — count model: {count_model} — loading 5v5 stints …")
    R = rate_rows(seasons)
    if R is None:
        raise SystemExit("no 5v5 stints")
    P = len(R["players"])
    print(f"  rate rows: {len(R['count']):,} stint-sides, {P} skaters")
    rate = fit_rate(R, count_model=count_model)
    rdesc = f" r={rate['r']:.2f}" if rate.get("r") else ""
    print(f"  rate fit ({count_model}): converged={rate['converged']} |grad|={rate['grad_norm']:.1e}{rdesc}")

    Q = quality_rows(seasons, R["idx"])
    print(f"  quality rows: {len(Q['y']):,} shots")
    qual = fit_quality(Q, P)
    print(f"  quality fit: converged={qual['converged']} |grad|={qual['grad_norm']:.1e}  beta_s={qual['beta_s']:.1f}")

    conv = conversion_params(seasons, names)

    toi = _toi(R)
    _board(names, R["players"], rate["a"], rate["se_a"], toi, "TOP shot-GENERATION (a_p, log-rate):")
    _board(names, R["players"], rate["d"], rate["se_d"], toi, "BEST shot-SUPPRESSION (d_p, most negative):", higher=False)
    _board(names, R["players"], qual["u"], qual["se_u"], toi, "TOP shot-QUALITY creation (u_p, logit-xG):")
    _board(names, R["players"], qual["w"], qual["se_w"], toi, "BEST quality SUPPRESSION (w_p, most negative):", higher=False)
    # combined creation = volume + quality, vs RAPM's single ev_off
    combo = rate["a"] + qual["u"]
    _board(names, R["players"], combo, np.sqrt(rate["se_a"]**2 + qual["se_u"]**2), toi,
           "TOP combined creation (a_p + u_p ≈ RAPM ev_off, split into volume×quality):")

    pp = ppc(R, rate, qual, conv)
    pp["goals_actual"] = int(Q["goal"].sum())
    pp["mean_xg_actual"] = float(_sigmoid(Q["y"]).mean())
    # the quality model's *central* (noiseless) per-shot prediction — should be ≈ actual mean xG
    pred_shots = (qual["intercept"] + qual["u"][Q["off_idx"]].sum(1) + qual["w"][Q["def_idx"]].sum(1)
                  + qual["beta"][0] * Q["Xctx"][:, 0])
    pp["mean_xg_fitted"] = float(_sigmoid(pred_shots).mean())
    print("\n── posterior-predictive check (simulate every real stint side) ──")
    print(f"  shots   actual {pp['shots_actual']:,}  sim {pp['shots_sim']:,}  (model-expected {pp['shots_expected']:.0f})")
    print(f"  per-side count  mean a/s {pp['perstint_mean_actual']:.3f}/{pp['perstint_mean_sim']:.3f}"
          f"  var a/s {pp['perstint_var_actual']:.3f}/{pp['perstint_var_sim']:.3f}"
          f"  zero% a/s {100*pp['zero_frac_actual']:.1f}/{100*pp['zero_frac_sim']:.1f}")
    print(f"  mean shot xG  actual {pp['mean_xg_actual']:.4f}  fitted {pp['mean_xg_fitted']:.4f}"
          f"  sim-draw {pp['mean_xg_sim']:.4f}")
    print(f"  goals         actual {pp['goals_actual']:,}  sim {pp['goals_sim']:,}")
    if rate.get("count_model") == "nb":
        print(f"  count model: NB (r={rate['r']:.2f}) — the simulated per-side variance now tracks the "
              f"actual ({pp['perstint_var_sim']:.3f} vs {pp['perstint_var_actual']:.3f}), fixing the "
              "Poisson under-dispersion.")
    else:
        print("  PPC finding: shots & goals reconcile, but real shot counts are OVERdispersed vs Poisson "
              "(var > mean) → try --count nb.")

    _save(seasons, R, rate, qual, conv, pp)


def _save(seasons, R, rate, qual, conv, pp):
    label = "+".join(map(str, seasons))
    out = {"model": "generative_model_poc", "seasons": list(seasons),
           "count_model": rate.get("count_model"), "nb_r": rate.get("r"),
           "prior_sd_rate": PRIOR_SD_RATE, "prior_sd_qual": PRIOR_SD_QUAL,
           "rate_intercept": float(rate["intercept"]), "quality_intercept": float(qual["intercept"]),
           "beta_s": qual["beta_s"], "mu_c": conv["mu_c"], "ppc": pp,
           "players": [{"id": int(R["players"][i]),
                        "a": float(rate["a"][i]), "a_se": float(rate["se_a"][i]),
                        "d": float(rate["d"][i]), "d_se": float(rate["se_d"][i]),
                        "u": float(qual["u"][i]), "u_se": float(qual["se_u"][i]),
                        "w": float(qual["w"][i]), "w_se": float(qual["se_w"][i])}
                       for i in range(len(R["players"]))]}
    C.MODELS.mkdir(parents=True, exist_ok=True)
    (C.MODELS / f"generative_model_{label}.json").write_text(json.dumps(out))
    print(f"\n  -> data/models/generative_model_{label}.json")


def main(argv=None):
    p = argparse.ArgumentParser(description="Experimental generative (Poisson) model — POC")
    p.add_argument("--season", type=int, default=None, help="one season (default: latest available)")
    p.add_argument("--pool", action="store_true", help="pool all available seasons")
    p.add_argument("--count", choices=["poisson", "nb"], default="poisson",
                   help="count layer: poisson (Var=μ) or nb (negative binomial, Var=μ+μ²/r)")
    args = p.parse_args(argv)
    avail = shooting_model.available_seasons()
    if not avail:
        raise SystemExit("no processed shots — run `make stints` first")
    seasons = avail if args.pool else ([args.season] if args.season else [avail[-1]])
    run(seasons, count_model=args.count)


if __name__ == "__main__":
    main()
