r"""EXPERIMENTAL generative (Poisson marked-process) model — SHOOTER-RESOLVED proof of concept.

This is the *generative* counterpart of the production additive model (docs/modeling.md). The
production pipeline decomposes observed goals; this one specifies how a stint *produces* them, so you
can both fit it AND simulate from it. It is not wired into the pipeline output — it's a POC.

The additive **linear model remains primary** (production). This generative model is an exploratory
ALTERNATIVE whose distinguishing feature is that it is **shooter-resolved**: the shot a player takes
is modeled separately from the shots he helps his TEAMMATES take, so each player splits into
SCORING (his own shots) and PLAYMAKING (the lift he gives the four teammates on the ice with him) —
without any pass/assist data, because "who shot" and "who was on the ice" are both observed. This is
what the production φ-partition cannot do (there, playmaking ends up ~95% collinear with total
offense). Full spec, inference, results, and open problems: docs/generative_model.md.

A stint side is modeled as a MARKED POISSON PROCESS with a BERNOULLI THINNING, in three layers. For
attacking on-ice skaters A, defending skaters B, defending goalie g, context x, length t seconds, and
EACH attacker j∈A treated as the focal shooter (teammates T = A\{j}). Symbols follow the glossary in
docs/generative_model.md; the code↔glossary map is at the end of this docstring.

────────────────────────────────────────────────────────────────────────────────────────────────
PARAMETERS  (per skater unless noted)
  rate layer        mu_rate              baseline log shot-rate (per hour)
                    shoot_j              how much j shoots himself           (SCORING volume)
                    create_p             UNIFIED creation — lifts teammates' shot rate AND is p's
                                           creator-credit propensity          (PLAYMAKING)
                    create_0             unassisted propensity (shooter self-created); one global scalar
                    def_d                opponent shot-rate suppression      (DEFENSE; <0 good)
                    beta_rate            context (home, O/D-zone, lead/trail)
  quality layer     mu_qual              baseline shot quality (logit mean xG)
                    qshoot_j             danger of j's own shots             (SCORING quality)
                    qcreate_c            danger the ONE creator c adds        (PLAYMAKING quality)
                    qdef_d               opponent danger suppression         (DEFENSE quality; <0 good)
                    beta_qual ; s        context ; Beta concentration (shot-to-shot spread)
  conversion layer  mu_conv              5v5 conversion baseline (recalibrated shooting-model intercept)
                    fin_j                finishing (goals/shot above xG)
                    gsave_g              goalie save effect (goals/shot above xG allowed)

GENERATIVE PROCESS (one stint side, per focal shooter j)
  rate_j = exp( mu_rate + shoot_j + Σ_{p∈T} create_p + Σ_{d∈B} def_d + beta_rate·x )    # j's rate/hour
  N_j    ~ Poisson( rate_j·t/3600 )  OR  NegBin( mean=rate_j·t/3600 , r )                # how many j takes
  for each of the N_j shots, with creator c ~ pi = softmax([create_0, create_T]):
     base   = mu_qual + qshoot_j + Σ_{d∈B} qdef_d + beta_qual·x
     qbar_c = sigmoid( base + qcreate_c )   (= sigmoid(base) if c = unassisted)          # mean quality
     q      ~ Beta( s·qbar_c , s·(1−qbar_c) )                                            # this chance's xG
     y      ~ Bernoulli( clip( q + mu_conv + fin_j + gsave_g , 0, 1 ) )                  # goal?

ASSIST-CREDIT ANCHOR (added to the Stage-1 fit — shares `create`)
  For each GOAL the primary assister is a conditional logit over {unassisted, 4 teammates}:
     pi_c = softmax([ create_0 , create_T ])_c
  added to the rate NLL, up-weighted by SHOTS_PER_GOAL (≈ shots per goal — see the constant). The SAME
  `create` thus blends 'I make my linemates shoot' (dense volume) with 'I'm credited with the setup'
  (sparse assist signal).

LIKELIHOOD  (factorizes into disjoint blocks; independent ridge priors)
  ℓ(θ) =  Σ_(side,j) log Pois/NB( N_j | rate_j·t/3600 )     ← rate    (mu_rate,shoot,create,def,beta_rate,r)
        + SHOTS_PER_GOAL · Σ_goals log pi[assister]         ← assist-credit anchor (shares create, create_0)
        + Σ_shots  log fracBern( xg | qbar )                ← quality (mu_qual,qshoot,qcreate,qdef,beta_qual,s)
                     (creator OBSERVED on goals; MARGINALIZED over fixed pi on non-goals)
        + Σ_shots  log Bernoulli( y | clip(xg+mu_conv+fin+gsave) )   ← conversion (mu_conv,fin,gsave)
        − ridge on every player block (= EB Normal(0,σ²) priors)

CODE↔GLOSSARY MAP: create_0→`psi0`; mu_rate/qual→ each fit's `intercept`, mu_conv→`conv["mu_c"]`;
  beta_rate/qual→`beta`; qbar→`sig5`; pi_c→`pi`; fin_j→`alpha`; gsave_g→`gamma`.
────────────────────────────────────────────────────────────────────────────────────────────────

Fit: MAP / penalized MLE by OPTIMIZATION (JAX autodiff gradient + scipy L-BFGS-B). Uncertainties via
the HESSIAN (Laplace): H = XᵀWX + diag(penalty); SE = sqrt(diag(H⁻¹)). The count layer is configurable
(`--count poisson|nb`). 5v5 only, for the POC. Usage:
  uv run --group experimental python -m yhattrick.models.generative_model            # latest season
  uv run --group experimental python -m yhattrick.models.generative_model --count nb # negative binomial
  uv run --group experimental python -m yhattrick.models.generative_model --pool     # all seasons
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, logsumexp

from .. import config as C
from . import shooting_model
from .player_onice_model import roster_names

jax.config.update("jax_enable_x64", True)   # float64: stable optimization + Hessian

# Ridge = a Normal(0, prior_sd²) prior on each player effect; ridge precision = 1/prior_sd². The
# create and qcreate loadings are PER-TEAMMATE effects (each loads on the 4 rows where the player is a
# teammate), so they get a tighter prior than the shooter's own loading.
PRIOR_SD_SHOOT = 0.30       # shoot_j / def_d on the log shot-rate scale
PRIOR_SD_CREATE = 0.12      # create_p — per-teammate lift; tighter (noisier high-leverage effect,
                            #   curbs high-minute pull on the shared teammate-shot volume)
PRIOR_SD_QSHOOT = 0.20      # qshoot_j / qdef_d on the logit-xG scale
PRIOR_SD_QCREATE = 0.25     # qcreate_c — danger the single (latent) creator adds
SHOTS_PER_GOAL = 16         # up-weight on the goal-assist credit term (Stage 1). We observe a shot's
                            #   creator only on the ~1-in-16 Fenwick shots that ARE goals (via the
                            #   primary assist), so weighting each observed goal-creator by ≈ shots/goal
                            #   makes it stand in for the shots whose creator we never see (inverse-
                            #   probability weighting). ≈ 1/(5v5 Fenwick goal rate); approximate, fixed.
EPS = 1e-6
SNIFF_MIN_TOI = 24000.0     # min on-ice seconds (5v5, ≈400 min) to appear in a leaderboard
N_TM = 4                    # teammates a player lifts at 5v5
N_DEF = 5                   # opposing shooters a defender faces at 5v5


# ── data loaders (5v5) ──────────────────────────────────────────────────────────────────────────

def _zone(atk_home, start_type, start_zone):
    """Attacking-team zone start: O/D flips for the away team. Returns (ozone, dzone) indicators."""
    if start_type != "faceoff" or start_zone not in ("O", "D"):
        return 0.0, 0.0
    az = start_zone if atk_home else ("O" if start_zone == "D" else "D")
    return (1.0, 0.0) if az == "O" else (0.0, 1.0)


RATE_CTX = ["home", "ozone", "dzone", "trail", "lead"]


def _load_5v5_stints(seasons):
    """All regular-season 5v5 non-overload stints (keep short <10s ones — the Poisson count handles
    them via the log-time offset)."""
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


def _shooter_counts(seasons):
    """{(nhl_game_id, stint_idx): {shooter_id: fenwick_count}} for 5v5 shots (the shooter-resolved
    response). A player's shots in a stint are his regardless of side (he's the attacker when he
    shoots)."""
    out: dict[tuple, dict] = defaultdict(lambda: defaultdict(int))
    for s in seasons:
        p = C.PROCESSED / "shots_onice" / f"{s}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["nhl_game_id", "stint_idx", "shooter_id", "strength"])
        d = d[(d.strength == "5v5") & d.shooter_id.notna()]
        g = d.groupby(["nhl_game_id", "stint_idx", "shooter_id"]).size()
        for (gid, sidx, pid), n in g.items():
            out[(int(gid), int(sidx))][int(pid)] = int(n)
    return out


def rate_rows(seasons):
    """Shooter-resolved rate design. For each 5v5 stint SIDE and each on-ice attacker j as the focal
    shooter: one row with j's Fenwick count in the stint (0 if none), j's index, the 4 teammate
    indices, the 5 defender indices, offset log(t/3600), and context. Also returns the shared player
    index and per-player 5v5 TOI."""
    df = _load_5v5_stints(seasons)
    if df.empty:
        return None
    counts = _shooter_counts(seasons)
    players = sorted(set().union(*df.home_skaters, *df.away_skaters))
    idx = {p: i for i, p in enumerate(players)}
    P = len(players)

    shooter, team, dff, ctx, cnt, off_t, gid = [], [], [], [], [], [], []
    toi = np.zeros(P)

    def emit_side(atk, dfd, atk_home, s, def_goalie):
        c = counts.get((int(s.nhl_game_id), int(s.stint_idx)), {})
        di = [idx[p] for p in dfd]
        oz, dz = _zone(atk_home, s.start_type, s.start_zone)
        lead = s.home_lead if atk_home else -s.home_lead
        row_ctx = [1.0 if atk_home else 0.0, oz, dz, 1.0 if lead < 0 else 0.0, 1.0 if lead > 0 else 0.0]
        for j in atk:
            shooter.append(idx[j])
            team.append([idx[p] for p in atk if p != j])
            dff.append(di)
            ctx.append(row_ctx)
            cnt.append(float(c.get(int(j), 0)))
            off_t.append(s.duration_s)
            gid.append(def_goalie)

    for s in df.itertuples():
        if len(s.home_skaters) != 5 or len(s.away_skaters) != 5:
            continue
        emit_side(s.home_skaters, s.away_skaters, True, s, s.away_goalie)
        emit_side(s.away_skaters, s.home_skaters, False, s, s.home_goalie)
        for p in (*s.home_skaters, *s.away_skaters):
            toi[idx[p]] += s.duration_s

    dur = np.asarray(off_t, dtype=np.float64)
    return {
        "players": players, "idx": idx,
        "shooter_idx": np.asarray(shooter, dtype=np.int64),
        "team_idx": np.asarray(team, dtype=np.int64),
        "def_idx": np.asarray(dff, dtype=np.int64),
        "Xctx": np.asarray(ctx, dtype=np.float64),
        "count": np.asarray(cnt, dtype=np.float64),
        "offset": np.log(np.clip(dur / 3600.0, 1e-9, None)),
        "dur": dur, "def_goalie": np.asarray(gid, dtype=object), "toi": toi,
    }


# ── quality data: each shot + (for goals) the observed primary creator ────────────────────────────

def quality_creator_rows(seasons, idx):
    """Per 5v5 Fenwick shot: shooter, 4 teammates, 5 defenders, context, logit-xG, goal flag, and a
    CREATOR LABEL — for goals, which teammate (0–3) got the primary assist, or 4 = unassisted (no
    primary assist, or assister off-ice / = shooter). Non-goal shots have label −1 (latent: marginalize
    over the creator at fit time). Join: shots_onice.event_idx == pbp goal sortOrder."""
    shooter, team, dff, ctx, y, goal, creator = [], [], [], [], [], [], []
    for s in seasons:
        p = C.PROCESSED / "shots_onice" / f"{s}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["nhl_game_id", "event_idx", "strength", "is_home", "xg",
                                        "goal", "shooter_id", "home_skaters", "away_skaters"])
        d = d[(d.strength == "5v5") & d.xg.notna() & d.shooter_id.notna()]
        for gid, sub in d.groupby("nhl_game_id"):
            a1 = {}
            pf = C.RAW_PBP / f"{int(gid)}.json"
            if pf.exists():
                plays = json.loads(pf.read_text()).get("plays", [])
                a1 = {pl.get("sortOrder"): pl.get("details", {}).get("assist1PlayerId")
                      for pl in plays if pl.get("typeDescKey") == "goal"}
            for r in sub.itertuples():
                hs, as_ = list(r.home_skaters), list(r.away_skaters)
                if len(hs) != 5 or len(as_) != 5:
                    continue
                atk, dfd = (hs, as_) if r.is_home == 1 else (as_, hs)
                sid = int(r.shooter_id)
                if sid not in atk or any(q not in idx for q in atk) or any(q not in idx for q in dfd):
                    continue
                mates = [q for q in atk if q != sid]                     # 4 teammates
                shooter.append(idx[sid]); team.append([idx[q] for q in mates])
                dff.append([idx[q] for q in dfd]); ctx.append([1.0 if r.is_home == 1 else 0.0])
                xg = float(np.clip(r.xg, EPS, 1 - EPS)); y.append(np.log(xg / (1 - xg)))
                goal.append(int(r.goal))
                if r.goal == 1:
                    ap = a1.get(int(r.event_idx))
                    creator.append(mates.index(int(ap)) if ap is not None and int(ap) in mates else 4)
                else:
                    creator.append(-1)
    return {"shooter_idx": np.asarray(shooter, dtype=np.int64), "team_idx": np.asarray(team, dtype=np.int64),
            "def_idx": np.asarray(dff, dtype=np.int64), "Xctx": np.asarray(ctx, dtype=np.float64),
            "y": np.asarray(y, dtype=np.float64), "goal": np.asarray(goal, dtype=np.int64),
            "creator": np.asarray(creator, dtype=np.int64)}


# ── the crossed design as a sparse matrix (for the exact analytic Hessian) ─────────────────────────

def _build_X(shooter, team, dfd, Xctx, P):
    """Sparse design, columns [intercept | shoot(P) | create(P) | def(P) | context(k)]."""
    n, k = len(shooter), Xctx.shape[1]
    ncol = 1 + 3 * P + k
    ar = np.arange(n)
    R, Cc, D = [], [], []

    def add_ind(col):
        R.append(ar); Cc.append(col); D.append(np.ones(n))

    add_ind(np.zeros(n, int))                           # intercept
    add_ind(1 + shooter)                                # shoot block
    for j in range(team.shape[1]):
        add_ind(1 + P + team[:, j])                     # play block
    for j in range(dfd.shape[1]):
        add_ind(1 + 2 * P + dfd[:, j])                  # def block
    for c in range(k):                                  # context (values, not indicators)
        R.append(ar); Cc.append(np.full(n, 1 + 3 * P + c)); D.append(Xctx[:, c])
    return sparse.csr_matrix((np.concatenate(D), (np.concatenate(R), np.concatenate(Cc))),
                             shape=(n, ncol))


def _pen_vec(P, k, lam_shoot, lam_create, lam_def):
    pen = np.zeros(1 + 3 * P + k)
    pen[1:1 + P] = lam_shoot
    pen[1 + P:1 + 2 * P] = lam_create
    pen[1 + 2 * P:1 + 3 * P] = lam_def
    return pen


def _se_from_hessian(H):
    return np.sqrt(np.clip(np.diag(np.linalg.inv(H)), 0.0, None))


# ── fits (JAX autodiff gradient + scipy L-BFGS-B; exact Hessian for SEs) ───────────────────────────

def _optimize(nll, x0, *data):
    """Minimize nll(th, *data) over th (scipy L-BFGS-B; grad via JAX). The big data arrays are passed
    as ARGUMENTS to the jitted value_and_grad — NOT closed over — so XLA streams them as inputs instead
    of baking them into the compiled program as multi-GB captured constants (which is slow to compile
    and doubles memory). Convert once to device arrays here and reuse across iterations."""
    vg = jax.jit(jax.value_and_grad(nll))
    dargs = [jnp.asarray(d) for d in data]

    def f(x):
        v, g = vg(jnp.asarray(x), *dargs)
        return float(v), np.asarray(g, dtype=np.float64)

    return minimize(f, x0, jac=True, method="L-BFGS-B",
                    options={"maxiter": 1500, "maxfun": 1500, "ftol": 1e-10, "gtol": 1e-7})


def fit_rate_create(R, g_team, g_cidx, shots_per_goal, count_model="poisson"):
    """UNIFIED CREATION. One `create` parameter that does double duty:
      (i) lifts teammates' shot rate in the Poisson/NB count layer, and
      (ii) is the creator-credit propensity in a conditional logit over the on-ice teammates,
           anchored to OBSERVED goal assisters (g_team = the 4 teammate indices, g_cidx = the credited
           column: 0 = unassisted/create_0, 1..4 = teammate position).
    Fit jointly; `shots_per_goal` up-weights the sparse assist-credit — each observed goal-creator stands
    in for the ≈ shots/goal whose creator we never see (inverse-probability weighting) — so `create`
    blends 'I make my linemates shoot' with 'I'm the one credited with creating' — the volume half also
    captures the unobserved passing the assist labels miss. Returns shoot/create/def + create_0 + SEs
    (Poisson Hessian, with the credit Fisher added to the create diagonal)."""
    P, k = len(R["players"]), R["Xctx"].shape[1]
    lsh, lcr, ldf = 1 / PRIOR_SD_SHOOT ** 2, 1 / PRIOR_SD_CREATE ** 2, 1 / PRIOR_SD_SHOOT ** 2
    nb = count_model == "nb"
    PS = 1 + 3 * P + k                                       # index of create_0 in th

    def split(th):
        return th[1:1 + P], th[1 + P:1 + 2 * P], th[1 + 2 * P:1 + 3 * P], th[1 + 3 * P:1 + 3 * P + k], th[PS]

    def nll(th, sh_i, tm_i, df_i, X, cnt, ofs, gt, gc):
        sh, cr, df, b, psi0 = split(th)
        eta = th[0] + sh[sh_i] + jnp.sum(cr[tm_i], 1) + jnp.sum(df[df_i], 1) + X @ b
        mu = jnp.exp(eta + ofs)
        if not nb:
            pois = jnp.sum(mu - cnt * jnp.log(mu))
        else:
            r = jnp.exp(th[-1])
            pois = -jnp.sum(gammaln(cnt + r) - gammaln(r) - gammaln(cnt + 1)
                            + r * jnp.log(r / (r + mu)) + cnt * jnp.log(mu / (r + mu)))
        logit5 = jnp.concatenate([jnp.full((gt.shape[0], 1), psi0), cr[gt]], axis=1)   # [create_0, create(4)]
        logpi = logit5 - logsumexp(logit5, axis=1)[:, None]
        credit = -jnp.sum(jnp.take_along_axis(logpi, gc[:, None], 1)[:, 0])
        pen = 0.5 * (lsh * jnp.sum(sh ** 2) + lcr * jnp.sum(cr ** 2) + ldf * jnp.sum(df ** 2))
        return pois + shots_per_goal * credit + pen

    x0 = np.zeros(1 + 3 * P + k + 1 + (1 if nb else 0))
    if nb:
        x0[-1] = 1.0
    res = _optimize(nll, x0, R["shooter_idx"], R["team_idx"], R["def_idx"], R["Xctx"], R["count"],
                    R["offset"], g_team, g_cidx)
    th = res.x
    sh, cr, df, b, psi0 = th[1:1 + P], th[1 + P:1 + 2 * P], th[1 + 2 * P:1 + 3 * P], th[1 + 3 * P:1 + 3 * P + k], float(th[PS])
    eta = th[0] + sh[R["shooter_idx"]] + cr[R["team_idx"]].sum(1) + df[R["def_idx"]].sum(1) + R["Xctx"] @ b
    mu = np.exp(eta + R["offset"])
    r = float(np.exp(th[-1])) if nb else None
    W = mu * r / (r + mu) if nb else mu
    Xs = _build_X(R["shooter_idx"], R["team_idx"], R["def_idx"], R["Xctx"], P)
    H = (Xs.T @ Xs.multiply(W[:, None])).toarray() + np.diag(_pen_vec(P, k, lsh, lcr, ldf))
    logit5 = np.concatenate([np.full((len(g_cidx), 1), psi0), cr[g_team]], 1)       # credit Fisher on create diag
    pi = np.exp(logit5 - logit5.max(1, keepdims=True)); pi /= pi.sum(1, keepdims=True)
    cinfo = np.zeros(P)
    for t in range(4):
        np.add.at(cinfo, g_team[:, t], shots_per_goal * pi[:, t + 1] * (1 - pi[:, t + 1]))
    di = np.arange(1 + P, 1 + 2 * P)
    H[di, di] += cinfo
    se = _se_from_hessian(H)
    return {"intercept": float(th[0]), "shoot": sh, "create": cr, "def": df, "beta": b, "psi0": psi0,
            "se_shoot": se[1:1 + P], "se_create": se[1 + P:1 + 2 * P], "se_def": se[1 + 2 * P:1 + 3 * P],
            "count_model": count_model, "r": r, "converged": bool(res.success),
            "grad_norm": float(np.max(np.abs(res.jac)))}


def fit_quality_creator(Q, P, create, psi0):
    """QUALITY fit given the creator distribution from the rate fit: pi = softmax([create_0,
    create[teammates]]) is FIXED (create already absorbed the assist-credit); estimate only
    qshoot/qcreate/qdef. A shot's danger depends on the SHOOTER (qshoot) + ONE creator's qcreate
    (+ qdef defenders + ctx). The creator is OBSERVED on goals (primary assister or 'unassisted') and
    MARGINALIZED over the fixed pi on non-goal shots. Coupled block, optimized with JAX/L-BFGS — one
    global fit, no EM. (SEs: diagonal Gauss-Newton — the marginalized block isn't a simple GLM.)"""
    n, k = len(Q["y"]), Q["Xctx"].shape[1]
    cre = Q["creator"]                                      # -1 latent, 0..3 teammate, 4 unassisted
    cidx_np = np.where(cre == 4, 0, np.clip(cre, 0, 3) + 1).astype(np.int64)  # col in [unassist, t0..t3]
    lqs, lqc, lqd = 1 / PRIOR_SD_QSHOOT ** 2, 1 / PRIOR_SD_QCREATE ** 2, 1 / PRIOR_SD_QSHOOT ** 2
    lg = np.concatenate([np.full((n, 1), psi0), create[Q["team_idx"]]], 1)   # FIXED creator distribution
    pi_np = np.exp(lg - lg.max(1, keepdims=True)); pi_np /= pi_np.sum(1, keepdims=True)

    def split(th):
        return th[0], th[1:1 + P], th[1 + P:1 + 2 * P], th[1 + 2 * P:1 + 3 * P], th[1 + 3 * P:1 + 3 * P + k]

    def nll(th, sh_i, tm_i, df_i, X, xg, goal, cidx, pi):
        mq, qs, qc, qd, b = split(th)
        base = mq + qs[sh_i] + jnp.sum(qd[df_i], 1) + X @ b
        sig5 = jnp.concatenate([jax.nn.sigmoid(base)[:, None],
                                jax.nn.sigmoid(base[:, None] + qc[tm_i])], axis=1)   # (n,5) col0=unassisted

        def fb(p):
            return xg * jnp.log(p + EPS) + (1 - xg) * jnp.log(1 - p + EPS)
        gs = jnp.take_along_axis(sig5, cidx[:, None], 1)[:, 0]       # goals: observed-creator quality
        marg = jnp.sum(pi * sig5, axis=1)                           # non-goals: marginalize over FIXED pi
        ll = jnp.where(goal, fb(gs), fb(marg))
        pen = 0.5 * (lqs * jnp.sum(qs ** 2) + lqc * jnp.sum(qc ** 2) + lqd * jnp.sum(qd ** 2))
        return -jnp.sum(ll) + pen

    res = _optimize(nll, np.zeros(1 + 3 * P + k), Q["shooter_idx"], Q["team_idx"], Q["def_idx"],
                    Q["Xctx"], _sigmoid(Q["y"]), Q["goal"].astype(bool), cidx_np, pi_np)
    mq, qs, qc, qd, b = (float(res.x[0]), res.x[1:1 + P], res.x[1 + P:1 + 2 * P],
                         res.x[1 + 2 * P:1 + 3 * P], res.x[1 + 3 * P:1 + 3 * P + k])
    # Beta concentration from residual MSE around the fitted mean
    base = mq + qs[Q["shooter_idx"]] + qd[Q["def_idx"]].sum(1) + Q["Xctx"] @ b
    sig5 = np.concatenate([_sigmoid(base)[:, None], _sigmoid(base[:, None] + qc[Q["team_idx"]])], 1)
    ci = cidx_np
    m = (pi_np * sig5).sum(1)                                    # marginal mean (non-goals), FIXED pi
    fitted = np.where(Q["goal"].astype(bool), sig5[np.arange(n), ci], m)
    mse = float(np.mean((_sigmoid(Q["y"]) - fitted) ** 2))
    s_conc = max(float(np.mean(fitted * (1 - fitted)) / max(mse, 1e-9) - 1.0), 1.0)

    # diagonal Gauss-Newton SE for qcreate (data-limited): observed-creator goal quality + non-goal marginal
    g = Q["goal"].astype(bool); tm = Q["team_idx"]
    info_qc = np.zeros(P)
    tg, sg, cig = tm[g], sig5[g], ci[g]
    is_tm = cig >= 1
    cr_pl = tg[np.arange(len(tg)), np.clip(cig - 1, 0, 3)]
    sc = sg[np.arange(len(sg)), cig]
    np.add.at(info_qc, cr_pl[is_tm], (sc * (1 - sc))[is_tm])
    pn, tn, sn, mn = pi_np[~g], tm[~g], sig5[~g], m[~g]
    den = np.clip(mn * (1 - mn), 1e-9, None)
    for t in range(4):
        np.add.at(info_qc, tn[:, t], (pn[:, t + 1] * sn[:, t + 1] * (1 - sn[:, t + 1])) ** 2 / den)
    se_qc = 1.0 / np.sqrt(info_qc + 1 / PRIOR_SD_QCREATE ** 2)
    n_create = np.zeros(P, dtype=np.int64)
    np.add.at(n_create, cr_pl[is_tm], 1)
    return {"intercept": mq, "qshoot": qs, "qcreate": qc, "qdef": qd, "beta": b, "beta_s": s_conc,
            "se_qcreate": se_qc, "n_create": n_create, "converged": bool(res.success),
            "grad_norm": float(np.max(np.abs(res.jac)))}


# ── conversion params (reuse the production shooting model) ─────────────────────────────────────────

def conversion_params(seasons, names):
    """Finishing (fin, code `alpha`) and goalie (gsave, code `gamma`) from the production shooting model,
    plus a per-shot intercept mu_conv (code `mu_c`) RECALIBRATED to THIS model's universe (5v5 Fenwick).
    The shooting model's own intercept is fit on a broader shot set, so reused directly it under-fills
    5v5 goals (~5%). We instead solve mu_conv so the deterministic expected goals reconcile exactly:
    mu_conv = (Σgoals − Σ(xg + fin_shooter + gsave_goalie))/N."""
    fin, gl, meta = shooting_model.fit(seasons, names)
    alpha = dict(zip(fin.player_id, fin.fin_per100 / 100.0)) if not fin.empty else {}
    gamma = dict(zip(gl.player_id, -gl.gsax_per100 / 100.0)) if not gl.empty else {}
    tot_goal = tot_pred = n = 0.0
    for s in seasons:
        p = C.PROCESSED / "shots_onice" / f"{s}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["strength", "is_home", "xg", "goal", "shooter_id",
                                        "home_goalie", "away_goalie"])
        d = d[(d.strength == "5v5") & d.xg.notna() & d.shooter_id.notna()]
        goalie = np.where(d.is_home.to_numpy() == 1, d.away_goalie.to_numpy(), d.home_goalie.to_numpy())
        a = np.array([alpha.get(int(x), 0.0) for x in d.shooter_id])
        g = np.array([gamma.get(x, 0.0) if pd.notna(x) else 0.0 for x in goalie])
        tot_goal += float(d.goal.sum())
        tot_pred += float((d.xg.to_numpy() + a + g).sum())
        n += len(d)
    mu_c = (tot_goal - tot_pred) / n if n else 0.0
    print(f"  conversion mu_conv recalibrated to 5v5 universe: {mu_c:+.5f}/shot "
          f"(Σgoals {tot_goal:.0f} vs Σ(xg+fin+gsave) {tot_pred:.0f})")
    return {"mu_c": mu_c, "alpha": alpha, "gamma": gamma}   # mu_c=mu_conv, alpha=fin, gamma=gsave


# ── attribution: merge the rate (volume) and quality loadings into goal-scale numbers ──────────────

def player_values(rate, qual, conv, players):
    """Deployment-free per-60 goal values from the fitted loadings + intercepts:
       scoring(j)    = exp(mu_rate+shoot)·sigmoid(mu_qual+qshoot) + exp(mu_rate+shoot)·fin    own shots
       playmaking(p) = N_TM·exp(mu_rate)·(exp(create)−1)·sigmoid(mu_qual+qcreate)             creation
       defense(d)    = N_DEF·[exp(mu_rate)·sigmoid(mu_qual) − exp(mu_rate+def)·sigmoid(mu_qual+qdef)]  suppression
    Playmaking = teammate shot-volume p adds (create) × their danger (qcreate). `creator_share` (pi) is
    reported separately (p's per-teammate-shot creator probability); it is NOT a factor in the
    playmaking value. All xG/60; defense >0 = suppresses (good). Code: ml=mu_rate, mq=mu_qual, cr=create."""
    ml, mq = rate["intercept"], qual["intercept"]
    base = np.exp(ml) * _sigmoid(mq)
    own = np.exp(ml + rate["shoot"]) * _sigmoid(mq + qual["qshoot"])
    cr = rate["create"]
    pi = np.exp(cr) / (np.exp(rate["psi0"]) + np.exp(cr) + (N_TM - 1))                       # creator share
    # playmaking = teammate shots p ADDS (volume lift from create) × their danger (qcreate)
    play = N_TM * np.exp(ml) * (np.exp(cr) - 1.0) * _sigmoid(mq + qual["qcreate"])
    deff = N_DEF * (base - np.exp(ml + rate["def"]) * _sigmoid(mq + qual["qdef"]))
    alpha = np.array([conv["alpha"].get(int(p), 0.0) for p in players])
    shots = np.exp(ml + rate["shoot"])
    scoring = own + shots * alpha
    return {"scoring": scoring, "playmaking": play, "defense": deff,
            "creator_share": pi, "own_xg": own, "own_shots": shots}


# ── simulation + posterior-predictive check ───────────────────────────────────────────────────────

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def ppc(R, rate, qual, conv, seed=0):
    """Forward-simulate every shooter-stint row from the fitted params and compare to actuals: shot
    count distribution (rate), shot quality (quality), and total goals (end-to-end). Shots are now
    shooter-attributed, so conversion uses the focal shooter's own finishing."""
    rng = np.random.default_rng(seed)
    players = R["players"]
    eta_r = (rate["intercept"] + rate["shoot"][R["shooter_idx"]] + rate["create"][R["team_idx"]].sum(1)
             + rate["def"][R["def_idx"]].sum(1) + R["Xctx"] @ rate["beta"])
    mu = np.exp(eta_r + R["offset"])
    if rate.get("count_model") == "nb" and rate.get("r"):
        r = rate["r"]
        N = rng.negative_binomial(r, r / (r + mu))
    else:
        N = rng.poisson(mu)

    # quality marginalized over the creator: base (shooter+def+ctx), then mix over {unassisted, 4
    # teammates} weighted by softmax([create_0, create]) (the UNIFIED creator dist), each lifting danger by qcreate.
    base = (qual["intercept"] + qual["qshoot"][R["shooter_idx"]] + qual["qdef"][R["def_idx"]].sum(1)
            + qual["beta"][0] * R["Xctx"][:, 0])
    sig5 = np.concatenate([_sigmoid(base)[:, None], _sigmoid(base[:, None] + qual["qcreate"][R["team_idx"]])], 1)
    lg = np.concatenate([np.full((len(base), 1), rate["psi0"]), rate["create"][R["team_idx"]]], 1)
    pi = np.exp(lg - lg.max(1, keepdims=True)); pi /= pi.sum(1, keepdims=True)
    qpred_marg = (pi * sig5).sum(1)
    alpha = np.array([conv["alpha"].get(int(players[s]), 0.0) for s in R["shooter_idx"]])
    gvec = np.array([conv["gamma"].get(g, 0.0) for g in R["def_goalie"]])

    rep = np.repeat(np.arange(len(N)), N)
    p_q = qpred_marg[rep]
    s = qual["beta_s"]
    q = rng.beta(s * p_q, s * (1 - p_q))
    p_goal = np.clip(q + conv["mu_c"] + alpha[rep] + gvec[rep], 0.0, 1.0)
    goals_sim = int(rng.binomial(1, p_goal).sum())
    return {
        "shots_actual": int(R["count"].sum()), "shots_sim": int(N.sum()),
        "shots_expected": float(mu.sum()),
        "perrow_mean_actual": float(R["count"].mean()), "perrow_mean_sim": float(N.mean()),
        "perrow_var_actual": float(R["count"].var()), "perrow_var_sim": float(N.var()),
        "zero_frac_actual": float((R["count"] == 0).mean()), "zero_frac_sim": float((N == 0).mean()),
        "mean_xg_sim": float(q.mean()), "goals_sim": goals_sim, "n_sim_shots": int(rep.size),
    }


# ── leaderboards + run ─────────────────────────────────────────────────────────────────────────────

def _board(names, players, val, toi, label, se=None, higher=True, n=12):
    elig = toi >= SNIFF_MIN_TOI
    order = np.argsort(val * (-1 if higher else 1))
    order = [i for i in order if elig[i]][:n]
    print(f"\n{label}")
    for i in order:
        pid = players[i]; nm = names.get(int(pid), {}).get("name", f"#{pid}")
        if se is not None:
            z = val[i] / se[i] if se[i] > 0 else 0.0
            flag = "" if abs(z) >= 2 else "  ⚠ low-conf"
            print(f"   {val[i]:+.3f} ±{1.96 * se[i]:.3f} (z={z:+.1f}){flag}  {nm:24s} ({toi[i] / 60:.0f} min)")
        else:
            print(f"   {val[i]:+.3f}  {nm:24s} ({toi[i] / 60:.0f} min)")


def run(seasons, count_model="poisson"):
    names = roster_names(seasons)
    print(f"[generative_model:shooter-resolved] seasons {seasons} — count {count_model} — loading 5v5 …")
    R = rate_rows(seasons)
    if R is None:
        raise SystemExit("no 5v5 stints")
    P = len(R["players"])
    print(f"  rate rows: {len(R['count']):,} shooter-stints, {P} skaters, {R['count'].sum():.0f} shots")

    # latent-creator quality rows (shots + observed goal creators); the goal subset anchors `create`
    Q = quality_creator_rows(seasons, R["idx"])
    ngoal = int(Q["goal"].sum()); nun = int((Q["creator"] == 4).sum())
    gmask = Q["goal"] == 1
    g_team = Q["team_idx"][gmask]
    g_lab = Q["creator"][gmask]
    g_cidx = np.where(g_lab == 4, 0, np.clip(g_lab, 0, 3) + 1).astype(np.int64)

    print(f"  fitting UNIFIED `create` (volume + goal-assist credit, shots_per_goal={SHOTS_PER_GOAL:g}) …")
    rate = fit_rate_create(R, g_team, g_cidx, SHOTS_PER_GOAL, count_model=count_model)
    rdesc = f" r={rate['r']:.2f}" if rate.get("r") else ""
    print(f"  rate fit ({count_model}): converged={rate['converged']} |grad|={rate['grad_norm']:.1e}{rdesc}"
          f"  create_0={rate['psi0']:+.3f}  ({ngoal:,} goals, {100*nun/max(ngoal,1):.0f}% unassisted)")

    print("  fitting quality (qshoot/qcreate/qdef) given the fixed creator distribution …")
    qual = fit_quality_creator(Q, P, rate["create"], rate["psi0"])
    print(f"  quality fit: converged={qual['converged']} |grad|={qual['grad_norm']:.1e}  beta_s={qual['beta_s']:.1f}")
    toi = R["toi"]
    el = toi >= SNIFF_MIN_TOI
    zc = np.abs(rate["create"]) > 2 * rate["se_create"]
    print(f"  identification: create |z|>2 in {int((zc & el).sum())}/{int(el.sum())} eligible; "
          f"qcreate |z|>2 in {int(((np.abs(qual['qcreate']) > 2 * qual['se_qcreate']) & el).sum())}/{int(el.sum())}")

    conv = conversion_params(seasons, names)
    vals = player_values(rate, qual, conv, R["players"])

    _board(names, R["players"], rate["shoot"], toi, "TOP shoot_p (own-shot volume — snipers):", rate["se_shoot"])
    _board(names, R["players"], rate["create"], toi, "TOP create (UNIFIED: teammate-shot volume + assist credit):", rate["se_create"])
    _board(names, R["players"], qual["qshoot"], toi, "TOP qshoot_p (own-shot danger):")
    _board(names, R["players"], qual["qcreate"], toi, "TOP qcreate (danger added per setup):", qual["se_qcreate"])
    _board(names, R["players"], vals["scoring"], toi, "TOP SCORING (goals/60, own shots):")
    _board(names, R["players"], vals["playmaking"], toi, "TOP PLAYMAKING (create × quality, xG/60):")
    _board(names, R["players"], vals["defense"], toi, "BEST DEFENSE (xG/60 suppressed):")

    pp = ppc(R, rate, qual, conv)
    pp["goals_actual"] = int(Q["goal"].sum())
    pp["mean_xg_actual"] = float(_sigmoid(Q["y"]).mean())
    print("\n── posterior-predictive check (simulate every shooter-stint row) ──")
    print(f"  shots   actual {pp['shots_actual']:,}  sim {pp['shots_sim']:,}  (model-expected {pp['shots_expected']:.0f})")
    print(f"  per-row count  mean a/s {pp['perrow_mean_actual']:.4f}/{pp['perrow_mean_sim']:.4f}"
          f"  var a/s {pp['perrow_var_actual']:.4f}/{pp['perrow_var_sim']:.4f}"
          f"  zero% a/s {100 * pp['zero_frac_actual']:.1f}/{100 * pp['zero_frac_sim']:.1f}")
    print(f"  mean shot xG  actual {pp['mean_xg_actual']:.4f}  sim-draw {pp['mean_xg_sim']:.4f}")
    print(f"  goals         actual {pp['goals_actual']:,}  sim {pp['goals_sim']:,}")

    _save(seasons, R, rate, qual, conv, vals, pp)


def _save(seasons, R, rate, qual, conv, vals, pp):
    label = "+".join(map(str, seasons))
    out = {"model": "generative_model_unified_create", "seasons": list(seasons),
           "count_model": rate.get("count_model"), "nb_r": rate.get("r"), "shots_per_goal": SHOTS_PER_GOAL,
           "rate_intercept": float(rate["intercept"]), "quality_intercept": float(qual["intercept"]),
           "beta_s": qual["beta_s"], "psi0": float(rate["psi0"]), "mu_c": conv["mu_c"], "ppc": pp,
           "players": [{"id": int(R["players"][i]), "toi_s": float(R["toi"][i]),
                        "shoot": float(rate["shoot"][i]), "create": float(rate["create"][i]),
                        "create_se": float(rate["se_create"][i]), "def": float(rate["def"][i]),
                        "qshoot": float(qual["qshoot"][i]), "qcreate": float(qual["qcreate"][i]),
                        "qcreate_se": float(qual["se_qcreate"][i]), "qdef": float(qual["qdef"][i]),
                        "n_create": int(qual["n_create"][i]), "creator_share": float(vals["creator_share"][i]),
                        "scoring": float(vals["scoring"][i]), "playmaking": float(vals["playmaking"][i]),
                        "defense": float(vals["defense"][i])}
                       for i in range(len(R["players"]))]}
    C.MODELS.mkdir(parents=True, exist_ok=True)
    (C.MODELS / f"generative_model_{label}.json").write_text(json.dumps(out))
    print(f"\n  -> data/models/generative_model_{label}.json")


def main(argv=None):
    p = argparse.ArgumentParser(description="Experimental shooter-resolved generative model — POC")
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
