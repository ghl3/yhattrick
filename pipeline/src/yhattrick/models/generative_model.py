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
                    shoot_j[s]           how much j shoots himself           (SCORING volume)
                    create_p[s]          UNIFIED creation — lifts teammates' shot rate AND is p's
                                           creator-credit propensity          (PLAYMAKING)
                    create_0             unassisted propensity (shooter self-created); one global scalar
                    def_d[s]             opponent shot-rate suppression      (DEFENSE; <0 good)
                    beta_rate            context (home, O/D-zone, lead/trail, season, position, AGE)
  quality layer     mu_qual              baseline shot quality (logit mean xG)
                    qshoot_j             danger of j's own shots             (SCORING quality)
                    qcreate_{F,D}        danger the ONE creator adds — POSITION-level (A1: per-player
                                           creation quality is unidentifiable, docs §7)
                    qdef_d               opponent danger suppression         (DEFENSE quality; <0 good)
                    beta_qual ; s        context ; Beta concentration (shot-to-shot spread, post-hoc MoM)
  conversion layer  a ; b                logit-conversion slope + intercept (per strength, UNPENALIZED →
                                           b's score eqn gives Σp=Σy exactly; + per-season offsets, F6)
                    fin_j                finishing — logit offset above xG on own shots
                    gsave_g              goalie save effect — logit offset (<0 good)

PLAYER CURVES (aging + drift — how skill moves over time; docs/generative_model.md "Player curves")
  [s] above marks the EV rate blocks that carry one STATE per (player, active-season) pair, tied
  across seasons by a random-walk prior: θ_first ~ N(0, prior_sd²), θ_next ~ N(θ_prev, rw_sd²·gap).
  One pooled fit yields each player's whole trajectory; his LAST state is the "current skill" read.
  On top, shared F/D AGING CURVES enter every stage as context columns — quadratic in
  z=(age−27)/10 for shoot/create/def (per bucket), finishing, and goalie age — plus D-intercept
  offsets (A2) so ridge shrinkage stops fighting positional baselines. MA rate blocks, quality and
  fin/gsave stay static levels (data too thin for drift). PROJECTION: advance ages to the target
  season, hold states at their last value (the RW mean), re-run the value attribution.

GENERATIVE PROCESS (one stint side, per focal shooter j)
  rate_j = exp( mu_rate + shoot_j + Σ_{p∈T} create_p + Σ_{d∈B} def_d + beta_rate·x )    # j's rate/hour
  N_j    ~ Poisson( rate_j·t/3600 )  OR  NegBin( mean=rate_j·t/3600 , r )                # how many j takes
  for each of the N_j shots, with creator c ~ pi = softmax([create_0, create_T]):
     base   = mu_qual + qshoot_j + Σ_{d∈B} qdef_d + beta_qual·x
     qbar_c = sigmoid( base + qcreate_c )   (= sigmoid(base) if c = unassisted)          # mean quality
     q      ~ Beta( s·qbar_c , s·(1−qbar_c) )                                            # this chance's xG
     y      ~ Bernoulli( sigmoid( a·logit(q) + b + fin_j + gsave_g ) )                   # goal?

ASSIST-CREDIT ANCHOR (added to the Stage-1 fit — shares `create`)
  For each GOAL the primary assister is a conditional logit over {unassisted, 4 teammates}:
     pi_c = softmax([ create_0 , create_T ])_c
  added to the rate NLL, up-weighted by spg = the bucket's shots-per-goal computed from the data
  (≈16 at 5v5; `--spg-scale` multiplies it for sensitivity checks). The SAME `create` thus blends
  'I make my linemates shoot' (dense volume) with 'I'm credited with the setup' (sparse assist
  signal). SEs on `create` are sandwich-corrected: the up-weighted pseudo-counts add curvature but
  not w× real information (F1).

LIKELIHOOD  (factorizes into disjoint blocks; independent ridge priors)
  ℓ(θ) =  Σ_(side,j) log Pois/NB( N_j | rate_j·t/3600 )     ← rate    (mu_rate,shoot,create,def,beta_rate,r)
        + spg · Σ_goals log pi[assister]                    ← assist-credit anchor (shares create, create_0)
        + Σ_shots  log fracBern( xg | qbar )                ← quality (mu_qual,qshoot,qcreate_{F,D},qdef,
                     (creator OBSERVED on goals; MARGINALIZED           beta_qual; s post-hoc)
                      over fixed pi on non-goals and off-ice-assister goals)
        + Σ_shots  log Bernoulli( y | sigmoid(a·logit(xg)+b+b_season+fin+gsave) ) ← conversion (fit
                     natively off the OBSERVED xg — no shooting-model reuse; a,b,b_season unpenalized)
        − ridge on every player block (= EB Normal(0,σ²) priors) − RW penalty on the EV state chains

CODE↔GLOSSARY MAP: create_0→`psi0`; mu_rate/qual→ each fit's `intercept`; a/b→`conv["a"]`/`conv["b"]`;
  beta_rate/qual→`beta`; qbar→`sig5`; pi_c→`pi`; fin_j→`conv["fin"]`; gsave_g→`conv["gsave"]`.
────────────────────────────────────────────────────────────────────────────────────────────────

Fit: MAP / penalized MLE by OPTIMIZATION (JAX autodiff gradient + scipy L-BFGS-B). Uncertainties via
the HESSIAN (Laplace): H = XᵀWX + K (K = level ridge + RW precision); dense inverse when small, sparse
splu column solves for the reported columns when the state expansion outgrows it; `se_create` is
sandwich-corrected (F1). The count layer is configurable (`--count poisson|nb`). The conversion ridge
prior SDs (fin, gsave) are NOT hand-set: a PRE-CALCULATION stage (estimate_conversion_prior_sds)
estimates the per-player talent SD from the data each fit — empirical Bayes, the logit analogue of
shooting_model._estimate_k — then holds it FIXED during the fit. Usage:
  uv run --group experimental python -m yhattrick.models.generative_model            # latest season
  uv run --group experimental python -m yhattrick.models.generative_model --count nb # negative binomial
  uv run --group experimental python -m yhattrick.models.generative_model --pool     # all seasons
                                       # (multi-season ⇒ RW drift states + projection block in the JSON)
  uv run --group experimental python -m yhattrick.models.generative_model --pool --spg-scale 0.5
                                       # A3: assist-credit weight sensitivity (rerun with 2.0, compare)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.sparse.linalg import splu

import jax
import jax.numpy as jnp

from .. import config as C
# The likelihood itself — parameter layouts, the sparse design + prior-precision matrices, the three
# objective factories, and the math primitives / prior constants — lives in generative_likelihood.
# This module builds the design arrays from the fact tables and minimizes the objective it returns.
from .generative_likelihood import (
    PRIOR_SD_SHOOT, PRIOR_SD_CREATE, PRIOR_SD_QSHOOT, PRIOR_SD_QCREATE,
    RW_SD_SHOOT, RW_SD_CREATE, RW_SD_DEF, AGE_PEAK, AGE_SCALE, DENSE_H_MAX,
    EV_STRENGTHS, MA_STRENGTHS, ALL_STRENGTHS, ARENA_SD, ARENA_RW_SD,
    RateLayout, QualLayout, ConvLayout, RateHypers, QualHypers,
    _sigmoid, _se_from_hessian, _build_X, _build_conv_X, _rate_penalty,
    make_rate_nll, make_quality_nll, make_conversion_nll,
)
# Data-in: the fact/dimension tables become per-fit design rows in generative_data; this module
# builds nothing itself, it fits the arrays that builder produces.
from .generative_data import (player_index, _age_position, _curve_val, rate_rows,
                              quality_creator_rows, conversion_rows, estimate_conversion_prior_sds)

jax.config.update("jax_enable_x64", True)   # float64: stable optimization + Hessian


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


# ── θ̂ checkpoint: warm starts + optimization-free re-exports ──────────────────────────────────────
# The optimizer's solution is the expensive artifact (hours on the pooled window); everything else is
# derived from it in minutes. After every full fit the raw solution vectors land in
# generative_ckpt.npz, keyed by (player, season) unit, ctx-column NAME, and (venue, season) arena
# state — so the next fit can WARM-START by key (new players/seasons/columns init cold; the optimum is
# unchanged, only the path to it shortens), and `--reexport` can skip optimization AND the SE Hessian
# outright when the fit signature matches exactly (export-side tweaks rerun in minutes, not hours).
# The holdout harness never warm-starts: its train fits must not touch a solution that saw test data.

def _ckpt_path():
    return C.MODELS / "generative_ckpt.npz"


def _arr(v, dtype):
    return np.asarray(v if v is not None else [], dtype=dtype)


def _rate_unit_keys(R):
    """(player id, season) key per unit — season −1 for static fits (units == players)."""
    pids = np.asarray(R["players"])
    if "unit_player" in R:
        return pids[R["unit_player"]], np.asarray(R["unit_season"], dtype=np.int64)
    return pids, np.full(len(pids), -1, dtype=np.int64)


def _stage_rate(rate, pids):
    """Checkpoint block for one rate fit (unprefixed keys)."""
    up = np.asarray(rate["unit_player"], dtype=np.int64)
    us = rate.get("unit_season")
    us = np.asarray(us, dtype=np.int64) if us is not None else np.full(len(up), -1, dtype=np.int64)
    return {"theta": np.asarray(rate["theta"], dtype=np.float64),
            "unit_pid": np.asarray(pids)[up], "unit_season": us,
            "n_ctx": np.asarray(len(rate["beta"])),          # ctx block SIZE (names may be absent)
            "ctx_names": _arr(rate.get("ctx_names"), str),
            "arena_venue": _arr(rate.get("arena_venue"), str),
            "arena_season": _arr(rate.get("arena_season"), np.int64),
            "has_a2": np.asarray(rate.get("a2_q") is not None),
            "nb": np.asarray(rate.get("r") is not None),
            "converged": np.asarray(rate["converged"]), "grad_norm": np.asarray(rate["grad_norm"]),
            "se_shoot_last": rate["se_shoot_last"], "se_create_last": rate["se_create_last"],
            "se_def_last": rate["se_def_last"]}


def _stage_qual(qual, pids):
    return {"theta": np.asarray(qual["theta"], dtype=np.float64), "pid": np.asarray(pids),
            "n_ctx": np.asarray(len(qual["beta"])),
            "ctx_names": _arr(qual.get("ctx_names"), str),
            "arena_venue": _arr(qual.get("arena_venue"), str),
            "arena_season": _arr(qual.get("arena_season"), np.int64),
            "converged": np.asarray(qual["converged"]), "grad_norm": np.asarray(qual["grad_norm"]),
            "se_qcreate": qual["se_qcreate"]}


def _stage_conv(conv, pids):
    return {"theta": np.asarray(conv["theta"], dtype=np.float64), "pid": np.asarray(pids),
            "gid": np.asarray(conv["goalies"]),
            "keys": np.asarray(list(conv["a"].keys()), dtype=str),
            "n_ctx": np.asarray(len(conv["beta"])),
            "ctx_names": _arr(conv.get("ctx_names"), str),
            "converged": np.asarray(conv["converged"]), "grad_norm": np.asarray(conv["grad_norm"]),
            "se_fin": conv["se_fin"], "se_gsave": conv["se_gsave"]}


def ckpt_save(seasons, count_model, spg_scale, rates, qual, conv, players, path=None,
              ma_anchor_scale=1.0, ma_create_prior_sd=None, ma_def_prior_sd=None):
    pids = np.asarray(players)
    z = {"seasons": np.asarray([int(s) for s in seasons]), "count_model": np.asarray(count_model),
         "spg_scale": np.asarray(float(spg_scale)),
         "ma_anchor_scale": np.asarray(float(ma_anchor_scale)),
         "ma_create_prior_sd": np.asarray(float(ma_create_prior_sd)
                                          if ma_create_prior_sd else np.nan),
         "ma_def_prior_sd": np.asarray(float(ma_def_prior_sd) if ma_def_prior_sd else np.nan)}
    for key, rate in rates.items():
        for k, v in _stage_rate(rate, pids).items():
            z[f"{key}_{k}"] = v
    for k, v in _stage_qual(qual, pids).items():
        z[f"qual_{k}"] = v
    for k, v in _stage_conv(conv, pids).items():
        z[f"conv_{k}"] = v
    p = path or _ckpt_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **z)
    return p


def ckpt_load(path=None):
    p = path or _ckpt_path()
    if not p.exists():
        return None
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _ck_stage(ck, key):
    """Extract one stage's unprefixed block from a loaded checkpoint (None if absent)."""
    if ck is None or f"{key}_theta" not in ck:
        return None
    pre = key + "_"
    return {k[len(pre):]: v for k, v in ck.items() if k.startswith(pre)}


def _warm_rate_x0(x0, w, R, NU, PS, QI, has_a2, nb):
    """Best-effort map of a checkpoint's rate θ onto this fit's layout. Keys: (pid, season) for the
    three unit blocks, column name for ctx, (venue, season) for arenas. Unmatched entries keep the
    cold init. Returns the number of units carried."""
    u_pid, u_sea = _rate_unit_keys(R)
    oth = np.asarray(w["theta"], dtype=np.float64)
    onu, ok, onar = len(w["unit_pid"]), int(w["n_ctx"]), len(w["arena_venue"])
    ops = 1 + 3 * onu + ok                                   # old psi0 index
    old = {(str(p), int(s)): j for j, (p, s) in enumerate(zip(w["unit_pid"], w["unit_season"]))}
    x0[0] = oth[0]
    hits = 0
    for i, (p, s) in enumerate(zip(u_pid, u_sea)):
        j = old.get((str(p), int(s)))
        if j is not None:
            hits += 1
            for b in range(3):
                x0[1 + b * NU + i] = oth[1 + b * onu + j]
    oc = {str(nm): float(v) for nm, v in zip(w["ctx_names"], oth[1 + 3 * onu:ops])}
    for j, nm in enumerate(R.get("ctx_names") or []):
        if nm in oc:
            x0[1 + 3 * NU + j] = oc[nm]
    x0[PS] = oth[ops]                                        # psi0 (create_0)
    oa = {(str(v), int(s)): float(x) for v, s, x in
          zip(w["arena_venue"], w["arena_season"], oth[ops + 1:ops + 1 + onar])}
    for j, (v, s) in enumerate(zip(R.get("arena_venue") or [], R.get("arena_season") or [])):
        x0[PS + 1 + j] = oa.get((str(v), int(s)), 0.0)
    pos = ops + 1 + onar                                     # old tail: [q?] [log r?]
    if bool(w["has_a2"]):
        if has_a2:
            x0[QI] = oth[pos]
        pos += 1
    if bool(w["nb"]) and nb:
        x0[-1] = oth[pos]
    return hits


def _reuse_theta_rate(w, n_th, R, has_a2, nb):
    u_pid, u_sea = _rate_unit_keys(R)
    ok = (w is not None and len(np.asarray(w["theta"])) == n_th
          and np.array_equal(np.asarray(w["unit_pid"]), u_pid)
          and np.array_equal(np.asarray(w["unit_season"]), u_sea)
          and [str(x) for x in w["ctx_names"]] == [str(x) for x in (R.get("ctx_names") or [])]
          and [str(x) for x in w["arena_venue"]] == [str(x) for x in (R.get("arena_venue") or [])]
          and bool(w["has_a2"]) == bool(has_a2) and bool(w["nb"]) == bool(nb))
    if not ok:
        raise ValueError("reuse: checkpoint does not match this rate fit exactly — run a full fit")
    return np.asarray(w["theta"], dtype=np.float64)


def _warm_qual_x0(x0, w, pids, ctx_names, ar_v, ar_s, P, k):
    """Map a checkpoint's quality θ onto this fit's layout (players by id, ctx by name, arenas by
    (venue, season)). Returns the number of players carried."""
    oth = np.asarray(w["theta"], dtype=np.float64)
    oP, ok, onar = len(w["pid"]), int(w["n_ctx"]), len(w["arena_venue"])
    old = {str(p): j for j, p in enumerate(w["pid"])}
    x0[0] = oth[0]
    hits = 0
    for i, p in enumerate(pids):
        j = old.get(str(p))
        if j is not None:
            hits += 1
            x0[1 + i] = oth[1 + j]                           # qshoot
            x0[3 + P + i] = oth[3 + oP + j]                  # qdef
    x0[1 + P:3 + P] = oth[1 + oP:3 + oP]                     # qcreate position pair
    oc = {str(nm): float(v) for nm, v in zip(w["ctx_names"], oth[3 + 2 * oP:3 + 2 * oP + ok])}
    for j, nm in enumerate(ctx_names or []):
        if nm in oc:
            x0[3 + 2 * P + j] = oc[nm]
    oa = {(str(v), int(s)): float(x) for v, s, x in
          zip(w["arena_venue"], w["arena_season"], oth[3 + 2 * oP + ok:3 + 2 * oP + ok + onar])}
    for j, (v, s) in enumerate(zip(ar_v or [], ar_s or [])):
        x0[3 + 2 * P + k + j] = oa.get((str(v), int(s)), 0.0)
    return hits


def _reuse_theta_qual(w, n_th, pids, Q):
    ok = (w is not None and pids is not None and len(np.asarray(w["theta"])) == n_th
          and np.array_equal(np.asarray(w["pid"]), np.asarray(pids))
          and [str(x) for x in w["ctx_names"]] == [str(x) for x in (Q.get("ctx_names") or [])]
          and [str(x) for x in w["arena_venue"]] == [str(x) for x in (Q.get("arena_venue") or [])])
    if not ok:
        raise ValueError("reuse: checkpoint does not match this quality fit exactly — run a full fit")
    return np.asarray(w["theta"], dtype=np.float64)


def _warm_conv_x0(x0, w, pids, gids, keys, ctx_names, S, kc, P):
    """Map a checkpoint's conversion θ onto this fit's layout (a/b by strength key, ctx by name,
    fin by player id, gsave by goalie id). Returns the number of shooters carried."""
    oth = np.asarray(w["theta"], dtype=np.float64)
    okeys = [str(x) for x in w["keys"]]
    oS, oP, okc = len(okeys), len(w["pid"]), int(w["n_ctx"])
    for i, kk in enumerate(keys):
        if kk in okeys:
            j = okeys.index(kk)
            x0[i] = oth[j]
            x0[S + i] = oth[oS + j]
    oc = {str(nm): float(v) for nm, v in zip(w["ctx_names"], oth[2 * oS:2 * oS + okc])}
    for j, nm in enumerate(ctx_names or []):
        if nm in oc:
            x0[2 * S + j] = oc[nm]
    o, oo = 2 * S + kc, 2 * oS + okc
    oldp = {str(p): j for j, p in enumerate(w["pid"])}
    hits = 0
    for i, p in enumerate(pids):
        j = oldp.get(str(p))
        if j is not None:
            hits += 1
            x0[o + i] = oth[oo + j]
    oldg = {str(g): j for j, g in enumerate(w["gid"])}
    for i, g in enumerate(gids):
        j = oldg.get(str(g))
        if j is not None:
            x0[o + P + i] = oth[oo + oP + j]
    return hits


def _reuse_theta_conv(w, n_th, pids, Cr, keys):
    ok = (w is not None and pids is not None and len(np.asarray(w["theta"])) == n_th
          and np.array_equal(np.asarray(w["pid"]), np.asarray(pids))
          and np.array_equal(np.asarray(w["gid"]), np.asarray(Cr["goalies"]))
          and [str(x) for x in w["keys"]] == list(keys)
          and [str(x) for x in w["ctx_names"]] == [str(x) for x in (Cr.get("ctx_names") or [])])
    if not ok:
        raise ValueError("reuse: checkpoint does not match this conversion fit exactly — run a full fit")
    return np.asarray(w["theta"], dtype=np.float64)


def _last_view(v, lu):
    """Per-player view of a per-unit array at each player's last state (0 where never active)."""
    out = np.zeros(len(lu))
    m = lu >= 0
    out[m] = v[lu[m]]
    return out


def _rate_ses(H, M, lay, lu, dense_max=DENSE_H_MAX):
    """Per-player SEs at each player's LAST state for the three rate blocks (columns from `lay`, a
    RateLayout). Dense inverse when the core Hessian has `dense_max` or fewer parameters; otherwise
    sparse splu column solves for just the reported columns. `create` (block 1) uses the sandwich
    Var_jj = h'Mh with h = H⁻¹e_j — the assist-credit's meat counts w² over the actual observed goals,
    so the up-weighted anchor adds curvature but not w× real information (F1). shoot/def use the plain
    (H⁻¹)_jj."""
    ncol = H.shape[0]
    P = len(lu)
    act = np.nonzero(lu >= 0)[0]
    cols = {b: lay.block_cols(b, lu[act]) for b in range(3)}
    se = [np.zeros(P) for _ in range(3)]
    if ncol <= dense_max:
        Hinv = np.linalg.inv(H.toarray())
        dj = np.clip(np.diag(Hinv), 0.0, None)
        for b in (0, 2):
            se[b][act] = np.sqrt(dj[cols[b]])
        hc = Hinv[:, cols[1]]
        se[1][act] = np.sqrt(np.clip(np.einsum("ij,ij->j", hc, M @ hc), 0.0, None))
    else:
        slu = splu(H.tocsc())
        for b in range(3):
            cj = cols[b]
            for i0 in range(0, len(cj), 512):
                sel = cj[i0:i0 + 512]
                E = np.zeros((ncol, len(sel)))
                E[sel, np.arange(len(sel))] = 1.0
                Hc = slu.solve(E)
                var = (np.einsum("ij,ij->j", Hc, M @ Hc) if b == 1
                       else Hc[sel, np.arange(len(sel))])
                se[b][act[i0:i0 + 512]] = np.sqrt(np.clip(var, 0.0, None))
    return se


def fit_rate_create(R, g_team, g_cidx, shots_per_goal, count_model="poisson", a2=None,
                    warm=None, reuse=False, create_prior_sd=None, def_prior_sd=None,
                    dense_max=DENSE_H_MAX):
    """UNIFIED CREATION over per-(player, season) STATES. One `create` parameter per unit that does
    double duty:
      (i) lifts teammates' shot rate in the Poisson/NB count layer, and
      (ii) is the creator-credit propensity in a conditional logit over the on-ice teammates,
           anchored to OBSERVED goal assisters (g_team = the 4 teammate unit indices per goal,
           g_cidx = the credited column: 0 = unassisted/create_0, 1..4 = teammate position).
    Fit jointly; `shots_per_goal` up-weights the sparse assist-credit — each observed goal-creator
    stands in for the ≈ shots/goal whose creator we never see (inverse-probability weighting) — so
    `create` blends 'I make my linemates shoot' with 'I'm the one credited with creating'.

    UNITS: when R carries the RW machinery (`states=True` rate rows), each block has one state per
    (player, active-season) pair, tied across seasons by the random-walk penalty (_rate_penalty):
    level ridge on the first state, N(prev, rw_sd²·gap) between consecutive states. Without it
    (the MA bucket, synthetic fixtures) units == players and the penalty is the old static ridge —
    the model degenerates exactly to the pre-curves behavior.

    SECONDARY ASSISTS (`a2` = (g2_team, g2_c1, g2_c2), unit-indexed like g_team): the (A1, A2) pair
    is a partial ranking of the teammates by creation involvement — an exploded-logit second stage
    over the 3 teammates left after masking A1's column. Because recorded A2s are part creation
    signal, part puck-touching noise, the second stage is a MIXTURE with a FITTED probability
    q = P(A2 reflects creation):  P(A2=c₂) = q·softmax(create_{T∖c₁})[c₂] + (1−q)/3.
    A bare weight on the A2 log-likelihood would be a pseudo-likelihood temperature (not MLE-
    fittable); q is a proper parameter, identified by A2's concordance with the create ordering
    already pinned by the counts + A1 anchor. The A2 term carries the same spg IPW weight, and its
    responsibility-weighted Fisher joins the create bread/meat.

    Returns per-unit shoot/create/def plus per-player `*_last` views (each player's most recent
    state — the "current skill" read) with SEs (`se_create_last` sandwich-corrected, F1).

    `warm` (a checkpoint stage block, see _ck_stage) initializes the optimizer at the previous
    solution mapped by key; `reuse=True` additionally SKIPS optimization and the SE Hessian,
    taking θ and SEs straight from the checkpoint — valid only for the exact same fit (enforced)."""
    P, k = len(R["players"]), R["Xctx"].shape[1]
    lsh = 1 / PRIOR_SD_SHOOT ** 2
    ldf = 1 / (def_prior_sd if def_prior_sd else PRIOR_SD_SHOOT) ** 2
    lcr = 1 / (create_prior_sd if create_prior_sd else PRIOR_SD_CREATE) ** 2
    nb = count_model == "nb"
    dmask = R.get("def_mask")
    if dmask is None:                                        # synthetic/legacy rows: all defenders real
        dmask = np.ones_like(R["def_idx"], dtype=np.float64)
    states = "unit_player" in R
    if states:
        NU = R["n_units"]
        sh_i, tm_i, df_i = R["shooter_unit"], R["team_unit"], R["def_unit"]
        up, fm = R["unit_player"], R["first_mask"]
        e_prev, e_next, e_gap = R["e_prev"], R["e_next"], R["e_gap"]
    else:
        NU = P
        sh_i, tm_i, df_i = R["shooter_idx"], R["team_idx"], R["def_idx"]
        up, fm = np.arange(P, dtype=np.int64), np.ones(P)
        e_prev = np.zeros(0, dtype=np.int64); e_next = np.zeros(0, dtype=np.int64); e_gap = np.ones(0)
    w_sh, w_cr, w_df = 1 / RW_SD_SHOOT ** 2, 1 / RW_SD_CREATE ** 2, 1 / RW_SD_DEF ** 2
    has_rw = len(e_prev) > 0
    n_ar = int(R.get("n_arenas", 0))                         # arena offsets: [PS+1, PS+1+n_ar)
    acol = R.get("arena_col")
    if acol is None:
        acol = np.full(len(R["count"]), -1, dtype=np.int32)
    la_ar, lrw_ar = 1 / ARENA_SD ** 2, 1 / ARENA_RW_SD ** 2  # (venue, season)-state nuisance prior
    a_ep = jnp.asarray(R.get("arena_e_prev", np.zeros(0, dtype=np.int64)))
    a_en = jnp.asarray(R.get("arena_e_next", np.zeros(0, dtype=np.int64)))
    a_iw = jnp.asarray(1.0 / np.maximum(R.get("arena_e_gap", np.ones(0)), 1e-9))
    has_ar_rw = int(a_ep.shape[0]) > 0
    has_a2 = a2 is not None and len(a2[0]) > 0               # A2 mixture stage; q at index PS+1+n_ar
    if has_a2:
        g2_team, g2_c1, g2_c2 = (np.asarray(a2[0], dtype=np.int64),
                                 np.asarray(a2[1], dtype=np.int64),
                                 np.asarray(a2[2], dtype=np.int64))
    else:
        g2_team = np.zeros((0, 4), dtype=np.int64)
        g2_c1 = g2_c2 = np.zeros(0, dtype=np.int64)

    lay = RateLayout(NU, k, n_ar, has_a2, nb)
    PS, QI = lay.PS, lay.QI                                  # psi0 index / A2-mixture-logit index

    hyp = RateHypers(nb=nb, has_a2=has_a2, has_rw=has_rw, has_ar_rw=has_ar_rw,
                     shots_per_goal=shots_per_goal, lsh=lsh, lcr=lcr, ldf=ldf,
                     w_sh=w_sh, w_cr=w_cr, w_df=w_df, la_ar=la_ar, lrw_ar=lrw_ar)
    nll = make_rate_nll(lay, hyp, (a_ep, a_en, a_iw))

    x0 = np.zeros(lay.n_theta)
    if nb:
        x0[-1] = 1.0                                         # NB dispersion log r starts at 1 (r ≈ e);
                                                             #   q's cold logit stays 0 ⇒ q = 0.5
    if reuse:
        th = _reuse_theta_rate(warm, len(x0), R, has_a2, nb)
        converged, gnorm, nit = bool(warm["converged"]), float(warm["grad_norm"]), 0
    else:
        if warm is not None:
            try:
                hits = _warm_rate_x0(x0, warm, R, NU, PS, QI, has_a2, nb)
                print(f"    warm start: {hits}/{NU} unit states carried from checkpoint")
            except Exception as e:                           # a stale checkpoint must never kill a fit
                print(f"    warm start skipped ({e}) — cold init")
                x0 = np.zeros_like(x0)
                if nb:
                    x0[-1] = 1.0
        res = _optimize(nll, x0, sh_i, tm_i, df_i, dmask, R["Xctx"], R["count"], R["offset"],
                        g_team, g_cidx, fm, e_prev, e_next, 1.0 / np.maximum(e_gap, 1e-9), acol,
                        g2_team, g2_c1, g2_c2)
        th, converged = res.x, bool(res.success)
        gnorm, nit = float(np.max(np.abs(res.jac))), int(res.nit)
    a2_q = float(_sigmoid(th[QI])) if has_a2 else None
    sh, cr, df = th[lay.shoot], th[lay.create], th[lay.defence]
    b, psi0 = th[lay.beta], float(th[lay.psi0])
    ar_vec = th[lay.arena]
    r = float(np.exp(th[-1])) if nb else None

    lu = np.full(P, -1, dtype=np.int64)
    lu[up] = np.arange(NU)                                   # last unit per player (units season-sorted)
    if reuse:                                                # same fit ⇒ SEs come from the checkpoint
        se_sh = np.asarray(warm["se_shoot_last"], dtype=np.float64)
        se_cr = np.asarray(warm["se_create_last"], dtype=np.float64)
        se_df = np.asarray(warm["se_def_last"], dtype=np.float64)
    else:
        eta = th[0] + sh[sh_i] + cr[tm_i].sum(1) + (df[df_i] * dmask).sum(1) + R["Xctx"] @ b
        if n_ar:
            eta = eta + np.where(acol >= 0, ar_vec[np.clip(acol, 0, None)], 0.0)
        mu = np.exp(eta + R["offset"])
        W = mu * r / (r + mu) if nb else mu

        # bread H = XᵀWX + penalty + w·(credit Fisher); meat M = XᵀWX + w²·(credit Fisher)  [F1]
        Xs = _build_X(sh_i, tm_i, df_i, R["Xctx"], lay, dmask)
        XtWX = (Xs.T @ Xs.multiply(W[:, None])).tocsr()
        logit5 = np.concatenate([np.full((len(g_cidx), 1), psi0), cr[g_team]], 1)
        pi = np.exp(logit5 - logit5.max(1, keepdims=True)); pi /= pi.sum(1, keepdims=True)
        cinfo = np.zeros(NU)                                 # per-goal creator Fisher info (weight 1)
        for t in range(4):
            np.add.at(cinfo, g_team[:, t], pi[:, t + 1] * (1 - pi[:, t + 1]))
        if has_a2:                                           # A2 Fisher, responsibility-weighted by the
            lg2 = cr[g2_team].astype(np.float64)             # mixture (noise share carries no info)
            lg2[np.arange(4)[None, :] == g2_c1[:, None]] = -1e30
            pi2 = np.exp(lg2 - lg2.max(1, keepdims=True)); pi2 /= pi2.sum(1, keepdims=True)
            plc = pi2[np.arange(len(g2_c2)), g2_c2]
            resp = a2_q * plc / (a2_q * plc + (1.0 - a2_q) / 3.0 + 1e-12)
            for t in range(4):
                valid = (g2_c1 != t).astype(np.float64)
                np.add.at(cinfo, g2_team[:, t], resp * pi2[:, t] * (1 - pi2[:, t]) * valid)
        ci = np.arange(lay.create.start, lay.create.stop)
        cdiag = sparse.csr_matrix((cinfo, (ci, ci)), shape=(lay.n_core, lay.n_core))
        H = XtWX + _rate_penalty(lay, fm, e_prev, e_next, e_gap) + shots_per_goal * cdiag
        M = XtWX + shots_per_goal ** 2 * cdiag
        se_sh, se_cr, se_df = _rate_ses(H, M, lay, lu, dense_max)
    out = {"intercept": float(th[0]), "beta": b, "psi0": psi0,
           "shoot": sh, "create": cr, "def": df,
           "unit_player": up, "unit_season": R.get("unit_season"), "last_unit": lu,
           "shoot_last": _last_view(sh, lu), "create_last": _last_view(cr, lu),
           "def_last": _last_view(df, lu),
           "se_shoot_last": se_sh, "se_create_last": se_cr, "se_def_last": se_df,
           "count_model": count_model, "r": r, "converged": converged,
           "grad_norm": gnorm, "nit": nit, "theta": np.asarray(th, dtype=np.float64).copy(),
           "ctx_names": R.get("ctx_names"),
           "arena_vec": ar_vec, "arena_venue": R.get("arena_venue"),
           "arena_season": R.get("arena_season"), "a2_q": a2_q, "n_a2": int(len(g2_c2))}
    if not states:                                           # static aliases (units == players)
        out["se_shoot"], out["se_create"], out["se_def"] = se_sh, se_cr, se_df
    return out


def fit_quality_creator(Q, P, creates, isD=None, warm=None, reuse=False, pids=None):
    """QUALITY fit, POOLED across strengths (EV+MA). Estimates one shared set of qshoot/qdef per player
    (danger is an intrinsic, strength-neutral skill — xG already encodes the man-advantage), with the
    strength environment absorbed by a `pp` context column (per-strength intercept). `qcreate` is a
    POSITION-LEVEL pair [F, D] (A1): per-player creation quality is not identifiable from ~20 observed
    setups each (docs §7), so the danger a creator adds is pooled to his position — every remaining
    parameter is one the data can actually estimate. The creator distribution
    pi = softmax([create_0, create[teammates]]) is FIXED from the rate fit and chosen PER ROW by that
    shot's strength: `creates` = {strength_label: (create_array, psi0[, team_cols])} where the optional
    `team_cols` (n,4) maps this strength's rows into `create_array` (the EV bucket passes per-season
    UNIT indices; without it Q["team_idx"] player indices are used). Creator is OBSERVED on goals
    (creator ≥ 0), MARGINALIZED over pi otherwise — including goals whose credited assister was not an
    on-ice teammate (creator = −1, F4). Defenders masked (≤5). SEs: diagonal Gauss-Newton."""
    n, k = len(Q["y"]), Q["Xctx"].shape[1]
    cre = Q["creator"]                                      # -1 latent, 0..3 teammate, 4 unassisted
    cidx_np = np.where(cre == 4, 0, np.clip(cre, 0, 3) + 1).astype(np.int64)  # col in [unassist, t0..t3]
    obs_np = (cre >= 0) & (Q["goal"] == 1)                  # rows with an OBSERVED creator label
    lqs, lqc, lqd = 1 / PRIOR_SD_QSHOOT ** 2, 1 / PRIOR_SD_QCREATE ** 2, 1 / PRIOR_SD_QSHOOT ** 2
    strength = Q.get("strength", np.zeros(n, dtype=np.int64))
    dmask = Q.get("def_mask")
    if dmask is None:
        dmask = np.ones_like(Q["def_idx"], dtype=np.float64)
    if isD is None:
        isD = np.zeros(P)
    tpos = isD[Q["team_idx"]].astype(np.int64)              # (n,4) teammate position: 0=F, 1=D
    n_ar = int(Q.get("n_arenas", 0))                        # arena offsets: last n_ar params
    acol = Q.get("arena_col")
    if acol is None:
        acol = np.full(n, -1, dtype=np.int32)
    la_ar, lrw_ar = 1 / ARENA_SD ** 2, 1 / ARENA_RW_SD ** 2
    a_ep = jnp.asarray(Q.get("arena_e_prev", np.zeros(0, dtype=np.int64)))
    a_en = jnp.asarray(Q.get("arena_e_next", np.zeros(0, dtype=np.int64)))
    a_iw = jnp.asarray(1.0 / np.maximum(Q.get("arena_e_gap", np.ones(0)), 1e-9))
    has_ar_rw = int(a_ep.shape[0]) > 0
    lg = np.zeros((n, 5))                                   # per-row creator logits by strength
    for st, spec in creates.items():
        cr_s, psi_s = spec[0], spec[1]
        tcols = spec[2] if len(spec) > 2 else Q["team_idx"]
        m = strength == st
        if m.any():
            lg[m] = np.concatenate([np.full((int(m.sum()), 1), psi_s), cr_s[tcols[m]]], 1)
    pi_np = np.exp(lg - lg.max(1, keepdims=True)); pi_np /= pi_np.sum(1, keepdims=True)
    lay = QualLayout(P, k, n_ar)

    hyp = QualHypers(has_ar_rw=has_ar_rw, lqs=lqs, lqc=lqc, lqd=lqd, la_ar=la_ar, lrw_ar=lrw_ar)
    nll = make_quality_nll(lay, hyp, (a_ep, a_en, a_iw))

    x0 = np.zeros(lay.n_theta)
    if reuse:
        th = _reuse_theta_qual(warm, len(x0), pids, Q)
        converged, gnorm, nit = bool(warm["converged"]), float(warm["grad_norm"]), 0
    else:
        if warm is not None and pids is not None:
            try:
                hits = _warm_qual_x0(x0, warm, pids, Q.get("ctx_names"), Q.get("arena_venue"),
                                     Q.get("arena_season"), P, k)
                print(f"    quality warm start: {hits}/{P} players carried from checkpoint")
            except Exception as e:                           # a stale checkpoint must never kill a fit
                print(f"    quality warm start skipped ({e}) — cold init")
                x0 = np.zeros_like(x0)
        res = _optimize(nll, x0, Q["shooter_idx"], Q["team_idx"], tpos,
                        Q["def_idx"], dmask, Q["Xctx"], _sigmoid(Q["y"]), obs_np, cidx_np, pi_np, acol)
        th, converged = res.x, bool(res.success)
        gnorm, nit = float(np.max(np.abs(res.jac))), int(res.nit)
    mq, qs, qc, qd, b = (float(th[lay.mq]), th[lay.qshoot], th[lay.qcreate], th[lay.qdef], th[lay.beta])
    arq_vec = th[lay.arena]
    # Beta concentration from residual MSE around the fitted mean
    base = mq + qs[Q["shooter_idx"]] + (qd[Q["def_idx"]] * dmask).sum(1) + Q["Xctx"] @ b
    if n_ar:
        base = base + np.where(acol >= 0, arq_vec[np.clip(acol, 0, None)], 0.0)
    sig5 = np.concatenate([_sigmoid(base)[:, None], _sigmoid(base[:, None] + qc[tpos])], 1)
    ci = cidx_np
    m = (pi_np * sig5).sum(1)                                    # marginal mean, FIXED pi
    fitted = np.where(obs_np, sig5[np.arange(n), ci], m)
    mse = float(np.mean((_sigmoid(Q["y"]) - fitted) ** 2))
    s_conc = max(float(np.mean(fitted * (1 - fitted)) / max(mse, 1e-9) - 1.0), 1.0)

    # diagonal Gauss-Newton SE for the two qcreate params: observed-creator goals + latent marginal
    tm = Q["team_idx"]
    info_qc = np.zeros(2)
    tg, sg, cig, tpg = tm[obs_np], sig5[obs_np], ci[obs_np], tpos[obs_np]
    is_tm = cig >= 1
    cr_pl = tg[np.arange(len(tg)), np.clip(cig - 1, 0, 3)]       # creator player (for n_create)
    cr_pos = tpg[np.arange(len(tpg)), np.clip(cig - 1, 0, 3)]    # creator position (for info)
    sc = sg[np.arange(len(sg)), cig]
    np.add.at(info_qc, cr_pos[is_tm], (sc * (1 - sc))[is_tm])
    lat = ~obs_np
    pn, sn, mn, tpn = pi_np[lat], sig5[lat], m[lat], tpos[lat]
    den = np.clip(mn * (1 - mn), 1e-9, None)
    for pos in (0, 1):
        gpos = sum(pn[:, t + 1] * sn[:, t + 1] * (1 - sn[:, t + 1]) * (tpn[:, t] == pos)
                   for t in range(4))
        info_qc[pos] += float(np.sum(gpos ** 2 / den))
    se_qc = (np.asarray(warm["se_qcreate"], dtype=np.float64) if reuse
             else 1.0 / np.sqrt(info_qc + lqc))
    n_create = np.zeros(P, dtype=np.int64)
    np.add.at(n_create, cr_pl[is_tm], 1)
    # per-strength quality intercept: EV = mq; PP = mq + (pp-column coefficient, Xctx col 1)
    mu_qual = {"ev": mq, "ma": mq + (float(b[1]) if len(b) > 1 else 0.0)}
    return {"intercept": mq, "mu_qual": mu_qual, "qshoot": qs, "qcreate": qc, "qdef": qd, "beta": b,
            "beta_s": s_conc, "se_qcreate": se_qc, "n_create": n_create, "converged": converged,
            "grad_norm": gnorm, "nit": nit, "theta": np.asarray(th, dtype=np.float64).copy(),
            "ctx_names": Q.get("ctx_names"),
            "arena_vec": arq_vec, "arena_venue": Q.get("arena_venue"),
            "arena_season": Q.get("arena_season")}


# ── conversion fit (native logit, keyed off the observed xG) ────────────────────────────────────────

def fit_conversion(Cr, P, warm=None, reuse=False, pids=None):
    """CONVERSION, fit NATIVELY inside this model (no shooting-model reuse). A Bernoulli goal model
    keyed off the shot's OBSERVED xG, on the logit scale:
        logit(p_goal_i) = a·logit(xg_i) + b + fin[shooter_i] + gsave[goalie_i]
    `a` (slope, recalibrates the xG→goal map — ≈1 if xG is well-calibrated on 5v5) and `b` (intercept,
    replaces the old mu_conv) are UNPENALIZED: leaving `b` free makes the MLE score equation Σ(y−p)=0,
    i.e. Σp = Σy, so deterministic expected goals still reconcile EXACTLY — now as a natural first-order
    condition, not a hand-solved constant. `fin` (per shooter) and `gsave` (per goalie, <0 = good) are
    log-odds offsets with EB ridge priors whose SDs come from the pre-calculation stage
    (estimate_conversion_prior_sds) — recomputed here each fit, then FIXED during it. Independent of
    Stages 1-2 (uses observed xg, not create/qcreate/qbar). SEs: diagonal Gauss-Newton (W = p(1−p))."""
    G = len(Cr["goalies"])
    prior_sd_fin, prior_sd_gsave = estimate_conversion_prior_sds(Cr, P)   # pre-calc: data-calibrated ridge
    lf, lg = 1 / prior_sd_fin ** 2, 1 / prior_sd_gsave ** 2
    slab = Cr.get("strength")
    if slab is None:
        slab = np.zeros(len(Cr["y"]), dtype=np.int64)
    present = sorted({int(x) for x in np.unique(slab)})     # strength labels present (0=EV, 1=MA)
    smap = {s: i for i, s in enumerate(present)}
    sidx = np.array([smap[int(x)] for x in slab], dtype=np.int64)   # compact 0..S-1
    S = len(present)
    keys = ["ev" if s == 0 else "ma" for s in present]      # a/b are per-strength; fin/gsave pooled
    Cctx = Cr.get("ctx")                                    # global block: season offsets + age curves
    kc = 0 if Cctx is None else Cctx.shape[1]
    if Cctx is None:
        Cctx = np.zeros((len(Cr["y"]), 0))
    lay = ConvLayout(S, kc, P, G)

    nll = make_conversion_nll(lay, lf, lg)

    x0 = np.zeros(lay.n_theta)
    x0[lay.a] = 1.0                                         # start each strength's slope at 1
    if reuse:
        th = _reuse_theta_conv(warm, len(x0), pids, Cr, keys)
        converged, gnorm, nit = bool(warm["converged"]), float(warm["grad_norm"]), 0
    else:
        if warm is not None and pids is not None:
            try:
                hits = _warm_conv_x0(x0, warm, pids, Cr["goalies"], keys, Cr.get("ctx_names"), S, kc, P)
                print(f"    conversion warm start: {hits}/{P} shooters carried from checkpoint")
            except Exception as e:                          # a stale checkpoint must never kill a fit
                print(f"    conversion warm start skipped ({e}) — cold init")
                x0 = np.zeros_like(x0); x0[lay.a] = 1.0
        res = _optimize(nll, x0, sidx, Cr["shooter_idx"], Cr["goalie_idx"], Cr["logit_xg"], Cctx, Cr["y"])
        th, converged = res.x, bool(res.success)
        gnorm, nit = float(np.max(np.abs(res.jac))), int(res.nit)
    a_v, b_v, c_v, fin, gsave = (th[lay.a], th[lay.b], th[lay.ctx], th[lay.fin], th[lay.gsave])

    eta = (a_v[sidx] * Cr["logit_xg"] + b_v[sidx] + Cctx @ c_v
           + fin[Cr["shooter_idx"]] + gsave[Cr["goalie_idx"]])
    p = _sigmoid(eta)
    if reuse:                                               # same fit ⇒ SEs come from the checkpoint
        se_fin = np.asarray(warm["se_fin"], dtype=np.float64)
        se_gsave = np.asarray(warm["se_gsave"], dtype=np.float64)
    else:
        W = p * (1 - p)
        X = _build_conv_X(Cr["logit_xg"], sidx, Cr["shooter_idx"], Cr["goalie_idx"], lay,
                          Cctx if kc else None)
        pen = np.zeros(lay.n_theta); pen[lay.fin] = lf; pen[lay.gsave] = lg
        H = (X.T @ X.multiply(W[:, None])).toarray() + np.diag(pen)
        se = _se_from_hessian(H)
        se_fin, se_gsave = se[lay.fin], se[lay.gsave]
    a = {k: float(v) for k, v in zip(keys, a_v)}
    b = {k: float(v) for k, v in zip(keys, b_v)}
    recon = "  ".join(f"{k}:Σp={p[sidx == i].sum():.0f}/Σy={Cr['y'][sidx == i].sum():.0f}"
                      for i, k in enumerate(keys))
    print(f"  conversion pre-calc EB prior SDs (per-fit): fin={prior_sd_fin:.3f} gsave={prior_sd_gsave:.3f} (logit)")
    print(f"  conversion (logit) fit: a={a} b={ {k: round(v,3) for k,v in b.items()} }  reconcile {recon}"
          f"  ({len(p):,} shots, {G} goalies, iters={nit})")
    out = {"a": a, "b": b, "fin": fin, "gsave": gsave, "goalies": Cr["goalies"],
           "beta": c_v, "ctx_names": Cr.get("ctx_names", []),
           "se_fin": se_fin, "se_gsave": se_gsave,
           "prior_sd_fin": prior_sd_fin, "prior_sd_gsave": prior_sd_gsave,
           "sum_p": float(p.sum()), "sum_y": float(Cr["y"].sum()), "n": int(len(p)),
           "converged": converged, "grad_norm": gnorm, "nit": nit,
           "theta": np.asarray(th, dtype=np.float64).copy()}
    if "season" in Cr:                                      # F6: per-season reconciliation (unpenalized
        out["recon_season"] = {int(s): [float(p[Cr["season"] == s].sum()),      # season offsets ⇒ exact)
                                        float(Cr["y"][Cr["season"] == s].sum())]
                               for s in np.unique(Cr["season"])}
    return out


# ── effective parameters: state + position offset + aging curve, per player ───────────────────────

def _coef_map(ctx_names, beta):
    """Named context coefficients (curve/offset extraction by column name; empty if unnamed)."""
    return {n: float(v) for n, v in zip(ctx_names or [], np.asarray(beta))}


def _block_curve(cm, blk, z, d, off_name=None):
    """One block's aging-curve + position-offset contribution at age-z and position d (0=F, 1=D):
    curve = F/D-split quadratic in z (coeffs `{blk}_zF … {blk}_z2D`), offset = `{blk}_D` (A2)."""
    w = [cm.get(f"{blk}_zF", 0.0), cm.get(f"{blk}_z2F", 0.0),
         cm.get(f"{blk}_zD", 0.0), cm.get(f"{blk}_z2D", 0.0)]
    return _curve_val(w, z, d) + d * cm.get(off_name or f"{blk}_D", 0.0)


def effective_params(rates, qual, conv, players, agepos, last_season, target=None):
    """Collapse the fitted model into per-player EFFECTIVE parameter arrays — last RW state (or
    static level) + position offset + aging curve — shaped exactly like the dicts player_values
    consumes. `target=None`: each player evaluated at his LAST active season ("current skill").
    `target=<season>`: ages advance to the target season while states stay at their last value (the
    RW mean) — the PROJECTION. Both use the reference-season environment (no season fixed-effects),
    so current and projected values are directly comparable. Missing birthdates stay at the curve
    reference (z = 0) and simply don't move under projection."""
    P = len(players)
    isD = agepos["isD"]
    slist = sorted(agepos["z"].keys())
    known = ~np.isnan(agepos["age"][slist[0]])
    z = np.zeros(P)
    for s in slist:
        m = last_season == s
        z[m] = agepos["z"][s][m]
    if target is not None:
        z = np.where(known & (last_season > 0), z + (target - last_season) / AGE_SCALE, z)
    rates_eff = {}
    for key, rate in rates.items():
        cm = _coef_map(rate.get("ctx_names"), rate["beta"])
        rates_eff[key] = {
            "intercept": rate["intercept"], "psi0": rate["psi0"],
            "value_env": rate.get("value_env"),
            "shoot": rate["shoot_last"] + _block_curve(cm, "shoot", z, isD),
            "create": rate["create_last"] + _block_curve(cm, "create", z, isD),
            "def": rate["def_last"] + _block_curve(cm, "def", z, isD),
        }
    qcm = _coef_map(qual.get("ctx_names"), qual["beta"])
    qc = np.asarray(qual["qcreate"])
    qual_eff = {"mu_qual": qual["mu_qual"], "beta_s": qual.get("beta_s"),
                "qshoot": qual["qshoot"] + isD * qcm.get("shooter_D", 0.0),
                "qcreate": qc[isD.astype(np.int64)] if qc.shape == (2,) else qc,   # A1: position pair
                "qcreate_pos": qc if qc.shape == (2,) else None,   # the [F, D] pair (creator classes)
                "qdef": qual["qdef"] + isD * qcm.get("def_D", 0.0)}
    ccm = _coef_map(conv.get("ctx_names"), conv.get("beta", []))
    conv_eff = {"a": conv["a"], "b": conv["b"],
                "fin": conv["fin"] + _block_curve(ccm, "fin", z, isD, off_name="shooter_D")}
    return rates_eff, qual_eff, conv_eff


def unit_effective(rate, agepos):
    """Per-unit (player, season) EFFECTIVE rate params — state + that season's curve value + D
    offset — the per-season skill trajectory behind the JSON `trend` block."""
    cm = _coef_map(rate.get("ctx_names"), rate["beta"])
    up, us = rate["unit_player"], rate["unit_season"]
    d = agepos["isD"][up]
    z = np.zeros(len(up))
    for s in sorted(agepos["z"].keys()):
        m = us == s
        z[m] = agepos["z"][s][up[m]]
    return {blk: rate[key] + _block_curve(cm, blk, z, d)
            for blk, key in (("shoot", "shoot"), ("create", "create"), ("def", "def"))}


# ── environment factor: the average-lineup multiplier the exported card values scale by ────────────

def value_environment(rate, season=None):
    """TOI-weighted mean ENVIRONMENT multiplier over the fit's own rows: everything in a row's
    log-rate beyond the intercept and the shooter's own effective shoot — teammates' create
    states, defenders' def states, the create/def age-position columns, game context, and season
    offsets. The value formulas multiply by this so card units are real-world per-60 rates: the
    reference cell (teammates at zero) runs ≈2× low because the league-average teammate's
    EFFECTIVE create is ≈ +0.17, not 0 (the 2026-07 Hyman-case audit). `season` restricts to that
    season's rows (the current-skill environment); arena states are excluded (≈0 mean by ridge)."""
    R = rate["R"]
    own = {"shoot_D", "shoot_zF", "shoot_z2F", "shoot_zD", "shoot_z2D"}
    beta = np.asarray(rate["beta"], dtype=np.float64).copy()
    for j, nm in enumerate(R.get("ctx_names") or []):
        if nm in own:
            beta[j] = 0.0
    cr, df = np.asarray(rate["create"]), np.asarray(rate["def"])
    dm = R.get("def_mask")
    if dm is None:
        dm = np.ones_like(R["def_idx"], dtype=np.float64)
    if "team_unit" in R:
        eta = cr[R["team_unit"]].sum(1) + (df[R["def_unit"]] * dm).sum(1)
    else:
        eta = cr[R["team_idx"]].sum(1) + (df[R["def_idx"]] * dm).sum(1)
    eta = eta + R["Xctx"] @ beta
    w = np.asarray(R["dur"], dtype=np.float64)
    if season is not None and "season_row" in R:
        m = np.asarray(R["season_row"]) == season
        if m.any():
            eta, w = eta[m], w[m]
    return float(np.sum(w * np.exp(eta)) / max(np.sum(w), 1e-9))


# ── fit-log helpers: one-line arena + aging-curve summaries printed during the fit ──────────────────


def _arena_report(fit, n=3, pct=True):
    """One-line summary of the fitted (venue, season) arena states: biggest/smallest venue means
    (rate: ≈% on shot counts; quality: logit-xG units) + the largest within-venue season drift."""
    vec = fit.get("arena_vec")
    if vec is None or not len(vec):
        return None
    g = pd.DataFrame({"v": fit["arena_venue"], "x": vec}).groupby("v").x
    m = g.mean().sort_values()
    fmt = (lambda x: f"{100 * x:+.1f}%") if pct else (lambda x: f"{x:+.3f}")
    lo = "  ".join(f"{v} {fmt(x)}" for v, x in m.head(n).items())
    hi = "  ".join(f"{v} {fmt(x)}" for v, x in m.tail(n)[::-1].items())
    return f"hi {hi}   lo {lo}   (max venue drift {fmt(float((g.max() - g.min()).max()))})"


def _curve_report(cm, blk):
    """One-line console summary of a fitted aging curve: curve value at 22/32 per position and the
    implied peak age (where the quadratic tops out, if it does)."""
    parts = []
    for pos in ("F", "D"):
        w1, w2 = cm.get(f"{blk}_z{pos}", 0.0), cm.get(f"{blk}_z2{pos}", 0.0)
        pk = AGE_PEAK - AGE_SCALE * w1 / (2 * w2) if w2 < -1e-9 else None
        v = {a: w1 * (a - AGE_PEAK) / AGE_SCALE + w2 * ((a - AGE_PEAK) / AGE_SCALE) ** 2
             for a in (22, 32)}
        parts.append(f"{pos}: 22y {v[22]:+.3f} 32y {v[32]:+.3f}"
                     + (f" peak {pk:.1f}" if pk is not None else ""))
    return "   ".join(parts)


def fit_all(seasons, count_model="poisson", spg_scale=1.0, warm=False, reexport=False,
            save_ckpt=False, ckpt_path=None, ma_anchor_scale=1.0, ma_create_prior_sd=None,
            ma_def_prior_sd=None):
    """Fit every stage on `seasons` and return the raw fit objects — the shared engine behind run()
    (which adds leaderboards, values, PPC, projection, JSON) and the held-out predictive harness
    (generative_holdout, which needs the fits without the reporting tail). Returns a dict with
    players/idx/agepos/Q/rates/spg/qual/conv/last_season.

    `warm` initializes each stage's optimizer from the θ̂ checkpoint (key-mapped; see ckpt_save) —
    exact refits converge in a handful of iterations, incremental refits (new season/players) in a
    fraction of a cold run. `reexport` skips optimization AND the SE Hessians entirely (requires a
    checkpoint from the exact same fit signature) — for regenerating reports/JSON after export-side
    changes. `save_ckpt` writes the checkpoint after fitting; `ckpt_path` overrides its location
    (run() uses the production path; the holdout harness keeps its own chain, so train-side fits
    never touch a solution that saw test data).

    `ma_anchor_scale` / `ma_create_prior_sd`: the MA (PP/PK) bucket's assist-anchor weight scale
    and create prior SD — hyperparameters selected by held-out validation (docs §7): on fixed PP
    units the shot counts pin only the unit's total creation, so the assist anchor alone splits
    it, and assists on the PP reflect ROLE (who touches the puck last), not creation skill. The EV
    bucket is untouched (linemate mixing identifies its split; validated at full weight)."""
    print(f"[generative_model:shooter-resolved] seasons {seasons} — count {count_model} — EV + PP/PK …")
    if ma_anchor_scale != 1.0 or ma_create_prior_sd or ma_def_prior_sd:
        print(f"  MA bucket: anchor×{ma_anchor_scale:g}"
              + (f", create prior sd {ma_create_prior_sd:g}" if ma_create_prior_sd else "")
              + (f", def prior sd {ma_def_prior_sd:g}" if ma_def_prior_sd else ""))
    ck = ckpt_load(ckpt_path) if (warm or reexport) else None
    if reexport:
        if ck is None:
            raise SystemExit("--reexport needs a θ̂ checkpoint from a prior full fit")
        ck_mas = float(ck.get("ma_anchor_scale", 1.0))
        ck_mcp = float(ck.get("ma_create_prior_sd", np.nan))
        if (list(map(int, ck["seasons"])) != [int(s) for s in seasons]
                or str(ck["count_model"]) != count_model
                or float(ck["spg_scale"]) != float(spg_scale)
                or ck_mas != float(ma_anchor_scale)
                or not (np.isnan(ck_mcp) and ma_create_prior_sd is None
                        or ck_mcp == float(ma_create_prior_sd or np.nan))):
            raise SystemExit("--reexport: checkpoint signature (seasons/count/spg-scale/MA hypers) "
                             "differs — run a full fit")
    elif ck is not None:
        print(f"  warm start ← checkpoint ({'+'.join(str(int(s)) for s in ck['seasons'])}, "
              f"{ck['count_model']})")
    players, idx = player_index(seasons)
    if not players:
        raise SystemExit("no stints")
    P = len(players)
    agepos = _age_position(players, seasons)
    print(f"  ages/positions: {P} players, {agepos['missing']} missing birthdates, "
          f"{int(agepos['isD'].sum())} D")

    # pooled quality rows (EV+MA); its goal subset (split by strength) anchors each bucket's `create`
    Q = quality_creator_rows(seasons, idx, ALL_STRENGTHS, agepos)
    cidx_all = np.where(Q["creator"] == 4, 0, np.clip(Q["creator"], 0, 3) + 1).astype(np.int64)

    rates, spg, creates = {}, {}, {}
    last_season = np.full(P, -1, dtype=np.int64)
    for key, strengths, dual in [("ev", EV_STRENGTHS, True), ("ma", MA_STRENGTHS, False)]:
        R = rate_rows(seasons, strengths, dual, players, idx, agepos, states=dual)
        if R is None or len(R["count"]) == 0:
            print(f"  [{key}] no stints — skipping"); continue
        slab = 0 if key == "ev" else 1
        gm = (Q["goal"] == 1) & (Q["strength"] == slab)
        ngoal = int(gm.sum()); nun = int(((Q["creator"] == 4) & (Q["strength"] == slab)).sum())
        aw = ma_anchor_scale if key == "ma" else 1.0         # per-bucket anchor weight (docs §7)
        spg[key] = aw * spg_scale * float(R["count"].sum()) / max(ngoal, 1)  # IPW weight (×A3 scale)
        anchor = gm & (Q["creator"] >= 0)                    # F4: off-ice-assister goals are latent
        gt, gc = Q["team_idx"][anchor], cidx_all[anchor]
        sord = {s: i for i, s in enumerate(R["seasons"])}
        # A2 stage: goals with an on-ice A1 (a column to mask) AND an on-ice A2
        anchor2 = (gm & (Q["creator"] >= 0) & (Q["creator"] <= 3)
                   & (Q["creator2"] >= 0) & (Q["creator2"] != Q["creator"]))
        gt2, g1c, gc2 = Q["team_idx"][anchor2], Q["creator"][anchor2], Q["creator2"][anchor2]
        if dual:                                             # anchor teammates → (player, season) units
            qs = np.array([sord[s] for s in Q["season"][anchor]], dtype=np.int64)
            gtu = R["unit_lut"][gt, qs[:, None]]
            ok = (gtu >= 0).all(1)
            if int((~ok).sum()):
                print(f"  [{key}] {int((~ok).sum())} anchor goals dropped (teammate w/o rate exposure)")
            gt, gc = gtu[ok], gc[ok]
            qs2 = np.array([sord[s] for s in Q["season"][anchor2]], dtype=np.int64)
            gtu2 = R["unit_lut"][gt2, qs2[:, None]]
            ok2 = (gtu2 >= 0).all(1)
            gt2, g1c, gc2 = gtu2[ok2], g1c[ok2], gc2[ok2]
        print(f"  [{key}] rate rows {len(R['count']):,}  shots {R['count'].sum():.0f}  goals {ngoal}  "
              f"spg {spg[key]:.1f}  anchored {len(gc)} ({100 * nun / max(ngoal, 1):.0f}% unassisted, "
              f"{len(gc2)} with A2)")
        rate = fit_rate_create(R, gt, gc, spg[key], count_model=count_model, a2=(gt2, g1c, gc2),
                               warm=_ck_stage(ck, key), reuse=reexport,
                               create_prior_sd=(ma_create_prior_sd if key == "ma" else None),
                               def_prior_sd=(ma_def_prior_sd if key == "ma" else None))
        rate["R"] = R
        rates[key] = rate
        last_season = np.maximum(last_season, R["last_season"])
        rdesc = f" r={rate['r']:.2f}" if rate.get("r") else ""
        nu = f" units={R['n_units']:,}" if dual and "n_units" in R else ""
        qd = f" a2_q={rate['a2_q']:.3f}" if rate.get("a2_q") is not None else ""
        print(f"    rate fit ({count_model}): converged={rate['converged']} |grad|={rate['grad_norm']:.1e}"
              f"{rdesc} create_0={rate['psi0']:+.3f}{nu}{qd} iters={rate['nit']}")
        cm = _coef_map(R["ctx_names"], rate["beta"])
        for blk in ("shoot", "create", "def"):
            print(f"    [{key}] {blk}-curve   {_curve_report(cm, blk)}")
        arep = _arena_report(rate)
        if arep:
            print(f"    [{key}] arena rate effects   {arep}")
        if dual:                                             # stage-2 pi via per-season unit states
            qs_all = np.array([sord[s] for s in Q["season"]], dtype=np.int64)
            tu_all = R["unit_lut"][Q["team_idx"], qs_all[:, None]]
            cre = np.append(rate["create"], 0.0)             # sentinel for unmapped teammates (no bump)
            tu_all = np.where(tu_all >= 0, tu_all, len(rate["create"]))
            creates[slab] = (cre, rate["psi0"], tu_all)
        else:
            creates[slab] = (rate["create"], rate["psi0"])
    if "ev" not in rates:
        raise SystemExit("no EV stints")

    # pooled quality fit (shared loadings; per-strength intercept; per-row creator dist by strength)
    qual = fit_quality_creator(Q, P, creates, isD=agepos["isD"], warm=_ck_stage(ck, "qual"),
                               reuse=reexport, pids=players)
    print(f"  quality fit (pooled EV+MA): converged={qual['converged']} |grad|={qual['grad_norm']:.1e} "
          f"beta_s={qual['beta_s']:.1f} mu_qual={ {k: round(v, 3) for k, v in qual['mu_qual'].items()} } "
          f"iters={qual['nit']}")
    qc, qse = qual["qcreate"], qual["se_qcreate"]
    print(f"  qcreate (position-level, A1): F {qc[0]:+.4f} ±{1.96 * qse[0]:.4f}   "
          f"D {qc[1]:+.4f} ±{1.96 * qse[1]:.4f}")
    qarep = _arena_report(qual, pct=False)
    if qarep:
        print(f"  arena quality (logit-xG) effects   {qarep}")

    # pooled conversion fit (shared fin/gsave; per-strength a/b; season offsets + curves in ctx)
    Cr = conversion_rows(seasons, idx, ALL_STRENGTHS, agepos)
    if Cr is None:
        raise SystemExit("no shots for the conversion fit")
    conv = fit_conversion(Cr, P, warm=_ck_stage(ck, "conv"), reuse=reexport, pids=players)
    ccm = _coef_map(conv.get("ctx_names"), conv.get("beta", []))
    if ccm:
        print(f"    fin-curve   {_curve_report(ccm, 'fin')}   goalie-age: z {ccm.get('g_z', 0.0):+.3f} "
              f"z² {ccm.get('g_z2', 0.0):+.3f}")
        if conv.get("recon_season"):
            rc = "  ".join(f"{s}:{v[0]:.0f}/{v[1]:.0f}" for s, v in sorted(conv["recon_season"].items()))
            print(f"    per-season Σp/Σy (F6): {rc}")
    if save_ckpt:
        cp = ckpt_save(seasons, count_model, spg_scale, rates, qual, conv, players, path=ckpt_path,
                       ma_anchor_scale=ma_anchor_scale, ma_create_prior_sd=ma_create_prior_sd,
                       ma_def_prior_sd=ma_def_prior_sd)
        print(f"  θ̂ checkpoint → {cp} (warm starts / --reexport)")
    return {"players": players, "idx": idx, "agepos": agepos, "Q": Q, "rates": rates, "spg": spg,
            "qual": qual, "conv": conv, "last_season": last_season}


# The value/PPC/leaderboard/JSON post-processing and the CLI live in generative_export; `run`/`main`
# are imported lazily here (only when this module is executed) so `python -m ...generative_model` still
# drives the fit + export while the import graph stays acyclic (export imports fit_all from here).
if __name__ == "__main__":
    from .generative_export import main
    main()
