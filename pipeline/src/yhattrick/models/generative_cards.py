r"""Player-card exporter for the generative model (Cards v2): GA/60 + WAR + trajectories.

Consumes the saved pooled fit (data/models/generative_model_<seasons>.json — must carry
`quality_ctx` and `strengths.*.rate_ctx`, exported since July 2026) plus the stints parquet, and
writes data/models/gen_cards.json for export_players.py to merge into the site JSONs (that side is
plain-JSON — JAX stays in this experimental group).

WHAT IT COMPUTES (per player)
  current-skill attribute values — the fit's effective values (last drift state + position offset
    + aging curve at his latest-season age), re-expressed in card units;
  GA/60 (vs replacement) — evaluate the model's closed-form production equations at the player's
    effective params and subtract the REPLACEMENT archetype's values (same zero point as WAR), so
    a freely-available player at his position is exactly 0 and regulars read positive:
        created/60   = [sc − sc_repl] + κ·[pm − pm_repl]
        prevented/60 = κ·[df − df_repl]
        GA/60        = created + prevented
    with sc/pm/df from the player_values formulas (docs/generative_model.md §2) and κ = league
    goals-per-xG from the conversion fit (≈1 by Σp=Σgoals; applied explicitly). PP (created) and
    PK (prevented) analogues from the MA bucket vs the PP/PK archetypes. TOI-weighted position
    means are still computed, but only to RANK players for the archetype band (ranks are
    baseline-invariant).
  trajectory — per-season values: that season's drift states + that season's age through the same
    formulas (baseline held at the current replacement level so seasons are comparable);
  projection — the fit's projected params → values → GA/60 (same baselines);
  WAR — Σ over his REAL stints of [E(GF−GA | actual lineup, with him) − E(… | him → replacement)],
    over EV + PP/PK, ÷ GOALS_PER_WIN. The rate model's exponential-additive form makes each swap
    closed-form: per stint side,
        E[GF]/h = cx · e^{Σ_A create} · e^{Σ_B def} · K_B · Σ_{j∈A} e^{shoot_j − create_j} · g_j
    where g_j = sigmoid(a·logit(q_own_j) + b + fin_j + gsave_goalie) is j's goals-per-shot and
    K_B = Π_{d∈B} sigmoid(mu_q + qdef_d)/sigmoid(mu_q) the defenders' quality factor. Swapping one
    player perturbs only his own term, the shared Σcreate / Σdef exponents, and K_B — O(1) with
    per-stint aggregates.
  v1 approximations (documented): defender quality enters multiplicatively (K_B) rather than
    inside each sigmoid; created shots convert at the shooter's g; arena states excluded (they
    ≈cancel in the swap difference); penalties are NOT included (production pipeline value shown
    separately on the card until the penalties stage lands).

CALIBRATION KNOBS (documented in docs/metrics.md)
  replacement level — per position (F/D), the TOI-weighted mean parameters of players in the
    REPL_BAND percentile band of GA/60 (default 8th–12th): an empirical "freely available player";
  GOALS_PER_WIN = 6.0 (standard rule of thumb; revisit with a season-estimated value).

Run:  make generative-cards   or
      uv run --group experimental python -m yhattrick.models.generative_cards [--fit <path>]
"""

from __future__ import annotations

import argparse
import json
from datetime import date

import numpy as np
import pandas as pd

from .. import config as C
from .generative_likelihood import (
    _sigmoid,
    creator_mix,
    marginal_goal_prob,
    EV_STRENGTHS,
    MA_STRENGTHS,
    AGE_PEAK,
    AGE_SCALE,
)
from .generative_features import side_rows

EV_GATE = 6000.0  # seconds (100 min) — EV card eligibility, matches export_players
MA_GATE = 2400.0  # seconds (40 min) — PP/PK card eligibility
GOALS_PER_WIN = 6.0  # goals → wins conversion (v1 constant)
REPL_BAND = (8.0, 12.0)  # GA/60 percentile band defining the replacement archetype
EPS = 1e-9
PRICED = ((5, 5), (5, 4))  # SHOOTER-relative (own, opp) skater counts the model prices; goals
# elsewhere are surfaced as `unpriced_goals` (SH offense, empty-net,
# extra-attacker, 5v3, OT…). NOTE the parquet `strength` column is
# HOME-vs-AWAY oriented — always re-orient by is_home.


def unpriced_goals(seasons, idx, last):
    """Per-player goals in situations the model does NOT price (transparency for the card — real
    goals, invisible to WAR until roadmap 5d-TODO(3) lands). Strength is re-oriented to the
    SHOOTER's side from is_home + situation skater counts (the raw `strength` label is
    home-vs-away). Returns (latest, window, latest per-situation detail)."""
    n = len(idx)
    up_last = np.zeros(n, dtype=np.int64)
    up_window = np.zeros(n, dtype=np.int64)
    detail = {}
    for s in seasons:
        f = C.PROCESSED / "shots_onice" / f"{int(s)}.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(
            f, columns=["shooter_id", "goal", "is_home", "sit_home_n", "sit_away_n"]
        )
        g = df[df.goal == 1]
        own = np.where(g.is_home, g.sit_home_n, g.sit_away_n).astype(int)
        opp = np.where(g.is_home, g.sit_away_n, g.sit_home_n).astype(int)
        for sid, o, q in zip(g.shooter_id, own, opp):
            if (o, q) in PRICED:
                continue
            i = idx.get(int(sid))
            if i is None:
                continue
            up_window[i] += 1
            if int(s) == last:
                up_last[i] += 1
                d = detail.setdefault(i, {})
                lbl = f"{o}v{q}"
                d[lbl] = d.get(lbl, 0) + 1
    return up_last, up_window, detail


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _curve_val4(coef, z, d):
    """Position-split quadratic curve value from an age_curves coef dict {zF,z2F,zD,z2D}."""
    return (1 - d) * (coef["zF"] * z + coef["z2F"] * z * z) + d * (
        coef["zD"] * z + coef["z2D"] * z * z
    )


def load_fit(path=None):
    """Load the pooled fit JSON (latest multi-season file by default) and index its players."""
    if path is None:
        cands = sorted(C.MODELS.glob("generative_model_*+*.json"), key=lambda p: p.stat().st_mtime)
        if not cands:
            raise SystemExit("no pooled generative_model_*.json — run the model with --pool first")
        path = cands[-1]
    fit = json.loads(path.read_text())
    if "quality_ctx" not in fit or "rate_ctx" not in fit["strengths"]["ev"]:
        raise SystemExit(
            f"{path.name} lacks quality_ctx/rate_ctx — rerun the pooled fit "
            "(exports added July 2026)"
        )
    fit["_path"] = str(path)
    return fit


# ── per-player effective parameter table (current skill + per-season) ────────────────────────────


def player_table(fit):
    """Vectorized per-player arrays from the fit JSON: ids, isD, age at last season, effective rate
    params (EV last state + MA static), quality/conversion effectives, values, TOI, trend."""
    ps = fit["players"]
    n = len(ps)
    qctx = fit["quality_ctx"]
    finc = fit["age_curves"]["fin"]
    t = {
        "id": np.array([p["id"] for p in ps], dtype=np.int64),
        "isD": np.array([1.0 if p["pos"] == "D" else 0.0 for p in ps]),
        "age": np.array([p["age"] if p.get("age") is not None else np.nan for p in ps]),
        "last_season": np.array([p.get("last_season") or -1 for p in ps], dtype=np.int64),
        "toi_ev": np.array([p["toi_ev"] for p in ps]),
        "toi_pp": np.array([p["toi_pp"] for p in ps]),
        "toi_pk": np.array([p["toi_pk"] for p in ps]),
        "trend": [p.get("trend") or {} for p in ps],
    }
    for k in (
        "ev_shoot",
        "ev_create",
        "ev_def",
        "pp_shoot",
        "pp_create",
        "pp_def",
        "ev_scoring",
        "ev_playmaking",
        "ev_defense",
        "pp_scoring",
        "pp_playmaking",
        "pk_defense",
        "ev_create_se",
        "qshoot",
        "qdef",
        "create_qual",  # per-player on-ice creation quality (0 when absent)
        "fin",
        "fin_se",
    ):
        t[k] = np.array([p.get(k, 0.0) or 0.0 for p in ps])
    d = t["isD"]
    t["qshoot_eff"] = t["qshoot"] + d * qctx.get("shooter_D", 0.0)
    t["qdef_eff"] = t["qdef"] + d * qctx.get("def_D", 0.0)
    z_last = np.where(np.isnan(t["age"]), 0.0, (t["age"] - AGE_PEAK) / AGE_SCALE)
    t["fin_eff"] = t["fin"] + _curve_val4(finc["coef"], z_last, d) + d * finc["d_offset"]
    t["z_last"] = z_last
    return t


def fin_eff_at(fit, t, z):
    """Finishing effective at age-z (the curve moves with age; the residual doesn't)."""
    finc = fit["age_curves"]["fin"]
    return t["fin"] + _curve_val4(finc["coef"], z, t["isD"]) + t["isD"] * finc["d_offset"]


# ── the closed-form value equations (docs/generative_model.md §2) ────────────────────────────────


def gbar_classes(fit, key, qshoot_eff, fin_eff, season=None):
    """(g0, gF, gD): the model's own goals-per-shot per CREATOR CLASS (unassisted / F-created /
    D-created), each the conversion curve marginalized over the fitted Beta shot-quality
    distribution (marginal_goal_prob). qcreate is parameterized with unassisted as the reference
    and created shots carry the (negative) position bumps — pricing every shot as unassisted is
    the ≈+13% level bug the 2026-07 audit found. `season` adds the quality model's own season
    offset (WAR rows are season-specific; the deployment-free values stay in the reference
    environment)."""
    mq = fit["mu_qual"]["ev" if key == "ev" else "ma"]
    if season is not None:
        mq = mq + fit["quality_ctx"].get(f"season_{season}", 0.0)
    a, b = (
        fit["conv"]["a"]["ev" if key == "ev" else "ma"],
        fit["conv"]["b"]["ev" if key == "ev" else "ma"],
    )
    s = float(fit["beta_s"])
    out = []
    for qc in (0.0, fit["qcreate"]["F"], fit["qcreate"]["D"]):
        qb = np.clip(_sigmoid(mq + qshoot_eff + qc), EPS, 1 - EPS)
        out.append(marginal_goal_prob(qb, s, a, b, fin_eff))
    return out


def gbar_class_deriv(fit, key, qshoot_eff, fin_eff, season=None, h=1e-3):
    """d(goals-per-shot)/d(logit-xg) per creator class — the local response to the per-player on-ice
    create_qual lift, which enters the mark's logit exactly like a shift in qshoot. Central finite
    difference on gbar_classes; used to apply each shooter's teammate-Σ(create_qual) lift (and its
    swap cross-term) linearly inside war_bucket. cq ≈ 0.01–0.1 logit, so the O(cq²) error is <1%."""
    gp = gbar_classes(fit, key, qshoot_eff + h, fin_eff, season)
    gm = gbar_classes(fit, key, qshoot_eff - h, fin_eff, season)
    return [(a2 - b2) / (2.0 * h) for a2, b2 in zip(gp, gm)]


def values_at(fit, t, key, sh, cr, df, fin_eff, n_def):
    """(scoring, playmaking, defense) per 60 at the given effective rate params, using the fit's
    intercepts for strength bucket `key` and the pooled quality/conversion effectives. Scoring's
    goals-per-shot is the creator-class + Beta marginal (mirrors player_values in the model)."""
    mu = fit["strengths"][key]["rate_intercept"]
    mq = fit["mu_qual"]["ev" if key == "ev" else "ma"]
    env = float((fit.get("value_env") or {}).get(key) or 1.0)  # average-environment factor
    qc_pos = np.where(t["isD"] > 0, fit["qcreate"]["D"], fit["qcreate"]["F"])
    shots = np.exp(mu + sh)
    g0, gF, gD = gbar_classes(fit, key, t["qshoot_eff"], fin_eff)
    w0, wf, wd = creator_mix(fit["strengths"][key]["psi0"], t["isD"], key)
    sc = env * shots * (w0 * g0 + wf * gF + wd * gD)
    pm = env * 4.0 * np.exp(mu) * (np.exp(cr) - 1.0) * _sigmoid(mq + qc_pos)
    base = np.exp(mu) * _sigmoid(mq)
    dfv = env * n_def * (base - np.exp(mu + df) * _sigmoid(mq + t["qdef_eff"]))
    return sc, pm, dfv


def kappa(fit):
    """League goals-per-xG at the average 5v5 shot (≈1 by construction; applied explicitly)."""
    mq = fit["mu_qual"]["ev"]
    a, b = fit["conv"]["a"]["ev"], fit["conv"]["b"]["ev"]
    q = _sigmoid(mq)
    return float(_sigmoid(a * _logit(np.array([q]))[0] + b) / q)


def _wmean(v, w, m):
    w = w * m
    return float(np.sum(v * w) / max(np.sum(w), EPS))


def baselines(t, vals, kap):
    """TOI-weighted position means of the modeled values — the 'baseline team' zero point."""
    out = {}
    for g, gm in (("F", t["isD"] < 0.5), ("D", t["isD"] > 0.5)):
        ev_el = (t["toi_ev"] >= EV_GATE) & gm
        pp_el = (t["toi_pp"] >= MA_GATE) & gm
        pk_el = (t["toi_pk"] >= MA_GATE) & gm
        out[g] = {
            "sc": _wmean(vals["sc"], t["toi_ev"], ev_el),
            "pm": _wmean(vals["pm"], t["toi_ev"], ev_el),
            "df": _wmean(vals["df"], t["toi_ev"], ev_el),
            "pp_sc": _wmean(vals["pp_sc"], t["toi_pp"], pp_el),
            "pp_pm": _wmean(vals["pp_pm"], t["toi_pp"], pp_el),
            "pk_df": _wmean(vals["pk_df"], t["toi_pk"], pk_el),
        }
    return out


def ga60_of(t, base, sc, pm, df, kap):
    """GA/60 vs the baseline team: difference each component against the player's position mean."""
    bs = np.where(t["isD"] > 0, base["D"]["sc"], base["F"]["sc"])
    bp = np.where(t["isD"] > 0, base["D"]["pm"], base["F"]["pm"])
    bd = np.where(t["isD"] > 0, base["D"]["df"], base["F"]["df"])
    return (sc - bs) + kap * (pm - bp) + kap * (df - bd)


def band_mean(arr, toi, elig, pct, isD, band=None):
    """Per-position TOI-weighted mean of `arr` over the replacement percentile band of `pct`
    among the CONTEXT-eligible (`elig`) players — the archetype builder for one parameter block.
    Falls back to the whole eligible group if the band is empty. Returns (per-player array mapped
    by position, {F/D: value})."""
    lo, hi = band or REPL_BAND
    v, meta = np.zeros(len(isD)), {}
    for g, gm in (("F", isD < 0.5), ("D", isD > 0.5)):
        b = gm & elig & (pct >= lo) & (pct <= hi)
        if b.sum() == 0:
            b = gm & elig
        mv = _wmean(arr, toi, b)
        v[gm] = mv
        meta[g] = round(mv, 4)
    return v, meta


def pct_within(v, elig, isD):
    """Percentile (0–100) within position group among eligible players; NaN if ineligible."""
    out = np.full(len(v), np.nan)
    for gm in (isD < 0.5, isD > 0.5):
        m = gm & elig
        if m.sum() > 1:
            r = np.argsort(np.argsort(v[m])).astype(float) / (m.sum() - 1)
            out[m] = np.round(r * 100.0)
    return out


# ── WAR engine ───────────────────────────────────────────────────────────────────────────────────


def war_rows(seasons, idx):
    """Per stint-side rows for the WAR accounting, one dict of arrays per (season, bucket): attacker
    index matrix (n,5), defender index matrix + mask, goalie ids, duration, RATE_CTX. MA keeps only the
    PP side. Thin wrapper over the shared `side_rows` builder (one orientation implementation)."""
    out = {}
    for s in seasons:
        for key, strengths, dual in (("ev", EV_STRENGTHS, True), ("ma", MA_STRENGTHS, False)):
            r = side_rows([s], strengths, dual, idx)
            if r is not None:
                out[(s, key)] = r
    return out


def bucket_E(rows, sh, cr, df, gcl, kq, cx, psi0, isD, gsave=None, cq=None, gcl_d=None):
    """E[GF] per side-row under the given parameter arrays — the same equation war_bucket uses
    for its base world (creator-marginalized ḡ per shooter from the row's actual teammates,
    goalie tilt, defenders' K_B). Exposed for the WAR audit's joint (whole-team) counterfactuals:
    pass replacement arrays to price an all-replacement attack. `cq`/`gcl_d` apply the per-player
    creation-quality lift (Σ teammate create_qual on each shooter's mark, linearised)."""
    A, B, dm = rows["atk"], rows["def"], rows["dmask"]
    gs = rows["gsave"] if gsave is None else gsave
    g0, gF, gD = gcl
    e_cr = np.exp(cr[A])
    fmask = (isD < 0.5).astype(np.float64)[A]
    p0 = float(np.exp(psi0))
    S_k = e_cr.sum(1)[:, None] - e_cr
    SF_k = (e_cr * fmask).sum(1)[:, None] - e_cr * fmask
    SD_k = S_k - SF_k
    Z = p0 + S_k
    gbar = (p0 * g0[A] + SF_k * gF[A] + SD_k * gD[A]) / Z
    if cq is not None and gcl_d is not None:  # teammate create_qual lift (linearised)
        g0d, gFd, gDd = gcl_d
        cqA = cq[A]
        delta = cqA.sum(1)[:, None] - cqA
        gbar = gbar + (p0 * g0d[A] + SF_k * gFd[A] + SD_k * gDd[A]) / Z * delta
    tj = np.exp(sh[A] - cr[A]) * gbar * np.exp((1.0 - g0[A]) * gs[:, None])
    lKB = (np.log(np.clip(kq[B], EPS, None)) * dm).sum(1)
    return cx * np.exp(cr[A].sum(1) + (df[B] * dm).sum(1) + lKB) * tj.sum(1) * rows["dur"] / 3600.0


def war_bucket(rows, P, sh, cr, df, gcl, kq, gsave, cx, repl, psi0, isD, cq=None, gcl_d=None):
    """GAR per player for ONE (season, bucket)'s stint-side rows — closed-form replacement swaps.

    Per side-row: E[GF]/h = cx·e^{Σcr_A}·e^{Σdf_B·mask}·K_B·Σ_j e^{sh_j − cr_j}·ḡ_j(row), where
    ḡ_j(row) is the model's own goals-per-shot for shooter j marginalized EXACTLY over the row's
    creator distribution π = softmax([ψ₀, create of j's four actual teammates]) with the three
    creator-class conversions `gcl` = (g0, gF, gD) (each already a Beta-marginal, see
    gbar_classes), tilted log-linearly for the row's goalie (·e^{(1−ḡ)·gsave}); K_B = Π_d κ_d.
    Each attacker slot's swap perturbs the shared e^{Σcr} and his own term. Per-player creation
    QUALITY (`cq` = create_qual) lifts each shooter's mark logit by Σ(cq of his on-ice teammates),
    so a swap ALSO shifts the four teammates' ḡ — a first-order cross term (applied
    linearly via `gcl_d` = d ḡ/d logit-xg per class, plus `repl["g0_d"/"gF_d"/"gD_d"]` for the swapped
    slot's own lift). Held fixed (cq off) this reduces exactly to the position-quality WAR. `repl` =
    per-player replacement arrays (sh, cr, df, g0, gF, gD, kq — each player's value = his position's
    CONTEXT-MATCHED archetype: EV band for EV, PP band for attack slots, PK band for defender slots).
    Returns (gar_atk, gar_def, E): per-player goals vs replacement (attack/defense booked separately:
    EV takes both; MA books attack=PP, defense=PK) and the per-row expected goals (audit raw material)."""
    A, B, dm = rows["atk"], rows["def"], rows["dmask"]
    hrs = rows["dur"] / 3600.0
    gs = rows["gsave"]  # per-row goalie gsave (n,)
    g0, gF, gD = gcl
    crA = cr[A].sum(1)
    dfB = (df[B] * dm).sum(1)
    lKB = (np.log(np.clip(kq[B], EPS, None)) * dm).sum(1)
    # per-shooter creator mix over the row's ACTUAL teammates (exact, model-defined)
    e_cr = np.exp(cr[A])  # (n,5)
    fmask = (isD < 0.5).astype(np.float64)[A]  # (n,5): attacker is a forward
    p0 = float(np.exp(psi0))
    S_k = e_cr.sum(1)[:, None] - e_cr  # (n,5) each shooter's 4 teammates
    SF_k = (e_cr * fmask).sum(1)[:, None] - e_cr * fmask
    SD_k = S_k - SF_k
    Z = p0 + S_k
    gbar0 = (p0 * g0[A] + SF_k * gF[A] + SD_k * gD[A]) / Z  # (n,5) creator-marginal ḡ per shooter
    # each shooter's teammate-Σ(create_qual) lifts his mark; ḡ shifts by ḡ' · δ (linear)
    use_cq = cq is not None and gcl_d is not None
    if use_cq:
        g0d, gFd, gDd = gcl_d
        cqA = cq[A]  # (n,5)
        delta = cqA.sum(1)[:, None] - cqA  # (n,5) each shooter's on-ice-teammate create_qual sum
        gbar_d = (p0 * g0d[A] + SF_k * gFd[A] + SD_k * gDd[A]) / Z  # dḡ/d(logit) per shooter (n,5)
        gbar = gbar0 + gbar_d * delta
        # u_k = sensitivity of shooter k's term to a unit lift in his teammates' cq (for cross term)
        uterm = np.exp(sh[A] - cr[A]) * gbar_d * np.exp((1.0 - g0[A]) * gs[:, None])
        Usum = uterm.sum(1)
    else:
        gbar = gbar0
    tj = np.exp(sh[A] - cr[A]) * gbar * np.exp((1.0 - g0[A]) * gs[:, None])
    T = tj.sum(1)
    E = cx * np.exp(crA + dfB + lKB) * T * hrs  # E[GF] per side-row (goals)
    gar_atk, gar_def = np.zeros(P), np.zeros(P)
    # attacker slots: swap p → repl(pos_p): Σcr shifts by (cr_r − cr_p); own term t_p → t_r; and
    # p's cq change shifts every teammate's ḡ — the Σ_{m≠k} cross term
    for k in range(A.shape[1]):
        p = A[:, k]
        gbar_r = (p0 * repl["g0"][p] + SF_k[:, k] * repl["gF"][p] + SD_k[:, k] * repl["gD"][p]) / Z[
            :, k
        ]
        cross = 0.0
        if use_cq:
            gbar_r_d = (
                p0 * repl["g0_d"][p] + SF_k[:, k] * repl["gF_d"][p] + SD_k[:, k] * repl["gD_d"][p]
            ) / Z[:, k]
            gbar_r = gbar_r + gbar_r_d * delta[:, k]  # swapped shooter keeps his teammates' lift
            dcq = repl["cq"][p] - cq[p]  # p's own cq change → shifts the OTHER shooters' ḡ
            cross = dcq * (Usum - uterm[:, k])
        t_r = np.exp(repl["sh"][p] - repl["cr"][p]) * gbar_r * np.exp((1.0 - repl["g0"][p]) * gs)
        E_swp = (
            cx * np.exp(crA - cr[p] + repl["cr"][p] + dfB + lKB) * (T - tj[:, k] + t_r + cross) * hrs
        )
        np.add.at(gar_atk, p, E - E_swp)
    # defender slots: swap d → repl: Σdf shifts, K_B ratio; sign: reducing opp GF is positive GAR
    for k in range(B.shape[1]):
        m = dm[:, k] > 0
        d_ = B[m, k]
        ratio = np.exp(repl["df"][d_] - df[d_]) * (
            np.clip(repl["kq"][d_], EPS, None) / np.clip(kq[d_], EPS, None)
        )
        np.add.at(gar_def, d_, E[m] * (ratio - 1.0))  # with him, opp scores E; with repl,
    return gar_atk, gar_def, E  # E·ratio — he saves E·(ratio−1)


# ── assembly ─────────────────────────────────────────────────────────────────────────────────────


def build(fit_path=None):
    fit = load_fit(fit_path)
    seasons = [int(s) for s in fit["seasons"]]
    t = player_table(fit)
    n = len(t["id"])
    idx = {int(p): i for i, p in enumerate(t["id"])}
    kap = kappa(fit)

    # current-skill values: recompute from effective params (consistency-checked vs the fit JSON)
    sc, pm, dfv = values_at(
        fit, t, "ev", t["ev_shoot"], t["ev_create"], t["ev_def"], t["fin_eff"], n_def=5
    )
    err = np.nanmax(np.abs(sc - t["ev_scoring"]))
    if err > 5e-3:
        print(
            f"  [note] recomputed ev_scoring differs from fit JSON by up to {err:.4f} — "
            "expected if the fit predates the marginal-conversion fix (cards use the "
            "recomputed values); otherwise check quality_ctx wiring"
        )
    pp_sc, pp_pm, _ = values_at(
        fit, t, "ma", t["pp_shoot"], t["pp_create"], t["pp_def"], t["fin_eff"], n_def=4
    )
    _, _, pk_df = values_at(
        fit, t, "ma", t["pp_shoot"], t["pp_create"], t["pp_def"], t["fin_eff"], n_def=4
    )
    vals = {"sc": sc, "pm": pm, "df": dfv, "pp_sc": pp_sc, "pp_pm": pp_pm, "pk_df": pk_df}
    base = baselines(t, vals, kap)
    ga60 = ga60_of(t, base, sc, pm, dfv, kap)
    bppsc = np.where(t["isD"] > 0, base["D"]["pp_sc"], base["F"]["pp_sc"])
    bpppm = np.where(t["isD"] > 0, base["D"]["pp_pm"], base["F"]["pp_pm"])
    bpkdf = np.where(t["isD"] > 0, base["D"]["pk_df"], base["F"]["pk_df"])
    pp_ga60 = (pp_sc - bppsc) + kap * (pp_pm - bpppm)
    pk_ga60 = kap * (pk_df - bpkdf)

    # replacement archetypes: CONTEXT-MATCHED bands — the "freely available player" for each role.
    # EV: the REPL_BAND GA/60 percentile band among EV regulars; PP: the same band of PP GA/60
    # among PP regulars; PK: the band of PK GA/60 among PK regulars. (Averaging EV-band players'
    # PP parameters would collect the ridge-shrunk ~zeros of players who never play PP — i.e. a
    # league-AVERAGE PP baseline, far too strong; the 2026-07 audit's fix-2 finding.)
    ev_el = t["toi_ev"] >= EV_GATE
    pp_el = t["toi_pp"] >= MA_GATE
    pk_el = t["toi_pk"] >= MA_GATE
    ga_pct = pct_within(ga60, ev_el, t["isD"])
    pp_pct = pct_within(pp_ga60, pp_el, t["isD"])
    pk_pct = pct_within(pk_ga60, pk_el, t["isD"])
    repl_params, repl_meta = {}, {"ev": {}, "pp": {}, "pk": {}}
    for ctx_key, toi, elig, pct, blocks in (
        (
            "ev",
            t["toi_ev"],
            ev_el,
            ga_pct,
            (
                ("sh", t["ev_shoot"]),
                ("cr", t["ev_create"]),
                ("df", t["ev_def"]),
                ("qs", t["qshoot_eff"]),
                ("qd", t["qdef_eff"]),
                ("fin", t["fin_eff"]),
            ),
        ),
        (
            "pp",
            t["toi_pp"],
            pp_el,
            pp_pct,
            (
                ("pp_sh", t["pp_shoot"]),
                ("pp_cr", t["pp_create"]),
                ("pp_qs", t["qshoot_eff"]),
                ("pp_fin", t["fin_eff"]),
            ),
        ),
        ("pk", t["toi_pk"], pk_el, pk_pct, (("pk_df", t["pp_def"]), ("pk_qd", t["qdef_eff"]))),
    ):
        for blk, arr in blocks:
            repl_params[blk], m = band_mean(arr, toi, elig, pct, t["isD"])
            for g, mv in m.items():
                repl_meta[ctx_key].setdefault(g, {})[blk] = mv

    # replacement VALUE levels (per-60, position-mapped): the archetypes' own modeled production.
    # Displayed card values are differenced against THESE — the same zero point WAR uses — so a
    # freely-available player reads 0 and regulars read positive. The position-mean versions above
    # exist only to rank players for the band (ranks are invariant to the baseline shift).
    zero = np.zeros(n)
    ti_r = {"isD": t["isD"], "qshoot_eff": repl_params["qs"], "qdef_eff": repl_params["qd"]}
    r_sc, r_pm, r_df = values_at(
        fit,
        ti_r,
        "ev",
        repl_params["sh"],
        repl_params["cr"],
        repl_params["df"],
        repl_params["fin"],
        n_def=5,
    )
    ti_pp = {"isD": t["isD"], "qshoot_eff": repl_params["pp_qs"], "qdef_eff": repl_params["qd"]}
    r_ppsc, r_pppm, _ = values_at(
        fit,
        ti_pp,
        "ma",
        repl_params["pp_sh"],
        repl_params["pp_cr"],
        zero,
        repl_params["pp_fin"],
        n_def=4,
    )
    ti_pk = {"isD": t["isD"], "qshoot_eff": zero, "qdef_eff": repl_params["pk_qd"]}
    _, _, r_pkdf = values_at(fit, ti_pk, "ma", zero, zero, repl_params["pk_df"], zero, n_def=4)
    rv = {}
    for g, gm in (("F", t["isD"] < 0.5), ("D", t["isD"] > 0.5)):
        i0 = int(np.argmax(gm))
        rv[g] = {
            "sc": float(r_sc[i0]),
            "pm": float(r_pm[i0]),
            "df": float(r_df[i0]),
            "pp_sc": float(r_ppsc[i0]),
            "pp_pm": float(r_pppm[i0]),
            "pk_df": float(r_pkdf[i0]),
        }
    rbase = {g: {"sc": v["sc"], "pm": v["pm"], "df": v["df"]} for g, v in rv.items()}
    created60 = (sc - r_sc) + kap * (pm - r_pm)
    prevented60 = kap * (dfv - r_df)
    ga60 = created60 + prevented60  # rebind: every displayed GA/60 is vs replacement
    pp_ga60 = (pp_sc - r_ppsc) + kap * (pp_pm - r_pppm)
    pk_ga60 = kap * (pk_df - r_pkdf)

    # WAR
    rows = war_rows(seasons, idx)
    gsave_map = {int(g["id"]): g["gsave"] for g in fit["goalies"]}
    gar = {"ev": np.zeros(n), "pp": np.zeros(n), "pk": np.zeros(n)}
    gar_season = {}
    last = int(max(seasons))
    toi_ev_last = np.zeros(n)  # latest-season 5v5 seconds (WAR pct gate)
    for s, key in sorted(rows):
        r = rows[(s, key)]
        rctx = fit["strengths"][key]["rate_ctx"]
        beta = np.array([rctx.get(c, 0.0) for c in ("home", "ozone", "dzone", "trail", "lead")])
        cx = np.exp(
            fit["strengths"][key]["rate_intercept"] + r["ctx"] @ beta + rctx.get(f"season_{s}", 0.0)
        )
        r["gsave"] = np.array(
            [gsave_map.get(int(g), 0.0) if g == g and g is not None else 0.0 for g in r["goalie"]]
        )
        mq = fit["mu_qual"]["ev" if key == "ev" else "ma"]
        zs = t["z_last"] - (t["last_season"] - s) / AGE_SCALE  # age-z in season s
        fin_s = fin_eff_at(fit, t, np.where(np.isnan(t["age"]), 0.0, zs))
        kq = _sigmoid(mq + t["qdef_eff"]) / max(_sigmoid(mq), EPS)
        if key == "ev":  # per-season EV states from the trend
            sh = np.array([tr.get(str(s), {}).get("shoot", 0.0) for tr in t["trend"]])
            cr = np.array([tr.get(str(s), {}).get("create", 0.0) for tr in t["trend"]])
            df_ = np.array([tr.get(str(s), {}).get("def", 0.0) for tr in t["trend"]])
            repl = {"sh": repl_params["sh"], "cr": repl_params["cr"], "df": repl_params["df"]}
            r_qs, r_fin, r_qd = repl_params["qs"], repl_params["fin"], repl_params["qd"]
        else:  # attack = PP archetype, defense = PK
            sh, cr, df_ = t["pp_shoot"], t["pp_create"], t["pp_def"]
            repl = {
                "sh": repl_params["pp_sh"],
                "cr": repl_params["pp_cr"],
                "df": repl_params["pk_df"],
            }
            r_qs, r_fin, r_qd = repl_params["pp_qs"], repl_params["pp_fin"], repl_params["pk_qd"]
        gcl = gbar_classes(fit, key, t["qshoot_eff"], fin_s, season=s)
        repl["g0"], repl["gF"], repl["gD"] = gbar_classes(fit, key, r_qs, r_fin, season=s)
        repl["kq"] = _sigmoid(mq + r_qd) / max(_sigmoid(mq), EPS)
        # per-player on-ice creation quality lifts each shooter's teammate-Σ(create_qual); enabled
        # only when the block is present, so WAR is byte-identical when create_qual is absent
        cq = t["create_qual"] if np.any(t["create_qual"]) else None
        gcl_d = None
        if cq is not None:
            gcl_d = gbar_class_deriv(fit, key, t["qshoot_eff"], fin_s, season=s)
            repl["g0_d"], repl["gF_d"], repl["gD_d"] = gbar_class_deriv(fit, key, r_qs, r_fin, season=s)
            repl["cq"] = np.zeros(n)  # a replacement player adds no creation quality (neutral)
        if key == "ev" and int(s) == last:
            np.add.at(toi_ev_last, r["atk"], np.broadcast_to(r["dur"][:, None], r["atk"].shape))
        ga, gd, _ = war_bucket(
            r,
            n,
            sh,
            cr,
            df_,
            gcl,
            kq,
            r["gsave"],
            cx,
            repl,
            fit["strengths"][key]["psi0"],
            t["isD"],
            cq=cq,
            gcl_d=gcl_d,
        )
        if key == "ev":
            gar["ev"] += ga + gd
        else:  # attacker slots = PP, defender = PK
            gar["pp"] += ga
            gar["pk"] += gd
        gs_ = gar_season.setdefault(
            int(s), {c: np.zeros(n) for c in ("ev_atk", "ev_def", "pp", "pk")}
        )
        ca, cd = ("ev_atk", "ev_def") if key == "ev" else ("pp", "pk")
        gs_[ca] += ga
        gs_[cd] += gd

    def _season_goals(s):
        gs_ = gar_season.get(s)
        return sum(gs_.values()) if gs_ else np.zeros(n)

    war_total = (gar["ev"] + gar["pp"] + gar["pk"]) / GOALS_PER_WIN
    war_latest = _season_goals(last) / GOALS_PER_WIN

    # card drill-down equation factors — each skill value split into the product/difference the
    # card modal displays (all exact: the value equals the displayed operation on the factors)
    env_ev = float((fit.get("value_env") or {}).get("ev") or 1.0)
    mu_ev_r = fit["strengths"]["ev"]["rate_intercept"]
    sc_shots60 = env_ev * np.exp(mu_ev_r + t["ev_shoot"])  # Scoring = shots60 × ḡ_mix
    sc_gps = sc / np.maximum(sc_shots60, EPS)
    qc_pos_arr = np.where(t["isD"] > 0, fit["qcreate"]["D"], fit["qcreate"]["F"])
    pm_q = _sigmoid(fit["mu_qual"]["ev"] + qc_pos_arr)  # Playmaking = extra shots × quality
    pm_extra = pm / np.maximum(pm_q, EPS)
    df_typ60 = env_ev * 5.0 * np.exp(mu_ev_r) * _sigmoid(fit["mu_qual"]["ev"])
    df_allowed60 = df_typ60 - dfv  # Defense = typical − with him

    # attribute display transforms. Displayed skill INCLUDES the player's age and position — the
    # age curves shape shrinkage, context, and projection, never a display normalizer. The ±
    # comparisons are against the TOI-weighted POSITION AVERAGE among regulars (no age matching).
    mq_ev, a_ev, b_ev = fit["mu_qual"]["ev"], fit["conv"]["a"]["ev"], fit["conv"]["b"]["ev"]
    qbar = float(np.clip(_sigmoid(mq_ev), EPS, 1 - EPS))  # league-average 5v5 shot quality
    lb = a_ev * float(_logit(np.array([qbar]))[0]) + b_ev

    def _posmean(arr):
        pv = np.zeros(n)
        for gm in (t["isD"] < 0.5, t["isD"] > 0.5):
            pv[gm] = _wmean(arr, t["toi_ev"], ev_el & gm)
        return pv

    p_his = _sigmoid(lb + t["fin_eff"])  # his conversion at the average shot
    p_pos = _posmean(p_his)
    fin100 = (p_his - p_pos) * 100.0  # goals/100 above the position average
    fin100_se = p_his * (1 - p_his) * t["fin_se"] * 100.0
    shooting_pct_vol = (sc_shots60 / np.maximum(_posmean(sc_shots60), EPS) - 1.0) * 100.0
    df_posmean = _posmean(dfv)
    defense_disp = dfv - df_posmean  # xG erased above the position average
    pm_se = pm * 0.0
    ok = np.abs(np.exp(t["ev_create"]) - 1.0) > 1e-6
    pm_se[ok] = (
        np.abs(pm[ok] * np.exp(t["ev_create"][ok]) / (np.exp(t["ev_create"][ok]) - 1.0))
        * t["ev_create_se"][ok]
    )

    # projection values
    projJ = {int(p["id"]): p for p in fit.get("projection", {}).get("players", [])}
    proj_season = fit.get("projection", {}).get("season")

    pp_el = t["toi_pp"] >= MA_GATE
    pk_el = t["toi_pk"] >= MA_GATE
    war_el = ev_el & (toi_ev_last >= 12000.0)  # WAR is a counting stat: rank it among
    pcts = {
        "war": pct_within(war_latest, war_el, t["isD"]),  # that season's REGULARS (≥200 EV min)
        "ga60": pct_within(ga60, ev_el, t["isD"]),
        "created60": pct_within(created60, ev_el, t["isD"]),
        "prevented60": pct_within(prevented60, ev_el, t["isD"]),
        "pp_ga60": pct_within(pp_ga60, pp_el, t["isD"]),
        "pk_ga60": pct_within(pk_ga60, pk_el, t["isD"]),
        "scoring": pct_within(sc, ev_el, t["isD"]),
        "shooting": pct_within(shooting_pct_vol, ev_el, t["isD"]),
        "finishing": pct_within(fin100, ev_el, t["isD"]),
        "playmaking": pct_within(pm, ev_el, t["isD"]),
        "defense": pct_within(defense_disp, ev_el, t["isD"]),
    }

    def _num(x, nd=3):
        return (
            None
            if x is None or (isinstance(x, float) and not np.isfinite(x))
            else round(float(x), nd)
        )

    def league_at_age(z, d):
        """Values of a POSITION-AVERAGE player at age-z (all residuals 0; position offsets + aging
        curves applied) — the trajectory chart's league-reference underlay."""
        ac = fit["age_curves"]

        def blk(name):
            c = ac[name]
            return _curve_val4(c["coef"], z, d) + d * c["d_offset"]

        ti = {
            "isD": np.array([d]),
            "qshoot_eff": np.array([d * fit["quality_ctx"].get("shooter_D", 0.0)]),
            "qdef_eff": np.array([d * fit["quality_ctx"].get("def_D", 0.0)]),
        }
        fe = np.array([blk("fin")])
        lsc, lpm, ldf = values_at(
            fit,
            ti,
            "ev",
            np.array([blk("ev_shoot")]),
            np.array([blk("ev_create")]),
            np.array([blk("ev_def")]),
            fe,
            n_def=5,
        )
        return float(lsc[0]), float(lpm[0]), float(ldf[0])

    up_last, up_window, up_detail = unpriced_goals(seasons, idx, last)

    players_out = {}
    for i in range(n):
        tr_out = []
        finc = fit["age_curves"]["fin"]
        for s_str, st in sorted(t["trend"][i].items()):
            s = int(s_str)
            zs = (
                t["z_last"][i] - (t["last_season"][i] - s) / AGE_SCALE
                if np.isfinite(t["age"][i])
                else 0.0
            )
            fe = (
                t["fin"][i]
                + _curve_val4(finc["coef"], zs, t["isD"][i])
                + t["isD"][i] * finc["d_offset"]
            )
            ti = {k: np.array([t[k][i]]) for k in ("isD", "qshoot_eff", "qdef_eff")}
            s_sc, s_pm, s_df = values_at(
                fit,
                ti,
                "ev",
                np.array([st["shoot"]]),
                np.array([st["create"]]),
                np.array([st["def"]]),
                np.array([fe]),
                n_def=5,
            )
            g60 = float(ga60_of({"isD": np.array([t["isD"][i]])}, rbase, s_sc, s_pm, s_df, kap)[0])
            age_s = t["age"][i] - (t["last_season"][i] - s) if np.isfinite(t["age"][i]) else None
            lsc, lpm, ldf = league_at_age(zs, t["isD"][i])
            lg60 = float(
                ga60_of(
                    {"isD": np.array([t["isD"][i]])},
                    rbase,
                    np.array([lsc]),
                    np.array([lpm]),
                    np.array([ldf]),
                    kap,
                )[0]
            )
            tr_out.append(
                {
                    "season": s,
                    "age": _num(age_s, 1),
                    "scoring": _num(s_sc[0]),
                    "playmaking": _num(s_pm[0]),
                    "defense": _num(s_df[0] - df_posmean[i]),
                    "ga60": _num(g60),
                    "lg_scoring": _num(lsc),
                    "lg_playmaking": _num(lpm),
                    "lg_defense": _num(ldf - df_posmean[i]),
                    "lg_ga60": _num(lg60),
                }
            )
        proj = None
        pj = projJ.get(int(t["id"][i]))
        if pj:
            pj_ga = ga60_of(
                {"isD": np.array([t["isD"][i]])},
                rbase,
                np.array([pj["ev_scoring"]]),
                np.array([pj["ev_playmaking"]]),
                np.array([pj["ev_defense"]]),
                kap,
            )[0]
            zt = (
                t["z_last"][i] + (proj_season - t["last_season"][i]) / AGE_SCALE
                if np.isfinite(t["age"][i]) and proj_season
                else 0.0
            )
            plsc, plpm, pldf = league_at_age(zt, t["isD"][i])
            plg60 = float(
                ga60_of(
                    {"isD": np.array([t["isD"][i]])},
                    rbase,
                    np.array([plsc]),
                    np.array([plpm]),
                    np.array([pldf]),
                    kap,
                )[0]
            )
            proj = {
                "season": proj_season,
                "scoring": _num(pj["ev_scoring"]),
                "playmaking": _num(pj["ev_playmaking"]),
                "defense": _num(pj["ev_defense"] - df_posmean[i]),
                "ga60": _num(pj_ga),
                "lg_scoring": _num(plsc),
                "lg_playmaking": _num(plpm),
                "lg_defense": _num(pldf - df_posmean[i]),
                "lg_ga60": _num(plg60),
            }
        A = {
            "war": {"v": _num(war_latest[i], 2), "pct": _num(pcts["war"][i], 0)},
            "ga60": {"v": _num(ga60[i]), "pct": _num(pcts["ga60"][i], 0)},
            "created60": {"v": _num(created60[i]), "pct": _num(pcts["created60"][i], 0)},
            "prevented60": {"v": _num(prevented60[i]), "pct": _num(pcts["prevented60"][i], 0)},
            "pp_ga60": {"v": _num(pp_ga60[i]), "pct": _num(pcts["pp_ga60"][i], 0)},
            "pk_ga60": {"v": _num(pk_ga60[i]), "pct": _num(pcts["pk_ga60"][i], 0)},
            "scoring": {"v": _num(sc[i]), "pct": _num(pcts["scoring"][i], 0)},
            "shooting": {"v": _num(shooting_pct_vol[i], 1), "pct": _num(pcts["shooting"][i], 0)},
            "finishing": {
                "v": _num(fin100[i], 2),
                "se": _num(fin100_se[i], 2),
                "pct": _num(pcts["finishing"][i], 0),
            },
            "playmaking": {
                "v": _num(pm[i]),
                "se": _num(pm_se[i]),
                "pct": _num(pcts["playmaking"][i], 0),
            },
            "defense": {"v": _num(defense_disp[i]), "pct": _num(pcts["defense"][i], 0)},
        }
        gs_last = gar_season.get(last)
        players_out[int(t["id"][i])] = {
            "pos": "D" if t["isD"][i] > 0 else "F",
            "age": _num(t["age"][i], 1),
            "last_season": int(t["last_season"][i]),
            "attrs": A,
            # components: the player's own modeled per-60 values (before the replacement baseline
            # is subtracted) — the card drill-downs rebuild the deltas from these + meta
            "components": {
                "pp_scoring": _num(pp_sc[i]),
                "pp_playmaking": _num(pp_pm[i]),
                "pk_defense": _num(pk_df[i]),
                "shots60": _num(sc_shots60[i], 1),
                "goals_per_shot": _num(sc_gps[i], 4),
                "pm_shots60": _num(pm_extra[i], 2),
                "pm_xg_per_shot": _num(pm_q[i], 4),
                "def_allowed60": _num(df_allowed60[i]),
                "def_erased60": _num(dfv[i]),
            },
            "war": {
                "latest": _num(war_latest[i], 2),
                "total": _num(war_total[i], 2),
                "by_season": {
                    s: _num(sum(g.values())[i] / GOALS_PER_WIN, 2) for s, g in gar_season.items()
                },
                "ev": _num(gar["ev"][i] / GOALS_PER_WIN, 2),
                "pp": _num(gar["pp"][i] / GOALS_PER_WIN, 2),
                "pk": _num(gar["pk"][i] / GOALS_PER_WIN, 2),
                # latest-season goals above replacement by component (WAR drill-down equation)
                **(
                    {"goals": {c: _num(gs_last[c][i], 1) for c in ("ev_atk", "ev_def", "pp", "pk")}}
                    if gs_last
                    else {}
                ),
            },
            "trajectory": tr_out,
            "projection": proj,
            **(
                {
                    "unpriced_goals": {
                        "latest": int(up_last[i]),
                        "window": int(up_window[i]),
                        "detail": up_detail.get(i, {}),
                    }
                }
                if up_window[i] > 0
                else {}
            ),
        }

    out = {
        "meta": {
            "fit": fit["_path"],
            "seasons": seasons,
            "generated": str(date.today()),
            "kappa": round(kap, 4),
            "goals_per_win": GOALS_PER_WIN,
            # position-average conversion per 100 at the average 5v5 shot (Finishing eq.)
            "fin_avg_p100": {
                g: round(float(p_pos[int(np.argmax(gm))]) * 100.0, 2)
                for g, gm in (("F", t["isD"] < 0.5), ("D", t["isD"] > 0.5))
            },
            # typical 5-defender share of on-ice xGA/60 (Defense drill-down equation)
            "lg_def_xga60": round(float(df_typ60), 3),
            "replacement": repl_meta,
            "repl_band_pct": list(REPL_BAND),
            # per-60 modeled production of the replacement archetypes — the zero point of
            # every card value (the drill-downs difference against these)
            "replacement_values": {
                g: {k: round(v, 4) for k, v in b.items()} for g, b in rv.items()
            },
            "baselines": {g: {k: round(v, 4) for k, v in b.items()} for g, b in base.items()},
            "age_curves": fit["age_curves"],
            "rw_sd": fit.get("rw_sd"),
            "latest_season": last,
        },
        "players": players_out,
    }
    path = C.MODELS / "gen_cards.json"
    path.write_text(json.dumps(out))
    league_war = float(war_latest[ev_el].sum())
    print(
        f"  gen_cards: {n} players  κ={kap:.3f}  league WAR({last}, eligible)={league_war:.0f}"
        f"  -> {path}"
    )
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Export generative player cards (GA/60 + WAR)")
    p.add_argument("--fit", type=str, default=None, help="path to a generative_model_*.json")
    args = p.parse_args(argv)
    build(args.fit)


if __name__ == "__main__":
    main()
