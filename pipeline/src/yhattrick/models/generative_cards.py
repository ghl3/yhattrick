r"""Player-card exporter for the generative model (Cards v2): GA/60 + WAR + trajectories.

Consumes the saved pooled fit (data/models/generative_model_<seasons>.json — must carry
`quality_ctx` and `strengths.*.rate_ctx`, exported since July 2026) plus the stints parquet, and
writes data/models/gen_cards.json for export_players.py to merge into the site JSONs (that side is
plain-JSON — JAX stays in this experimental group).

WHAT IT COMPUTES (per player)
  current-skill attribute values — the fit's effective values (last drift state + position offset
    + aging curve at his latest-season age), re-expressed in card units;
  GA/60 (baseline team) — evaluate the model's closed-form production equations at the player's
    effective params and subtract the TOI-weighted position mean, so a league-average player at
    his position is exactly 0:
        GA/60 = [sc − sc̄_pos] + κ·[pm − p̄m_pos] + κ·[df − d̄f_pos]
    with sc/pm/df from the player_values formulas (docs/generative_model.md §2) and κ = league
    goals-per-xG from the conversion fit (≈1 by Σp=Σgoals; applied explicitly). PP/PK analogues
    from the MA bucket.
  trajectory — per-season values: that season's drift states + that season's age through the same
    formulas (baseline held at the current position means so seasons are comparable);
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

from .. import config as C
from .generative_model import (_load_stints, _sigmoid, _zone, creator_mix, marginal_goal_prob,
                               EV_STRENGTHS, MA_STRENGTHS, AGE_PEAK, AGE_SCALE)

EV_GATE = 6000.0            # seconds (100 min) — EV card eligibility, matches export_players
MA_GATE = 2400.0            # seconds (40 min) — PP/PK card eligibility
GOALS_PER_WIN = 6.0         # goals → wins conversion (v1 constant)
REPL_BAND = (8.0, 12.0)     # GA/60 percentile band defining the replacement archetype
EPS = 1e-9


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _curve_val4(coef, z, d):
    """Position-split quadratic curve value from an age_curves coef dict {zF,z2F,zD,z2D}."""
    return ((1 - d) * (coef["zF"] * z + coef["z2F"] * z * z)
            + d * (coef["zD"] * z + coef["z2D"] * z * z))


def load_fit(path=None):
    """Load the pooled fit JSON (latest multi-season file by default) and index its players."""
    if path is None:
        cands = sorted(C.MODELS.glob("generative_model_*+*.json"), key=lambda p: p.stat().st_mtime)
        if not cands:
            raise SystemExit("no pooled generative_model_*.json — run the model with --pool first")
        path = cands[-1]
    fit = json.loads(path.read_text())
    if "quality_ctx" not in fit or "rate_ctx" not in fit["strengths"]["ev"]:
        raise SystemExit(f"{path.name} lacks quality_ctx/rate_ctx — rerun the pooled fit "
                         "(exports added July 2026)")
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
    for k in ("ev_shoot", "ev_create", "ev_def", "pp_shoot", "pp_create", "pp_def",
              "ev_scoring", "ev_playmaking", "ev_defense", "pp_scoring", "pp_playmaking",
              "pk_defense", "ev_create_se", "qshoot", "qdef", "fin", "fin_se"):
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
    a, b = fit["conv"]["a"]["ev" if key == "ev" else "ma"], fit["conv"]["b"]["ev" if key == "ev" else "ma"]
    s = float(fit["beta_s"])
    out = []
    for qc in (0.0, fit["qcreate"]["F"], fit["qcreate"]["D"]):
        qb = np.clip(_sigmoid(mq + qshoot_eff + qc), EPS, 1 - EPS)
        out.append(marginal_goal_prob(qb, s, a, b, fin_eff))
    return out


def values_at(fit, t, key, sh, cr, df, fin_eff, n_def):
    """(scoring, playmaking, defense) per 60 at the given effective rate params, using the fit's
    intercepts for strength bucket `key` and the pooled quality/conversion effectives. Scoring's
    goals-per-shot is the creator-class + Beta marginal (mirrors player_values in the model)."""
    mu = fit["strengths"][key]["rate_intercept"]
    mq = fit["mu_qual"]["ev" if key == "ev" else "ma"]
    qc_pos = np.where(t["isD"] > 0, fit["qcreate"]["D"], fit["qcreate"]["F"])
    shots = np.exp(mu + sh)
    g0, gF, gD = gbar_classes(fit, key, t["qshoot_eff"], fin_eff)
    w0, wf, wd = creator_mix(fit["strengths"][key]["psi0"], t["isD"], key)
    sc = shots * (w0 * g0 + wf * gF + wd * gD)
    pm = 4.0 * np.exp(mu) * (np.exp(cr) - 1.0) * _sigmoid(mq + qc_pos)
    base = np.exp(mu) * _sigmoid(mq)
    dfv = n_def * (base - np.exp(mu + df) * _sigmoid(mq + t["qdef_eff"]))
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
        out[g] = {"sc": _wmean(vals["sc"], t["toi_ev"], ev_el),
                  "pm": _wmean(vals["pm"], t["toi_ev"], ev_el),
                  "df": _wmean(vals["df"], t["toi_ev"], ev_el),
                  "pp_sc": _wmean(vals["pp_sc"], t["toi_pp"], pp_el),
                  "pp_pm": _wmean(vals["pp_pm"], t["toi_pp"], pp_el),
                  "pk_df": _wmean(vals["pk_df"], t["toi_pk"], pk_el)}
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
    """Per stint-side rows for the WAR accounting, one dict of arrays per (season, bucket):
    attacker index matrix (n,5), defender index matrix + mask, goalie ids, duration, context
    columns [home, ozone, dzone, trail, lead]. MA keeps only the PP side."""
    out = {}
    for s in seasons:
        for key, strengths, dual in (("ev", EV_STRENGTHS, True), ("ma", MA_STRENGTHS, False)):
            df = _load_stints([s], strengths)
            if df.empty:
                continue
            atk, dfd, dmask, gid, dur, ctx = [], [], [], [], [], []

            def emit(A, B, home, st, goalie):
                if any(q not in idx for q in (*A, *B)):
                    return
                nb = len(B)
                dfd.append([idx[q] for q in B] + [0] * (5 - nb))
                dmask.append([1.0] * nb + [0.0] * (5 - nb))
                atk.append([idx[q] for q in A])
                oz, dz = _zone(home, st.start_type, st.start_zone)
                lead = st.home_lead if home else -st.home_lead
                ctx.append([1.0 if home else 0.0, oz, dz,
                            1.0 if lead < 0 else 0.0, 1.0 if lead > 0 else 0.0])
                gid.append(goalie); dur.append(st.duration_s)

            for st in df.itertuples():
                hn, an = len(st.home_skaters), len(st.away_skaters)
                if dual:
                    if hn == 5 and an == 5:
                        emit(st.home_skaters, st.away_skaters, True, st, st.away_goalie)
                        emit(st.away_skaters, st.home_skaters, False, st, st.home_goalie)
                elif hn != an and max(hn, an) == 5:
                    home = hn > an
                    A, B = (st.home_skaters, st.away_skaters) if home else (st.away_skaters, st.home_skaters)
                    emit(A, B, home, st, st.away_goalie if home else st.home_goalie)
            if atk:
                out[(s, key)] = {"atk": np.asarray(atk, dtype=np.int64),
                                 "def": np.asarray(dfd, dtype=np.int64),
                                 "dmask": np.asarray(dmask), "goalie": np.asarray(gid, dtype=object),
                                 "dur": np.asarray(dur, dtype=np.float64),
                                 "ctx": np.asarray(ctx, dtype=np.float64)}
    return out


def bucket_E(rows, sh, cr, df, gcl, kq, cx, psi0, isD, gsave=None):
    """E[GF] per side-row under the given parameter arrays — the same equation war_bucket uses
    for its base world (creator-marginalized ḡ per shooter from the row's actual teammates,
    goalie tilt, defenders' K_B). Exposed for the WAR audit's joint (whole-team) counterfactuals:
    pass replacement arrays to price an all-replacement attack."""
    A, B, dm = rows["atk"], rows["def"], rows["dmask"]
    gs = rows["gsave"] if gsave is None else gsave
    g0, gF, gD = gcl
    e_cr = np.exp(cr[A])
    fmask = (isD < 0.5).astype(np.float64)[A]
    p0 = float(np.exp(psi0))
    S_k = e_cr.sum(1)[:, None] - e_cr
    SF_k = (e_cr * fmask).sum(1)[:, None] - e_cr * fmask
    SD_k = S_k - SF_k
    gbar = (p0 * g0[A] + SF_k * gF[A] + SD_k * gD[A]) / (p0 + S_k)
    tj = np.exp(sh[A] - cr[A]) * gbar * np.exp((1.0 - g0[A]) * gs[:, None])
    lKB = (np.log(np.clip(kq[B], EPS, None)) * dm).sum(1)
    return (cx * np.exp(cr[A].sum(1) + (df[B] * dm).sum(1) + lKB)
            * tj.sum(1) * rows["dur"] / 3600.0)


def war_bucket(rows, P, sh, cr, df, gcl, kq, gsave, cx, repl, psi0, isD):
    """GAR per player for ONE (season, bucket)'s stint-side rows — closed-form replacement swaps.

    Per side-row: E[GF]/h = cx·e^{Σcr_A}·e^{Σdf_B·mask}·K_B·Σ_j e^{sh_j − cr_j}·ḡ_j(row), where
    ḡ_j(row) is the model's own goals-per-shot for shooter j marginalized EXACTLY over the row's
    creator distribution π = softmax([ψ₀, create of j's four actual teammates]) with the three
    creator-class conversions `gcl` = (g0, gF, gD) (each already a Beta-marginal, see
    gbar_classes), tilted log-linearly for the row's goalie (·e^{(1−ḡ)·gsave}); K_B = Π_d κ_d.
    Each attacker slot's swap perturbs the shared e^{Σcr} and his own term (his teammates' creator
    mixes are held fixed — a second-order cross term); each defender slot's swap perturbs e^{Σdf}
    and K_B. `repl` = per-player replacement arrays (sh, cr, df, g0, gF, gD, kq — each player's
    value = his position's CONTEXT-MATCHED archetype: EV band for EV, PP band for attack slots,
    PK band for defender slots). Returns (gar_atk, gar_def, E): per-player goals vs replacement
    (attack/defense booked separately: EV takes both; MA books attack=PP, defense=PK) and the
    per-row expected goals (the audit's raw material)."""
    A, B, dm = rows["atk"], rows["def"], rows["dmask"]
    hrs = rows["dur"] / 3600.0
    gs = rows["gsave"]                                       # per-row goalie gsave (n,)
    g0, gF, gD = gcl
    crA = cr[A].sum(1)
    dfB = (df[B] * dm).sum(1)
    lKB = (np.log(np.clip(kq[B], EPS, None)) * dm).sum(1)
    # per-shooter creator mix over the row's ACTUAL teammates (exact, model-defined)
    e_cr = np.exp(cr[A])                                     # (n,5)
    fmask = (isD < 0.5).astype(np.float64)[A]                # (n,5): attacker is a forward
    p0 = float(np.exp(psi0))
    S_k = e_cr.sum(1)[:, None] - e_cr                        # (n,5) each shooter's 4 teammates
    SF_k = (e_cr * fmask).sum(1)[:, None] - e_cr * fmask
    SD_k = S_k - SF_k
    Z = p0 + S_k
    gbar = (p0 * g0[A] + SF_k * gF[A] + SD_k * gD[A]) / Z    # (n,5) creator-marginal ḡ per shooter
    tj = np.exp(sh[A] - cr[A]) * gbar * np.exp((1.0 - g0[A]) * gs[:, None])
    T = tj.sum(1)
    E = cx * np.exp(crA + dfB + lKB) * T * hrs               # E[GF] per side-row (goals)
    gar_atk, gar_def = np.zeros(P), np.zeros(P)
    # attacker slots: swap p → repl(pos_p): Σcr shifts by (cr_r − cr_p); own term t_p → t_r
    for k in range(A.shape[1]):
        p = A[:, k]
        gbar_r = (p0 * repl["g0"][p] + SF_k[:, k] * repl["gF"][p]
                  + SD_k[:, k] * repl["gD"][p]) / Z[:, k]
        t_r = (np.exp(repl["sh"][p] - repl["cr"][p]) * gbar_r
               * np.exp((1.0 - repl["g0"][p]) * gs))
        E_swp = cx * np.exp(crA - cr[p] + repl["cr"][p] + dfB + lKB) * (T - tj[:, k] + t_r) * hrs
        np.add.at(gar_atk, p, E - E_swp)
    # defender slots: swap d → repl: Σdf shifts, K_B ratio; sign: reducing opp GF is positive GAR
    for k in range(B.shape[1]):
        m = dm[:, k] > 0
        d_ = B[m, k]
        ratio = np.exp(repl["df"][d_] - df[d_]) * (np.clip(repl["kq"][d_], EPS, None)
                                                   / np.clip(kq[d_], EPS, None))
        np.add.at(gar_def, d_, E[m] * (ratio - 1.0))         # with him, opp scores E; with repl,
    return gar_atk, gar_def, E                               # E·ratio — he saves E·(ratio−1)


# ── assembly ─────────────────────────────────────────────────────────────────────────────────────

def build(fit_path=None):
    fit = load_fit(fit_path)
    seasons = [int(s) for s in fit["seasons"]]
    t = player_table(fit)
    n = len(t["id"])
    idx = {int(p): i for i, p in enumerate(t["id"])}
    kap = kappa(fit)

    # current-skill values: recompute from effective params (consistency-checked vs the fit JSON)
    sc, pm, dfv = values_at(fit, t, "ev", t["ev_shoot"], t["ev_create"], t["ev_def"],
                            t["fin_eff"], n_def=5)
    err = np.nanmax(np.abs(sc - t["ev_scoring"]))
    if err > 5e-3:
        print(f"  [note] recomputed ev_scoring differs from fit JSON by up to {err:.4f} — "
              "expected if the fit predates the marginal-conversion fix (cards use the "
              "recomputed values); otherwise check quality_ctx wiring")
    pp_sc, pp_pm, _ = values_at(fit, t, "ma", t["pp_shoot"], t["pp_create"], t["pp_def"],
                                t["fin_eff"], n_def=4)
    _, _, pk_df = values_at(fit, t, "ma", t["pp_shoot"], t["pp_create"], t["pp_def"],
                            t["fin_eff"], n_def=4)
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
            ("ev", t["toi_ev"], ev_el, ga_pct,
             (("sh", t["ev_shoot"]), ("cr", t["ev_create"]), ("df", t["ev_def"]),
              ("qs", t["qshoot_eff"]), ("qd", t["qdef_eff"]), ("fin", t["fin_eff"]))),
            ("pp", t["toi_pp"], pp_el, pp_pct,
             (("pp_sh", t["pp_shoot"]), ("pp_cr", t["pp_create"]),
              ("pp_qs", t["qshoot_eff"]), ("pp_fin", t["fin_eff"]))),
            ("pk", t["toi_pk"], pk_el, pk_pct,
             (("pk_df", t["pp_def"]), ("pk_qd", t["qdef_eff"])))):
        for blk, arr in blocks:
            repl_params[blk], m = band_mean(arr, toi, elig, pct, t["isD"])
            for g, mv in m.items():
                repl_meta[ctx_key].setdefault(g, {})[blk] = mv

    # WAR
    rows = war_rows(seasons, idx)
    gsave_map = {int(g["id"]): g["gsave"] for g in fit["goalies"]}
    gar = {"ev": np.zeros(n), "pp": np.zeros(n), "pk": np.zeros(n)}
    gar_season = {}
    for (s, key) in sorted(rows):
        r = rows[(s, key)]
        rctx = fit["strengths"][key]["rate_ctx"]
        beta = np.array([rctx.get(c, 0.0) for c in ("home", "ozone", "dzone", "trail", "lead")])
        cx = np.exp(fit["strengths"][key]["rate_intercept"] + r["ctx"] @ beta
                    + rctx.get(f"season_{s}", 0.0))
        r["gsave"] = np.array([gsave_map.get(int(g), 0.0) if g == g and g is not None else 0.0
                               for g in r["goalie"]])
        mq = fit["mu_qual"]["ev" if key == "ev" else "ma"]
        zs = t["z_last"] - (t["last_season"] - s) / AGE_SCALE            # age-z in season s
        fin_s = fin_eff_at(fit, t, np.where(np.isnan(t["age"]), 0.0, zs))
        kq = _sigmoid(mq + t["qdef_eff"]) / max(_sigmoid(mq), EPS)
        if key == "ev":                                     # per-season EV states from the trend
            sh = np.array([tr.get(str(s), {}).get("shoot", 0.0) for tr in t["trend"]])
            cr = np.array([tr.get(str(s), {}).get("create", 0.0) for tr in t["trend"]])
            df_ = np.array([tr.get(str(s), {}).get("def", 0.0) for tr in t["trend"]])
            repl = {"sh": repl_params["sh"], "cr": repl_params["cr"], "df": repl_params["df"]}
            r_qs, r_fin, r_qd = repl_params["qs"], repl_params["fin"], repl_params["qd"]
        else:                                               # attack = PP archetype, defense = PK
            sh, cr, df_ = t["pp_shoot"], t["pp_create"], t["pp_def"]
            repl = {"sh": repl_params["pp_sh"], "cr": repl_params["pp_cr"],
                    "df": repl_params["pk_df"]}
            r_qs, r_fin, r_qd = repl_params["pp_qs"], repl_params["pp_fin"], repl_params["pk_qd"]
        gcl = gbar_classes(fit, key, t["qshoot_eff"], fin_s, season=s)
        repl["g0"], repl["gF"], repl["gD"] = gbar_classes(fit, key, r_qs, r_fin, season=s)
        repl["kq"] = _sigmoid(mq + r_qd) / max(_sigmoid(mq), EPS)
        ga, gd, _ = war_bucket(r, n, sh, cr, df_, gcl, kq, r["gsave"], cx, repl,
                               fit["strengths"][key]["psi0"], t["isD"])
        if key == "ev":
            gar["ev"] += ga + gd
        else:                                               # attacker slots = PP, defender = PK
            gar["pp"] += ga
            gar["pk"] += gd
        gar_season.setdefault(int(s), np.zeros(n))
        gar_season[int(s)] += ga + gd

    war_total = (gar["ev"] + gar["pp"] + gar["pk"]) / GOALS_PER_WIN
    last = int(max(seasons))
    war_latest = gar_season.get(last, np.zeros(n)) / GOALS_PER_WIN

    # attribute display transforms
    mq_ev, a_ev, b_ev = fit["mu_qual"]["ev"], fit["conv"]["a"]["ev"], fit["conv"]["b"]["ev"]
    qbar = float(np.clip(_sigmoid(mq_ev), EPS, 1 - EPS))     # league-average 5v5 shot quality
    lb = a_ev * float(_logit(np.array([qbar]))[0]) + b_ev
    p0 = _sigmoid(lb)
    fin100 = (_sigmoid(lb + t["fin"]) - p0) * 100.0          # goals per 100 shots above baseline
    fin100_se = p0 * (1 - p0) * t["fin_se"] * 100.0
    shooting_pct_vol = (np.exp(t["ev_shoot"]) - 1.0) * 100.0  # % shots vs positional baseline
    pm_se = pm * 0.0
    ok = np.abs(np.exp(t["ev_create"]) - 1.0) > 1e-6
    pm_se[ok] = np.abs(pm[ok] * np.exp(t["ev_create"][ok])
                       / (np.exp(t["ev_create"][ok]) - 1.0)) * t["ev_create_se"][ok]

    # projection values
    projJ = {int(p["id"]): p for p in fit.get("projection", {}).get("players", [])}
    proj_season = fit.get("projection", {}).get("season")

    pp_el = t["toi_pp"] >= MA_GATE
    pk_el = t["toi_pk"] >= MA_GATE
    pcts = {"war": pct_within(war_latest, ev_el, t["isD"]),
            "ga60": pct_within(ga60, ev_el, t["isD"]),
            "pp_ga60": pct_within(pp_ga60, pp_el, t["isD"]),
            "pk_ga60": pct_within(pk_ga60, pk_el, t["isD"]),
            "scoring": pct_within(sc, ev_el, t["isD"]),
            "shooting": pct_within(shooting_pct_vol, ev_el, t["isD"]),
            "finishing": pct_within(fin100, ev_el, t["isD"]),
            "playmaking": pct_within(pm, ev_el, t["isD"]),
            "defense": pct_within(dfv, ev_el, t["isD"])}

    def _num(x, nd=3):
        return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), nd)

    def league_at_age(z, d):
        """Values of a POSITION-AVERAGE player at age-z (all residuals 0; position offsets + aging
        curves applied) — the trajectory chart's league-reference underlay."""
        ac = fit["age_curves"]
        def blk(name):
            c = ac[name]
            return _curve_val4(c["coef"], z, d) + d * c["d_offset"]
        ti = {"isD": np.array([d]),
              "qshoot_eff": np.array([d * fit["quality_ctx"].get("shooter_D", 0.0)]),
              "qdef_eff": np.array([d * fit["quality_ctx"].get("def_D", 0.0)])}
        fe = np.array([blk("fin")])
        lsc, lpm, ldf = values_at(fit, ti, "ev", np.array([blk("ev_shoot")]),
                                  np.array([blk("ev_create")]), np.array([blk("ev_def")]),
                                  fe, n_def=5)
        return float(lsc[0]), float(lpm[0]), float(ldf[0])

    players_out = {}
    for i in range(n):
        tr_out = []
        finc = fit["age_curves"]["fin"]
        for s_str, st in sorted(t["trend"][i].items()):
            s = int(s_str)
            zs = (t["z_last"][i] - (t["last_season"][i] - s) / AGE_SCALE
                  if np.isfinite(t["age"][i]) else 0.0)
            fe = (t["fin"][i] + _curve_val4(finc["coef"], zs, t["isD"][i])
                  + t["isD"][i] * finc["d_offset"])
            ti = {k: np.array([t[k][i]]) for k in ("isD", "qshoot_eff", "qdef_eff")}
            s_sc, s_pm, s_df = values_at(fit, ti, "ev", np.array([st["shoot"]]),
                                         np.array([st["create"]]), np.array([st["def"]]),
                                         np.array([fe]), n_def=5)
            g60 = float(ga60_of({"isD": np.array([t["isD"][i]])}, base, s_sc, s_pm, s_df, kap)[0])
            age_s = t["age"][i] - (t["last_season"][i] - s) if np.isfinite(t["age"][i]) else None
            lsc, lpm, ldf = league_at_age(zs, t["isD"][i])
            tr_out.append({"season": s, "age": _num(age_s, 1), "scoring": _num(s_sc[0]),
                           "playmaking": _num(s_pm[0]), "defense": _num(s_df[0]),
                           "ga60": _num(g60),
                           "lg_scoring": _num(lsc), "lg_playmaking": _num(lpm),
                           "lg_defense": _num(ldf)})
        proj = None
        pj = projJ.get(int(t["id"][i]))
        if pj:
            pj_ga = ga60_of({"isD": np.array([t["isD"][i]])}, base,
                            np.array([pj["ev_scoring"]]), np.array([pj["ev_playmaking"]]),
                            np.array([pj["ev_defense"]]), kap)[0]
            zt = (t["z_last"][i] + (proj_season - t["last_season"][i]) / AGE_SCALE
                  if np.isfinite(t["age"][i]) and proj_season else 0.0)
            plsc, plpm, pldf = league_at_age(zt, t["isD"][i])
            proj = {"season": proj_season, "scoring": _num(pj["ev_scoring"]),
                    "playmaking": _num(pj["ev_playmaking"]), "defense": _num(pj["ev_defense"]),
                    "ga60": _num(pj_ga), "lg_scoring": _num(plsc), "lg_playmaking": _num(plpm),
                    "lg_defense": _num(pldf)}
        A = {"war": {"v": _num(war_latest[i], 2), "pct": _num(pcts["war"][i], 0)},
             "ga60": {"v": _num(ga60[i]), "pct": _num(pcts["ga60"][i], 0)},
             "pp_ga60": {"v": _num(pp_ga60[i]), "pct": _num(pcts["pp_ga60"][i], 0)},
             "pk_ga60": {"v": _num(pk_ga60[i]), "pct": _num(pcts["pk_ga60"][i], 0)},
             "scoring": {"v": _num(sc[i]), "pct": _num(pcts["scoring"][i], 0)},
             "shooting": {"v": _num(shooting_pct_vol[i], 1), "pct": _num(pcts["shooting"][i], 0)},
             "finishing": {"v": _num(fin100[i], 2), "se": _num(fin100_se[i], 2),
                           "pct": _num(pcts["finishing"][i], 0)},
             "playmaking": {"v": _num(pm[i]), "se": _num(pm_se[i]), "pct": _num(pcts["playmaking"][i], 0)},
             "defense": {"v": _num(dfv[i]), "pct": _num(pcts["defense"][i], 0)}}
        players_out[int(t["id"][i])] = {
            "pos": "D" if t["isD"][i] > 0 else "F", "age": _num(t["age"][i], 1),
            "last_season": int(t["last_season"][i]),
            "attrs": A,
            "war": {"latest": _num(war_latest[i], 2), "total": _num(war_total[i], 2),
                    "by_season": {s: _num(g[i] / GOALS_PER_WIN, 2) for s, g in gar_season.items()},
                    "ev": _num(gar["ev"][i] / GOALS_PER_WIN, 2),
                    "pp": _num(gar["pp"][i] / GOALS_PER_WIN, 2),
                    "pk": _num(gar["pk"][i] / GOALS_PER_WIN, 2)},
            "trajectory": tr_out, "projection": proj}

    out = {"meta": {"fit": fit["_path"], "seasons": seasons, "generated": str(date.today()),
                    "kappa": round(kap, 4), "goals_per_win": GOALS_PER_WIN,
                    "replacement": repl_meta, "repl_band_pct": list(REPL_BAND),
                    "baselines": {g: {k: round(v, 4) for k, v in b.items()}
                                  for g, b in base.items()},
                    "age_curves": fit["age_curves"], "rw_sd": fit.get("rw_sd"),
                    "latest_season": last},
           "players": players_out}
    path = C.MODELS / "gen_cards.json"
    path.write_text(json.dumps(out))
    league_war = float(war_latest[ev_el].sum())
    print(f"  gen_cards: {n} players  κ={kap:.3f}  league WAR({last}, eligible)={league_war:.0f}"
          f"  -> {path}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Export generative player cards (GA/60 + WAR)")
    p.add_argument("--fit", type=str, default=None, help="path to a generative_model_*.json")
    args = p.parse_args(argv)
    build(args.fit)


if __name__ == "__main__":
    main()
