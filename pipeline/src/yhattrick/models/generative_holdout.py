r"""Held-out-season predictive harness for the generative model (roadmap #1).

Fits the full model on seasons ≤ --train-through, projects every player to the following season
(the RW-mean state + aging curve at next-season age), then scores the projection against what
ACTUALLY happened that season — on the season's real lineups — versus simpler reads of the same
training fit. This converts "the trajectories look right" into "they predict better", and is the
tool for tuning RW_SD_* (rerun with different constants, compare).

CANDIDATE per-player EV parameter sets (all from the ONE training fit; identical treatment of
intercept/context, so differences isolate the player-skill read):
  league-avg    every player at the prior mean — the "no player skill" floor (it keeps the full
                context incl. position/age composition, which needs no skill knowledge)
  pooled-mean   TOI-weighted mean of the player's per-season effective states — the static-pooled
                read (what you'd estimate with no drift and no projection)
  last-state    the player's LAST effective state (state + curve at his last age) — drift, no aging
  projection    last RW state + curve at TARGET-season age — the model's actual forecast

The skill candidates carry each player's aging curve + position offset inside their EFFECTIVE
parameters, so the holdout context applies only the base columns (home/zone/score) + the league
environment nowcast (final training season's FE) — applying the AGE/position columns too would
double-count terms with large nonzero league means (a ~2× Σ-shots inflation; the per-candidate
block-mean diagnostic line exists to catch exactly that class of bug).

SCORING (EV rate stage — own-shot generation on the held-out season's real stint rows):
  1. Row-level Poisson deviance of observed focal-shooter Fenwick counts under each candidate's
     rates (per-1000-row scale; lower is better). Real lineups ⇒ deployment identical across
     candidates. Arena states are omitted for all candidates (the target season's venue states
     don't exist in training — common treatment).
  2. Player-level own-shots/60: predicted (candidate rates summed over the player's actual rows)
     vs observed, TOI-weighted correlation + MAE over eligible players (≥ MIN_TOI_EVAL in the
     target season AND seen in training). Also scored: the naive baseline (the player's raw
     final-training-season shots/60 — the "just use last year's stats" bar).

Players unseen in training (rookies) sit at the prior mean for every candidate; the eligibility
gate keeps the player-level table honest.

The training side (candidate vectors, naive aggregates, context coefficients) is cached to
data/models/holdout_fit_<train>.npz so scoring iterations don't repay the ~hours-long fit:
  uv run --group experimental python -m yhattrick.models.generative_holdout            # 2021-24 → 2025
  uv run --group experimental python -m yhattrick.models.generative_holdout --rescore  # reuse cache
Output: printed tables + data/models/holdout_<target>.json.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from .. import config as C
from . import generative_model as G
from . import generative_data as D
from . import generative_likelihood as L
from .generative_features import RATE_CTX

MIN_TOI_EVAL = 12000.0  # ≥ 200 EV minutes in the target season to enter the player-level table
CAND_ORDER = ("league-avg", "pooled-mean", "last-state", "projection")
BLOCKS = ("shoot", "create", "def")


def poisson_deviance(N, mu):
    """Total Poisson deviance 2Σ[μ − N + N·log(N/μ)] (the 0·log0 term is 0)."""
    return float(2.0 * np.sum(mu - N + np.where(N > 0, N * np.log(np.maximum(N, 1e-12) / mu), 0.0)))


def wpearson(x, y, w):
    """Weighted Pearson correlation."""
    w = w / w.sum()
    mx, my = np.sum(w * x), np.sum(w * y)
    cov = np.sum(w * (x - mx) * (y - my))
    return float(cov / np.sqrt(np.sum(w * (x - mx) ** 2) * np.sum(w * (y - my) ** 2)))


def wmae(x, y, w):
    """Weighted mean absolute error."""
    return float(np.sum(w * np.abs(x - y)) / w.sum())


def candidates(M, target):
    """Per-player (training-index) EV shoot/create/def parameter sets — see module docstring."""
    rates, qual, conv = M["rates"], M["qual"], M["conv"]
    players, agepos, ls = M["players"], M["agepos"], M["last_season"]
    P = len(players)
    re_last, _, _ = G.effective_params(rates, qual, conv, players, agepos, ls)
    re_proj, _, _ = G.effective_params(rates, qual, conv, players, agepos, ls, target=target)
    ev = rates["ev"]
    ue = G.unit_effective(ev, agepos)
    tw, up = ev["R"]["toi_unit"], ev["unit_player"]
    pooled = {}
    for blk in BLOCKS:
        ws = np.bincount(up, weights=tw, minlength=P)
        vs = np.bincount(up, weights=ue[blk] * tw, minlength=P)
        pooled[blk] = np.where(ws > 0, vs / np.maximum(ws, 1e-9), 0.0)
    return {
        "league-avg": {b: np.zeros(P) for b in BLOCKS},
        "pooled-mean": pooled,
        "last-state": {b: re_last["ev"][b] for b in BLOCKS},
        "projection": {b: re_proj["ev"][b] for b in BLOCKS},
    }


def _hyper_tag(
    ma_anchor_scale,
    ma_create_prior_sd,
    ma_def_prior_sd=None,
    ev_anchor_scale=1.0,
    create_prior_center=None,
):
    """Cache-file suffix for non-default hyperparameters (sweep candidates keep separate caches)."""
    tag = ""
    if ma_anchor_scale != 1.0:
        tag += f"_a{ma_anchor_scale:g}"
    if ma_create_prior_sd:
        tag += f"_p{ma_create_prior_sd:g}"
    if ma_def_prior_sd:
        tag += f"_d{ma_def_prior_sd:g}"
    if ev_anchor_scale != 1.0:
        tag += f"_ea{ev_anchor_scale:g}"
    if create_prior_center:
        tag += f"_cc{create_prior_center}"
    return tag


def _fit_train_side(
    train,
    train_through,
    target,
    count_model,
    spg_scale,
    ma_anchor_scale=1.0,
    ma_create_prior_sd=None,
    ma_def_prior_sd=None,
    ev_anchor_scale=1.0,
    create_prior_center=None,
):
    """Run the training fit and reduce it to the slim scoring inputs; cached as an npz so scoring
    can iterate without re-fitting. The train side keeps its OWN θ̂ checkpoint chain
    (holdout_ckpt.npz): sweep candidates warm-start each other — never a fit that saw test data."""
    M = G.fit_all(
        train,
        count_model=count_model,
        spg_scale=spg_scale,
        ma_anchor_scale=ma_anchor_scale,
        ma_create_prior_sd=ma_create_prior_sd,
        ma_def_prior_sd=ma_def_prior_sd,
        ev_anchor_scale=ev_anchor_scale,
        create_prior_center=create_prior_center,
        warm=True,
        save_ckpt=True,
        ckpt_path=C.MODELS / "holdout_ckpt.npz",
    )
    evt = M["rates"]["ev"]
    Rt = evt["R"]
    P = len(M["players"])
    mlast = Rt["season_row"] == train_through  # naive baseline: final-season raw rates
    tm_last = np.zeros(P)  # teammate shots while on ice (create side)
    for t in range(Rt["team_idx"].shape[1]):
        np.add.at(tm_last, Rt["team_idx"][mlast, t], Rt["count"][mlast])
    ts = {
        "players": np.array(M["players"], dtype=np.int64),
        "intercept": np.float64(evt["intercept"]),
        "ctx_names": np.array(evt["ctx_names"]),
        "beta": np.asarray(evt["beta"], dtype=np.float64),
        "s_last": np.bincount(Rt["shooter_idx"][mlast], weights=Rt["count"][mlast], minlength=P),
        "t_last": np.bincount(Rt["shooter_idx"][mlast], weights=Rt["dur"][mlast], minlength=P),
        "tm_last": tm_last,
        "a2_q": np.float64(evt.get("a2_q") if evt.get("a2_q") is not None else np.nan),
        "train": np.array(train, dtype=np.int64),
        "ma_anchor_scale": np.float64(ma_anchor_scale),
        "ma_create_prior_sd": np.float64(ma_create_prior_sd if ma_create_prior_sd else np.nan),
        "ma_def_prior_sd": np.float64(ma_def_prior_sd if ma_def_prior_sd else np.nan),
        "ev_anchor_scale": np.float64(ev_anchor_scale),
    }
    for name, c in candidates(M, target).items():
        for blk in BLOCKS:
            ts[f"cand_{name}_{blk}"] = c[blk]
    mat = M["rates"].get("ma")
    if mat is not None:  # MA track: EFFECTIVE per-player params
        re_last, _, _ = G.effective_params(
            M["rates"], M["qual"], M["conv"], M["players"], M["agepos"], M["last_season"]
        )
        for blk in BLOCKS:
            ts[f"ma_{blk}"] = re_last["ma"][blk]
        ts["ma_intercept"] = np.float64(mat["intercept"])
        ts["ma_ctx_names"] = np.array(mat["ctx_names"])
        ts["ma_beta"] = np.asarray(mat["beta"], dtype=np.float64)
    path = C.MODELS / (
        f"holdout_fit_{train_through}"
        f"{_hyper_tag(ma_anchor_scale, ma_create_prior_sd, ma_def_prior_sd, ev_anchor_scale, create_prior_center)}.npz"
    )
    C.MODELS.mkdir(parents=True, exist_ok=True)
    np.savez(path, **ts)
    print(f"[holdout] training side cached -> {path.name}")
    return ts


def calibration_slope(N, offset, base, term, iters=40):
    """Out-of-sample calibration slope γ for one parameter block: the 1-D Poisson MLE of
    μ = exp(base + γ·term + offset) on held-out rows, all other blocks held at training values.
    γ = 1 ⇔ the block's fitted spread predicts at face value; γ < 1 ⇔ the fit's spread is wider
    than what it can demonstrate out of sample (e.g. the PP create split inside long-lived units,
    where assists reflect role rather than creation). Newton on the concave log-likelihood."""
    g = 1.0
    for _ in range(iters):
        mu = np.exp(base + g * term + offset)
        step = float(np.sum(term * (N - mu))) / max(float(np.sum(term * term * mu)), 1e-12)
        g += float(np.clip(step, -0.5, 0.5))
        if abs(step) < 1e-9:
            break
    return float(g)


def evaluate(
    train_through,
    target=None,
    count_model="nb",
    spg_scale=1.0,
    rescore=False,
    ma_anchor_scale=1.0,
    ma_create_prior_sd=None,
    ma_def_prior_sd=None,
    ev_anchor_scale=1.0,
    create_prior_center=None,
):
    sd = C.PROCESSED / "shots_onice"
    avail = sorted(int(f.stem) for f in sd.glob("*.parquet")) if sd.exists() else []
    train = [s for s in avail if s <= train_through]
    target = target or train_through + 1
    if target not in avail:
        raise SystemExit(f"target season {target} not in processed data {avail}")
    tag = _hyper_tag(
        ma_anchor_scale, ma_create_prior_sd, ma_def_prior_sd, ev_anchor_scale, create_prior_center
    )
    cache = C.MODELS / f"holdout_fit_{train_through}{tag}.npz"
    if rescore and cache.exists():
        print(f"[holdout] rescoring from {cache.name}")
        z = np.load(cache)
        ts = {k: z[k] for k in z.files}
    else:
        print(f"[holdout] train {train} → target {target} — fitting …")
        ts = _fit_train_side(
            train,
            train_through,
            target,
            count_model,
            spg_scale,
            ma_anchor_scale=ma_anchor_scale,
            ma_create_prior_sd=ma_create_prior_sd,
            ma_def_prior_sd=ma_def_prior_sd,
            ev_anchor_scale=ev_anchor_scale,
            create_prior_center=create_prior_center,
        )
    players_t = ts["players"]
    idx_t = {int(p): i for i, p in enumerate(players_t)}

    # held-out EV rows on the target season's own player index (rookies included, at the prior mean)
    players_h, idx_h = D.player_index([target])
    agepos_h = D._age_position(players_h, [target])
    Rh = D.rate_rows(
        [target], L.EV_STRENGTHS, True, players_h, idx_h, agepos_h, states=False, arenas=False
    )
    Ph = len(players_h)
    seen = np.array([p in idx_t for p in players_h])
    print(
        f"[holdout] target rows {len(Rh['count']):,}  players {Ph} ({int(seen.sum())} seen in training)"
    )

    # context by NAME: base columns for the skill candidates (their effective params already carry
    # curve + position offsets — see docstring); full columns for the no-skill league-avg floor
    cmt = G._coef_map([str(n) for n in ts["ctx_names"]], ts["beta"])
    nowcast = cmt.get(f"season_{train_through}", 0.0)
    ctx_base = np.full(len(Rh["count"]), nowcast)
    ctx_full = ctx_base.copy()
    base_names = set(RATE_CTX)
    for j, nm in enumerate(Rh["ctx_names"]):
        if nm in cmt:
            if nm in base_names:
                ctx_base = ctx_base + cmt[nm] * Rh["Xctx"][:, j]
            ctx_full = ctx_full + cmt[nm] * Rh["Xctx"][:, j]

    def to_hold(vec_t):
        out = np.zeros(Ph)
        for i, p in enumerate(players_h):
            j = idx_t.get(int(p))
            if j is not None:
                out[i] = vec_t[j]
        return out

    N = Rh["count"]
    toi_h = Rh["toi_atk"]
    obs_shots = np.bincount(Rh["shooter_idx"], weights=N, minlength=Ph)
    obs_rate = np.where(toi_h > 0, obs_shots / np.maximum(toi_h, 1.0) * 3600.0, 0.0)
    obs_tm = np.zeros(Ph)  # create side: teammate shots while on ice
    for t in range(Rh["team_idx"].shape[1]):
        np.add.at(obs_tm, Rh["team_idx"][:, t], N)
    obs_tm_rate = np.where(toi_h > 0, obs_tm / np.maximum(toi_h, 1.0) * 3600.0, 0.0)
    elig = (toi_h >= MIN_TOI_EVAL) & seen
    w = toi_h
    mu0 = float(ts["intercept"])

    out = {
        "train": [int(s) for s in ts["train"]],
        "target": int(target),
        "count_model": count_model,
        "n_rows": int(len(N)),
        "n_players": Ph,
        "n_seen": int(seen.sum()),
        "n_eligible": int(elig.sum()),
        "rw_sd": {"shoot": L.RW_SD_SHOOT, "create": L.RW_SD_CREATE, "def": L.RW_SD_DEF},
        "candidates": {},
    }
    if np.isfinite(float(ts.get("a2_q", np.nan))):
        out["a2_q"] = float(ts["a2_q"])
        print(f"[holdout] training fit A2 mixture q = {out['a2_q']:.3f}")
    print(
        f"\n[holdout] eligible for the player table: {int(elig.sum())} "
        f"(≥{MIN_TOI_EVAL / 60:.0f} EV min in {target} + trained)"
    )
    print(
        f"{'candidate':14s} {'row-dev/1k':>11s} {'Σμ/ΣN':>8s} {'corr':>7s} {'MAE/60':>7s} "
        f"{'tm-corr':>8s} {'tm-MAE':>7s}"
    )
    for name in CAND_ORDER:
        sh = to_hold(ts[f"cand_{name}_shoot"])
        cr = to_hold(ts[f"cand_{name}_create"])
        df = to_hold(ts[f"cand_{name}_def"])
        ctx = ctx_full if name == "league-avg" else ctx_base
        eta = (
            mu0
            + sh[Rh["shooter_idx"]]
            + cr[Rh["team_idx"]].sum(1)
            + (df[Rh["def_idx"]] * Rh["def_mask"]).sum(1)
            + ctx
        )
        mu = np.exp(eta + Rh["offset"])
        dev = poisson_deviance(N, mu)
        pred_shots = np.bincount(Rh["shooter_idx"], weights=mu, minlength=Ph)
        pred_rate = np.where(toi_h > 0, pred_shots / np.maximum(toi_h, 1.0) * 3600.0, 0.0)
        pred_tm = np.zeros(Ph)  # create side: predicted teammate shots
        for t in range(Rh["team_idx"].shape[1]):
            np.add.at(pred_tm, Rh["team_idx"][:, t], mu)
        pred_tm_rate = np.where(toi_h > 0, pred_tm / np.maximum(toi_h, 1.0) * 3600.0, 0.0)
        corr = wpearson(pred_rate[elig], obs_rate[elig], w[elig])
        mae = wmae(pred_rate[elig], obs_rate[elig], w[elig])
        tcorr = wpearson(pred_tm_rate[elig], obs_tm_rate[elig], w[elig])
        tmae = wmae(pred_tm_rate[elig], obs_tm_rate[elig], w[elig])
        out["candidates"][name] = {
            "row_deviance_per_1k": dev / len(N) * 1000.0,
            "sum_mu": float(mu.sum()),
            "sum_N": float(N.sum()),
            "rate_corr": corr,
            "rate_mae60": mae,
            "tm_corr": tcorr,
            "tm_mae60": tmae,
        }
        wm = w[seen] / max(w[seen].sum(), 1e-9)  # block-mean diagnostic (level-bug canary)
        print(
            f"{name:14s} {dev / len(N) * 1000.0:11.3f} {mu.sum() / N.sum():8.3f} "
            f"{corr:7.3f} {mae:7.3f} {tcorr:8.3f} {tmae:7.3f}   blocks sh {np.sum(wm * sh[seen]):+.2f} "
            f"cr {np.sum(wm * cr[seen]):+.2f} df {np.sum(wm * df[seen]):+.2f}"
        )

    # naive baseline: the player's raw final-training-season rates predict his target rates
    naive_t = np.where(ts["t_last"] > 0, ts["s_last"] / np.maximum(ts["t_last"], 1.0) * 3600.0, 0.0)
    naive_h = to_hold(naive_t)
    en = elig & (to_hold(ts["t_last"]) >= MIN_TOI_EVAL)
    corr = wpearson(naive_h[en], obs_rate[en], w[en])
    mae = wmae(naive_h[en], obs_rate[en], w[en])
    out["naive_last_season"] = {"rate_corr": corr, "rate_mae60": mae, "n": int(en.sum())}
    tcorr = tmae = None
    if "tm_last" in ts:
        naive_tm = to_hold(
            np.where(ts["t_last"] > 0, ts["tm_last"] / np.maximum(ts["t_last"], 1.0) * 3600.0, 0.0)
        )
        tcorr = wpearson(naive_tm[en], obs_tm_rate[en], w[en])
        tmae = wmae(naive_tm[en], obs_tm_rate[en], w[en])
        out["naive_last_season"].update(tm_corr=tcorr, tm_mae60=tmae)
    tstr = f" {tcorr:8.3f} {tmae:7.3f}" if tcorr is not None else ""
    print(
        f"{'naive-' + str(train_through):14s} {'—':>11s} {'—':>8s} {corr:7.3f} {mae:7.3f}{tstr}"
        f"   (n={int(en.sum())}, needs {train_through} TOI too)"
    )
    lm = float(np.sum(w[elig] * obs_rate[elig]) / w[elig].sum())
    out["league_mean_mae60"] = wmae(np.full(int(elig.sum()), lm), obs_rate[elig], w[elig])
    print(f"{'league-mean':14s} {'—':>11s} {'—':>8s} {'0.000':>7s} {out['league_mean_mae60']:7.3f}")

    # ── calibration slopes γ (face-value honesty per block) + the MA track ────────────────────────
    # γ = 1 ⇔ that block's fitted spread predicts held-out reality at face value. EV uses the
    # last-state candidate (the "current skill" read the cards/WAR consume); MA uses the bucket's
    # static effective params. This is the selection criterion for the MA anchor/prior sweep.
    out["ma_anchor_scale"] = float(ts.get("ma_anchor_scale", 1.0))
    out["ev_anchor_scale"] = float(ts.get("ev_anchor_scale", 1.0))
    mcp = float(ts.get("ma_create_prior_sd", np.nan))
    out["ma_create_prior_sd"] = None if np.isnan(mcp) else mcp
    mdp = float(ts.get("ma_def_prior_sd", np.nan))
    out["ma_def_prior_sd"] = None if np.isnan(mdp) else mdp
    gamma = {"ev": {}, "ma": {}}
    ev_terms = {
        "shoot": to_hold(ts["cand_last-state_shoot"])[Rh["shooter_idx"]],
        "create": to_hold(ts["cand_last-state_create"])[Rh["team_idx"]].sum(1),
        "def": (to_hold(ts["cand_last-state_def"])[Rh["def_idx"]] * Rh["def_mask"]).sum(1),
    }
    ev_all = ev_terms["shoot"] + ev_terms["create"] + ev_terms["def"]
    for blk, term in ev_terms.items():
        gamma["ev"][blk] = calibration_slope(N, Rh["offset"], mu0 + ctx_base + ev_all - term, term)
    if "ma_shoot" in ts:
        Rm = D.rate_rows(
            [target], L.MA_STRENGTHS, False, players_h, idx_h, agepos_h, states=False, arenas=False
        )
        cmm = G._coef_map([str(n) for n in ts["ma_ctx_names"]], ts["ma_beta"])
        ctx_ma = np.full(len(Rm["count"]), cmm.get(f"season_{train_through}", 0.0))
        for j, nm in enumerate(Rm["ctx_names"]):
            if nm in cmm and nm in set(RATE_CTX):
                ctx_ma = ctx_ma + cmm[nm] * Rm["Xctx"][:, j]
        ma_terms = {
            "shoot": to_hold(ts["ma_shoot"])[Rm["shooter_idx"]],
            "create": to_hold(ts["ma_create"])[Rm["team_idx"]].sum(1),
            "def": (to_hold(ts["ma_def"])[Rm["def_idx"]] * Rm["def_mask"]).sum(1),
        }
        ma_all = ma_terms["shoot"] + ma_terms["create"] + ma_terms["def"]
        mu0_ma = float(ts["ma_intercept"])
        Nm = Rm["count"]
        for blk, term in ma_terms.items():
            gamma["ma"][blk] = calibration_slope(
                Nm, Rm["offset"], mu0_ma + ctx_ma + ma_all - term, term
            )
        mu_ma = np.exp(mu0_ma + ctx_ma + ma_all + Rm["offset"])
        out["ma_track"] = {
            "n_rows": int(len(Nm)),
            "row_deviance_per_1k": poisson_deviance(Nm, mu_ma) / len(Nm) * 1000.0,
            "sum_mu": float(mu_ma.sum()),
            "sum_N": float(Nm.sum()),
        }
        print(
            f"\n[MA track] rows {len(Nm):,}  Σμ/ΣN {mu_ma.sum() / max(Nm.sum(), 1.0):.3f}  "
            f"dev/1k {out['ma_track']['row_deviance_per_1k']:.3f}"
        )
    else:
        print("\n[MA track] skipped — cache predates the MA params (re-run without --rescore)")
    out["gamma"] = gamma
    gs = "  ".join(f"{b}:{k} {v:+.3f}" for b in ("ev", "ma") for k, v in gamma[b].items())
    print(f"[γ] calibration slopes (1 = face value honest): {gs}")
    cal = {
        "train_through": int(train_through),
        "target": int(target),
        "count_model": count_model,
        "spg_scale": spg_scale,
        "ma_anchor_scale": out["ma_anchor_scale"],
        "ev_anchor_scale": out["ev_anchor_scale"],
        "ma_create_prior_sd": out["ma_create_prior_sd"],
        "ma_def_prior_sd": out["ma_def_prior_sd"],
        "gamma": gamma,
    }
    cpath = C.MODELS / f"holdout_calibration{tag}.json"
    cpath.write_text(json.dumps(cal, indent=1))
    print(f"  -> {cpath}")

    path = C.MODELS / f"holdout_{target}{tag}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"  -> {path}")
    return out


# ══ go/no-go: does per-player creation QUALITY (create_qual) beat the position pair? ══════════════
# The rate harness above tests the create VOLUME half (teammate-shots track). This tests the QUALITY
# half: a per-player on-ice xG lift `cq[p]` (dense RAPM-on-quality — every shot p is on the ice for
# is worth +cq[p] logit-xG) vs the model's fallback of one lift per POSITION (qcreate_{F,D}). Both
# candidates come from ONE create_qual=True fit and differ ONLY in that channel (per-player cq vs its
# F/D mean), exactly as the rate track derives all candidates from one fit. Prediction is the on-ice
# level sigmoid(base): the sparse creator-credit bump (the qc mixture) is a shared nuisance that would
# need projected anchor states, so it is dropped from BOTH sides to isolate the dense cq channel and
# keep the A/B fair. Aggregation is per-player teammate xG-per-shot over held-out EV shots where the
# player is an on-ice non-shooter (the create side), mirroring the rate track's teammate-shots read.
MIN_TMSHOTS_EVAL = 400  # ≥ this many held-out EV teammate-shots to enter the quality gate table


def _fit_quality_gate_side(train, train_through, target, count_model, ma_kwargs):
    """Fit the model WITH per-player create_qual on the training seasons and reduce it to the slim
    quality-scoring inputs (cached npz). Also caches the naive teammate-xG bar: each player's mean
    on-ice teammate xG in the final training season (the model-free 'use last year' read)."""
    M = G.fit_all(
        train,
        count_model=count_model,
        create_qual=True,
        warm=True,
        save_ckpt=True,
        # per-train_through checkpoint: each split keeps its OWN chain so a fit never warm-starts
        # from one that saw its held-out target (the 2021-24 fit must not seed the 2021-23 → 2024 split)
        ckpt_path=C.MODELS / f"holdout_cq_ckpt_{train_through}.npz",
        **ma_kwargs,
    )
    q = M["qual"]
    players = np.asarray(M["players"], dtype=np.int64)
    isD = np.asarray(M["agepos"]["isD"], dtype=bool)
    # naive bar: per-player mean on-ice teammate xG in the final training season (EV only)
    idx_tr = {int(p): i for i, p in enumerate(players)}
    Qtr = D.quality_creator_rows([train_through], idx_tr, G.ALL_STRENGTHS, M["agepos"], arenas=False)
    ev = Qtr["strength"] == 0
    P = len(players)
    xg_tr = _sigmoid_np(Qtr["y"][ev])
    naive_sum = np.zeros(P)
    naive_cnt = np.zeros(P)
    tmi = Qtr["team_idx"][ev]
    for t in range(tmi.shape[1]):
        np.add.at(naive_sum, tmi[:, t], xg_tr)
        np.add.at(naive_cnt, tmi[:, t], 1.0)
    ts = {
        "players": players,
        "isD": isD,
        "mq": np.float64(q["intercept"]),
        "qshoot": np.asarray(q["qshoot"], dtype=np.float64),
        "qcreate": np.asarray(q["qcreate"], dtype=np.float64),
        "qdef": np.asarray(q["qdef"], dtype=np.float64),
        "create_qual": np.asarray(q["create_qual"], dtype=np.float64),
        "beta": np.asarray(q["beta"], dtype=np.float64),
        "ctx_names": np.array(q["ctx_names"]),
        "naive_tmxg": np.where(naive_cnt > 0, naive_sum / np.maximum(naive_cnt, 1.0), np.nan),
        "naive_cnt": naive_cnt,
        "train": np.array(train, dtype=np.int64),
        "train_through": np.int64(train_through),
    }
    path = C.MODELS / f"qgate_fit_{train_through}.npz"
    C.MODELS.mkdir(parents=True, exist_ok=True)
    np.savez(path, **ts)
    print(f"[qgate] training side cached -> {path.name}")
    return ts


def _sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def quality_gate(
    train_through,
    target=None,
    count_model="nb",
    rescore=False,
    ma_anchor_scale=0.25,
    ma_create_prior_sd=0.04,
    ma_def_prior_sd=0.10,
    ev_anchor_scale=0.25,
    create_prior_center="position-mean",
):
    """Held-out go/no-go for the per-player creation-quality parameter: ship create_qual only if the
    per-player candidate beats the position-level pair at predicting held-out teammate xG."""
    sd = C.PROCESSED / "shots_onice"
    avail = sorted(int(f.stem) for f in sd.glob("*.parquet")) if sd.exists() else []
    train = [s for s in avail if s <= train_through]
    target = target or train_through + 1
    if target not in avail:
        raise SystemExit(f"target season {target} not in processed data {avail}")
    ma_kwargs = dict(
        ma_anchor_scale=ma_anchor_scale,
        ma_create_prior_sd=ma_create_prior_sd,
        ma_def_prior_sd=ma_def_prior_sd,
        ev_anchor_scale=ev_anchor_scale,
        create_prior_center=create_prior_center,
    )
    cache = C.MODELS / f"qgate_fit_{train_through}.npz"
    if rescore and cache.exists():
        print(f"[qgate] rescoring from {cache.name}")
        z = np.load(cache, allow_pickle=False)
        ts = {k: z[k] for k in z.files}
    else:
        print(f"[qgate] train {train} → target {target} — fitting create_qual=True …")
        ts = _fit_quality_gate_side(train, train_through, target, count_model, ma_kwargs)

    players_t = ts["players"]
    idx_t = {int(p): i for i, p in enumerate(players_t)}
    isD_t = ts["isD"].astype(bool)
    cq_t = ts["create_qual"]
    # position means of the fitted cq (the position-level counterfactual): trained players only
    muF = float(cq_t[~isD_t].mean())
    muD = float(cq_t[isD_t].mean())
    print(f"[qgate] fitted create_qual: F mean {muF:+.4f} std {cq_t[~isD_t].std():.4f}  "
          f"D mean {muD:+.4f} std {cq_t[isD_t].std():.4f}")

    # held-out quality rows on the target's own player index (rookies at the position mean)
    players_h, idx_h = D.player_index([target])
    agepos_h = D._age_position(players_h, [target])
    isD_h = np.asarray(agepos_h["isD"], dtype=bool)
    Qh = D.quality_creator_rows([target], idx_h, G.ALL_STRENGTHS, agepos_h, arenas=False)
    ev = Qh["strength"] == 0  # EV only: the 5v5 chance-creation signal (matches the rate track)
    Ph = len(players_h)
    seen = np.array([int(p) in idx_t for p in players_h])
    print(f"[qgate] target EV shots {int(ev.sum()):,}  players {Ph} ({int(seen.sum())} seen)")

    def to_hold(vec_t, fill_pos=False):
        """Map a train-index vector onto the held-out player index; unseen → position mean (fill_pos)
        or 0."""
        out = np.where(isD_h, muD, muF) if fill_pos else np.zeros(Ph)
        for i, p in enumerate(players_h):
            j = idx_t.get(int(p))
            if j is not None:
                out[i] = vec_t[j]
        return out

    qshoot_h = to_hold(ts["qshoot"])
    qdef_h = to_hold(ts["qdef"])
    cq_pp = to_hold(cq_t, fill_pos=True)  # per-player candidate (unseen → position mean)
    cq_pos = np.where(isD_h, muD, muF)  # position-level candidate: every player at his F/D mean

    # context by NAME (quality per-player params carry no curves, so apply ALL matched columns
    # + the final-training-season league nowcast), on EV rows only
    cmt = G._coef_map([str(n) for n in ts["ctx_names"]], ts["beta"])
    nowcast = cmt.get(f"season_{int(ts['train_through'])}", 0.0)
    ctx = np.full(int(ev.sum()), nowcast)
    for j, nm in enumerate(Qh["ctx_names"]):
        if nm in cmt:
            ctx = ctx + cmt[nm] * Qh["Xctx"][ev, j]

    sh_i = Qh["shooter_idx"][ev]
    tm_i = Qh["team_idx"][ev]
    df_i = Qh["def_idx"][ev]
    dm = Qh["def_mask"][ev]
    obs_xg = _sigmoid_np(Qh["y"][ev])
    base0 = float(ts["mq"]) + qshoot_h[sh_i] + (qdef_h[df_i] * dm).sum(1) + ctx  # shared, no cq

    # observed per-player teammate xG (the create side: mean xG of shots the player is on ice for
    # but did not take) + shot weight
    obs_sum = np.zeros(Ph)
    cnt = np.zeros(Ph)
    for t in range(tm_i.shape[1]):
        np.add.at(obs_sum, tm_i[:, t], obs_xg)
        np.add.at(cnt, tm_i[:, t], 1.0)
    obs_tmxg = np.where(cnt > 0, obs_sum / np.maximum(cnt, 1.0), 0.0)
    elig = seen & (cnt >= MIN_TMSHOTS_EVAL)
    w = cnt
    print(f"[qgate] eligible (seen + ≥{MIN_TMSHOTS_EVAL} EV teammate-shots): {int(elig.sum())}")

    def pred_player(cq_h):
        """Per-player predicted teammate xG-per-shot under an on-ice cq vector."""
        px = _sigmoid_np(base0 + cq_h[tm_i].sum(1))
        s = np.zeros(Ph)
        for t in range(tm_i.shape[1]):
            np.add.at(s, tm_i[:, t], px)
        return np.where(cnt > 0, s / np.maximum(cnt, 1.0), 0.0), px

    pp_pred, pp_px = pred_player(cq_pp)  # per-player candidate
    pos_pred, pos_px = pred_player(cq_pos)  # position-level candidate

    def rowdev(px):  # per-1k fractional-Bernoulli deviance (per-shot honesty check)
        d = -2.0 * np.sum(obs_xg * np.log(px + 1e-9) + (1 - obs_xg) * np.log(1 - px + 1e-9))
        return float(d) / len(obs_xg) * 1000.0

    def _wcorr(pred, m):
        return wpearson(pred[m], obs_tmxg[m], w[m])

    res = {
        "per-player": {
            "tmxg_corr": _wcorr(pp_pred, elig),
            "tmxg_mae": wmae(pp_pred[elig], obs_tmxg[elig], w[elig]),
            "row_dev_per_1k": rowdev(pp_px),
        },
        "position-level": {
            "tmxg_corr": _wcorr(pos_pred, elig),
            "tmxg_mae": wmae(pos_pred[elig], obs_tmxg[elig], w[elig]),
            "row_dev_per_1k": rowdev(pos_px),
        },
    }

    # naive bar: last training season's per-player teammate xG predicts the target's
    naive_h = to_hold(np.where(np.isnan(ts["naive_tmxg"]), 0.0, ts["naive_tmxg"]))
    naive_ok = elig & (to_hold(ts["naive_cnt"]) >= MIN_TMSHOTS_EVAL)
    res["naive_last_season"] = {
        "tmxg_corr": (_wcorr(naive_h, naive_ok) if naive_ok.sum() > 2 else float("nan")),
        "tmxg_mae": (wmae(naive_h[naive_ok], obs_tmxg[naive_ok], w[naive_ok]) if naive_ok.any() else float("nan")),
        "n": int(naive_ok.sum()),
    }

    # honesty check: does a player's OWN fitted cq predict his observed teammate-xg deviation out of
    # sample? This is the clean, MEAN-CENTERED calibration — corr of the own-cq deviation (from the
    # position mean) with the observed teammate-xg deviation (from the position-level prediction).
    # It isolates the per-player creation signal from the linemate-clustering confound that pollutes
    # a prediction-space slope, and — being centered — is immune to the level offset from omitting the
    # shared qc creator bump. > 0 ⇔ the signal is directionally real; ≈ 0 ⇔ noise / linemate-only.
    ei = np.flatnonzero(elig)
    own_dev = (cq_pp - cq_pos)[ei]  # player's own cq minus his position mean
    obs_dev = (obs_tmxg - pos_pred)[ei]  # observed teammate-xg minus the position-level prediction
    own_cal_corr = wpearson(own_dev, obs_dev, w[ei])
    # informational: mean-centered slope of the observed deviation on the cq-driven prediction
    # deviation (honest magnitude read; a level offset from the dropped qc bump makes it noisy, so it
    # is reported, not gated)
    dpred = (pp_pred - pos_pred)[ei]
    ww = w[ei] / w[ei].sum()
    mdp, mdo = float(np.sum(ww * dpred)), float(np.sum(ww * obs_dev))
    cal_slope = float(
        np.sum(ww * (dpred - mdp) * (obs_dev - mdo)) / max(np.sum(ww * (dpred - mdp) ** 2), 1e-12)
    )

    # noise-robust verdict: a paired bootstrap over eligible players. A bare Δcorr>0 rubber-stamps
    # noise (a random cq clears it ~half the time); require the improvement to survive resampling.
    B, ne = 2000, len(ei)
    rng = np.random.RandomState(12345)
    dboot = np.empty(B)
    for b in range(B):
        j = ei[rng.randint(0, ne, ne)]  # resample eligible players with replacement
        dboot[b] = wpearson(pp_pred[j], obs_tmxg[j], w[j]) - wpearson(pos_pred[j], obs_tmxg[j], w[j])
    p_boot = float(np.mean(dboot > 0))
    lo, hi = float(np.percentile(dboot, 5)), float(np.percentile(dboot, 95))
    dcorr = res["per-player"]["tmxg_corr"] - res["position-level"]["tmxg_corr"]
    # PASS iff the per-player rank edge is real (survives resampling) AND the own-cq signal is
    # directionally honest (not a linemate-clustering mirage)
    verdict = "PASS" if (p_boot >= 0.95 and lo > 0 and own_cal_corr > 0.15) else "FAIL"
    res["own_cal_corr"] = own_cal_corr
    res["cal_slope"] = cal_slope

    print(f"\n{'candidate':16s} {'tmxg-corr':>10s} {'tmxg-MAE':>9s} {'row-dev/1k':>11s}")
    for name in ("position-level", "per-player", "naive_last_season"):
        r = res[name]
        rd = f"{r['row_dev_per_1k']:11.3f}" if "row_dev_per_1k" in r else f"{'—':>11s}"
        print(f"{name:16s} {r['tmxg_corr']:10.3f} {r['tmxg_mae']:9.4f} {rd}")
    print(
        f"\n[qgate] Δcorr(per-player − position) = {dcorr:+.4f}  "
        f"bootstrap 90% CI [{lo:+.4f}, {hi:+.4f}]  P(per-player wins) = {p_boot:.3f}"
    )
    print(
        f"[qgate] own-cq calibration corr = {own_cal_corr:+.3f} (>.15 ⇒ signal real)  "
        f"centered slope = {cal_slope:+.3f}"
    )
    print(f"[qgate] GATE: {verdict}  (PASS ⇔ CI-low > 0, P(win) ≥ .95, own-cq corr > .15)")

    out = {
        "train": [int(s) for s in ts["train"]],
        "target": int(target),
        "count_model": count_model,
        "n_eligible": int(elig.sum()),
        "min_tmshots": MIN_TMSHOTS_EVAL,
        "cq_std_F": float(cq_t[~isD_t].std()),
        "cq_std_D": float(cq_t[isD_t].std()),
        "candidates": res,
        "delta_corr": dcorr,
        "boot_ci90": [lo, hi],
        "boot_p_win": p_boot,
        "own_cal_corr": own_cal_corr,
        "cal_slope": cal_slope,
        "verdict": verdict,
    }
    path = C.MODELS / f"qgate_{target}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"  -> {path}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Held-out-season predictive harness (generative model)")
    p.add_argument("--train-through", type=int, default=2024, help="last training season")
    p.add_argument("--target", type=int, default=None, help="held-out season (default: next)")
    p.add_argument("--count", choices=["poisson", "nb"], default="nb")
    p.add_argument("--spg-scale", type=float, default=1.0)
    p.add_argument(
        "--rescore",
        action="store_true",
        help="reuse the cached training side (holdout_fit_<train>.npz) — skip the fit",
    )
    p.add_argument(
        "--ma-anchor-scale",
        type=float,
        default=1.0,
        help="sweep candidate: scale the MA bucket's assist-anchor weight",
    )
    p.add_argument(
        "--ma-create-prior",
        type=float,
        default=None,
        help="sweep candidate: create prior SD for the MA bucket",
    )
    p.add_argument(
        "--ma-def-prior",
        type=float,
        default=None,
        help="sweep candidate: def prior SD for the MA bucket",
    )
    p.add_argument(
        "--ev-anchor-scale",
        type=float,
        default=1.0,
        help="sweep candidate: scale the EV bucket's assist-anchor weight (Kapanen class, §5e)",
    )
    p.add_argument(
        "--create-prior-center",
        choices=["position-mean"],
        default=None,
        help="sweep candidate: re-center the EV create ridge on the F/D position mean (§5e lever 3)",
    )
    p.add_argument(
        "--quality-gate",
        action="store_true",
        help="go/no-go: does per-player create_qual beat the position pair at predicting held-out "
        "teammate xG? (fits create_qual=True; uses the production MA/EV anchor settings)",
    )
    args = p.parse_args(argv)
    if args.quality_gate:
        quality_gate(
            args.train_through,
            args.target,
            count_model=args.count,
            rescore=args.rescore,
        )
        return
    evaluate(
        args.train_through,
        args.target,
        count_model=args.count,
        spg_scale=args.spg_scale,
        rescore=args.rescore,
        ma_anchor_scale=args.ma_anchor_scale,
        ma_create_prior_sd=args.ma_create_prior,
        ma_def_prior_sd=args.ma_def_prior,
        ev_anchor_scale=args.ev_anchor_scale,
        create_prior_center=args.create_prior_center,
    )


if __name__ == "__main__":
    main()
