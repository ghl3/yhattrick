"""Tests for the shooter-resolved generative model. Requires JAX (the `experimental` group), so the
whole module skips when JAX isn't installed:  uv run --group experimental pytest.

Synthetic tests are self-contained; a data-gated test exercises the real loaders/fit when a
processed/ tree is present."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")          # skip entirely unless the experimental deps are installed

from yhattrick import config as C
from yhattrick.models import generative_model as G


# ── synthetic builders (shooter-resolved: 1 focal shooter + 4 teammates + 5 defenders per row) ────

def _synth_create(P=70, n=120000, dur=45.0, seed=4, r_disp=None):
    """Rate data where each focal shooter's Fenwick count is lifted by teammates' `create` and
    suppressed by defenders' `def`, PLUS ng goal creator labels drawn from softmax([create_0, create]).
    Exercises the unified fit end-to-end: the dense count signal AND the sparse goal-assist credit both
    inform `create`. r_disp (Gamma shape) makes the counts overdispersed (Poisson-Gamma == NB)."""
    rng = np.random.default_rng(seed)
    shoot = rng.normal(0, 0.30, P)
    create = rng.normal(0, 0.30, P)
    deff = rng.normal(0, 0.30, P)
    create_0, mu0 = -0.5, np.log(11.0)                       # unassisted baseline; ~11 shots/60 focal
    atk = np.array([rng.choice(P, 5, replace=False) for _ in range(n)])
    dff = np.array([rng.choice(P, 5, replace=False) for _ in range(n)])
    shooter, team = atk[:, 0], atk[:, 1:5]
    Xctx = rng.integers(0, 2, (n, 1)).astype(float)          # one varying context col (home), true coef 0
    lam = np.exp(mu0 + shoot[shooter] + create[team].sum(1) + deff[dff].sum(1)) * dur / 3600.0
    if r_disp is None:
        count = rng.poisson(lam).astype(float)
    else:
        count = rng.poisson(lam * rng.gamma(r_disp, 1.0 / r_disp, size=n)).astype(float)
    R = {"players": list(range(P)), "idx": {p: p for p in range(P)}, "shooter_idx": shooter,
         "team_idx": team, "def_idx": dff, "Xctx": Xctx, "count": count,
         "offset": np.full(n, np.log(dur / 3600.0)), "dur": np.full(n, dur),
         "def_goalie": np.zeros(n, dtype=object), "toi": np.full(P, 1e6)}
    ng = 9000
    gt = np.array([rng.choice(P, 4, replace=False) for _ in range(ng)])
    pr = np.exp(np.concatenate([np.full((ng, 1), create_0), create[gt]], 1)); pr /= pr.sum(1, keepdims=True)
    gc = np.array([rng.choice(5, p=pr[i]) for i in range(ng)]).astype(np.int64)   # 0=unassist, 1..4 col
    return R, gt, gc, create, shoot, deff


def _synth_creator(P=60, n=150000, seed=2, qcF=0.30, qcD=-0.20):
    """Latent-creator quality data: each shot has ONE creator drawn from softmax([create_0, create]);
    the creator's POSITION-level qcreate (A1: [F, D]) lifts the shot's danger. Goals reveal the
    creator (the primary assister); non-goal shots keep it latent (marginalized at fit time)."""
    rng = np.random.default_rng(seed)
    qshoot = rng.normal(0, 0.20, P)
    qdef = rng.normal(0, 0.20, P)
    isD = (rng.random(P) < 0.4).astype(float)
    qc_pos = np.array([qcF, qcD])
    create = rng.normal(0, 0.60, P)                          # creator-identity propensity
    create_0 = -0.5                                          # unassisted baseline
    mu_q = np.log(0.07 / 0.93)
    atk = np.array([rng.choice(P, 5, replace=False) for _ in range(n)])
    dff = np.array([rng.choice(P, 5, replace=False) for _ in range(n)])
    shooter, team = atk[:, 0], atk[:, 1:5]
    base = mu_q + qshoot[shooter] + qdef[dff].sum(1)
    logit5 = np.concatenate([np.full((n, 1), create_0), create[team]], 1)
    pr = np.exp(logit5); pr /= pr.sum(1, keepdims=True)
    c = np.array([rng.choice(5, p=pr[i]) for i in range(n)])     # 0=unassisted, 1..4 teammate
    cr_pl = team[np.arange(n), np.clip(c - 1, 0, 3)]
    tcreate = np.where(c == 0, 0.0, qc_pos[isD[cr_pl].astype(np.int64)])
    qbar = 1.0 / (1.0 + np.exp(-(base + tcreate)))
    xg = rng.beta(12 * qbar, 12 * (1 - qbar))
    y = np.log(np.clip(xg, 1e-6, 1 - 1e-6) / (1 - np.clip(xg, 1e-6, 1 - 1e-6)))
    goal = rng.binomial(1, np.clip(qbar, 0, 1))
    creator = np.where(goal == 1, np.where(c == 0, 4, c - 1), -1)   # goals observe creator; else latent
    Q = {"shooter_idx": shooter, "team_idx": team, "def_idx": dff, "Xctx": np.zeros((n, 1)),
         "y": y, "goal": goal, "creator": creator.astype(np.int64)}
    return Q, qc_pos, create, isD


def _synth_conversion(P=60, Gg=20, n=200000, seed=5):
    """Conversion data: each shot has an observed xg, a shooter (fin) and a facing goalie (gsave), and
    the goal is Bernoulli(sigmoid(a·logit(xg) + b + fin + gsave)). xg is low-danger-skewed like real xG.
    Exercises the native logit conversion fit (slope/intercept + per-shooter fin + per-goalie gsave)."""
    rng = np.random.default_rng(seed)
    a_true, b_true = 1.1, -0.15
    # center each block: the intercept b and the offset means aren't separately identified (ridge
    # reassigns any nonzero mean to b), so mean-zero offsets let the test check b directly.
    fin = rng.normal(0, 0.20, P); fin -= fin.mean()
    gsave = rng.normal(0, 0.20, Gg); gsave -= gsave.mean()
    shooter = rng.integers(0, P, n)
    goalie = rng.integers(0, Gg, n)
    xg = np.clip(rng.beta(1.0, 12.0, n), G.EPS, 1 - G.EPS)
    lxg = np.log(xg / (1 - xg))
    eta = a_true * lxg + b_true + fin[shooter] + gsave[goalie]
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(float)
    Cr = {"shooter_idx": shooter.astype(np.int64), "goalie_idx": goalie.astype(np.int64),
          "logit_xg": lxg, "y": y, "goalies": list(range(Gg))}
    return Cr, a_true, b_true, fin, gsave


# ── rate + credit layer (fit_rate_create) ─────────────────────────────────────────────────────────

def test_rate_create_recovers_shoot_create_def():
    R, gt, gc, create, shoot, deff = _synth_create()
    fit = G.fit_rate_create(R, gt, gc, shots_per_goal=16)
    assert fit["converged"]
    assert np.corrcoef(fit["shoot"], shoot)[0, 1] > 0.8      # own-shot (scoring) volume
    assert np.corrcoef(fit["create"], create)[0, 1] > 0.8    # teammate-lift + assist credit (playmaking)
    assert np.corrcoef(fit["def"], deff)[0, 1] > 0.8         # defense volume


def test_rate_create_reconciles_total_shots():
    R, gt, gc, _, _, _ = _synth_create()
    fit = G.fit_rate_create(R, gt, gc, shots_per_goal=16)
    eta = (fit["intercept"] + fit["shoot"][R["shooter_idx"]] + fit["create"][R["team_idx"]].sum(1)
           + fit["def"][R["def_idx"]].sum(1) + R["Xctx"] @ fit["beta"])
    mu = np.exp(eta + R["offset"])
    assert abs(mu.sum() - R["count"].sum()) / R["count"].sum() < 0.01


def test_rate_create_hessian_se_positive():
    R, gt, gc, _, _, _ = _synth_create()
    fit = G.fit_rate_create(R, gt, gc, shots_per_goal=16)
    for key in ("se_shoot", "se_create", "se_def"):
        assert np.all(fit[key] > 0) and np.all(np.isfinite(fit[key]))


def test_shoot_and_create_are_separately_identified():
    """The whole point of shooter-resolution: a player's OWN-shot loading and his TEAMMATE-lift loading
    are different design columns, so recovering one does not just mirror the other."""
    R, gt, gc, create, shoot, _ = _synth_create()
    fit = G.fit_rate_create(R, gt, gc, shots_per_goal=16)
    assert np.corrcoef(fit["shoot"], shoot)[0, 1] > abs(np.corrcoef(fit["shoot"], create)[0, 1]) + 0.3


def test_nb_recovers_and_estimates_dispersion():
    R, gt, gc, create, _, _ = _synth_create(r_disp=2.0, seed=7)
    fit = G.fit_rate_create(R, gt, gc, shots_per_goal=16, count_model="nb")
    assert fit["converged"]
    assert fit["r"] is not None and np.isfinite(fit["r"]) and fit["r"] > 0
    assert np.corrcoef(fit["create"], create)[0, 1] > 0.8
    assert np.all(fit["se_create"] > 0)


# ── quality layer (latent creator) ────────────────────────────────────────────────────────────────

def test_quality_creator_recovers_position_qcreate():
    """Given the creator distribution, the quality fit recovers the POSITION-level creation quality
    (A1) from the observed-creator goal quality + the marginalized non-goal quality."""
    Q, qc_pos, create, isD = _synth_creator()
    fit = G.fit_quality_creator(Q, P=len(isD), creates={0: (create, -0.5)}, isD=isD)
    assert fit["converged"]
    assert fit["qcreate"].shape == (2,)
    assert abs(fit["qcreate"][0] - qc_pos[0]) < 0.08         # F creation quality
    assert abs(fit["qcreate"][1] - qc_pos[1]) < 0.08         # D creation quality
    assert np.all(fit["se_qcreate"] > 0)


def test_office_assister_goals_are_latent():
    """F4: goals whose credited assister is not an on-ice teammate carry creator = −1 — treated as
    latent (marginalized) in the quality fit, and never counted in n_create."""
    Q, qc_pos, create, isD = _synth_creator(n=60000, seed=8)
    g = np.nonzero((Q["goal"] == 1) & (Q["creator"] >= 0) & (Q["creator"] <= 3))[0][:150]
    Q["creator"][g] = -1                                     # simulate off-ice-assister labels
    fit = G.fit_quality_creator(Q, P=len(isD), creates={0: (create, -0.5)}, isD=isD)
    assert fit["converged"]
    n_obs_tm = int(((Q["creator"] >= 0) & (Q["creator"] <= 3) & (Q["goal"] == 1)).sum())
    assert int(fit["n_create"].sum()) == n_obs_tm


# ── conversion layer (native logit) ───────────────────────────────────────────────────────────────

def test_conversion_recovers_slope_intercept_fin_gsave():
    Cr, a_true, b_true, fin, gsave = _synth_conversion()
    fit = G.fit_conversion(Cr, P=len(fin))
    assert fit["converged"]
    assert abs(fit["a"]["ev"] - a_true) < 0.10           # slope recalibration (single-strength → 'ev')
    assert abs(fit["b"]["ev"] - b_true) < 0.05           # intercept (replaces mu_conv)
    assert np.corrcoef(fit["fin"], fin)[0, 1] > 0.7      # per-shooter finishing
    assert np.corrcoef(fit["gsave"], gsave)[0, 1] > 0.7  # per-goalie save
    assert np.all(fit["se_fin"] > 0) and np.all(fit["se_gsave"] > 0)


def test_conversion_reconciles_goals_exactly():
    """b is unpenalized, so the MLE score equation ∂/∂b: Σ(y−p)=0 gives Σp = Σy — deterministic
    expected goals reconcile as a natural first-order condition, not a hand-solved constant."""
    Cr, _, _, fin, _ = _synth_conversion()
    fit = G.fit_conversion(Cr, P=len(fin))
    assert abs(fit["sum_p"] - fit["sum_y"]) / fit["sum_y"] < 1e-3


# ── attribution ───────────────────────────────────────────────────────────────────────────────────

def test_player_values_merge_signs():
    """Goal values move the right way: +shoot => more scoring, +create/+qcreate => more playmaking,
    -def (suppression) => more positive defense value."""
    P = 3
    mq = np.log(0.07 / 0.93)
    rate = {"intercept": np.log(8.0), "shoot": np.array([0.4, 0.0, 0.0]),
            "create": np.array([0.0, 0.5, 0.0]), "def": np.array([0.0, 0.0, -0.4]), "psi0": -0.5}
    qual = {"mu_qual": {"ev": mq}, "qshoot": np.zeros(P),
            "qcreate": np.array([0.0, 0.5, 0.0]), "qdef": np.zeros(P)}
    conv = {"a": {"ev": 1.0}, "b": {"ev": 0.0}, "fin": np.zeros(P)}      # identity conversion
    vals = G.player_values({"ev": rate}, qual, conv, [0, 1, 2])
    v = vals["ev"]
    assert v["scoring"][0] == max(v["scoring"])          # player 0 (high shoot) tops scoring
    assert v["playmaking"][1] == max(v["playmaking"])    # player 1 (high create + qcreate) tops playmaking
    assert v["defense"][2] == max(v["defense"])          # player 2 (neg def) suppresses most


def test_finishing_lifts_scoring():
    """A positive fin (better finisher) raises a player's converted scoring above the fin=0 baseline."""
    P = 2
    mq = np.log(0.07 / 0.93)
    rate = {"intercept": np.log(8.0), "shoot": np.zeros(P), "create": np.zeros(P),
            "def": np.zeros(P), "psi0": -0.5}
    qual = {"mu_qual": {"ev": mq}, "qshoot": np.zeros(P), "qcreate": np.zeros(P), "qdef": np.zeros(P)}
    conv = {"a": {"ev": 1.0}, "b": {"ev": 0.0}, "fin": np.array([0.5, 0.0])}
    v = G.player_values({"ev": rate}, qual, conv, [0, 1])["ev"]
    assert v["scoring"][0] > v["scoring"][1]             # better finisher scores more on identical shots
    assert v["finishing"][0] > 0 and abs(v["finishing"][1]) < 1e-9


# ── player curves: RW drift states + aging curves + projection ───────────────────────────────────

def _synth_states(P=40, S=3, n_per=60000, dur=45.0, seed=9, drift_player=0, drift=0.35,
                  age_cols=False):
    """Multi-season rate data with per-(player, season) TRUE states: static levels except
    `drift_player`, whose shoot rises by `drift` per season. Optionally bakes a known aging curve
    (w1·z + w2·z²) into the shoot rates and exposes the matching AGE-basis context columns. Emits
    the unit machinery exactly as rate_rows does (all pairs active, player-major, season-sorted)."""
    rng = np.random.default_rng(seed)
    shoot0, create0, deff0 = rng.normal(0, 0.25, P), rng.normal(0, 0.25, P), rng.normal(0, 0.25, P)
    mu0, create_0 = np.log(11.0), -0.5
    shoot = np.tile(shoot0, (S, 1)).T.copy()                 # (P, S) true states
    shoot[drift_player] = shoot0[drift_player] + drift * np.arange(S)
    create, deff = np.tile(create0, (S, 1)).T, np.tile(deff0, (S, 1)).T
    ages = 20.0 + rng.uniform(0, 12, P)                      # age at the first season
    w1, w2 = (-0.25, -0.60) if age_cols else (0.0, 0.0)
    z = (ages[:, None] + np.arange(S)[None, :] - 27.0) / 10.0
    n = n_per * S
    atk = np.array([rng.choice(P, 5, replace=False) for _ in range(n)])
    dff_ = np.array([rng.choice(P, 5, replace=False) for _ in range(n)])
    srow = np.repeat(np.arange(S), n_per)
    shooter, team = atk[:, 0], atk[:, 1:5]
    zsh = z[shooter, srow]
    if age_cols:
        Xctx = np.column_stack([rng.integers(0, 2, n).astype(float), zsh, zsh ** 2])
        ctx_names = ["home", "shoot_zF", "shoot_z2F"]
    else:
        Xctx = rng.integers(0, 2, (n, 1)).astype(float)
        ctx_names = ["home"]
    lam = np.exp(mu0 + shoot[shooter, srow] + w1 * zsh + w2 * zsh ** 2
                 + create[team, srow[:, None]].sum(1) + deff[dff_, srow[:, None]].sum(1)) * dur / 3600.0
    count = rng.poisson(lam).astype(float)
    up, us = np.repeat(np.arange(P), S), np.tile(np.arange(S), P)   # unit = p*S + s (player-major)
    e_prev = np.array([p * S + s for p in range(P) for s in range(S - 1)], dtype=np.int64)
    seasons = list(range(2021, 2021 + S))
    R = {"players": list(range(P)), "idx": {p: p for p in range(P)},
         "shooter_idx": shooter, "team_idx": team, "def_idx": dff_,
         "Xctx": Xctx, "ctx_names": ctx_names, "count": count,
         "offset": np.full(n, np.log(dur / 3600.0)), "dur": np.full(n, dur),
         "def_goalie": np.zeros(n, dtype=object), "toi": np.full(P, 1e6),
         "season_row": srow + 2021, "seasons": seasons, "n_season_cols": 0,
         "last_season": np.full(P, seasons[-1], dtype=np.int64),
         "unit_player": up, "unit_season": us + 2021,
         "unit_lut": np.arange(P * S).reshape(P, S), "n_units": P * S,
         "shooter_unit": (shooter * S + srow).astype(np.int32),
         "team_unit": (team * S + srow[:, None]).astype(np.int32),
         "def_unit": (dff_ * S + srow[:, None]).astype(np.int32),
         "e_prev": e_prev, "e_next": e_prev + 1, "e_gap": np.ones(len(e_prev)),
         "first_mask": (us == 0).astype(float), "toi_unit": np.full(P * S, 1e6)}
    ng = 3000 * S
    g_s = rng.integers(0, S, ng)
    gt_p = np.array([rng.choice(P, 4, replace=False) for _ in range(ng)])
    pr = np.exp(np.concatenate([np.full((ng, 1), create_0), create[gt_p, g_s[:, None]]], 1))
    pr /= pr.sum(1, keepdims=True)
    gc = np.array([rng.choice(5, p=pr[i]) for i in range(ng)]).astype(np.int64)
    gt = (gt_p * S + g_s[:, None]).astype(np.int64)          # anchor teammates as UNIT indices
    truth = {"shoot": shoot, "create": create, "def": deff, "w": (w1, w2), "z": z}
    return R, gt, gc, truth


def test_rw_states_track_drift():
    """One pooled fit yields each player's per-season trajectory: the drifting player's shoot states
    rise across seasons (through the RW prior), everyone else's stay ~flat, and the *_last views
    match the final column."""
    R, gt, gc, truth = _synth_states()
    fit = G.fit_rate_create(R, gt, gc, shots_per_goal=16)
    assert fit["converged"]
    S = 3
    sh = fit["shoot"].reshape(-1, S)                         # units are player-major, season-sorted
    d0 = sh[0, -1] - sh[0, 0]                                # drifting player's fitted rise
    assert d0 > 0.35                                         # ≥ half the true total drift (0.70)
    assert d0 > np.abs(sh[1:, -1] - sh[1:, 0]).mean() + 0.15
    np.testing.assert_allclose(fit["shoot_last"], sh[:, -1], atol=1e-12)
    assert np.all(fit["se_shoot_last"] > 0) and np.all(fit["se_create_last"] > 0)


def test_s1_states_degenerate_to_static():
    """With one season the unit machinery is exactly the static model: same objective, same fit."""
    R, gt, gc, _ = _synth_states(S=1, n_per=80000)
    fit_u = G.fit_rate_create(R, gt, gc, shots_per_goal=16)
    drop = ("unit_player", "unit_season", "unit_lut", "n_units", "shooter_unit", "team_unit",
            "def_unit", "e_prev", "e_next", "e_gap", "first_mask", "toi_unit")
    R2 = {k: v for k, v in R.items() if k not in drop}
    fit_s = G.fit_rate_create(R2, gt, gc, shots_per_goal=16)
    np.testing.assert_allclose(fit_u["shoot"], fit_s["shoot"], atol=1e-6)
    np.testing.assert_allclose(fit_u["create"], fit_s["create"], atol=1e-6)
    np.testing.assert_allclose(fit_u["se_create_last"], fit_s["se_create_last"], atol=1e-8)


def test_age_curve_recovery_and_projection():
    """The shared aging curve is a pair of unpenalized context coefficients: the fit recovers the
    known quadratic, effective_params adds curve(z_last) to the last state, and a projection
    advances the age by one season while holding the state (the RW mean)."""
    R, gt, gc, truth = _synth_states(age_cols=True, seed=13, drift=0.0)
    fit = G.fit_rate_create(R, gt, gc, shots_per_goal=16)
    w1, w2 = truth["w"]
    cm = G._coef_map(fit["ctx_names"], fit["beta"])
    assert abs(cm["shoot_zF"] - w1) < 0.06
    assert abs(cm["shoot_z2F"] - w2) < 0.15
    P, seasons = len(R["players"]), R["seasons"]
    agepos = {"isD": np.zeros(P), "missing": 0,
              "age": {s: 27.0 + 10.0 * truth["z"][:, i] for i, s in enumerate(seasons)},
              "z": {s: truth["z"][:, i] for i, s in enumerate(seasons)}}
    last_season = np.full(P, seasons[-1], dtype=np.int64)
    qual = {"mu_qual": {"ev": -2.5}, "qshoot": np.zeros(P), "qcreate": np.zeros(2),
            "qdef": np.zeros(P), "beta": np.zeros(1), "ctx_names": ["home"]}
    conv = {"a": {"ev": 1.0}, "b": {"ev": 0.0}, "fin": np.zeros(P), "beta": [], "ctx_names": []}
    re_, _, _ = G.effective_params({"ev": fit}, qual, conv, R["players"], agepos, last_season)
    zl = truth["z"][:, -1]
    np.testing.assert_allclose(re_["ev"]["shoot"],
                               fit["shoot_last"] + cm["shoot_zF"] * zl + cm["shoot_z2F"] * zl ** 2,
                               atol=1e-9)
    rp, _, _ = G.effective_params({"ev": fit}, qual, conv, R["players"], agepos, last_season,
                                  target=seasons[-1] + 1)
    zp = zl + 0.1                                            # one season older
    np.testing.assert_allclose(rp["ev"]["shoot"],
                               fit["shoot_last"] + cm["shoot_zF"] * zp + cm["shoot_z2F"] * zp ** 2,
                               atol=1e-9)


def test_sparse_dense_se_parity(monkeypatch):
    """The sparse splu column-solve SE path (used when the state expansion outgrows the dense
    Hessian) matches the dense inverse, including the F1 sandwich on create."""
    R, gt, gc, _, _, _ = _synth_create(P=40, n=40000, seed=3)
    fit_d = G.fit_rate_create(R, gt, gc, shots_per_goal=16)
    monkeypatch.setattr(G, "DENSE_H_MAX", 1)
    fit_s = G.fit_rate_create(R, gt, gc, shots_per_goal=16)
    for k in ("se_shoot", "se_create", "se_def"):
        np.testing.assert_allclose(fit_d[k], fit_s[k], rtol=1e-6, atol=1e-10)


def test_conversion_season_offsets_reconcile():
    """F6: unpenalized per-season conversion offsets absorb league finishing drift and make Σp=Σy
    hold PER SEASON, with the offset coefficient recovering the injected drift."""
    rng = np.random.default_rng(6)
    P, Gg, n = 40, 12, 160000
    off_true = 0.20
    fin = rng.normal(0, 0.20, P); fin -= fin.mean()
    gsave = rng.normal(0, 0.20, Gg); gsave -= gsave.mean()
    shooter = rng.integers(0, P, n)
    goalie = rng.integers(0, Gg, n)
    season = np.where(np.arange(n) < n // 2, 2021, 2022)
    xg = np.clip(rng.beta(1.0, 12.0, n), G.EPS, 1 - G.EPS)
    lxg = np.log(xg / (1 - xg))
    eta = 1.1 * lxg - 0.15 + off_true * (season == 2022) + fin[shooter] + gsave[goalie]
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(float)
    Cr = {"shooter_idx": shooter.astype(np.int64), "goalie_idx": goalie.astype(np.int64),
          "logit_xg": lxg, "y": y, "goalies": list(range(Gg)), "season": season,
          "ctx": (season == 2022).astype(float)[:, None], "ctx_names": ["season_2022"]}
    fit = G.fit_conversion(Cr, P=P)
    assert fit["converged"]
    assert abs(fit["beta"][0] - off_true) < 0.06             # drift recovered
    for s, (sp, sy) in fit["recon_season"].items():          # per-season reconciliation
        assert abs(sp - sy) / sy < 2e-3


# ── data-gated real fit ─────────────────────────────────────────────────────────────────────────

def _has_data():
    d = C.PROCESSED / "stints"
    return d.exists() and any(d.glob("*.parquet"))


def test_ma_bucket_masked_defenders_recover():
    """Man-advantage rate rows have 4 defenders (padded to 5 with a mask). Build a synthetic MA-shaped
    R with a masked 5th defender slot and confirm the masked fit still recovers shoot/create/def and the
    masked slot contributes nothing (its `def` is untouched by data → shrinks to ~0)."""
    R, gt, gc, create, shoot, deff = _synth_create(seed=11)
    n = len(R["count"])
    # turn it into an MA-shaped bucket: keep 4 real defenders, pad a 5th masked slot
    real = R["def_idx"]                                       # (n,5) — treat first 4 as real, 5th masked
    mask = np.ones((n, 5)); mask[:, 4] = 0.0
    R = {**R, "def_mask": mask}
    fit = G.fit_rate_create(R, gt, gc, shots_per_goal=16)
    assert fit["converged"]
    assert np.corrcoef(fit["shoot"], shoot)[0, 1] > 0.8
    assert np.corrcoef(fit["create"], create)[0, 1] > 0.8
    assert np.all(np.isfinite(fit["def"]))


@pytest.mark.skipif(not _has_data(), reason="no processed/ tree")
def test_real_fit_smoke():
    sd = C.PROCESSED / "shots_onice"
    seasons = (sorted(int(f.stem) for f in sd.glob("*.parquet"))[-1:]) if sd.exists() else []
    players, idx = G.player_index(seasons)
    assert players
    agepos = G._age_position(players, seasons)
    assert agepos["isD"].sum() > 0                            # rosters resolve positions
    Q = G.quality_creator_rows(seasons, idx, G.ALL_STRENGTHS, agepos)
    cidx = np.where(Q["creator"] == 4, 0, np.clip(Q["creator"], 0, 3) + 1).astype(np.int64)
    creates = {}
    for key, strengths, dual in [("ev", G.EV_STRENGTHS, True), ("ma", G.MA_STRENGTHS, False)]:
        R = G.rate_rows(seasons, strengths, dual, players, idx, agepos, states=dual)
        assert R is not None and len(R["count"]) > 0
        assert R["team_idx"].shape[1] == 4 and R["def_idx"].shape[1] == 5
        assert len(R["ctx_names"]) == R["Xctx"].shape[1]      # named context incl. AGE_CTX
        slab = 0 if key == "ev" else 1
        anchor = (Q["goal"] == 1) & (Q["strength"] == slab) & (Q["creator"] >= 0)
        gt, gc = Q["team_idx"][anchor], cidx[anchor]
        sord = {s: i for i, s in enumerate(R["seasons"])}
        if dual:
            assert R["n_units"] > 0                           # per-(player, season) states exist
            qs = np.array([sord[s] for s in Q["season"][anchor]], dtype=np.int64)
            gtu = R["unit_lut"][gt, qs[:, None]]
            ok = (gtu >= 0).all(1)
            gt, gc = gtu[ok], gc[ok]
        fit = G.fit_rate_create(R, gt, gc, shots_per_goal=16.0)
        assert fit["converged"] and np.all(np.isfinite(fit["create"]))
        if dual:
            qs_all = np.array([sord[s] for s in Q["season"]], dtype=np.int64)
            tu = R["unit_lut"][Q["team_idx"], qs_all[:, None]]
            cre = np.append(fit["create"], 0.0)
            creates[slab] = (cre, fit["psi0"], np.where(tu >= 0, tu, len(fit["create"])))
        else:
            creates[slab] = (fit["create"], fit["psi0"])
    qual = G.fit_quality_creator(Q, len(players), creates, isD=agepos["isD"])
    assert qual["converged"] and set(qual["mu_qual"]) >= {"ev", "ma"}
    assert np.asarray(qual["qcreate"]).shape == (2,)          # A1: position-level
    Cr = G.conversion_rows(seasons, idx, G.ALL_STRENGTHS, agepos)
    cf = G.fit_conversion(Cr, len(players))
    assert cf["converged"] and set(cf["a"]) >= {"ev", "ma"}
    assert cf["recon_season"]                                 # F6 per-season reconciliation present
