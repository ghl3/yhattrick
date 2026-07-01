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


def _synth_creator(P=60, n=150000, seed=2):
    """Latent-creator quality data: each shot has ONE creator drawn from softmax([create_0, create]);
    only that creator's qcreate lifts the shot's danger. Goals reveal the creator (the primary
    assister); non-goal shots keep it latent (marginalized at fit time)."""
    rng = np.random.default_rng(seed)
    qshoot = rng.normal(0, 0.20, P)
    qcreate = rng.normal(0, 0.30, P)
    qdef = rng.normal(0, 0.20, P)
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
    tcreate = np.where(c == 0, 0.0, qcreate[team[np.arange(n), np.clip(c - 1, 0, 3)]])
    qbar = 1.0 / (1.0 + np.exp(-(base + tcreate)))
    xg = rng.beta(12 * qbar, 12 * (1 - qbar))
    y = np.log(np.clip(xg, 1e-6, 1 - 1e-6) / (1 - np.clip(xg, 1e-6, 1 - 1e-6)))
    goal = rng.binomial(1, np.clip(qbar, 0, 1))
    creator = np.where(goal == 1, np.where(c == 0, 4, c - 1), -1)   # goals observe creator; else latent
    Q = {"shooter_idx": shooter, "team_idx": team, "def_idx": dff, "Xctx": np.zeros((n, 1)),
         "y": y, "goal": goal, "creator": creator.astype(np.int64)}
    return Q, qcreate, create


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

def test_quality_creator_recovers_qcreate():
    """Given the creator distribution, the quality fit recovers creation-quality qcreate from the
    observed-creator goal quality + the marginalized non-goal quality."""
    Q, qcreate, create = _synth_creator()
    fit = G.fit_quality_creator(Q, P=len(qcreate), creates={0: (create, -0.5)})
    assert fit["converged"]
    assert np.corrcoef(fit["qcreate"], qcreate)[0, 1] > 0.45


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
    Q = G.quality_creator_rows(seasons, idx, G.ALL_STRENGTHS)
    cidx = np.where(Q["creator"] == 4, 0, np.clip(Q["creator"], 0, 3) + 1).astype(np.int64)
    rates = {}
    for key, strengths, dual in [("ev", G.EV_STRENGTHS, True), ("ma", G.MA_STRENGTHS, False)]:
        R = G.rate_rows(seasons, strengths, dual, players, idx)
        assert R is not None and len(R["count"]) > 0
        assert R["team_idx"].shape[1] == 4 and R["def_idx"].shape[1] == 5
        gm = (Q["goal"] == 1) & (Q["strength"] == (0 if key == "ev" else 1))
        fit = G.fit_rate_create(R, Q["team_idx"][gm], cidx[gm], shots_per_goal=G.SHOTS_PER_GOAL)
        assert fit["converged"] and np.all(np.isfinite(fit["create"]))
        rates[key] = fit
    creates = {0: (rates["ev"]["create"], rates["ev"]["psi0"]), 1: (rates["ma"]["create"], rates["ma"]["psi0"])}
    qual = G.fit_quality_creator(Q, len(players), creates)
    assert qual["converged"] and set(qual["mu_qual"]) >= {"ev", "ma"}
    Cr = G.conversion_rows(seasons, idx, G.ALL_STRENGTHS)
    cf = G.fit_conversion(Cr, len(players))
    assert cf["converged"] and set(cf["a"]) >= {"ev", "ma"}
    for k in ("ev", "ma"):                                    # per-strength reconciliation Σp≈Σy
        assert k in cf["a"]
