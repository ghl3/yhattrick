"""Tests for the card exporter's pure math: the WAR engine's closed-form replacement swaps,
GA/60 baselining, and percentile gating. (The full build() is data-gated and runs via its CLI.)"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

from yhattrick.models import generative_cards as GC


def _mini_rows(n=4000, P=30, seed=3):
    """Synthetic stint-side rows: 5 attackers, 5 defenders drawn from P players, 45 s each."""
    rng = np.random.default_rng(seed)
    picks = np.array([rng.choice(P, 10, replace=False) for _ in range(n)])
    return {"atk": picks[:, :5], "def": picks[:, 5:], "dmask": np.ones((n, 5)),
            "gsave": np.zeros(n), "dur": np.full(n, 45.0), "ctx": np.zeros((n, 5))}


def _repl_like(P, sh=0.0, cr=0.0, df=0.0, g_logit=-2.5, kq=1.0):
    return {"sh": np.full(P, sh), "cr": np.full(P, cr), "df": np.full(P, df),
            "g_logit": np.full(P, g_logit), "kq": np.full(P, kq)}


def test_war_bucket_signs_and_replacement_zero():
    """A player strictly better than replacement gets positive GAR in every slot type; a player AT
    replacement level gets ~0; a strictly worse player goes negative."""
    P = 30
    rows = _mini_rows(P=P)
    sh = np.zeros(P); cr = np.zeros(P); df = np.zeros(P)
    g_logit = np.full(P, -2.5); kq = np.ones(P)
    sh[1] = 0.4; cr[1] = 0.3; df[1] = -0.3                   # player 1: better everywhere
    sh[2] = -0.4; cr[2] = -0.2; df[2] = 0.3; kq[2] = 1.05    # player 2: worse everywhere
    repl = _repl_like(P)                                     # replacement == the average (zeros)
    ga, gd = GC.war_bucket(rows, P, sh, cr, df, g_logit, kq, rows["gsave"],
                           np.full(len(rows["dur"]), np.exp(2.4)), repl)
    gar = ga + gd
    assert gar[1] > 0.5                                      # clearly positive
    assert gar[2] < -0.3                                     # clearly negative
    others = np.delete(gar, [1, 2])
    assert np.abs(others).max() < 1e-9                       # at-replacement players are exactly 0


def test_war_bucket_swap_consistency():
    """The swap algebra matches a brute-force recomputation of E[GF] with the player replaced."""
    P = 12
    rng = np.random.default_rng(7)
    rows = _mini_rows(n=200, P=P, seed=9)
    sh, cr, df = rng.normal(0, 0.3, P), rng.normal(0, 0.2, P), rng.normal(0, 0.2, P)
    g_logit = rng.normal(-2.5, 0.2, P)
    kq = np.exp(rng.normal(0, 0.05, P))
    repl = _repl_like(P, sh=-0.2, cr=-0.1, df=0.1, g_logit=-2.7, kq=1.02)
    cx = np.full(len(rows["dur"]), np.exp(2.4))

    def e_gf(A, B, pvec):
        sh_, cr_, df_, gl_, kq_ = pvec
        out = 0.0
        for i in range(len(A)):
            crA = cr_[A[i]].sum(); dfB = df_[B[i]].sum(); K = np.prod(kq_[B[i]])
            T = np.sum(np.exp(sh_[A[i]] - cr_[A[i]]) * GC._sigmoid(gl_[A[i]]))
            out += cx[i] * np.exp(crA + dfB) * K * T * rows["dur"][i] / 3600.0
        return out

    ga, gd = GC.war_bucket(rows, P, sh, cr, df, g_logit, kq, rows["gsave"], cx, repl)
    # brute force for player 0: replace him wherever he appears, on both sides of the ledger
    p = 0
    base = e_gf(rows["atk"], rows["def"], (sh, cr, df, g_logit, kq))
    shr, crr, dfr, glr, kqr = (v.copy() for v in (sh, cr, df, g_logit, kq))
    shr[p], crr[p], dfr[p] = repl["sh"][p], repl["cr"][p], repl["df"][p]
    glr[p], kqr[p] = repl["g_logit"][p], repl["kq"][p]
    swapped = e_gf(rows["atk"], rows["def"], (shr, crr, dfr, glr, kqr))
    # ga[p] books his attacker-slot GF loss; gd[p] books the defender-slot GA saving — the brute
    # force difference of THIS side's totals equals ga[p] − gd[p] (defender swap raises E here)
    assert abs((base - swapped) - (ga[p] - gd[p])) < 1e-8


def test_pct_within_gates_and_groups():
    v = np.array([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])
    isD = np.array([0, 0, 0, 0, 1, 1, 1, 1.0])
    elig = np.array([True, True, True, False, True, True, True, True])
    p = GC.pct_within(v, elig, isD)
    assert np.isnan(p[3])                                    # ineligible → no percentile
    assert p[0] == 0 and p[2] == 100                         # within-F ranking
    assert p[4] == 0 and p[7] == 100                         # within-D ranking, separate scale


def test_ga60_baseline_zero_mean():
    """GA/60 is exactly TOI-weighted zero within each position group by construction."""
    rng = np.random.default_rng(1)
    n = 300
    t = {"isD": (rng.random(n) < 0.35).astype(float),
         "toi_ev": rng.uniform(6000, 90000, n),
         "toi_pp": np.zeros(n), "toi_pk": np.zeros(n)}
    vals = {"sc": rng.normal(0.4, 0.1, n), "pm": rng.normal(0.5, 0.2, n),
            "df": rng.normal(0.0, 0.2, n),
            "pp_sc": np.zeros(n), "pp_pm": np.zeros(n), "pk_df": np.zeros(n)}
    base = GC.baselines(t, vals, kap=1.0)
    ga = GC.ga60_of(t, base, vals["sc"], vals["pm"], vals["df"], 1.0)
    for gm in (t["isD"] < 0.5, t["isD"] > 0.5):
        m = gm & (t["toi_ev"] >= GC.EV_GATE)
        assert abs(np.average(ga[m], weights=t["toi_ev"][m])) < 1e-9
