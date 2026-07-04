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
    return {
        "atk": picks[:, :5],
        "def": picks[:, 5:],
        "dmask": np.ones((n, 5)),
        "gsave": np.zeros(n),
        "dur": np.full(n, 45.0),
        "ctx": np.zeros((n, 5)),
    }


PSI0 = -1.3


def _repl_like(P, sh=0.0, cr=0.0, df=0.0, g=0.08, kq=1.0):
    return {
        "sh": np.full(P, sh),
        "cr": np.full(P, cr),
        "df": np.full(P, df),
        "g0": np.full(P, g),
        "gF": np.full(P, g * 0.95),
        "gD": np.full(P, g * 0.8),
        "kq": np.full(P, kq),
    }


def _gcl(P, g=0.08):
    """Creator-class conversions: unassisted reference; created shots slightly less dangerous."""
    return (np.full(P, g), np.full(P, g * 0.95), np.full(P, g * 0.8))


def _e_gf_bruteforce(rows, pvec, cx, psi0, isD):
    """Reference E[GF]: per row, per shooter, exact creator mix over his four teammates."""
    sh_, cr_, df_, (g0_, gF_, gD_), kq_ = pvec
    A, B, gs = rows["atk"], rows["def"], rows["gsave"]
    out = 0.0
    p0 = np.exp(psi0)
    for i in range(len(A)):
        crA = cr_[A[i]].sum()
        dfB = df_[B[i]].sum()
        K = np.prod(kq_[B[i]])
        T = 0.0
        for k in range(5):
            j = A[i][k]
            mates = np.delete(A[i], k)
            w = np.exp(cr_[mates])
            wf = w[isD[mates] < 0.5].sum()
            wd = w[isD[mates] > 0.5].sum()
            gbar = (p0 * g0_[j] + wf * gF_[j] + wd * gD_[j]) / (p0 + w.sum())
            T += np.exp(sh_[j] - cr_[j]) * gbar * np.exp((1.0 - g0_[j]) * gs[i])
        out += cx[i] * np.exp(crA + dfB) * K * T * rows["dur"][i] / 3600.0
    return out


def test_war_bucket_signs_and_replacement_zero():
    """A player strictly better than replacement gets positive GAR in every slot type; a player AT
    replacement level gets ~0; a strictly worse player goes negative."""
    P = 30
    rows = _mini_rows(P=P)
    isD = (np.arange(P) % 3 == 0).astype(float)
    sh = np.zeros(P)
    cr = np.zeros(P)
    df = np.zeros(P)
    kq = np.ones(P)
    sh[1] = 0.4
    cr[1] = 0.3
    df[1] = -0.3  # player 1: better everywhere
    sh[2] = -0.4
    cr[2] = -0.2
    df[2] = 0.3
    kq[2] = 1.05  # player 2: worse everywhere
    repl = _repl_like(P)  # replacement == the average (zeros)
    ga, gd, E = GC.war_bucket(
        rows,
        P,
        sh,
        cr,
        df,
        _gcl(P),
        kq,
        rows["gsave"],
        np.full(len(rows["dur"]), np.exp(2.4)),
        repl,
        PSI0,
        isD,
    )
    gar = ga + gd
    assert gar[1] > 0.5  # clearly positive
    assert gar[2] < -0.3  # clearly negative
    others = np.delete(gar, [1, 2])
    # at-replacement players are ~0 up to the frozen cross-π term (their creator-mix weight on the
    # swapped players' rows) — second-order small
    assert np.abs(others).max() < 0.02
    assert np.all(E > 0) and len(E) == len(rows["dur"])


def test_war_bucket_swap_consistency():
    """The swap algebra matches a brute-force recomputation of E[GF] with the player replaced
    (holding teammates' creator mixes fixed, as the engine documents)."""
    P = 12
    rng = np.random.default_rng(7)
    rows = _mini_rows(n=200, P=P, seed=9)
    isD = (rng.random(P) < 0.4).astype(float)
    sh, cr, df = rng.normal(0, 0.3, P), rng.normal(0, 0.2, P), rng.normal(0, 0.2, P)
    g0 = 0.08 * np.exp(rng.normal(0, 0.15, P))
    gcl = (g0, g0 * 0.95, g0 * 0.8)
    kq = np.exp(rng.normal(0, 0.05, P))
    rows["gsave"] = rng.normal(0, 0.05, len(rows["dur"]))
    repl = _repl_like(P, sh=-0.2, cr=-0.1, df=0.1, g=0.07, kq=1.02)
    cx = np.full(len(rows["dur"]), np.exp(2.4))

    ga, gd, _E = GC.war_bucket(rows, P, sh, cr, df, gcl, kq, rows["gsave"], cx, repl, PSI0, isD)
    # brute force for player 0: replace him wherever he appears, on both sides of the ledger —
    # with his CREATOR-MIX contribution to teammates held at fitted values (the engine's
    # documented frozen cross-π), i.e. the swap only touches his own term + shared exponents
    p = 0
    base = _e_gf_bruteforce(rows, (sh, cr, df, gcl, kq), cx, PSI0, isD)
    # swapped world: p's own params replaced everywhere EXCEPT inside other shooters' mixes
    A, B, gs = rows["atk"], rows["def"], rows["gsave"]
    p0 = np.exp(PSI0)
    swapped = 0.0
    for i in range(len(A)):
        in_atk = p in set(A[i])
        crA = cr[A[i]].sum()
        dfB = df[B[i]].sum()
        K = np.prod(kq[B[i]])
        if in_atk:
            crA = crA - cr[p] + repl["cr"][p]
        if p in set(B[i]):
            dfB = dfB - df[p] + repl["df"][p]
            K = K / kq[p] * repl["kq"][p]
        T = 0.0
        for k in range(5):
            j = A[i][k]
            mates = np.delete(A[i], k)
            w = np.exp(cr[mates])  # frozen cross-π: fitted cr everywhere
            wf = w[isD[mates] < 0.5].sum()
            wd = w[isD[mates] > 0.5].sum()
            Z = p0 + w.sum()
            if j == p:
                gbar = (p0 * repl["g0"][p] + wf * repl["gF"][p] + wd * repl["gD"][p]) / Z
                T += (
                    np.exp(repl["sh"][p] - repl["cr"][p])
                    * gbar
                    * np.exp((1 - repl["g0"][p]) * gs[i])
                )
            else:
                gbar = (p0 * gcl[0][j] + wf * gcl[1][j] + wd * gcl[2][j]) / Z
                T += np.exp(sh[j] - cr[j]) * gbar * np.exp((1.0 - gcl[0][j]) * gs[i])
        swapped += cx[i] * np.exp(crA + dfB) * K * T * rows["dur"][i] / 3600.0
    assert abs((base - swapped) - (ga[p] - gd[p])) < 1e-8


def test_pct_within_gates_and_groups():
    v = np.array([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])
    isD = np.array([0, 0, 0, 0, 1, 1, 1, 1.0])
    elig = np.array([True, True, True, False, True, True, True, True])
    p = GC.pct_within(v, elig, isD)
    assert np.isnan(p[3])  # ineligible → no percentile
    assert p[0] == 0 and p[2] == 100  # within-F ranking
    assert p[4] == 0 and p[7] == 100  # within-D ranking, separate scale


def test_ga60_baseline_zero_mean():
    """GA/60 is exactly TOI-weighted zero within each position group by construction."""
    rng = np.random.default_rng(1)
    n = 300
    t = {
        "isD": (rng.random(n) < 0.35).astype(float),
        "toi_ev": rng.uniform(6000, 90000, n),
        "toi_pp": np.zeros(n),
        "toi_pk": np.zeros(n),
    }
    vals = {
        "sc": rng.normal(0.4, 0.1, n),
        "pm": rng.normal(0.5, 0.2, n),
        "df": rng.normal(0.0, 0.2, n),
        "pp_sc": np.zeros(n),
        "pp_pm": np.zeros(n),
        "pk_df": np.zeros(n),
    }
    base = GC.baselines(t, vals, kap=1.0)
    ga = GC.ga60_of(t, base, vals["sc"], vals["pm"], vals["df"], 1.0)
    for gm in (t["isD"] < 0.5, t["isD"] > 0.5):
        m = gm & (t["toi_ev"] >= GC.EV_GATE)
        assert abs(np.average(ga[m], weights=t["toi_ev"][m])) < 1e-9


def test_band_mean_context_matched():
    """The archetype builder draws only from CONTEXT-eligible band members — PP-less players in
    the EV band must not leak (ridge-shrunk zeros) into the PP archetype."""
    n = 40
    isD = np.zeros(n)
    toi_pp = np.zeros(n)
    toi_pp[:10] = 10000.0  # only 10 PP regulars
    arr = np.full(n, 0.0)
    arr[:10] = -0.5  # PP regulars carry real (negative) values
    pct = np.full(n, np.nan)
    pct[:10] = np.linspace(0, 100, 10)  # their pp_ga60 percentiles
    elig = toi_pp >= GC.MA_GATE
    v, meta = GC.band_mean(arr, toi_pp, elig, pct, isD, band=(5, 30))
    assert meta["F"] == -0.5  # from PP regulars, NOT the zeros
    assert np.allclose(v[isD < 0.5], -0.5)


def test_marginal_goal_prob_matches_mc_and_limit():
    """Quadrature marginal ≡ Monte-Carlo Beta average; s→∞ degenerates to the point evaluation."""
    from yhattrick.models.generative_likelihood import marginal_goal_prob

    rng = np.random.default_rng(0)
    a, b = 1.14, 0.27
    for qbar, s, fin in ((0.087, 14.2, 0.0), (0.05, 14.2, 0.2), (0.15, 8.0, -0.3)):
        x = rng.beta(s * qbar, s * (1 - qbar), 1_000_000)
        mc = np.mean(GC._sigmoid(a * np.log(x / (1 - x)) + b + fin))
        gq = marginal_goal_prob(np.array([qbar]), s, a, b, np.array([fin]))[0]
        assert abs(gq - mc) < 5e-4
    pt = GC._sigmoid(a * np.log(0.087 / 0.913) + b)
    assert abs(marginal_goal_prob(np.array([0.087]), 1e6, a, b)[0] - pt) < 1e-4
