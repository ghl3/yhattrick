"""Tests for the model-math leaf (yhattrick.models.generative_likelihood).

Two things live here that the end-to-end recovery tests in test_generative_model can't isolate:
the parameter LAYOUTS (the θ-vector slicing every stage shares with its standard-error code) and the
value PRIMITIVES the WAR consumers import. The three objective factories are exercised end-to-end by
the recovery tests (which fit through them); here we lock the layout arithmetic and one stage's
objective against its written-down definition.
"""

import numpy as np
import jax.numpy as jnp

from yhattrick.models import generative_likelihood as L

# ── parameter layouts ────────────────────────────────────────────────────────────────────────────


def test_rate_layout_blocks_are_contiguous_and_sized():
    # NU=5 units, k=3 context cols, 2 arena states, plus the A2-mixture logit and the NB dispersion.
    lay = L.RateLayout(NU=5, k=3, n_ar=2, has_a2=True, nb=True)
    assert lay.shoot == slice(1, 6)  # intercept at 0, then three per-unit blocks
    assert lay.create == slice(6, 11)
    assert lay.defence == slice(11, 16)
    assert lay.beta == slice(16, 19)  # PS = 1 + 3·5 + 3 = 19
    assert lay.psi0 == 19
    assert lay.arena == slice(20, 22)
    assert lay.QI == 22  # the A2 logit sits after the arena block
    assert lay.n_theta == 24  # + q + log r
    assert lay.n_core == 19  # the SE Hessian spans only intercept+blocks+beta
    # block_cols maps a set of unit indices to their θ columns in block b (0 shoot, 1 create, 2 def).
    np.testing.assert_array_equal(lay.block_cols(0, [0, 4]), [1, 5])
    np.testing.assert_array_equal(lay.block_cols(1, [0, 4]), [6, 10])
    np.testing.assert_array_equal(lay.block_cols(2, [2]), [13])


def test_rate_layout_without_optional_tails():
    # No arena / no A2 / Poisson: θ is intercept + 3 blocks + beta + psi0 only.
    lay = L.RateLayout(NU=4, k=2)
    assert lay.n_ar == 0 and lay.has_a2 is False and lay.nb is False
    assert lay.psi0 == lay.PS == 1 + 3 * 4 + 2
    assert lay.n_theta == lay.PS + 1  # just psi0 past the SE core
    assert lay.arena == slice(lay.PS + 1, lay.PS + 1)  # empty


def test_qual_layout_blocks():
    lay = L.QualLayout(P=6, k=4, n_ar=3)
    assert lay.mq == 0
    assert lay.qshoot == slice(1, 7)
    assert lay.qcreate == slice(7, 9)  # position pair [F, D]
    assert lay.qdef == slice(9, 15)
    assert lay.beta == slice(15, 19)
    assert lay.arena == slice(19, 22)
    assert lay.n_theta == 22


def test_conv_layout_blocks():
    lay = L.ConvLayout(S=2, kc=3, P=5, G=4)
    assert lay.a == slice(0, 2)  # per-strength slope
    assert lay.b == slice(2, 4)  # per-strength intercept
    assert lay.ctx == slice(4, 7)  # season offsets + curves
    assert lay.fin == slice(7, 12)
    assert lay.gsave == slice(12, 16)
    assert lay.n_theta == 16


# ── objective value against its definition ────────────────────────────────────────────────────────


def test_conversion_nll_equals_bce_plus_ridge():
    """The conversion objective is exactly Σ[softplus(η) − y·η] + ½(lf·Σfin² + lg·Σgsave²) with
    η = a·logit(xg) + b + fin[shooter] + gsave[goalie]. Compute both and compare."""
    lay = L.ConvLayout(S=1, kc=0, P=2, G=2)
    lf, lg = 1.0 / 0.20**2, 1.0 / 0.25**2
    nll = L.make_conversion_nll(lay, lf, lg)

    th = np.array([1.10, -0.30, 0.05, -0.02, 0.10, -0.04])  # [a | b | fin(2) | gsave(2)]
    si = np.array([0, 0, 0])
    sh = np.array([0, 1, 0])
    go = np.array([1, 0, 1])
    lxg = np.array([-1.0, 0.5, 0.2])
    Cx = np.zeros((3, 0))
    y = np.array([0.0, 1.0, 0.0])

    a, b, fin, gsave = th[0], th[1], th[2:4], th[4:6]
    eta = a * lxg + b + fin[sh] + gsave[go]
    expected = np.sum(np.logaddexp(0.0, eta) - y * eta) + 0.5 * (
        lf * np.sum(fin**2) + lg * np.sum(gsave**2)
    )

    got = float(nll(jnp.asarray(th), si, sh, go, lxg, Cx, y))
    assert abs(got - expected) < 1e-9


# ── value primitives (imported by the WAR consumers) ────────────────────────────────────────────────


def test_marginal_goal_prob_approaches_point_evaluation_for_large_concentration():
    """As the Beta concentration s → ∞ the shot-quality distribution collapses to its mean qbar, so
    the marginal goals-per-shot must approach the point evaluation sigmoid(a·logit(qbar) + b)."""
    qbar = np.array([0.08, 0.15])
    a, b = 1.05, -0.20
    point = L._sigmoid(a * np.log(qbar / (1 - qbar)) + b)
    marg = L.marginal_goal_prob(qbar, s=1e6, a=a, b=b)
    np.testing.assert_allclose(marg, point, rtol=1e-3)


def test_creator_mix_weights_sum_to_one_and_favor_forwards_for_a_defenseman():
    """The reference-environment creator weights (unassisted, F-created, D-created) partition 1, and
    at EV a defenseman (3F+1D teammates) gets a larger forward-created share than a forward (2F+2D).
    """
    isD = np.array([0.0, 1.0])  # a forward, then a defenseman
    w0, w_f, w_d = L.creator_mix(psi0=0.0, isD=isD, key="ev")
    np.testing.assert_allclose(w0 + w_f + w_d, 1.0)
    assert w_f[1] > w_f[0]
