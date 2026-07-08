"""Core model math for the generative player model — the objective (likelihood), the parameter
layouts, the sparse design and prior-precision matrices, and the shared math primitives. This module
holds everything the fit needs to *evaluate* the model, with no data loading and no optimizer: the
fit engine (`generative_model`) builds the design arrays, asks here for an objective, and minimizes
it; the WAR consumers (`generative_cards`, `generative_war_audit`) import the value primitives.

Three stages, each a penalized negative log-likelihood built by a `make_*_nll` factory that closes
over its layout + flags + precision weights and returns `nll(th, *data)` — a JAX-jittable function of
the flat parameter vector whose big data arrays stream in as positional arguments.

Parameter symbols (docs/generative_model.md ↔ code):

    rate stage       mu_rate → th[0] (intercept)   shoot_j/create_p/def_d → per-unit blocks
                     create_0 → psi0 (th[lay.psi0])   beta_rate → th[lay.beta]
                     q (secondary-assist mixture share) → th[lay.QI]   r (NB dispersion) → exp(th[-1])
    quality stage    mu_qual → th[0]   qshoot_j/qdef_d → per-player blocks
                     qcreate_{F,D} → th[lay.qcreate] (position pair)   beta_qual → th[lay.beta]
    conversion stage a/b (slope/intercept, per strength) → th[lay.a]/th[lay.b]
                     fin_j → th[lay.fin]   gsave_g → th[lay.gsave]   season offsets + curves → th[lay.ctx]
    arena            per-(venue, season) scorer-bias offsets → th[lay.arena] (nuisance; ridge + RW)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.special import betaincinv

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, logsumexp

jax.config.update("jax_enable_x64", True)  # float64: stable optimization + Hessian

# ── ridge / random-walk / basis constants ──────────────────────────────────────────────────────────
# Ridge = a Normal(0, prior_sd²) prior on each player effect; ridge precision = 1/prior_sd². The
# create and qcreate loadings are PER-TEAMMATE effects (each loads on the 4 rows where the player is a
# teammate), so they get a tighter prior than the shooter's own loading.
PRIOR_SD_SHOOT = 0.30  # shoot_j / def_d on the log shot-rate scale
PRIOR_SD_CREATE = 0.12  # create_p — per-teammate lift; tighter (noisier high-leverage effect,
#   curbs high-minute pull on the shared teammate-shot volume)
PRIOR_SD_QSHOOT = 0.20  # qshoot_j / qdef_d on the logit-xG scale
PRIOR_SD_QCREATE = 0.25  # qcreate_c — danger the single (latent) creator adds
PRIOR_SD_QCREATE_PL = 0.08  # create_qual (per-player on-ice xG lift) — tight: each loads on 4× rows
# The conversion-stage prior SDs (fin, gsave) are NOT hand-set: a pre-calculation stage estimates the
# per-player TALENT SD from the data each fit (empirical Bayes; see estimate_conversion_prior_sds) and
# feeds it in as the ridge prior. These two constants are only the FALLBACK values used when too few
# high-volume shooters/goalies clear the min-shots gate to estimate the talent SD reliably. Whether
# estimated or fallback, from the fit's perspective they are FIXED prior hyperparameters — not
# model-fitted parameters like fin_j/gsave_g themselves.
PRIOR_SD_FIN = 0.20  # FALLBACK finishing prior SD (logit-conversion scale) if unestimable
PRIOR_SD_GSAVE = 0.20  # FALLBACK goalie   prior SD (logit-conversion scale, <0 good) if unestimable
PRIOR_SD_FLOOR = 0.02  # floor on the estimated prior SD (keeps the ridge finite if talent var ≈ 0)
MIN_SHOTS_FIN_EST = 200  # min shots for a shooter to enter the finishing talent-SD estimate
MIN_SHOTS_GSAVE_EST = 1000  # min shots faced for a goalie to enter the save talent-SD estimate
# Random-walk drift prior SDs (per season, on each block's own scale): a player's per-season state
# moves N(prev, rw_sd²·gap). Small = states glued across seasons (static model at 0); large = each
# season fit independently. Hand-set; the EB analogue (estimate within-player drift variance from
# the data) is a documented follow-up.
RW_SD_SHOOT = 0.10  # own-shot volume drifts the most (role/deployment changes)
RW_SD_CREATE = 0.05  # per-teammate creation lift — tighter, like its level prior
RW_SD_DEF = 0.10
AGE_PEAK = 27.0  # age basis B(a) = [z, z²], z = (a − AGE_PEAK)/AGE_SCALE; B(27) = 0 so the
AGE_SCALE = 10.0  #   curve is a deviation from peak-age production (missing DOB ⇒ z = 0)
DENSE_H_MAX = 8000  # params ≤ this: dense Hessian inverse; above: sparse splu column solves
EPS = 1e-6
SNIFF_MIN_TOI = 24000.0  # min on-ice seconds (5v5, ≈400 min) to appear in an EV leaderboard
SNIFF_MIN_TOI_MA = 2400.0  # min man-advantage seconds (≈40 min, matching the production PP/PK gate)
N_TM = 4  # teammates the focal shooter has — the attacking side always has 5 skaters
#   (at EV and on the PP), so this is 4 in every modeled bucket
N_DEF_EV = 5  # defenders faced at even strength (5v5)
N_DEF_MA = 4  # defenders faced on the man-advantage (the PK unit has 4)
MAX_DEF = 5  # padded defender width; PK rows fill 4 real slots + 1 masked (see def_mask)
ARENA_SD = 0.05  # ridge SD on each (venue, season) state — biases are small, shrink to 0
ARENA_RW_SD = 0.03  # per-season random-walk SD between a venue's consecutive states
# Strength buckets: the strengths in each, and whether BOTH sides attack (dual, EV) or only the
# more-skaters side (man-advantage, MA). {5v4,4v5} is ONE PP environment (home/away mirror image);
# the `home` context column carries the asymmetry, so it is a single PP intercept — extensible later.
EV_STRENGTHS = ("5v5",)
MA_STRENGTHS = ("5v4", "4v5")
ALL_STRENGTHS = EV_STRENGTHS + MA_STRENGTHS


# ── math primitives ─────────────────────────────────────────────────────────────────────────────────


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


_GL_X, _GL_W = np.polynomial.legendre.leggauss(40)  # fixed nodes on (0,1) for the Beta marginal
_GL_X = 0.5 * (_GL_X + 1.0)
_GL_W = 0.5 * _GL_W


def marginal_goal_prob(qbar, s, a, b, fin=0.0):
    """The model's own goals-per-shot: E[sigmoid(a·logit(x) + b + fin)] under the model's fitted
    shot-quality distribution x ~ Beta(s·qbar, s·(1−qbar)) (Stage 2's mean qbar + concentration
    beta_s). The point shortcut sigmoid(a·logit(qbar)+b+fin) evaluates the conversion curve at the
    MEAN quality and overstates goals over the right-skewed quality distribution (a Jensen gap of
    ~+13% league-wide); this computes the marginal the PPC simulator computes by Monte Carlo, but
    deterministically. Quadrature runs in CDF space — E[f(X)] = ∫₀¹ f(Q(u)) du with Q the Beta
    quantile function — so the α<1 density singularity (low-quality shooters) costs no accuracy
    and s → ∞ recovers the point evaluation exactly. Vectorized over players."""
    qbar = np.clip(np.atleast_1d(np.asarray(qbar, dtype=np.float64)), EPS, 1 - EPS)
    al, be = s * qbar, s * (1.0 - qbar)  # (P,)
    x = np.clip(betaincinv(al[:, None], be[:, None], _GL_X[None, :]), EPS, 1 - EPS)  # (P, K)
    lx = np.log(x / (1.0 - x))
    f = _sigmoid(a * lx + b + np.atleast_1d(np.asarray(fin, dtype=np.float64))[:, None])
    return np.sum(_GL_W[None, :] * f, axis=1)


def creator_mix(psi0, isD, key):
    """Creator-class weights (unassisted, F-created, D-created) per shooter, in the model's own
    REFERENCE environment (teammate creates = 0): w0 = e^psi0/(e^psi0 + 4); the rest splits by the
    on-ice teammate position mix — EV 5v5 rosters are 3F+2D (an F shooter has 2F+2D teammates, a D
    has 3F+1D); the MA bucket uses the typical 4F+1D PP unit. A documented convention for the
    deployment-free VALUES; the WAR engine uses each stint's actual teammates instead.
    """
    isD = np.asarray(isD, dtype=np.float64)
    w0 = float(np.exp(psi0) / (np.exp(psi0) + 4.0))
    n_f = np.where(isD > 0, 3.0, 2.0) if key == "ev" else np.where(isD > 0, 4.0, 3.0)
    w_f = (1.0 - w0) * n_f / 4.0
    return w0, w_f, (1.0 - w0) - w_f


# ── parameter-vector layouts ─────────────────────────────────────────────────────────────────────────
# One source of truth for how each fit's flat θ is sliced into named blocks. The objective, the sparse
# design used for the SE Hessian, the prior-precision matrix and the SE column-solve all index the same
# offsets, so they live on the layout rather than being recomputed at each site (a class of drift bug).


@dataclass(frozen=True)
class RateLayout:
    """Rate fit θ = [intercept | shoot(NU) | create(NU) | def(NU) | beta(k) | psi0 | arena(n_ar)
    | q? | log r?]. `n_core` (= PS = 1+3·NU+k) is the SE-Hessian dimension; psi0, arena, q and r sit
    past it and carry no per-player SE. `block_cols(b, units)` gives the θ columns of block b
    (0 shoot, 1 create, 2 def) for a set of unit indices."""

    NU: int
    k: int
    n_ar: int = 0
    has_a2: bool = False
    nb: bool = False

    @property
    def PS(self) -> int:  # index of psi0 = width of the SE core
        return 1 + 3 * self.NU + self.k

    @property
    def n_core(self) -> int:
        return self.PS

    @property
    def QI(self) -> int:  # index of the A2 mixture logit q (valid iff has_a2)
        return self.PS + 1 + self.n_ar

    @property
    def n_theta(self) -> int:
        return self.PS + 1 + self.n_ar + (1 if self.has_a2 else 0) + (1 if self.nb else 0)

    @property
    def shoot(self) -> slice:
        return slice(1, 1 + self.NU)

    @property
    def create(self) -> slice:
        return slice(1 + self.NU, 1 + 2 * self.NU)

    @property
    def defence(self) -> slice:
        return slice(1 + 2 * self.NU, 1 + 3 * self.NU)

    @property
    def beta(self) -> slice:
        return slice(1 + 3 * self.NU, self.PS)

    @property
    def psi0(self) -> int:
        return self.PS

    @property
    def arena(self) -> slice:
        return slice(self.PS + 1, self.PS + 1 + self.n_ar)

    def block_offset(self, b: int) -> int:  # b: 0 shoot, 1 create, 2 def
        return 1 + b * self.NU

    def block_cols(self, b: int, units):
        return self.block_offset(b) + np.asarray(units)


@dataclass(frozen=True)
class QualLayout:
    """Quality fit θ = [mq | qshoot(P) | qcreate(2) | qdef(P) | create_qual(P?) | beta(k) | arena].
    qcreate is a position-level pair [F, D]. `create_qual` (present iff `has_cq`) is a
    per-player on-ice creation-QUALITY block: each teammate on the ice lifts the shot's xG, mirroring
    how `create` lifts the shot RATE in Stage 1 — a dense RAPM-on-quality signal (every shot, no
    creator marginalization). Width 0 when off, so the layout degenerates exactly to the position-pair
    model."""

    P: int
    k: int
    n_ar: int = 0
    has_cq: bool = False

    @property
    def _ncq(self) -> int:
        return self.P if self.has_cq else 0

    @property
    def n_theta(self) -> int:
        return 3 + 2 * self.P + self._ncq + self.k + self.n_ar

    @property
    def mq(self) -> int:
        return 0

    @property
    def qshoot(self) -> slice:
        return slice(1, 1 + self.P)

    @property
    def qcreate(self) -> slice:
        return slice(1 + self.P, 3 + self.P)

    @property
    def qdef(self) -> slice:
        return slice(3 + self.P, 3 + 2 * self.P)

    @property
    def create_qual(self) -> slice:
        return slice(3 + 2 * self.P, 3 + 2 * self.P + self._ncq)

    @property
    def beta(self) -> slice:
        return slice(3 + 2 * self.P + self._ncq, 3 + 2 * self.P + self._ncq + self.k)

    @property
    def arena(self) -> slice:
        b = 3 + 2 * self.P + self._ncq + self.k
        return slice(b, b + self.n_ar)


@dataclass(frozen=True)
class ConvLayout:
    """Conversion fit θ = [a(S) | b(S) | ctx(kc) | fin(P) | gsave(G)]. a/b are per-strength and
    unpenalized; fin/gsave carry the empirical-Bayes ridge."""

    S: int
    kc: int
    P: int
    G: int

    @property
    def n_theta(self) -> int:
        return 2 * self.S + self.kc + self.P + self.G

    @property
    def a(self) -> slice:
        return slice(0, self.S)

    @property
    def b(self) -> slice:
        return slice(self.S, 2 * self.S)

    @property
    def ctx(self) -> slice:
        return slice(2 * self.S, 2 * self.S + self.kc)

    @property
    def fin_off(self) -> int:
        return 2 * self.S + self.kc

    @property
    def fin(self) -> slice:
        return slice(self.fin_off, self.fin_off + self.P)

    @property
    def gsave(self) -> slice:
        return slice(self.fin_off + self.P, self.n_theta)


# ── the crossed design as a sparse matrix (for the exact analytic Hessian) ─────────────────────────


def _build_X(shooter, team, dfd, Xctx, lay, def_mask=None):
    """Sparse design, columns [intercept | shoot(NU) | create(NU) | def(NU) | context(k)] laid out by
    `lay` (a RateLayout). `def_mask` (n × width) zeroes padded/PK-absent defender slots so masked
    entries contribute nothing."""
    n, k = len(shooter), Xctx.shape[1]
    ar = np.arange(n)
    R, Cc, D = [], [], []

    def add_ind(col, val=None):
        R.append(ar)
        Cc.append(col)
        D.append(np.ones(n) if val is None else val)

    add_ind(np.zeros(n, int))  # intercept
    add_ind(lay.block_offset(0) + shooter)  # shoot block
    for j in range(team.shape[1]):
        add_ind(lay.block_offset(1) + team[:, j])  # create (play) block
    for j in range(dfd.shape[1]):  # def block (masked: padded slots contribute 0)
        add_ind(
            lay.block_offset(2) + dfd[:, j],
            None if def_mask is None else def_mask[:, j],
        )
    for c in range(k):  # context (values, not indicators)
        R.append(ar)
        Cc.append(np.full(n, lay.beta.start + c))
        D.append(Xctx[:, c])
    return sparse.csr_matrix(
        (np.concatenate(D), (np.concatenate(R), np.concatenate(Cc))),
        shape=(n, lay.n_core),
    )


def _build_conv_X(logit_xg, sidx, shooter, goalie, lay, Cctx=None):
    """Sparse conversion design for the Hessian, columns
    [slope_s (S) | intercept_s (S) | ctx(kc) | fin(P) | gsave(G)] laid out by `lay` (a ConvLayout) —
    the slope/intercept are per-strength (`sidx` in 0..S-1), so a shot loads its own strength's slope
    (value=logit_xg) and intercept (value=1); `Cctx` is the optional global context block (season
    offsets, age curves)."""
    n = len(shooter)
    ar = np.arange(n)
    rows = [ar, ar, ar, ar]
    cols = [
        lay.a.start + sidx,
        lay.b.start + sidx,
        lay.fin_off + shooter,
        lay.fin_off + lay.P + goalie,
    ]
    data = [logit_xg, np.ones(n), np.ones(n), np.ones(n)]
    for c in range(lay.kc):
        rows.append(ar)
        cols.append(np.full(n, lay.ctx.start + c))
        data.append(Cctx[:, c])
    return sparse.csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, lay.n_theta),
    )


def _rate_penalty(lay, first_mask, e_prev, e_next, e_gap, levels):
    """Sparse prior-precision matrix for the rate fit's player blocks, laid out by `lay` (a
    RateLayout). Per block: a level ridge on each player's FIRST state (`first_mask`) + the
    random-walk precision between his consecutive states — for an edge with weight
    w = 1/(rw_sd²·gap), add w to both diagonal entries and −w to the off-diagonals (the precision of
    θ_next − θ_prev ~ N(0, rw_sd²·gap)). With no edges (static or single season) this is exactly the
    old diagonal ridge. `levels` = (shoot, create, def) LEVEL precisions (1/prior_sd²) supplied by the
    caller so they are the SAME ridge the objective penalises with — a per-bucket create/def prior
    override (the MA bucket) then shrinks the point estimate and its standard error consistently. The
    random-walk precisions are the shared RW_SD_* constants."""
    blocks = [
        (levels[0], 1.0 / RW_SD_SHOOT**2),
        (levels[1], 1.0 / RW_SD_CREATE**2),
        (levels[2], 1.0 / RW_SD_DEF**2),
    ]
    rows, cols, vals = [], [], []
    au = np.arange(lay.NU)
    for bi, (lev, wrw) in enumerate(blocks):
        off = lay.block_offset(bi)
        rows.append(off + au)
        cols.append(off + au)
        vals.append(lev * first_mask)
        if len(e_prev):
            w = wrw / e_gap
            for a, b in ((e_prev, e_prev), (e_next, e_next)):
                rows.append(off + a)
                cols.append(off + b)
                vals.append(w)
            for a, b in ((e_prev, e_next), (e_next, e_prev)):
                rows.append(off + a)
                cols.append(off + b)
                vals.append(-w)
    return sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(lay.n_core, lay.n_core),
    )


def _se_from_hessian(H):
    return np.sqrt(np.clip(np.diag(np.linalg.inv(H)), 0.0, None))


# ── objective factories (JAX-jittable nll(th, *data) closures) ──────────────────────────────────────
# Each factory closes over the layout + static flags + precision weights and returns the stage's
# penalized negative log-likelihood. The optimizer streams the big per-row arrays in as positional
# arguments (never closed over), so XLA keeps them as inputs instead of baked-in constants.


@dataclass(frozen=True)
class RateHypers:
    """Static config the rate objective closes over: the count-model / stage flags, the up-weight on
    the assist-credit anchor (`shots_per_goal`), and the ridge/random-walk/arena precisions.
    """

    nb: bool
    has_a2: bool
    has_rw: bool
    has_ar_rw: bool
    shots_per_goal: float
    lsh: float
    lcr: float
    ldf: float
    w_sh: float
    w_cr: float
    w_df: float
    la_ar: float
    lrw_ar: float


@dataclass(frozen=True)
class QualHypers:
    """Static config the quality objective closes over: the arena random-walk flag and the
    qshoot/qcreate/qdef + arena ridge precisions."""

    has_ar_rw: bool
    lqs: float
    lqc: float
    lqd: float
    la_ar: float
    lrw_ar: float
    lcq: float = 0.0  # create_qual ridge precision (0 when the block is off)


def make_rate_nll(lay: RateLayout, h: RateHypers, arena_rw, create_center=None):
    """Rate-stage objective: Poisson/NB shot-count NLL + assist-credit anchor (spg-weighted
    conditional logit over the on-ice teammates, plus the optional A2 mixture stage) + ridge/RW/arena
    penalties. `arena_rw` = (a_ep, a_en, a_iw) the venue-season random-walk edges (jnp arrays).
    `create_center` (per-unit, or None = 0): the `create` level ridge shrinks first states toward THIS
    target instead of 0 — set to the position (F/D) mean so a weakly-identified forward defaults to the
    forward baseline, not to "suppresses offense" (roadmap §5e). A fixed constant (not a free
    parameter), so the level stays pinned and SE curvature is unchanged. Returns nll(th, *data)."""
    n_ar, PS, QI = lay.n_ar, lay.PS, lay.QI
    nb, has_a2, has_rw, has_ar_rw = h.nb, h.has_a2, h.has_rw, h.has_ar_rw
    lsh, lcr, ldf = h.lsh, h.lcr, h.ldf
    w_sh, w_cr, w_df = h.w_sh, h.w_cr, h.w_df
    la_ar, lrw_ar = h.la_ar, h.lrw_ar
    shots_per_goal = h.shots_per_goal
    a_ep, a_en, a_iw = arena_rw
    ccen = 0.0 if create_center is None else jnp.asarray(create_center)

    def split(th):
        return (
            th[lay.shoot],
            th[lay.create],
            th[lay.defence],
            th[lay.beta],
            th[lay.psi0],
        )

    def nll(
        th,
        sh_j,
        tm_j,
        df_j,
        dm,
        X,
        cnt,
        ofs,
        gt,
        gc,
        fmj,
        ep,
        en,
        iw,
        acl,
        g2t,
        g2a,
        g2b,
    ):
        sh, cr, df, b, psi0 = split(th)
        eta = th[0] + sh[sh_j] + jnp.sum(cr[tm_j], 1) + jnp.sum(df[df_j] * dm, 1) + X @ b
        if n_ar:  # venue recording-bias offset (−1 = ref)
            ar = th[PS + 1 : PS + 1 + n_ar]
            eta = eta + jnp.where(acl >= 0, ar[jnp.clip(acl, 0, None)], 0.0)
        mu = jnp.exp(eta + ofs)
        if not nb:
            pois = jnp.sum(mu - cnt * jnp.log(mu))
        else:
            r = jnp.exp(th[-1])
            pois = -jnp.sum(
                gammaln(cnt + r)
                - gammaln(r)
                - gammaln(cnt + 1)
                + r * jnp.log(r / (r + mu))
                + cnt * jnp.log(mu / (r + mu))
            )
        logit5 = jnp.concatenate(
            [jnp.full((gt.shape[0], 1), psi0), cr[gt]], axis=1
        )  # [create_0, create(4)]
        logpi = logit5 - logsumexp(logit5, axis=1)[:, None]
        credit = -jnp.sum(jnp.take_along_axis(logpi, gc[:, None], 1)[:, 0])
        if has_a2:  # A2 mixture stage (A1's column masked)
            q = jax.nn.sigmoid(th[QI])
            lg2 = jnp.where(jnp.arange(4)[None, :] == g2a[:, None], -1e30, cr[g2t])
            logpi2 = lg2 - logsumexp(lg2, axis=1)[:, None]
            pl2 = jnp.exp(jnp.take_along_axis(logpi2, g2b[:, None], 1)[:, 0])
            credit = credit - jnp.sum(jnp.log(q * pl2 + (1.0 - q) / 3.0 + 1e-12))
        pen = 0.5 * (
            lsh * jnp.sum(fmj * sh**2)
            + lcr * jnp.sum(fmj * (cr - ccen) ** 2)
            + ldf * jnp.sum(fmj * df**2)
        )
        if has_rw:  # RW: θ_next ~ N(θ_prev, rw_sd²·gap)
            pen += 0.5 * (
                w_sh * jnp.sum(iw * (sh[en] - sh[ep]) ** 2)
                + w_cr * jnp.sum(iw * (cr[en] - cr[ep]) ** 2)
                + w_df * jnp.sum(iw * (df[en] - df[ep]) ** 2)
            )
        if n_ar:  # arena nuisance prior: ridge + season RW
            pen += 0.5 * la_ar * jnp.sum(ar**2)
            if has_ar_rw:
                pen += 0.5 * lrw_ar * jnp.sum(a_iw * (ar[a_en] - ar[a_ep]) ** 2)
        return pois + shots_per_goal * credit + pen

    return nll


def make_quality_nll(lay: QualLayout, h: QualHypers, arena_rw):
    """Quality-stage objective: fractional-Bernoulli xG quasi-likelihood with the creator observed on
    goals and marginalized over the FIXED creator distribution pi otherwise + ridge/arena penalties.
    `arena_rw` = (a_ep, a_en, a_iw). Returns nll(th, *data)."""
    n_ar = lay.n_ar
    has_cq = lay.has_cq
    has_ar_rw = h.has_ar_rw
    lqs, lqc, lqd, lcq = h.lqs, h.lqc, h.lqd, h.lcq
    la_ar, lrw_ar = h.la_ar, h.lrw_ar
    a_ep, a_en, a_iw = arena_rw

    def split(th):  # [mq | qshoot(P) | qcreate(2) | qdef(P) | create_qual(P?) | beta(k) | arena]
        return th[lay.mq], th[lay.qshoot], th[lay.qcreate], th[lay.qdef], th[lay.beta]

    def nll(th, sh_i, tm_i, tp_i, df_i, dm, X, xg, obs, cidx, pi, acl):
        mq, qs, qc, qd, b = split(th)
        base = mq + qs[sh_i] + jnp.sum(qd[df_i] * dm, 1) + X @ b
        if has_cq:  # each on-ice teammate lifts the shot's xG (dense RAPM-on-quality)
            cq = th[lay.create_qual]
            base = base + jnp.sum(cq[tm_i], 1)
        if n_ar:  # venue location-bias offset (−1 = ref)
            arq = th[lay.arena]
            base = base + jnp.where(acl >= 0, arq[jnp.clip(acl, 0, None)], 0.0)
        sig5 = jnp.concatenate(
            [jax.nn.sigmoid(base)[:, None], jax.nn.sigmoid(base[:, None] + qc[tp_i])],
            axis=1,
        )  # (n,5) col0=unassisted

        def fb(p):
            return xg * jnp.log(p + EPS) + (1 - xg) * jnp.log(1 - p + EPS)

        gs = jnp.take_along_axis(sig5, cidx[:, None], 1)[:, 0]  # observed-creator quality
        marg = jnp.sum(pi * sig5, axis=1)  # latent: marginalize over FIXED pi
        ll = jnp.where(obs, fb(gs), fb(marg))
        pen = 0.5 * (lqs * jnp.sum(qs**2) + lqc * jnp.sum(qc**2) + lqd * jnp.sum(qd**2))
        if has_cq:  # create_qual EB ridge (tight — 4× on-ice leverage)
            pen = pen + 0.5 * lcq * jnp.sum(cq**2)
        if n_ar:  # arena nuisance prior: ridge + season RW
            pen += 0.5 * la_ar * jnp.sum(arq**2)
            if has_ar_rw:
                pen += 0.5 * lrw_ar * jnp.sum(a_iw * (arq[a_en] - arq[a_ep]) ** 2)
        return -jnp.sum(ll) + pen

    return nll


def make_conversion_nll(lay: ConvLayout, lf: float, lg: float):
    """Conversion-stage objective: Bernoulli goal cross-entropy on the observed xG (a/b/context
    unpenalized so Σp = Σy holds) + fin/gsave EB ridge (precisions lf, lg). Returns nll(th, *data).
    """

    def split(th):  # [a(S) | b(S) | c(kc) | fin(P) | gsave(G)]
        return th[lay.a], th[lay.b], th[lay.ctx], th[lay.fin], th[lay.gsave]

    def nll(th, si, sh_i, g_i, lxg, Cx, y):
        a, b, c, fin, gsave = split(th)
        eta = a[si] * lxg + b[si] + Cx @ c + fin[sh_i] + gsave[g_i]
        bce = jnp.sum(jax.nn.softplus(eta) - y * eta)  # = −Σ[y·log p + (1−y)·log(1−p)]
        pen = 0.5 * (lf * jnp.sum(fin**2) + lg * jnp.sum(gsave**2))
        return bce + pen  # a, b, c all unpenalized

    return nll
