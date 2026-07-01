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
                    shoot_j              how much j shoots himself           (SCORING volume)
                    create_p             UNIFIED creation — lifts teammates' shot rate AND is p's
                                           creator-credit propensity          (PLAYMAKING)
                    create_0             unassisted propensity (shooter self-created); one global scalar
                    def_d                opponent shot-rate suppression      (DEFENSE; <0 good)
                    beta_rate            context (home, O/D-zone, lead/trail)
  quality layer     mu_qual              baseline shot quality (logit mean xG)
                    qshoot_j             danger of j's own shots             (SCORING quality)
                    qcreate_c            danger the ONE creator c adds        (PLAYMAKING quality)
                    qdef_d               opponent danger suppression         (DEFENSE quality; <0 good)
                    beta_qual ; s        context ; Beta concentration (shot-to-shot spread)
  conversion layer  a ; b                logit-conversion slope + intercept (b replaces the old mu_conv;
                                           both global, UNPENALIZED → b's score eqn gives Σp=Σy exactly)
                    fin_j                finishing — logit offset above xG on own shots
                    gsave_g              goalie save effect — logit offset (<0 good)

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
  added to the rate NLL, up-weighted by SHOTS_PER_GOAL (≈ shots per goal — see the constant). The SAME
  `create` thus blends 'I make my linemates shoot' (dense volume) with 'I'm credited with the setup'
  (sparse assist signal).

LIKELIHOOD  (factorizes into disjoint blocks; independent ridge priors)
  ℓ(θ) =  Σ_(side,j) log Pois/NB( N_j | rate_j·t/3600 )     ← rate    (mu_rate,shoot,create,def,beta_rate,r)
        + SHOTS_PER_GOAL · Σ_goals log pi[assister]         ← assist-credit anchor (shares create, create_0)
        + Σ_shots  log fracBern( xg | qbar )                ← quality (mu_qual,qshoot,qcreate,qdef,beta_qual,s)
                     (creator OBSERVED on goals; MARGINALIZED over fixed pi on non-goals)
        + Σ_shots  log Bernoulli( y | sigmoid(a·logit(xg)+b+fin+gsave) ) ← conversion (a,b,fin,gsave; fit
                     natively off the OBSERVED xg — no shooting-model reuse; a,b unpenalized)
        − ridge on every player block (= EB Normal(0,σ²) priors)

CODE↔GLOSSARY MAP: create_0→`psi0`; mu_rate/qual→ each fit's `intercept`; a/b→`conv["a"]`/`conv["b"]`;
  beta_rate/qual→`beta`; qbar→`sig5`; pi_c→`pi`; fin_j→`conv["fin"]`; gsave_g→`conv["gsave"]`.
────────────────────────────────────────────────────────────────────────────────────────────────

Fit: MAP / penalized MLE by OPTIMIZATION (JAX autodiff gradient + scipy L-BFGS-B). Uncertainties via
the HESSIAN (Laplace): H = XᵀWX + diag(penalty); SE = sqrt(diag(H⁻¹)). The count layer is configurable
(`--count poisson|nb`). 5v5 only, for the POC. The conversion ridge prior SDs (fin, gsave) are NOT
hand-set: a PRE-CALCULATION stage (estimate_conversion_prior_sds) estimates the per-player talent SD
from the data each fit — empirical Bayes, the logit analogue of shooting_model._estimate_k — then holds
it FIXED during the fit. Usage:
  uv run --group experimental python -m yhattrick.models.generative_model            # latest season
  uv run --group experimental python -m yhattrick.models.generative_model --count nb # negative binomial
  uv run --group experimental python -m yhattrick.models.generative_model --pool     # all seasons
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, logsumexp

from .. import config as C
from .player_onice_model import roster_names

jax.config.update("jax_enable_x64", True)   # float64: stable optimization + Hessian

# Ridge = a Normal(0, prior_sd²) prior on each player effect; ridge precision = 1/prior_sd². The
# create and qcreate loadings are PER-TEAMMATE effects (each loads on the 4 rows where the player is a
# teammate), so they get a tighter prior than the shooter's own loading.
PRIOR_SD_SHOOT = 0.30       # shoot_j / def_d on the log shot-rate scale
PRIOR_SD_CREATE = 0.12      # create_p — per-teammate lift; tighter (noisier high-leverage effect,
                            #   curbs high-minute pull on the shared teammate-shot volume)
PRIOR_SD_QSHOOT = 0.20      # qshoot_j / qdef_d on the logit-xG scale
PRIOR_SD_QCREATE = 0.25     # qcreate_c — danger the single (latent) creator adds
# The conversion-stage prior SDs (fin, gsave) are NOT hand-set: a pre-calculation stage estimates the
# per-player TALENT SD from the data each fit (empirical Bayes; see estimate_conversion_prior_sds) and
# feeds it in as the ridge prior. These two constants are only the FALLBACK values used when too few
# high-volume shooters/goalies clear the min-shots gate to estimate the talent SD reliably. Whether
# estimated or fallback, from the fit's perspective they are FIXED prior hyperparameters — not
# model-fitted parameters like fin_j/gsave_g themselves.
PRIOR_SD_FIN = 0.20         # FALLBACK finishing prior SD (logit-conversion scale) if unestimable
PRIOR_SD_GSAVE = 0.20       # FALLBACK goalie   prior SD (logit-conversion scale, <0 good) if unestimable
PRIOR_SD_FLOOR = 0.02       # floor on the estimated prior SD (keeps the ridge finite if talent var ≈ 0)
MIN_SHOTS_FIN_EST = 200     # min shots for a shooter to enter the finishing talent-SD estimate
MIN_SHOTS_GSAVE_EST = 1000  # min shots faced for a goalie to enter the save talent-SD estimate
SHOTS_PER_GOAL = 16         # up-weight on the goal-assist credit term (Stage 1). We observe a shot's
                            #   creator only on the ~1-in-16 Fenwick shots that ARE goals (via the
                            #   primary assist), so weighting each observed goal-creator by ≈ shots/goal
                            #   makes it stand in for the shots whose creator we never see (inverse-
                            #   probability weighting). ≈ 1/(5v5 Fenwick goal rate); approximate, fixed.
EPS = 1e-6
SNIFF_MIN_TOI = 24000.0     # min on-ice seconds (5v5, ≈400 min) to appear in an EV leaderboard
SNIFF_MIN_TOI_MA = 2400.0   # min man-advantage seconds (≈40 min, matching the production PP/PK gate)
N_TM = 4                    # teammates the focal shooter has — the attacking side always has 5 skaters
                            #   (at EV and on the PP), so this is 4 in every modeled bucket
N_DEF_EV = 5                # defenders faced at even strength (5v5)
N_DEF_MA = 4                # defenders faced on the man-advantage (the PK unit has 4)
MAX_DEF = 5                 # padded defender width; PK rows fill 4 real slots + 1 masked (see def_mask)
# Strength buckets (Spec-like): the strengths in each, and whether BOTH sides attack (dual, EV) or only
# the more-skaters side (man-advantage, MA). {5v4,4v5} is ONE PP environment (home/away mirror image);
# the `home` context column carries the asymmetry, so it is a single PP intercept — extensible later.
EV_STRENGTHS = ("5v5",)
MA_STRENGTHS = ("5v4", "4v5")
ALL_STRENGTHS = EV_STRENGTHS + MA_STRENGTHS


# ── data loaders (5v5) ──────────────────────────────────────────────────────────────────────────

def _zone(atk_home, start_type, start_zone):
    """Attacking-team zone start: O/D flips for the away team. Returns (ozone, dzone) indicators."""
    if start_type != "faceoff" or start_zone not in ("O", "D"):
        return 0.0, 0.0
    az = start_zone if atk_home else ("O" if start_zone == "D" else "D")
    return (1.0, 0.0) if az == "O" else (0.0, 1.0)


RATE_CTX = ["home", "ozone", "dzone", "trail", "lead"]


def _load_stints(seasons, strengths):
    """Regular-season non-overload stints in the given strength states (e.g. ('5v5',) or
    ('5v4','4v5')). Keeps short <10s stints (the Poisson offset handles them). Adds a `season` column
    for the season fixed-effect."""
    frames = []
    for s in seasons:
        p = C.PROCESSED / "stints" / f"{s}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if "overload" not in df.columns:
            df["overload"] = False
        reg = (df.nhl_game_id // 10000) % 100 == 2
        sub = df[reg & df.strength.isin(strengths) & (~df.overload) & (df.duration_s >= 1)].copy()
        sub["season"] = s
        frames.append(sub)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def player_index(seasons):
    """Shared player index across ALL modeled strengths (EV+MA), so the per-strength rate fits align to
    one player list. Returns (players, idx)."""
    df = _load_stints(seasons, ALL_STRENGTHS)
    if df.empty:
        return [], {}
    players = sorted(set().union(*df.home_skaters, *df.away_skaters))
    return players, {p: i for i, p in enumerate(players)}


def _season_cols(seasons):
    """Map season -> season-indicator column (first season = reference, dropped). Empty for one season."""
    sl = sorted(set(seasons))
    return {s: i for i, s in enumerate(sl[1:])}, max(len(sl) - 1, 0)


def _shooter_counts(seasons, strengths):
    """{(nhl_game_id, stint_idx): {shooter_id: fenwick_count}} for shots in the given strengths (the
    shooter-resolved response). A player's shots in a stint are his regardless of side."""
    out: dict[tuple, dict] = defaultdict(lambda: defaultdict(int))
    for s in seasons:
        p = C.PROCESSED / "shots_onice" / f"{s}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["nhl_game_id", "stint_idx", "shooter_id", "strength"])
        d = d[d.strength.isin(strengths) & d.shooter_id.notna()]
        g = d.groupby(["nhl_game_id", "stint_idx", "shooter_id"]).size()
        for (gid, sidx, pid), n in g.items():
            out[(int(gid), int(sidx))][int(pid)] = int(n)
    return out


def rate_rows(seasons, strengths, dual, players, idx):
    """Shooter-resolved rate design for ONE strength bucket, on the SHARED player index. For each stint
    side that attacks (EV/dual: both sides; MA: only the more-skaters side) and each on-ice attacker j
    as focal shooter: one row with j's Fenwick count, j's index, the 4 teammate indices, the (≤5)
    defender indices padded to width 5 with a `def_mask`, offset log(t/3600), and context (5 base cols +
    season indicators). Returns per-player attacking/defending TOI in this bucket."""
    df = _load_stints(seasons, strengths)
    if df.empty:
        return None
    counts = _shooter_counts(seasons, strengths)
    P = len(players)
    scol, nseas = _season_cols(seasons)

    shooter, team, dff, dmask, ctx, cnt, off_t, gid = [], [], [], [], [], [], [], []
    toi_atk, toi_def = np.zeros(P), np.zeros(P)

    def emit_side(atk, dfd, atk_home, s, def_goalie):
        c = counts.get((int(s.nhl_game_id), int(s.stint_idx)), {})
        di = [idx[p] for p in dfd]
        nd = len(di)
        di_pad = di + [0] * (MAX_DEF - nd)                     # pad to width 5 (masked slots point at 0)
        mrow = [1.0] * nd + [0.0] * (MAX_DEF - nd)
        oz, dz = _zone(atk_home, s.start_type, s.start_zone)
        lead = s.home_lead if atk_home else -s.home_lead
        base = [1.0 if atk_home else 0.0, oz, dz, 1.0 if lead < 0 else 0.0, 1.0 if lead > 0 else 0.0]
        seas = [0.0] * nseas
        if s.season in scol:
            seas[scol[s.season]] = 1.0
        row_ctx = base + seas
        for j in atk:
            shooter.append(idx[j]); team.append([idx[p] for p in atk if p != j])
            dff.append(di_pad); dmask.append(mrow); ctx.append(row_ctx)
            cnt.append(float(c.get(int(j), 0))); off_t.append(s.duration_s); gid.append(def_goalie)

    for s in df.itertuples():
        hn, an = len(s.home_skaters), len(s.away_skaters)
        dur = s.duration_s
        if dual:
            if hn != 5 or an != 5:
                continue
            emit_side(s.home_skaters, s.away_skaters, True, s, s.away_goalie)
            emit_side(s.away_skaters, s.home_skaters, False, s, s.home_goalie)
            for p in (*s.home_skaters, *s.away_skaters):
                toi_atk[idx[p]] += dur; toi_def[idx[p]] += dur
        else:
            if hn == an:
                continue                                       # no man-advantage
            atk_home = hn > an
            atk, dfd = (s.home_skaters, s.away_skaters) if atk_home else (s.away_skaters, s.home_skaters)
            emit_side(atk, dfd, atk_home, s, s.away_goalie if atk_home else s.home_goalie)
            for p in atk:
                toi_atk[idx[p]] += dur                         # PP-attacker time
            for p in dfd:
                toi_def[idx[p]] += dur                         # PK-defender time

    dur = np.asarray(off_t, dtype=np.float64)
    return {
        "players": players, "idx": idx, "dual": dual,
        "shooter_idx": np.asarray(shooter, dtype=np.int64),
        "team_idx": np.asarray(team, dtype=np.int64),
        "def_idx": np.asarray(dff, dtype=np.int64),
        "def_mask": np.asarray(dmask, dtype=np.float64),
        "Xctx": np.asarray(ctx, dtype=np.float64),
        "count": np.asarray(cnt, dtype=np.float64),
        "offset": np.log(np.clip(dur / 3600.0, 1e-9, None)),
        "dur": dur, "def_goalie": np.asarray(gid, dtype=object),
        "toi_atk": toi_atk, "toi_def": toi_def, "toi": toi_atk, "n_season_cols": nseas,
    }


# ── quality data: each shot + (for goals) the observed primary creator ────────────────────────────

def quality_creator_rows(seasons, idx, strengths):
    """Per Fenwick shot in the given strengths (POOLED across EV+MA): shooter, 4 teammates, ≤5 defenders
    (padded to 5 with a `def_mask`), context [is_home, pp, season…], logit-xG, goal flag, a CREATOR
    LABEL, and a strength label (0=EV, 1=MA). MA keeps only shots taken by the more-skaters (PP) side —
    so teammates=4, defenders=4 — consistent with the rate stage (shorthanded shots are out of scope).
    Creator: for goals, which teammate (0–3) got the primary assist, or 4=unassisted; non-goals −1
    (latent). Join: shots_onice.event_idx == pbp goal sortOrder."""
    scol, nseas = _season_cols(seasons)
    shooter, team, dff, dmask, ctx, y, goal, creator, slab = [], [], [], [], [], [], [], [], []
    for s in seasons:
        p = C.PROCESSED / "shots_onice" / f"{s}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["nhl_game_id", "event_idx", "strength", "is_home", "xg",
                                        "goal", "shooter_id", "home_skaters", "away_skaters"])
        d = d[d.strength.isin(strengths) & d.xg.notna() & d.shooter_id.notna()]
        for gid, sub in d.groupby("nhl_game_id"):
            a1 = {}
            pf = C.RAW_PBP / f"{int(gid)}.json"
            if pf.exists():
                plays = json.loads(pf.read_text()).get("plays", [])
                a1 = {pl.get("sortOrder"): pl.get("details", {}).get("assist1PlayerId")
                      for pl in plays if pl.get("typeDescKey") == "goal"}
            for r in sub.itertuples():
                hs, as_ = list(r.home_skaters), list(r.away_skaters)
                atk, dfd = (hs, as_) if r.is_home == 1 else (as_, hs)
                is_ev = r.strength == "5v5"
                if len(atk) != 5:                        # EV: 5v5; MA: shooter must be on the 5 (PP) side
                    continue
                if is_ev and len(dfd) != 5:
                    continue
                sid = int(r.shooter_id)
                if sid not in atk or any(q not in idx for q in atk) or any(q not in idx for q in dfd):
                    continue
                mates = [q for q in atk if q != sid]                     # 4 teammates
                nd = len(dfd)
                di = [idx[q] for q in dfd] + [0] * (MAX_DEF - nd)
                mrow = [1.0] * nd + [0.0] * (MAX_DEF - nd)
                seas = [0.0] * nseas
                if s in scol:
                    seas[scol[s]] = 1.0
                shooter.append(idx[sid]); team.append([idx[q] for q in mates])
                dff.append(di); dmask.append(mrow)
                ctx.append([1.0 if r.is_home == 1 else 0.0, 0.0 if is_ev else 1.0] + seas)
                xg = float(np.clip(r.xg, EPS, 1 - EPS)); y.append(np.log(xg / (1 - xg)))
                goal.append(int(r.goal)); slab.append(0 if is_ev else 1)
                if r.goal == 1:
                    ap = a1.get(int(r.event_idx))
                    creator.append(mates.index(int(ap)) if ap is not None and int(ap) in mates else 4)
                else:
                    creator.append(-1)
    return {"shooter_idx": np.asarray(shooter, dtype=np.int64), "team_idx": np.asarray(team, dtype=np.int64),
            "def_idx": np.asarray(dff, dtype=np.int64), "def_mask": np.asarray(dmask, dtype=np.float64),
            "Xctx": np.asarray(ctx, dtype=np.float64),
            "y": np.asarray(y, dtype=np.float64), "goal": np.asarray(goal, dtype=np.int64),
            "creator": np.asarray(creator, dtype=np.int64), "strength": np.asarray(slab, dtype=np.int64)}


# ── conversion data: each shot's shooter, facing goalie, observed xG, goal flag ────────────────────

def conversion_rows(seasons, idx, strengths):
    """Per Fenwick shot (POOLED across the given strengths) for the CONVERSION fit: shooter index,
    facing-goalie index, logit of the OBSERVED xG, goal flag, and a strength label (0=EV, 1=MA). For MA,
    keeps only PP-side shots (shooter's team has the man-advantage), matching rate/quality. Rows whose
    shooter isn't in `idx`, or with a missing facing goalie, are dropped. Uses observed `xg` (fixed),
    never `qbar`, so the stage stays independent of Stages 1-2. `fin`/`gsave` are POOLED across
    strengths (strength is handled by per-strength slope/intercept in fit_conversion)."""
    frames = []
    for s in seasons:
        p = C.PROCESSED / "shots_onice" / f"{s}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["strength", "is_home", "xg", "goal",
                                        "shooter_id", "home_goalie", "away_goalie"])
        d = d[d.strength.isin(strengths) & d.xg.notna() & d.shooter_id.notna()].copy()
        # skater counts from the "HvA" strength string; shooter's own count vs the defenders'
        hn = d.strength.str.slice(0, 1).astype(int).to_numpy()
        an = d.strength.str.slice(2, 3).astype(int).to_numpy()
        is_home = d.is_home.to_numpy() == 1
        shooter_n = np.where(is_home, hn, an)
        def_n = np.where(is_home, an, hn)
        d["slab"] = np.where(shooter_n == def_n, 0, 1)         # 0 = EV (equal), 1 = MA (man-advantage)
        d = d[shooter_n >= def_n]                              # EV, or PP-side shots only (drop shorthanded)
        d["goalie_id"] = np.where(d.is_home.to_numpy() == 1, d.away_goalie.to_numpy(), d.home_goalie.to_numpy())
        d = d[d.goalie_id.notna()]
        d["sidx"] = d.shooter_id.astype(int).map(idx)
        d = d[d.sidx.notna()]
        frames.append(d[["sidx", "goalie_id", "xg", "goal", "slab"]])
    if not frames or sum(len(f) for f in frames) == 0:
        return None
    D = pd.concat(frames, ignore_index=True)
    goalies = sorted(int(g) for g in D.goalie_id.astype(int).unique())
    gmap = {g: i for i, g in enumerate(goalies)}
    xg = np.clip(D.xg.to_numpy(float), EPS, 1 - EPS)
    return {"shooter_idx": D.sidx.astype(int).to_numpy(np.int64),
            "goalie_idx": D.goalie_id.astype(int).map(gmap).to_numpy(np.int64),
            "logit_xg": np.log(xg / (1 - xg)),
            "y": D.goal.to_numpy(np.float64),
            "strength": D.slab.to_numpy(np.int64),
            "goalies": goalies}


# ── the crossed design as a sparse matrix (for the exact analytic Hessian) ─────────────────────────

def _build_X(shooter, team, dfd, Xctx, P, def_mask=None):
    """Sparse design, columns [intercept | shoot(P) | create(P) | def(P) | context(k)]. `def_mask`
    (n × width) zeroes padded/PK-absent defender slots so masked entries contribute nothing."""
    n, k = len(shooter), Xctx.shape[1]
    ncol = 1 + 3 * P + k
    ar = np.arange(n)
    R, Cc, D = [], [], []

    def add_ind(col, val=None):
        R.append(ar); Cc.append(col); D.append(np.ones(n) if val is None else val)

    add_ind(np.zeros(n, int))                           # intercept
    add_ind(1 + shooter)                                # shoot block
    for j in range(team.shape[1]):
        add_ind(1 + P + team[:, j])                     # play block
    for j in range(dfd.shape[1]):                       # def block (masked: padded slots contribute 0)
        add_ind(1 + 2 * P + dfd[:, j], None if def_mask is None else def_mask[:, j])
    for c in range(k):                                  # context (values, not indicators)
        R.append(ar); Cc.append(np.full(n, 1 + 3 * P + c)); D.append(Xctx[:, c])
    return sparse.csr_matrix((np.concatenate(D), (np.concatenate(R), np.concatenate(Cc))),
                             shape=(n, ncol))


def _build_conv_X(logit_xg, sidx, shooter, goalie, S, P, G):
    """Sparse conversion design for the Hessian, columns
    [slope_s (S) | intercept_s (S) | fin(P) | gsave(G)] — the slope/intercept are per-strength (`sidx`
    in 0..S-1), so a shot loads its own strength's slope (value=logit_xg) and intercept (value=1)."""
    n = len(shooter)
    ar = np.arange(n)
    rows = [ar, ar, ar, ar]
    cols = [sidx, S + sidx, 2 * S + shooter, 2 * S + P + goalie]
    data = [logit_xg, np.ones(n), np.ones(n), np.ones(n)]
    return sparse.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(n, 2 * S + P + G))


def _pen_vec(P, k, lam_shoot, lam_create, lam_def):
    pen = np.zeros(1 + 3 * P + k)
    pen[1:1 + P] = lam_shoot
    pen[1 + P:1 + 2 * P] = lam_create
    pen[1 + 2 * P:1 + 3 * P] = lam_def
    return pen


def _se_from_hessian(H):
    return np.sqrt(np.clip(np.diag(np.linalg.inv(H)), 0.0, None))


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


def fit_rate_create(R, g_team, g_cidx, shots_per_goal, count_model="poisson"):
    """UNIFIED CREATION. One `create` parameter that does double duty:
      (i) lifts teammates' shot rate in the Poisson/NB count layer, and
      (ii) is the creator-credit propensity in a conditional logit over the on-ice teammates,
           anchored to OBSERVED goal assisters (g_team = the 4 teammate indices, g_cidx = the credited
           column: 0 = unassisted/create_0, 1..4 = teammate position).
    Fit jointly; `shots_per_goal` up-weights the sparse assist-credit — each observed goal-creator stands
    in for the ≈ shots/goal whose creator we never see (inverse-probability weighting) — so `create`
    blends 'I make my linemates shoot' with 'I'm the one credited with creating' — the volume half also
    captures the unobserved passing the assist labels miss. Returns shoot/create/def + create_0 + SEs
    (Poisson Hessian, with the credit Fisher added to the create diagonal)."""
    P, k = len(R["players"]), R["Xctx"].shape[1]
    lsh, lcr, ldf = 1 / PRIOR_SD_SHOOT ** 2, 1 / PRIOR_SD_CREATE ** 2, 1 / PRIOR_SD_SHOOT ** 2
    nb = count_model == "nb"
    PS = 1 + 3 * P + k                                       # index of create_0 in th
    dmask = R.get("def_mask")
    if dmask is None:                                        # synthetic/legacy rows: all defenders real
        dmask = np.ones_like(R["def_idx"], dtype=np.float64)

    def split(th):
        return th[1:1 + P], th[1 + P:1 + 2 * P], th[1 + 2 * P:1 + 3 * P], th[1 + 3 * P:1 + 3 * P + k], th[PS]

    def nll(th, sh_i, tm_i, df_i, dm, X, cnt, ofs, gt, gc):
        sh, cr, df, b, psi0 = split(th)
        eta = th[0] + sh[sh_i] + jnp.sum(cr[tm_i], 1) + jnp.sum(df[df_i] * dm, 1) + X @ b
        mu = jnp.exp(eta + ofs)
        if not nb:
            pois = jnp.sum(mu - cnt * jnp.log(mu))
        else:
            r = jnp.exp(th[-1])
            pois = -jnp.sum(gammaln(cnt + r) - gammaln(r) - gammaln(cnt + 1)
                            + r * jnp.log(r / (r + mu)) + cnt * jnp.log(mu / (r + mu)))
        logit5 = jnp.concatenate([jnp.full((gt.shape[0], 1), psi0), cr[gt]], axis=1)   # [create_0, create(4)]
        logpi = logit5 - logsumexp(logit5, axis=1)[:, None]
        credit = -jnp.sum(jnp.take_along_axis(logpi, gc[:, None], 1)[:, 0])
        pen = 0.5 * (lsh * jnp.sum(sh ** 2) + lcr * jnp.sum(cr ** 2) + ldf * jnp.sum(df ** 2))
        return pois + shots_per_goal * credit + pen

    x0 = np.zeros(1 + 3 * P + k + 1 + (1 if nb else 0))
    if nb:
        x0[-1] = 1.0
    res = _optimize(nll, x0, R["shooter_idx"], R["team_idx"], R["def_idx"], dmask, R["Xctx"], R["count"],
                    R["offset"], g_team, g_cidx)
    th = res.x
    sh, cr, df, b, psi0 = th[1:1 + P], th[1 + P:1 + 2 * P], th[1 + 2 * P:1 + 3 * P], th[1 + 3 * P:1 + 3 * P + k], float(th[PS])
    eta = th[0] + sh[R["shooter_idx"]] + cr[R["team_idx"]].sum(1) + (df[R["def_idx"]] * dmask).sum(1) + R["Xctx"] @ b
    mu = np.exp(eta + R["offset"])
    r = float(np.exp(th[-1])) if nb else None
    W = mu * r / (r + mu) if nb else mu
    Xs = _build_X(R["shooter_idx"], R["team_idx"], R["def_idx"], R["Xctx"], P, dmask)
    H = (Xs.T @ Xs.multiply(W[:, None])).toarray() + np.diag(_pen_vec(P, k, lsh, lcr, ldf))
    logit5 = np.concatenate([np.full((len(g_cidx), 1), psi0), cr[g_team]], 1)       # credit Fisher on create diag
    pi = np.exp(logit5 - logit5.max(1, keepdims=True)); pi /= pi.sum(1, keepdims=True)
    cinfo = np.zeros(P)
    for t in range(4):
        np.add.at(cinfo, g_team[:, t], shots_per_goal * pi[:, t + 1] * (1 - pi[:, t + 1]))
    di = np.arange(1 + P, 1 + 2 * P)
    H[di, di] += cinfo
    se = _se_from_hessian(H)
    return {"intercept": float(th[0]), "shoot": sh, "create": cr, "def": df, "beta": b, "psi0": psi0,
            "se_shoot": se[1:1 + P], "se_create": se[1 + P:1 + 2 * P], "se_def": se[1 + 2 * P:1 + 3 * P],
            "count_model": count_model, "r": r, "converged": bool(res.success),
            "grad_norm": float(np.max(np.abs(res.jac)))}


def fit_quality_creator(Q, P, creates):
    """QUALITY fit, POOLED across strengths (EV+MA). Estimates one shared set of qshoot/qcreate/qdef
    (danger is an intrinsic, strength-neutral skill — xG already encodes the man-advantage), with the
    strength environment absorbed by a `pp` context column (per-strength intercept). The creator
    distribution pi = softmax([create_0, create[teammates]]) is FIXED from the rate fit and chosen
    PER ROW by that shot's strength (`creates` = {strength_label: (create_array, psi0)}). Creator is
    OBSERVED on goals, MARGINALIZED over pi on non-goals. Defenders masked (≤5). SEs: diagonal
    Gauss-Newton."""
    n, k = len(Q["y"]), Q["Xctx"].shape[1]
    cre = Q["creator"]                                      # -1 latent, 0..3 teammate, 4 unassisted
    cidx_np = np.where(cre == 4, 0, np.clip(cre, 0, 3) + 1).astype(np.int64)  # col in [unassist, t0..t3]
    lqs, lqc, lqd = 1 / PRIOR_SD_QSHOOT ** 2, 1 / PRIOR_SD_QCREATE ** 2, 1 / PRIOR_SD_QSHOOT ** 2
    strength = Q.get("strength", np.zeros(n, dtype=np.int64))
    dmask = Q.get("def_mask")
    if dmask is None:
        dmask = np.ones_like(Q["def_idx"], dtype=np.float64)
    lg = np.zeros((n, 5))                                   # per-row creator logits by strength
    for st, (cr_s, psi_s) in creates.items():
        m = strength == st
        if m.any():
            lg[m] = np.concatenate([np.full((int(m.sum()), 1), psi_s), cr_s[Q["team_idx"][m]]], 1)
    pi_np = np.exp(lg - lg.max(1, keepdims=True)); pi_np /= pi_np.sum(1, keepdims=True)

    def split(th):
        return th[0], th[1:1 + P], th[1 + P:1 + 2 * P], th[1 + 2 * P:1 + 3 * P], th[1 + 3 * P:1 + 3 * P + k]

    def nll(th, sh_i, tm_i, df_i, dm, X, xg, goal, cidx, pi):
        mq, qs, qc, qd, b = split(th)
        base = mq + qs[sh_i] + jnp.sum(qd[df_i] * dm, 1) + X @ b
        sig5 = jnp.concatenate([jax.nn.sigmoid(base)[:, None],
                                jax.nn.sigmoid(base[:, None] + qc[tm_i])], axis=1)   # (n,5) col0=unassisted

        def fb(p):
            return xg * jnp.log(p + EPS) + (1 - xg) * jnp.log(1 - p + EPS)
        gs = jnp.take_along_axis(sig5, cidx[:, None], 1)[:, 0]       # goals: observed-creator quality
        marg = jnp.sum(pi * sig5, axis=1)                           # non-goals: marginalize over FIXED pi
        ll = jnp.where(goal, fb(gs), fb(marg))
        pen = 0.5 * (lqs * jnp.sum(qs ** 2) + lqc * jnp.sum(qc ** 2) + lqd * jnp.sum(qd ** 2))
        return -jnp.sum(ll) + pen

    res = _optimize(nll, np.zeros(1 + 3 * P + k), Q["shooter_idx"], Q["team_idx"], Q["def_idx"], dmask,
                    Q["Xctx"], _sigmoid(Q["y"]), Q["goal"].astype(bool), cidx_np, pi_np)
    mq, qs, qc, qd, b = (float(res.x[0]), res.x[1:1 + P], res.x[1 + P:1 + 2 * P],
                         res.x[1 + 2 * P:1 + 3 * P], res.x[1 + 3 * P:1 + 3 * P + k])
    # Beta concentration from residual MSE around the fitted mean
    base = mq + qs[Q["shooter_idx"]] + (qd[Q["def_idx"]] * dmask).sum(1) + Q["Xctx"] @ b
    sig5 = np.concatenate([_sigmoid(base)[:, None], _sigmoid(base[:, None] + qc[Q["team_idx"]])], 1)
    ci = cidx_np
    m = (pi_np * sig5).sum(1)                                    # marginal mean (non-goals), FIXED pi
    fitted = np.where(Q["goal"].astype(bool), sig5[np.arange(n), ci], m)
    mse = float(np.mean((_sigmoid(Q["y"]) - fitted) ** 2))
    s_conc = max(float(np.mean(fitted * (1 - fitted)) / max(mse, 1e-9) - 1.0), 1.0)

    # diagonal Gauss-Newton SE for qcreate (data-limited): observed-creator goal quality + non-goal marginal
    g = Q["goal"].astype(bool); tm = Q["team_idx"]
    info_qc = np.zeros(P)
    tg, sg, cig = tm[g], sig5[g], ci[g]
    is_tm = cig >= 1
    cr_pl = tg[np.arange(len(tg)), np.clip(cig - 1, 0, 3)]
    sc = sg[np.arange(len(sg)), cig]
    np.add.at(info_qc, cr_pl[is_tm], (sc * (1 - sc))[is_tm])
    pn, tn, sn, mn = pi_np[~g], tm[~g], sig5[~g], m[~g]
    den = np.clip(mn * (1 - mn), 1e-9, None)
    for t in range(4):
        np.add.at(info_qc, tn[:, t], (pn[:, t + 1] * sn[:, t + 1] * (1 - sn[:, t + 1])) ** 2 / den)
    se_qc = 1.0 / np.sqrt(info_qc + 1 / PRIOR_SD_QCREATE ** 2)
    n_create = np.zeros(P, dtype=np.int64)
    np.add.at(n_create, cr_pl[is_tm], 1)
    # per-strength quality intercept: EV = mq; PP = mq + (pp-column coefficient, Xctx col 1)
    mu_qual = {"ev": mq, "ma": mq + (float(b[1]) if len(b) > 1 else 0.0)}
    return {"intercept": mq, "mu_qual": mu_qual, "qshoot": qs, "qcreate": qc, "qdef": qd, "beta": b,
            "beta_s": s_conc, "se_qcreate": se_qc, "n_create": n_create, "converged": bool(res.success),
            "grad_norm": float(np.max(np.abs(res.jac)))}


# ── conversion PRE-CALCULATION: empirical-Bayes prior SDs (recomputed each fit, then held fixed) ─────

def _eb_prior_sd(count, exp_goals, made, var, min_shots, vbar, fallback):
    """Empirical-Bayes prior SD for one conversion offset block (finishing OR goalie) — the ridge
    analogue of shooting_model._estimate_k. For each entity aggregate its shots: N, expected goals
    Σxg, actual goals, and summed Bernoulli variance Σxg(1−xg). The per-shot residual rate
    r = (goals − Σxg)/N has, ACROSS high-volume entities, a volume-weighted spread equal to
    (true talent variance) + (mean sampling variance); subtract the latter (method of moments) to
    isolate the talent variance on the goals/shot scale. Map that to the LOGIT scale the fit uses via
    the local link derivative (a small logit offset δ shifts a shot's goal prob by ≈ p(1−p)·δ, so
    r ≈ vbar·offset ⇒ sd_logit = sd_prob / vbar). Falls back to `fallback` when <2 entities clear
    `min_shots`. This is computed OUTSIDE the fit and is a FIXED hyperparameter within it."""
    m = count >= min_shots
    if int(m.sum()) < 2:
        return fallback
    N, E, M, V = count[m].astype(float), exp_goals[m], made[m], var[m]
    r = (M - E) / N                                         # per-shot residual rate (goals/shot above xG)
    W = N / N.sum()
    rbar = float(np.sum(W * r))
    wvar = float(np.sum(W * (r - rbar) ** 2))               # observed spread of r
    msamp = float(np.sum(W * (V / N ** 2)))                 # mean sampling variance of r
    tau2_prob = max(wvar - msamp, 0.0)                      # talent variance (goals/shot), floored at 0
    sd_logit = np.sqrt(tau2_prob) / max(vbar, 1e-9)         # map goals/shot → logit scale
    return float(max(sd_logit, PRIOR_SD_FLOOR))


def estimate_conversion_prior_sds(Cr, P):
    """Pre-calculation stage for the conversion fit: estimate the finishing and goalie prior SDs from
    THIS fit's data (empirical Bayes), so the ridge shrinkage is data-calibrated rather than hand-set.
    Recomputed each fit and then held FIXED during the fit. Returns (sd_fin, sd_gsave) on the logit
    scale. Talent SD is estimated only from high-volume entities (MIN_SHOTS_*), then applied to all."""
    G = len(Cr["goalies"])
    xg = _sigmoid(Cr["logit_xg"])                           # recover observed xg from its logit
    v = xg * (1.0 - xg)
    vbar = float(v.mean()) if len(v) else 1e-9              # mean per-shot Bernoulli variance p(1−p)

    def blocks(idx, n):
        return (np.bincount(idx, minlength=n).astype(float),      # N
                np.bincount(idx, weights=xg, minlength=n),         # Σxg
                np.bincount(idx, weights=Cr["y"], minlength=n),    # goals
                np.bincount(idx, weights=v, minlength=n))          # ΣV

    sc, se_, sm, sv = blocks(Cr["shooter_idx"], P)
    gc, ge, gm, gv = blocks(Cr["goalie_idx"], G)
    sd_fin = _eb_prior_sd(sc, se_, sm, sv, MIN_SHOTS_FIN_EST, vbar, PRIOR_SD_FIN)
    sd_gsave = _eb_prior_sd(gc, ge, gm, gv, MIN_SHOTS_GSAVE_EST, vbar, PRIOR_SD_GSAVE)
    return sd_fin, sd_gsave


# ── conversion fit (native logit, keyed off the observed xG) ────────────────────────────────────────

def fit_conversion(Cr, P):
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

    def split(th):
        return th[0:S], th[S:2 * S], th[2 * S:2 * S + P], th[2 * S + P:2 * S + P + G]

    def nll(th, si, sh_i, g_i, lxg, y):
        a, b, fin, gsave = split(th)
        eta = a[si] * lxg + b[si] + fin[sh_i] + gsave[g_i]
        bce = jnp.sum(jax.nn.softplus(eta) - y * eta)       # = −Σ[y·log p + (1−y)·log(1−p)]
        pen = 0.5 * (lf * jnp.sum(fin ** 2) + lg * jnp.sum(gsave ** 2))
        return bce + pen

    x0 = np.zeros(2 * S + P + G)
    x0[0:S] = 1.0                                           # start each strength's slope at 1
    res = _optimize(nll, x0, sidx, Cr["shooter_idx"], Cr["goalie_idx"], Cr["logit_xg"], Cr["y"])
    a_v, b_v = res.x[0:S], res.x[S:2 * S]
    fin, gsave = res.x[2 * S:2 * S + P], res.x[2 * S + P:2 * S + P + G]

    eta = a_v[sidx] * Cr["logit_xg"] + b_v[sidx] + fin[Cr["shooter_idx"]] + gsave[Cr["goalie_idx"]]
    p = _sigmoid(eta); W = p * (1 - p)
    X = _build_conv_X(Cr["logit_xg"], sidx, Cr["shooter_idx"], Cr["goalie_idx"], S, P, G)
    pen = np.zeros(2 * S + P + G); pen[2 * S:2 * S + P] = lf; pen[2 * S + P:] = lg
    H = (X.T @ X.multiply(W[:, None])).toarray() + np.diag(pen)
    se = _se_from_hessian(H)
    a = {k: float(v) for k, v in zip(keys, a_v)}
    b = {k: float(v) for k, v in zip(keys, b_v)}
    recon = "  ".join(f"{k}:Σp={p[sidx == i].sum():.0f}/Σy={Cr['y'][sidx == i].sum():.0f}"
                      for i, k in enumerate(keys))
    print(f"  conversion pre-calc EB prior SDs (per-fit): fin={prior_sd_fin:.3f} gsave={prior_sd_gsave:.3f} (logit)")
    print(f"  conversion (logit) fit: a={a} b={ {k: round(v,3) for k,v in b.items()} }  reconcile {recon}"
          f"  ({len(p):,} shots, {G} goalies)")
    return {"a": a, "b": b, "fin": fin, "gsave": gsave, "goalies": Cr["goalies"],
            "se_fin": se[2 * S:2 * S + P], "se_gsave": se[2 * S + P:2 * S + P + G],
            "prior_sd_fin": prior_sd_fin, "prior_sd_gsave": prior_sd_gsave,
            "sum_p": float(p.sum()), "sum_y": float(Cr["y"].sum()), "n": int(len(p)),
            "converged": bool(res.success), "grad_norm": float(np.max(np.abs(res.jac)))}


# ── attribution: merge the rate (volume) and quality loadings into goal-scale numbers ──────────────

def player_values(rates, qual, conv, players):
    """Per-strength deployment-free per-60 goal values. `rates` = {"ev": rate_fit, "ma": rate_fit}.
    For each strength (rate loadings are per-strength; quality/finishing loadings are POOLED, only the
    intercept splits):
       scoring(j)    = exp(mu_rate+shoot) · sigmoid(a·logit(sigmoid(mu_qual+qshoot)) + b + fin)   own shots, converted
       playmaking(p) = N_TM · exp(mu_rate) · (exp(create)−1) · sigmoid(mu_qual+qcreate)           teammate xG added
       defense(d)    = N_DEF · [exp(mu_rate)·sigmoid(mu_qual) − exp(mu_rate+def)·sigmoid(mu_qual+qdef)]  suppression
    EV → ev_scoring/playmaking/defense (N_DEF=5). MA → pp_scoring/pp_playmaking + pk_defense (N_DEF=4;
    the MA `def` loadings ARE the penalty-killers). All per 60; defense >0 = suppresses (good)."""
    out = {}
    qshoot, qcreate, qdef, fin = qual["qshoot"], qual["qcreate"], qual["qdef"], conv["fin"]
    for key, rate in rates.items():
        ml, mq = rate["intercept"], qual["mu_qual"][key]
        n_def = N_DEF_EV if key == "ev" else N_DEF_MA
        a, b = conv["a"][key], conv["b"][key]
        cr, shoot, defn = rate["create"], rate["shoot"], rate["def"]
        shots = np.exp(ml + shoot)
        q_own = np.clip(_sigmoid(mq + qshoot), EPS, 1 - EPS)
        lq = np.log(q_own / (1 - q_own))
        p_own = _sigmoid(a * lq + b + fin)
        p_own0 = _sigmoid(a * lq + b)
        base = np.exp(ml) * _sigmoid(mq)
        out[key] = {
            "scoring": shots * p_own,                          # own shots, converted to goals
            "finishing": shots * (p_own - p_own0),             # goals above xG-implied conversion
            "own_xg": shots * _sigmoid(mq + qshoot), "own_shots": shots,
            "playmaking": N_TM * np.exp(ml) * (np.exp(cr) - 1.0) * _sigmoid(mq + qcreate),
            "defense": n_def * (base - np.exp(ml + defn) * _sigmoid(mq + qdef)),
            "creator_share": np.exp(cr) / (np.exp(rate["psi0"]) + np.exp(cr) + (N_TM - 1)),
        }
    return out


# ── simulation + posterior-predictive check ───────────────────────────────────────────────────────

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def ppc(R, rate, qual, conv, key, seed=0):
    """Forward-simulate every shooter-stint row of one strength bucket (`key` ∈ {'ev','ma'}) and compare
    to actuals: shot count (rate), shot quality (pooled quality at the strength intercept), total goals
    (conversion at the strength's a/b). Defenders masked (≤5)."""
    rng = np.random.default_rng(seed)
    dmask = R.get("def_mask")
    if dmask is None:
        dmask = np.ones_like(R["def_idx"], dtype=np.float64)
    eta_r = (rate["intercept"] + rate["shoot"][R["shooter_idx"]] + rate["create"][R["team_idx"]].sum(1)
             + (rate["def"][R["def_idx"]] * dmask).sum(1) + R["Xctx"] @ rate["beta"])
    mu = np.exp(eta_r + R["offset"])
    if rate.get("count_model") == "nb" and rate.get("r"):
        r = rate["r"]
        N = rng.negative_binomial(r, r / (r + mu))
    else:
        N = rng.poisson(mu)

    # quality marginalized over the creator, at this strength's quality intercept (mu_qual[key])
    base = (qual["mu_qual"][key] + qual["qshoot"][R["shooter_idx"]] + (qual["qdef"][R["def_idx"]] * dmask).sum(1)
            + qual["beta"][0] * R["Xctx"][:, 0])
    sig5 = np.concatenate([_sigmoid(base)[:, None], _sigmoid(base[:, None] + qual["qcreate"][R["team_idx"]])], 1)
    lg = np.concatenate([np.full((len(base), 1), rate["psi0"]), rate["create"][R["team_idx"]]], 1)
    pi = np.exp(lg - lg.max(1, keepdims=True)); pi /= pi.sum(1, keepdims=True)
    qpred_marg = (pi * sig5).sum(1)
    gsave_map = {g: conv["gsave"][i] for i, g in enumerate(conv["goalies"])}
    fin_row = conv["fin"][R["shooter_idx"]]                                    # per shooter-stint row
    gsv_row = np.array([gsave_map.get(int(g), 0.0) if pd.notna(g) else 0.0 for g in R["def_goalie"]])

    rep = np.repeat(np.arange(len(N)), N)
    p_q = qpred_marg[rep]
    s = qual["beta_s"]
    q = rng.beta(s * p_q, s * (1 - p_q))                                      # this chance's drawn xG
    qc = np.clip(q, EPS, 1 - EPS)
    eta = conv["a"][key] * np.log(qc / (1 - qc)) + conv["b"][key] + fin_row[rep] + gsv_row[rep]
    p_goal = _sigmoid(eta)                                                    # logit conversion; bounded
    goals_sim = int(rng.binomial(1, p_goal).sum())
    return {
        "shots_actual": int(R["count"].sum()), "shots_sim": int(N.sum()),
        "shots_expected": float(mu.sum()),
        "perrow_mean_actual": float(R["count"].mean()), "perrow_mean_sim": float(N.mean()),
        "perrow_var_actual": float(R["count"].var()), "perrow_var_sim": float(N.var()),
        "zero_frac_actual": float((R["count"] == 0).mean()), "zero_frac_sim": float((N == 0).mean()),
        "mean_xg_sim": float(q.mean()), "goals_sim": goals_sim, "n_sim_shots": int(rep.size),
    }


# ── leaderboards + run ─────────────────────────────────────────────────────────────────────────────

def _board(names, players, val, toi, label, se=None, higher=True, n=12, min_toi=SNIFF_MIN_TOI):
    elig = toi >= min_toi
    order = np.argsort(val * (-1 if higher else 1))
    order = [i for i in order if elig[i]][:n]
    print(f"\n{label}")
    for i in order:
        pid = players[i]; nm = names.get(int(pid), {}).get("name", f"#{pid}")
        if se is not None:
            z = val[i] / se[i] if se[i] > 0 else 0.0
            flag = "" if abs(z) >= 2 else "  ⚠ low-conf"
            print(f"   {val[i]:+.3f} ±{1.96 * se[i]:.3f} (z={z:+.1f}){flag}  {nm:24s} ({toi[i] / 60:.0f} min)")
        else:
            print(f"   {val[i]:+.3f}  {nm:24s} ({toi[i] / 60:.0f} min)")


def run(seasons, count_model="poisson"):
    names = roster_names(seasons)
    print(f"[generative_model:shooter-resolved] seasons {seasons} — count {count_model} — EV + PP/PK …")
    players, idx = player_index(seasons)
    if not players:
        raise SystemExit("no stints")
    P = len(players)

    # pooled quality rows (EV+MA); its goal subset (split by strength) anchors each bucket's `create`
    Q = quality_creator_rows(seasons, idx, ALL_STRENGTHS)
    cidx_all = np.where(Q["creator"] == 4, 0, np.clip(Q["creator"], 0, 3) + 1).astype(np.int64)

    rates, spg = {}, {}
    for key, strengths, dual in [("ev", EV_STRENGTHS, True), ("ma", MA_STRENGTHS, False)]:
        R = rate_rows(seasons, strengths, dual, players, idx)
        if R is None or len(R["count"]) == 0:
            print(f"  [{key}] no stints — skipping"); continue
        slab = 0 if key == "ev" else 1
        gm = (Q["goal"] == 1) & (Q["strength"] == slab)
        ngoal = int(gm.sum()); nun = int(((Q["creator"] == 4) & (Q["strength"] == slab)).sum())
        spg[key] = float(R["count"].sum()) / max(ngoal, 1)     # per-strength shots-per-goal (IPW weight)
        print(f"  [{key}] rate rows {len(R['count']):,}  shots {R['count'].sum():.0f}  goals {ngoal}  "
              f"spg {spg[key]:.1f}")
        rate = fit_rate_create(R, Q["team_idx"][gm], cidx_all[gm], spg[key], count_model=count_model)
        rate["R"] = R
        rates[key] = rate
        rdesc = f" r={rate['r']:.2f}" if rate.get("r") else ""
        print(f"    rate fit ({count_model}): converged={rate['converged']} |grad|={rate['grad_norm']:.1e}"
              f"{rdesc} create_0={rate['psi0']:+.3f} ({100 * nun / max(ngoal, 1):.0f}% unassisted)")
    if "ev" not in rates:
        raise SystemExit("no EV stints")

    # pooled quality fit (shared loadings; per-strength intercept; per-row creator dist by strength)
    creates = {0: (rates["ev"]["create"], rates["ev"]["psi0"])}
    if "ma" in rates:
        creates[1] = (rates["ma"]["create"], rates["ma"]["psi0"])
    qual = fit_quality_creator(Q, P, creates)
    print(f"  quality fit (pooled EV+MA): converged={qual['converged']} |grad|={qual['grad_norm']:.1e} "
          f"beta_s={qual['beta_s']:.1f} mu_qual={ {k: round(v, 3) for k, v in qual['mu_qual'].items()} }")

    # pooled conversion fit (shared fin/gsave; per-strength a/b)
    Cr = conversion_rows(seasons, idx, ALL_STRENGTHS)
    if Cr is None:
        raise SystemExit("no shots for the conversion fit")
    conv = fit_conversion(Cr, P)
    vals = player_values(rates, qual, conv, players)

    # pooled (shared) skill leaderboards — once
    ev_toi = rates["ev"]["R"]["toi"]
    _board(names, players, qual["qshoot"], ev_toi, "TOP qshoot (own-shot danger, pooled):")
    _board(names, players, qual["qcreate"], ev_toi, "TOP qcreate (setup danger, pooled):", qual["se_qcreate"])
    _board(names, players, conv["fin"], ev_toi, "TOP FINISHING (log-odds above xG, pooled):", conv["se_fin"])
    fz = (np.abs(conv["fin"]) > 2 * conv["se_fin"]) & (ev_toi >= SNIFF_MIN_TOI)
    print(f"  conversion identification: fin |z|>2 in {int(fz.sum())}/{int((ev_toi >= SNIFF_MIN_TOI).sum())} "
          f"eligible (weak-signal — expected)")

    # per-strength rate/value leaderboards + posterior-predictive check
    ppcs = {}
    for key in rates:
        R = rates[key]["R"]
        off, dfn = R["toi_atk"], R["toi_def"]
        gate = SNIFF_MIN_TOI if key == "ev" else SNIFF_MIN_TOI_MA
        lbl = "EV" if key == "ev" else "PP"
        print(f"\n──[{lbl}] rate/value leaderboards ──")
        _board(names, players, rates[key]["create"], off, f"[{lbl}] TOP create (playmaking volume):", rates[key]["se_create"], min_toi=gate)
        _board(names, players, vals[key]["scoring"], off, f"[{lbl}] TOP SCORING (goals/60, own shots):", min_toi=gate)
        _board(names, players, vals[key]["playmaking"], off, f"[{lbl}] TOP PLAYMAKING (xG/60):", min_toi=gate)
        _board(names, players, vals[key]["defense"], dfn, f"[{'EV' if key == 'ev' else 'PK'}] BEST DEFENSE (xG/60 suppressed):", min_toi=gate)
        pp = ppc(R, rates[key], qual, conv, key)
        slab = 0 if key == "ev" else 1
        pp["goals_actual"] = int(((Q["goal"] == 1) & (Q["strength"] == slab)).sum())
        ppcs[key] = pp
        print(f"  [{lbl}] PPC — shots a/s {pp['shots_actual']:,}/{pp['shots_sim']:,} "
              f"(model-exp {pp['shots_expected']:.0f})  goals a/s {pp['goals_actual']:,}/{pp['goals_sim']:,}")

    _save(seasons, players, rates, qual, conv, vals, ppcs, spg)


def _save(seasons, players, rates, qual, conv, vals, ppcs, spg):
    label = "+".join(map(str, seasons))
    ev, ma = rates.get("ev"), rates.get("ma")
    out = {"model": "generative_model_ev_ma", "seasons": list(seasons),
           "count_model": ev.get("count_model"), "nb_r": ev.get("r"), "shots_per_goal": spg,
           "quality_intercept": float(qual["intercept"]), "mu_qual": qual["mu_qual"],
           "beta_s": qual["beta_s"],
           "conv": {"a": conv["a"], "b": conv["b"], "prior_sd_fin": conv["prior_sd_fin"],
                    "prior_sd_gsave": conv["prior_sd_gsave"], "sum_p": conv["sum_p"], "sum_y": conv["sum_y"]},
           "strengths": {k: {"rate_intercept": float(rates[k]["intercept"]), "psi0": float(rates[k]["psi0"]),
                             "ppc": ppcs.get(k)} for k in rates},
           "players": [], "goalies": [{"id": int(conv["goalies"][j]), "gsave": float(conv["gsave"][j]),
                                       "gsave_se": float(conv["se_gsave"][j])}
                                      for j in range(len(conv["goalies"]))]}
    for i in range(len(players)):
        rec = {"id": int(players[i]),
               "toi_ev": float(ev["R"]["toi_atk"][i]) if ev else 0.0,
               "toi_pp": float(ma["R"]["toi_atk"][i]) if ma else 0.0,
               "toi_pk": float(ma["R"]["toi_def"][i]) if ma else 0.0,
               "qshoot": float(qual["qshoot"][i]), "qcreate": float(qual["qcreate"][i]),
               "qcreate_se": float(qual["se_qcreate"][i]), "qdef": float(qual["qdef"][i]),
               "n_create": int(qual["n_create"][i]),
               "fin": float(conv["fin"][i]), "fin_se": float(conv["se_fin"][i]),
               "ev_defense": float(vals["ev"]["defense"][i]) if "ev" in vals else 0.0,
               "pk_defense": float(vals["ma"]["defense"][i]) if "ma" in vals else 0.0}
        for key, pre in [("ev", "ev"), ("ma", "pp")]:
            if key in rates:
                rec[f"{pre}_shoot"] = float(rates[key]["shoot"][i])
                rec[f"{pre}_create"] = float(rates[key]["create"][i])
                rec[f"{pre}_create_se"] = float(rates[key]["se_create"][i])
                rec[f"{pre}_def"] = float(rates[key]["def"][i])
                rec[f"{pre}_scoring"] = float(vals[key]["scoring"][i])
                rec[f"{pre}_playmaking"] = float(vals[key]["playmaking"][i])
                rec[f"{pre}_finishing"] = float(vals[key]["finishing"][i])
        out["players"].append(rec)
    C.MODELS.mkdir(parents=True, exist_ok=True)
    (C.MODELS / f"generative_model_{label}.json").write_text(json.dumps(out))
    print(f"\n  -> data/models/generative_model_{label}.json")


def main(argv=None):
    p = argparse.ArgumentParser(description="Experimental shooter-resolved generative model — POC")
    p.add_argument("--season", type=int, default=None, help="one season (default: latest available)")
    p.add_argument("--pool", action="store_true", help="pool all available seasons")
    p.add_argument("--count", choices=["poisson", "nb"], default="poisson",
                   help="count layer: poisson (Var=μ) or nb (negative binomial, Var=μ+μ²/r)")
    args = p.parse_args(argv)
    sd = C.PROCESSED / "shots_onice"
    avail = sorted(int(f.stem) for f in sd.glob("*.parquet")) if sd.exists() else []
    if not avail:
        raise SystemExit("no processed shots — run `make stints` first")
    seasons = avail if args.pool else ([args.season] if args.season else [avail[-1]])
    run(seasons, count_model=args.count)


if __name__ == "__main__":
    main()
