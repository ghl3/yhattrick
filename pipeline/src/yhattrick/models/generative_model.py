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
                    shoot_j[s]           how much j shoots himself           (SCORING volume)
                    create_p[s]          UNIFIED creation — lifts teammates' shot rate AND is p's
                                           creator-credit propensity          (PLAYMAKING)
                    create_0             unassisted propensity (shooter self-created); one global scalar
                    def_d[s]             opponent shot-rate suppression      (DEFENSE; <0 good)
                    beta_rate            context (home, O/D-zone, lead/trail, season, position, AGE)
  quality layer     mu_qual              baseline shot quality (logit mean xG)
                    qshoot_j             danger of j's own shots             (SCORING quality)
                    qcreate_{F,D}        danger the ONE creator adds — POSITION-level (A1: per-player
                                           creation quality is unidentifiable, docs §7)
                    qdef_d               opponent danger suppression         (DEFENSE quality; <0 good)
                    beta_qual ; s        context ; Beta concentration (shot-to-shot spread, post-hoc MoM)
  conversion layer  a ; b                logit-conversion slope + intercept (per strength, UNPENALIZED →
                                           b's score eqn gives Σp=Σy exactly; + per-season offsets, F6)
                    fin_j                finishing — logit offset above xG on own shots
                    gsave_g              goalie save effect — logit offset (<0 good)

PLAYER CURVES (aging + drift — how skill moves over time; docs/generative_model.md "Player curves")
  [s] above marks the EV rate blocks that carry one STATE per (player, active-season) pair, tied
  across seasons by a random-walk prior: θ_first ~ N(0, prior_sd²), θ_next ~ N(θ_prev, rw_sd²·gap).
  One pooled fit yields each player's whole trajectory; his LAST state is the "current skill" read.
  On top, shared F/D AGING CURVES enter every stage as context columns — quadratic in
  z=(age−27)/10 for shoot/create/def (per bucket), finishing, and goalie age — plus D-intercept
  offsets (A2) so ridge shrinkage stops fighting positional baselines. MA rate blocks, quality and
  fin/gsave stay static levels (data too thin for drift). PROJECTION: advance ages to the target
  season, hold states at their last value (the RW mean), re-run the value attribution.

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
  added to the rate NLL, up-weighted by spg = the bucket's shots-per-goal computed from the data
  (≈16 at 5v5; `--spg-scale` multiplies it for sensitivity checks). The SAME `create` thus blends
  'I make my linemates shoot' (dense volume) with 'I'm credited with the setup' (sparse assist
  signal). SEs on `create` are sandwich-corrected: the up-weighted pseudo-counts add curvature but
  not w× real information (F1).

LIKELIHOOD  (factorizes into disjoint blocks; independent ridge priors)
  ℓ(θ) =  Σ_(side,j) log Pois/NB( N_j | rate_j·t/3600 )     ← rate    (mu_rate,shoot,create,def,beta_rate,r)
        + spg · Σ_goals log pi[assister]                    ← assist-credit anchor (shares create, create_0)
        + Σ_shots  log fracBern( xg | qbar )                ← quality (mu_qual,qshoot,qcreate_{F,D},qdef,
                     (creator OBSERVED on goals; MARGINALIZED           beta_qual; s post-hoc)
                      over fixed pi on non-goals and off-ice-assister goals)
        + Σ_shots  log Bernoulli( y | sigmoid(a·logit(xg)+b+b_season+fin+gsave) ) ← conversion (fit
                     natively off the OBSERVED xg — no shooting-model reuse; a,b,b_season unpenalized)
        − ridge on every player block (= EB Normal(0,σ²) priors) − RW penalty on the EV state chains

CODE↔GLOSSARY MAP: create_0→`psi0`; mu_rate/qual→ each fit's `intercept`; a/b→`conv["a"]`/`conv["b"]`;
  beta_rate/qual→`beta`; qbar→`sig5`; pi_c→`pi`; fin_j→`conv["fin"]`; gsave_g→`conv["gsave"]`.
────────────────────────────────────────────────────────────────────────────────────────────────

Fit: MAP / penalized MLE by OPTIMIZATION (JAX autodiff gradient + scipy L-BFGS-B). Uncertainties via
the HESSIAN (Laplace): H = XᵀWX + K (K = level ridge + RW precision); dense inverse when small, sparse
splu column solves for the reported columns when the state expansion outgrows it; `se_create` is
sandwich-corrected (F1). The count layer is configurable (`--count poisson|nb`). The conversion ridge
prior SDs (fin, gsave) are NOT hand-set: a PRE-CALCULATION stage (estimate_conversion_prior_sds)
estimates the per-player talent SD from the data each fit — empirical Bayes, the logit analogue of
shooting_model._estimate_k — then holds it FIXED during the fit. Usage:
  uv run --group experimental python -m yhattrick.models.generative_model            # latest season
  uv run --group experimental python -m yhattrick.models.generative_model --count nb # negative binomial
  uv run --group experimental python -m yhattrick.models.generative_model --pool     # all seasons
                                       # (multi-season ⇒ RW drift states + projection block in the JSON)
  uv run --group experimental python -m yhattrick.models.generative_model --pool --spg-scale 0.5
                                       # A3: assist-credit weight sensitivity (rerun with 2.0, compare)
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.sparse.linalg import splu
from scipy.special import betaincinv

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
# The goal-assist credit up-weight (Stage 1) is NOT a constant: it is the per-strength shots-per-goal
# ratio, computed from the data each run (`spg` in run()). We observe a shot's creator only on the
# ~1-in-16 Fenwick shots that ARE goals (via the primary assist), so weighting each observed
# goal-creator by ≈ shots/goal makes it stand in for the shots whose creator we never see
# (inverse-probability weighting). `--spg-scale` multiplies it for sensitivity checks (A3).
# Random-walk drift prior SDs (per season, on each block's own scale): a player's per-season state
# moves N(prev, rw_sd²·gap). Small = states glued across seasons (static model at 0); large = each
# season fit independently. Hand-set; the EB analogue (estimate within-player drift variance from
# the data) is a documented follow-up.
RW_SD_SHOOT = 0.10          # own-shot volume drifts the most (role/deployment changes)
RW_SD_CREATE = 0.05         # per-teammate creation lift — tighter, like its level prior
RW_SD_DEF = 0.10
AGE_PEAK = 27.0             # age basis B(a) = [z, z²], z = (a − AGE_PEAK)/AGE_SCALE; B(27) = 0 so the
AGE_SCALE = 10.0            #   curve is a deviation from peak-age production (missing DOB ⇒ z = 0)
DENSE_H_MAX = 8000          # params ≤ this: dense Hessian inverse; above: sparse splu column solves
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


# ── game venues (the arena recording-bias offsets) ──────────────────────────────────────────────

# Same building, new sponsor name → alias to the CURRENT name so a venue's bias states stay one
# random-walk chain across the rename (scorer crew doesn't change with the signage). Real building
# moves stay split on purpose: Gila River → Mullett → Delta Center, Joe Louis → Little Caesars,
# Nassau/Barclays → UBS are different rinks with different crews. Covers 2016+ for the backfill.
_VENUE_ALIAS = {
    "STAPLES Center": "Crypto.com Arena",
    "FLA Live Arena": "Amerant Bank Arena",
    "BB&T Center": "Amerant Bank Arena",
    "Amalie Arena": "Benchmark International Arena",
    "PNC Arena": "Lenovo Center",
    "Wells Fargo Center": "Xfinity Mobile Arena",
    "Xcel Energy Center": "Grand Casino Arena",
    "Pepsi Center": "Ball Arena",
    "Scottrade Center": "Enterprise Center",
    "Verizon Center": "Capital One Arena",
    "Air Canada Centre": "Scotiabank Arena",
    "Bell MTS Place": "Canada Life Centre",
    "MTS Centre": "Canada Life Centre",
    "First Niagara Center": "KeyBank Center",
    "Consol Energy Center": "PPG Paints Arena",
}


def _game_venues(seasons):
    """{nhl_game_id: venue name} for all downloaded games in the seasons. NHL scorekeeping varies by
    building (shot counts AND recorded locations), so the rate/quality stages carry per-venue
    offsets. Venue comes from raw pbp `venue.default` (building-keyed, so Arizona's moves and
    relocations split correctly); cached to interim/game_venue/<season>.parquet on first build."""
    out = {}
    for s in sorted(set(seasons)):
        cache = C.INTERIM / "game_venue" / f"{s}.parquet"
        if cache.exists():
            df = pd.read_parquet(cache)
        else:
            rows = []
            for pf in sorted(C.RAW_PBP.glob(f"{s}0*.json")):
                try:
                    d = json.loads(pf.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                v = (d.get("venue") or {}).get("default") or (d.get("homeTeam") or {}).get("abbrev")
                if v:
                    rows.append({"nhl_game_id": int(pf.stem), "venue": str(v)})
            df = pd.DataFrame(rows, columns=["nhl_game_id", "venue"])
            cache.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache)
        out.update({int(g): _VENUE_ALIAS.get(v, v)
                    for g, v in zip(df.nhl_game_id.astype(int), df.venue)})
    return out


ARENA_MIN_GAMES = 20        # venues with fewer games in the fit window (outdoor/neutral sites)
                            #   get no offset — too few games to estimate one
ARENA_SD = 0.05             # ridge SD on each (venue, season) state — biases are small, shrink to 0
                            #   (a nuisance prior; also what identifies the block, so no reference
                            #   venue is needed)
ARENA_RW_SD = 0.03          # per-season random-walk SD between a venue's consecutive states —
                            #   scorer crews persist but do change; stable bias ⇒ states glue into
                            #   one pooled effect, a crew change ⇒ the data can move it


def _arena_index(seasons, min_games=ARENA_MIN_GAMES):
    """(Venue, SEASON)-state index for the arena recording-bias offsets. Rink bias is a scorer-crew
    phenomenon — persistent across adjacent seasons, movable when crews change — so each major venue
    gets one state per season it hosts games, ridged to zero (ARENA_SD) and smoothed across seasons
    by a random-walk penalty (ARENA_RW_SD). Rare venues (< min_games in the window: outdoor/neutral
    sites) map to −1 (no offset). Returns (game→venue map, {(venue, season): col}, pair machinery
    dict or None)."""
    ven = _game_venues(seasons)
    if not ven:
        return {}, {}, None
    counts = pd.Series(list(ven.values())).value_counts()
    majors = {v for v, c in counts.items() if c >= min_games}
    sset = set(int(s) for s in seasons)
    pairs = sorted({(v, int(str(g)[:4])) for g, v in ven.items()
                    if v in majors and int(str(g)[:4]) in sset})
    if not pairs:
        return ven, {}, None
    pmap = {p: i for i, p in enumerate(pairs)}
    pv = [p[0] for p in pairs]
    ps = np.array([p[1] for p in pairs], dtype=np.int64)
    same = np.array([pv[i + 1] == pv[i] for i in range(len(pv) - 1)], dtype=bool)
    e_prev = np.nonzero(same)[0].astype(np.int64)
    e_next = e_prev + 1
    mach = {"venue": pv, "season": ps, "e_prev": e_prev, "e_next": e_next,
            "e_gap": (ps[e_next] - ps[e_prev]).astype(np.float64)}
    return ven, pmap, mach


# ── age & position (the shared aging-curve inputs) ──────────────────────────────────────────────

def _birthdates(ids):
    """Birthdates for a list of player/goalie ids from the raw landing JSONs (NaT if missing)."""
    born = []
    for pid in ids:
        f = C.RAW_PLAYERS / f"{int(pid)}.json"
        b = json.loads(f.read_text()).get("birthDate") if f.exists() else None
        born.append(b)
    return pd.to_datetime(pd.Series(born), errors="coerce")


def _season_age(born, season):
    """Float age at Jan 1 of season+1 (mid-season) for a datetime Series; NaN if unknown."""
    return (pd.Timestamp(year=int(season) + 1, month=1, day=1) - born).dt.days.to_numpy() / 365.25


def _age_position(players, seasons):
    """Per-player ages and positions for the aging curves. Returns
    {"age": {season: (P,) float, NaN if unknown}, "z": {season: (P,) float, 0 if unknown},
     "isD": (P,) float 0/1, "missing": int}. Age = years at Jan 1 of season+1 (mid-season);
    z = (age − AGE_PEAK)/AGE_SCALE. A missing birthdate maps to z = 0 — the player sits AT the
    curve's reference point (contributes no age signal) instead of biasing the curve. Position from
    the season rosters; D = 1, anything else (F, unknown) = 0."""
    P = len(players)
    born = _birthdates(players)
    missing = int(born.isna().sum())
    names = roster_names(list(seasons))
    isD = np.array([1.0 if names.get(int(p), {}).get("pos") == "D" else 0.0 for p in players])
    age, z = {}, {}
    for s in sorted(set(seasons)):
        a = _season_age(born, s)
        age[s] = a
        z[s] = np.where(np.isnan(a), 0.0, (a - AGE_PEAK) / AGE_SCALE)
    return {"age": age, "z": z, "isD": isD, "missing": missing}


def _age_cols(z, d):
    """Position-split age-basis columns for skaters: [F·z, F·z², D·z, D·z²] (last axis).
    `z`/`d` broadcast — scalars or arrays."""
    z, d = np.asarray(z, dtype=np.float64), np.asarray(d, dtype=np.float64)
    f = 1.0 - d
    return np.stack([f * z, f * z * z, d * z, d * z * z], axis=-1)


def _curve_val(w, z, d):
    """Evaluate a fitted position-split age curve (coeffs w = [Fz, Fz², Dz, Dz²]) at z for
    position d (0=F, 1=D). Broadcasts over arrays."""
    return (1.0 - d) * (w[0] * z + w[1] * z * z) + d * (w[2] * z + w[3] * z * z)


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


AGE_CTX = ["shoot_D", "create_D", "def_D",
           "shoot_zF", "shoot_z2F", "shoot_zD", "shoot_z2D",
           "create_zF", "create_z2F", "create_zD", "create_z2D",
           "def_zF", "def_z2F", "def_zD", "def_z2D"]


def rate_rows(seasons, strengths, dual, players, idx, agepos=None, states=False, arenas=True):
    """Shooter-resolved rate design for ONE strength bucket, on the SHARED player index. For each stint
    side that attacks (EV/dual: both sides; MA: only the more-skaters side) and each on-ice attacker j
    as focal shooter: one row with j's Fenwick count, j's index, the 4 teammate indices, the (≤5)
    defender indices padded to width 5 with a `def_mask`, offset log(t/3600), and context. Context =
    5 base cols + season indicators + (when `agepos` is given) the position-offset and age-basis
    columns for the shoot/create/def blocks (AGE_CTX — the shared F/D aging curves + D intercepts).
    `ctx_names` names every context column so downstream code extracts curve coefficients by name.
    Also returns per-player attacking/defending TOI and each player's last active season.

    `states=True` (the EV bucket) additionally emits the per-(player, season) UNIT machinery for the
    random-walk drift states: `unit_player`/`unit_season` (compact index over active pairs, sorted by
    player then season), `shooter_unit`/`team_unit`/`def_unit` gathers (int32; masked def slots → 0),
    the RW edge list (`e_prev`, `e_next` unit positions, `e_gap` season gaps), `first_mask` (which
    unit carries the level ridge), per-unit attacking TOI, and `unit_lut` ((P, nS) player×season →
    unit, −1 if inactive) for mapping goal rows in run()."""
    df = _load_stints(seasons, strengths)
    if df.empty:
        return None
    counts = _shooter_counts(seasons, strengths)
    P = len(players)
    scol, nseas = _season_cols(seasons)
    slist = sorted(set(seasons))
    AC = {s: _age_cols(agepos["z"][s], agepos["isD"]) for s in slist} if agepos else None
    isD = agepos["isD"] if agepos else None
    ven, acol_of, amach = _arena_index(seasons) if arenas else ({}, {}, None)

    shooter, team, dff, dmask, ctx, cnt, off_t, gid, seas_row, arena = [], [], [], [], [], [], [], [], [], []
    toi_atk, toi_def = np.zeros(P), np.zeros(P)
    last_season = np.full(P, -1, dtype=np.int64)

    def emit_side(atk, dfd, atk_home, s, def_goalie):
        ac_i = acol_of.get((ven.get(int(s.nhl_game_id)), s.season), -1)   # −1 = rare/unknown venue
        c = counts.get((int(s.nhl_game_id), int(s.stint_idx)), {})
        ai = [idx[p] for p in atk]
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
        if AC is not None:
            ac = AC[s.season]
            a_ac = ac[ai]                                      # (5, 4) attacker age-basis rows
            a_sum = a_ac.sum(0)
            d_cols = list(ac[di].sum(0)) if nd else [0.0] * 4  # defender age-basis sum (all real)
            a_d = isD[ai]
            a_dsum = float(a_d.sum())
            d_dsum = float(isD[di].sum()) if nd else 0.0
        for t_i, j in enumerate(atk):
            row = base + seas
            if AC is not None:                                 # AGE_CTX: [D-offsets | shoot|create|def basis]
                row = row + [float(a_d[t_i]), a_dsum - float(a_d[t_i]), d_dsum] \
                          + list(a_ac[t_i]) + list(a_sum - a_ac[t_i]) + d_cols
            shooter.append(idx[j]); team.append([idx[p] for p in atk if p != j])
            dff.append(di_pad); dmask.append(mrow); ctx.append(row)
            cnt.append(float(c.get(int(j), 0))); off_t.append(s.duration_s); gid.append(def_goalie)
            seas_row.append(s.season); arena.append(ac_i)

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
                last_season[idx[p]] = max(last_season[idx[p]], s.season)
        else:
            if hn == an:
                continue                                       # no man-advantage
            atk_home = hn > an
            atk, dfd = (s.home_skaters, s.away_skaters) if atk_home else (s.away_skaters, s.home_skaters)
            emit_side(atk, dfd, atk_home, s, s.away_goalie if atk_home else s.home_goalie)
            for p in atk:
                toi_atk[idx[p]] += dur                         # PP-attacker time
                last_season[idx[p]] = max(last_season[idx[p]], s.season)
            for p in dfd:
                toi_def[idx[p]] += dur                         # PK-defender time
                last_season[idx[p]] = max(last_season[idx[p]], s.season)

    dur = np.asarray(off_t, dtype=np.float64)
    ctx_names = RATE_CTX + [f"season_{s}" for s in slist[1:]] + (AGE_CTX if AC is not None else [])
    out = {
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
        "season_row": np.asarray(seas_row, dtype=np.int64), "seasons": slist,
        "ctx_names": ctx_names, "last_season": last_season,
    }
    if acol_of:
        out["arena_col"] = np.asarray(arena, dtype=np.int32)
        out["n_arenas"] = len(acol_of)
        out["arena_venue"] = amach["venue"]; out["arena_season"] = amach["season"]
        out["arena_e_prev"] = amach["e_prev"]; out["arena_e_next"] = amach["e_next"]
        out["arena_e_gap"] = amach["e_gap"]
    if states:
        out.update(_unit_machinery(out, P, slist))
    return out


def _unit_machinery(R, P, slist):
    """Build the per-(player, season) unit index for the RW drift states of one bucket: which pairs
    are active (appear in any row as shooter/teammate/defender), compact unit ids sorted by (player,
    season), unit-indexed row gathers, RW edges between a player's consecutive active seasons, the
    first-state mask, and per-unit attacking TOI (from the focal-shooter rows)."""
    sord = {s: i for i, s in enumerate(slist)}
    nS = len(slist)
    sh, tm, dfi = R["shooter_idx"], R["team_idx"], R["def_idx"]
    dmk = R["def_mask"]
    srow = np.array([sord[s] for s in R["season_row"]], dtype=np.int64)
    active = np.zeros((P, nS), dtype=bool)
    active[sh, srow] = True
    for j in range(tm.shape[1]):
        active[tm[:, j], srow] = True
    for j in range(dfi.shape[1]):
        m = dmk[:, j] > 0
        active[dfi[m, j], srow[m]] = True
    up, us = np.nonzero(active)                                # sorted by player, then season
    lut = np.full((P, nS), -1, dtype=np.int64)
    lut[up, us] = np.arange(len(up))
    same = up[1:] == up[:-1]                                   # consecutive units of the same player
    e_prev = np.nonzero(same)[0].astype(np.int64)
    e_next = e_prev + 1
    seas_arr = np.array(slist, dtype=np.int64)
    first_mask = np.ones(len(up))
    first_mask[e_next] = 0.0                                   # only each player's first state is ridged
    toi_unit = np.zeros(len(up))
    np.add.at(toi_unit, lut[sh, srow], R["dur"])
    return {
        "unit_player": up.astype(np.int64), "unit_season": seas_arr[us],
        "unit_lut": lut, "n_units": int(len(up)),
        "shooter_unit": lut[sh, srow].astype(np.int32),
        "team_unit": lut[tm, srow[:, None]].astype(np.int32),
        "def_unit": np.where(dmk > 0, lut[dfi, srow[:, None]], 0).astype(np.int32),
        "e_prev": e_prev, "e_next": e_next,
        "e_gap": (seas_arr[us[e_next]] - seas_arr[us[e_prev]]).astype(np.float64),
        "first_mask": first_mask, "toi_unit": toi_unit,
    }


# ── quality data: each shot + (for goals) the observed primary creator ────────────────────────────

def quality_creator_rows(seasons, idx, strengths, agepos=None, arenas=True):
    """Per Fenwick shot in the given strengths (POOLED across EV+MA): shooter, 4 teammates, ≤5 defenders
    (padded to 5 with a `def_mask`), context [is_home, pp, season…, shooter_D, def_D], logit-xG, goal
    flag, a CREATOR LABEL, a strength label (0=EV, 1=MA), and the row's season (for mapping teammates
    to the rate stage's per-season units). MA keeps only shots taken by the more-skaters (PP) side —
    so teammates=4, defenders=4 — consistent with the rate stage (shorthanded shots are out of scope).
    Creator: for goals, which teammate (0–3) got the primary assist, 4=unassisted, or −1 when the
    credited assister is not an on-ice teammate (data glitch / goalie assist — latent, and excluded
    from the assist-credit anchor; F4). Non-goals −1 (latent).
    Join: shots_onice.event_idx == pbp goal sortOrder."""
    scol, nseas = _season_cols(seasons)
    isD = agepos["isD"] if agepos else None
    ven, acol_of, amach = _arena_index(seasons) if arenas else ({}, {}, None)
    shooter, team, dff, dmask, ctx, y, goal, creator, creator2, slab, seas_row, arena = \
        [], [], [], [], [], [], [], [], [], [], [], []
    for s in seasons:
        p = C.PROCESSED / "shots_onice" / f"{s}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["nhl_game_id", "event_idx", "strength", "is_home", "xg",
                                        "goal", "shooter_id", "home_skaters", "away_skaters"])
        d = d[d.strength.isin(strengths) & d.xg.notna() & d.shooter_id.notna()]
        for gid, sub in d.groupby("nhl_game_id"):
            a1, a2 = {}, {}
            ac_i = acol_of.get((ven.get(int(gid)), s), -1)    # −1 = rare/unknown venue
            pf = C.RAW_PBP / f"{int(gid)}.json"
            if pf.exists():
                plays = json.loads(pf.read_text()).get("plays", [])
                for pl in plays:
                    if pl.get("typeDescKey") == "goal":
                        det = pl.get("details", {})
                        a1[pl.get("sortOrder")] = det.get("assist1PlayerId")
                        a2[pl.get("sortOrder")] = det.get("assist2PlayerId")
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
                row = [1.0 if r.is_home == 1 else 0.0, 0.0 if is_ev else 1.0] + seas
                if isD is not None:                          # A2: position offsets (appended last so
                    row += [float(isD[idx[sid]]),            #   the pp column stays at index 1)
                            float(sum(isD[idx[q]] for q in dfd))]
                ctx.append(row)
                xg = float(np.clip(r.xg, EPS, 1 - EPS)); y.append(np.log(xg / (1 - xg)))
                goal.append(int(r.goal)); slab.append(0 if is_ev else 1); seas_row.append(s)
                arena.append(ac_i)
                if r.goal == 1:
                    ap = a1.get(int(r.event_idx))
                    if ap is None:
                        creator.append(4)                    # genuinely unassisted
                    elif int(ap) in mates:
                        creator.append(mates.index(int(ap)))
                    else:
                        creator.append(-1)                   # assister not an on-ice teammate (data
                    ap2 = a2.get(int(r.event_idx))           # glitch / goalie assist): latent, and
                    creator2.append(mates.index(int(ap2))    # excluded from the assist-credit anchor
                                    if ap2 is not None and int(ap2) in mates else -1)
                else:
                    creator.append(-1)
                    creator2.append(-1)
    out = {"shooter_idx": np.asarray(shooter, dtype=np.int64), "team_idx": np.asarray(team, dtype=np.int64),
           "def_idx": np.asarray(dff, dtype=np.int64), "def_mask": np.asarray(dmask, dtype=np.float64),
           "Xctx": np.asarray(ctx, dtype=np.float64),
           "y": np.asarray(y, dtype=np.float64), "goal": np.asarray(goal, dtype=np.int64),
           "creator": np.asarray(creator, dtype=np.int64), "strength": np.asarray(slab, dtype=np.int64),
           "creator2": np.asarray(creator2, dtype=np.int64),
           "season": np.asarray(seas_row, dtype=np.int64),
           "ctx_names": ["home", "pp"] + [f"season_{s}" for s in sorted(set(seasons))[1:]]
                        + (["shooter_D", "def_D"] if isD is not None else [])}
    if acol_of:
        out["arena_col"] = np.asarray(arena, dtype=np.int32)
        out["n_arenas"] = len(acol_of)
        out["arena_venue"] = amach["venue"]; out["arena_season"] = amach["season"]
        out["arena_e_prev"] = amach["e_prev"]; out["arena_e_next"] = amach["e_next"]
        out["arena_e_gap"] = amach["e_gap"]
    return out


# ── conversion data: each shot's shooter, facing goalie, observed xG, goal flag ────────────────────

def conversion_rows(seasons, idx, strengths, agepos=None):
    """Per Fenwick shot (POOLED across the given strengths) for the CONVERSION fit: shooter index,
    facing-goalie index, logit of the OBSERVED xG, goal flag, a strength label (0=EV, 1=MA), and the
    season. For MA, keeps only PP-side shots (shooter's team has the man-advantage), matching
    rate/quality. Rows whose shooter isn't in `idx`, or with a missing facing goalie, are dropped.
    Uses observed `xg` (fixed), never `qbar`, so the stage stays independent of Stages 1-2.
    `fin`/`gsave` are POOLED across strengths (per-strength slope/intercept in fit_conversion).
    With `agepos`, also builds the global context block `ctx` (named in `ctx_names`): per-season
    intercept offsets (F6 — league finishing drift, first season = reference), the shooter position
    offset + F/D age basis (finishing curve), and the goalie age basis (goalie aging curve)."""
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
        d["season"] = s
        frames.append(d[["sidx", "goalie_id", "xg", "goal", "slab", "season"]])
    if not frames or sum(len(f) for f in frames) == 0:
        return None
    D = pd.concat(frames, ignore_index=True)
    goalies = sorted(int(g) for g in D.goalie_id.astype(int).unique())
    gmap = {g: i for i, g in enumerate(goalies)}
    xg = np.clip(D.xg.to_numpy(float), EPS, 1 - EPS)
    sidx = D.sidx.astype(int).to_numpy(np.int64)
    gidx = D.goalie_id.astype(int).map(gmap).to_numpy(np.int64)
    srow = D.season.to_numpy(np.int64)
    out = {"shooter_idx": sidx, "goalie_idx": gidx,
           "logit_xg": np.log(xg / (1 - xg)),
           "y": D.goal.to_numpy(np.float64),
           "strength": D.slab.to_numpy(np.int64),
           "season": srow, "goalies": goalies}
    if agepos is not None:
        slist = sorted(set(seasons))
        cols = [(srow == s).astype(np.float64) for s in slist[1:]]      # per-season offsets (F6)
        names = [f"season_{s}" for s in slist[1:]]
        zsh = np.zeros(len(D)); dsh = agepos["isD"][sidx]
        gborn = _birthdates(goalies)
        zg = np.zeros(len(D))
        for s in slist:
            m = srow == s
            zsh[m] = agepos["z"][s][sidx[m]]
            ag = _season_age(gborn, s)
            zg[m] = np.where(np.isnan(ag), 0.0, (ag - AGE_PEAK) / AGE_SCALE)[gidx[m]]
        cols += [dsh] + list(_age_cols(zsh, dsh).T) + [zg, zg * zg]
        names += ["shooter_D", "fin_zF", "fin_z2F", "fin_zD", "fin_z2D", "g_z", "g_z2"]
        out["ctx"] = np.column_stack(cols) if cols else np.zeros((len(D), 0))
        out["ctx_names"] = names
    return out


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


def _build_conv_X(logit_xg, sidx, shooter, goalie, S, P, G, Cctx=None):
    """Sparse conversion design for the Hessian, columns
    [slope_s (S) | intercept_s (S) | ctx(kc) | fin(P) | gsave(G)] — the slope/intercept are
    per-strength (`sidx` in 0..S-1), so a shot loads its own strength's slope (value=logit_xg) and
    intercept (value=1); `Cctx` is the optional global context block (season offsets, age curves)."""
    n = len(shooter)
    ar = np.arange(n)
    kc = 0 if Cctx is None else Cctx.shape[1]
    rows = [ar, ar, ar, ar]
    cols = [sidx, S + sidx, 2 * S + kc + shooter, 2 * S + kc + P + goalie]
    data = [logit_xg, np.ones(n), np.ones(n), np.ones(n)]
    for c in range(kc):
        rows.append(ar); cols.append(np.full(n, 2 * S + c)); data.append(Cctx[:, c])
    return sparse.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(n, 2 * S + kc + P + G))


def _rate_penalty(NU, k, first_mask, e_prev, e_next, e_gap):
    """Sparse prior-precision matrix for the rate fit's player blocks. Per block: a level ridge on
    each player's FIRST state (`first_mask`) + the random-walk precision between his consecutive
    states — for an edge with weight w = 1/(rw_sd²·gap), add w to both diagonal entries and −w to
    the off-diagonals (the precision of θ_next − θ_prev ~ N(0, rw_sd²·gap)). With no edges (static
    or single season) this is exactly the old diagonal ridge."""
    blocks = [(1.0 / PRIOR_SD_SHOOT ** 2, 1.0 / RW_SD_SHOOT ** 2),
              (1.0 / PRIOR_SD_CREATE ** 2, 1.0 / RW_SD_CREATE ** 2),
              (1.0 / PRIOR_SD_SHOOT ** 2, 1.0 / RW_SD_DEF ** 2)]
    rows, cols, vals = [], [], []
    au = np.arange(NU)
    for bi, (lev, wrw) in enumerate(blocks):
        off = 1 + bi * NU
        rows.append(off + au); cols.append(off + au); vals.append(lev * first_mask)
        if len(e_prev):
            w = wrw / e_gap
            for a, b in ((e_prev, e_prev), (e_next, e_next)):
                rows.append(off + a); cols.append(off + b); vals.append(w)
            for a, b in ((e_prev, e_next), (e_next, e_prev)):
                rows.append(off + a); cols.append(off + b); vals.append(-w)
    ncol = 1 + 3 * NU + k
    return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(ncol, ncol))


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


# ── θ̂ checkpoint: warm starts + optimization-free re-exports ──────────────────────────────────────
# The optimizer's solution is the expensive artifact (hours on the pooled window); everything else is
# derived from it in minutes. After every full fit the raw solution vectors land in
# generative_ckpt.npz, keyed by (player, season) unit, ctx-column NAME, and (venue, season) arena
# state — so the next fit can WARM-START by key (new players/seasons/columns init cold; the optimum is
# unchanged, only the path to it shortens), and `--reexport` can skip optimization AND the SE Hessian
# outright when the fit signature matches exactly (export-side tweaks rerun in minutes, not hours).
# The holdout harness never warm-starts: its train fits must not touch a solution that saw test data.

def _ckpt_path():
    return C.MODELS / "generative_ckpt.npz"


def _arr(v, dtype):
    return np.asarray(v if v is not None else [], dtype=dtype)


def _rate_unit_keys(R):
    """(player id, season) key per unit — season −1 for static fits (units == players)."""
    pids = np.asarray(R["players"])
    if "unit_player" in R:
        return pids[R["unit_player"]], np.asarray(R["unit_season"], dtype=np.int64)
    return pids, np.full(len(pids), -1, dtype=np.int64)


def _stage_rate(rate, pids):
    """Checkpoint block for one rate fit (unprefixed keys)."""
    up = np.asarray(rate["unit_player"], dtype=np.int64)
    us = rate.get("unit_season")
    us = np.asarray(us, dtype=np.int64) if us is not None else np.full(len(up), -1, dtype=np.int64)
    return {"theta": np.asarray(rate["theta"], dtype=np.float64),
            "unit_pid": np.asarray(pids)[up], "unit_season": us,
            "n_ctx": np.asarray(len(rate["beta"])),          # ctx block SIZE (names may be absent)
            "ctx_names": _arr(rate.get("ctx_names"), str),
            "arena_venue": _arr(rate.get("arena_venue"), str),
            "arena_season": _arr(rate.get("arena_season"), np.int64),
            "has_a2": np.asarray(rate.get("a2_q") is not None),
            "nb": np.asarray(rate.get("r") is not None),
            "converged": np.asarray(rate["converged"]), "grad_norm": np.asarray(rate["grad_norm"]),
            "se_shoot_last": rate["se_shoot_last"], "se_create_last": rate["se_create_last"],
            "se_def_last": rate["se_def_last"]}


def _stage_qual(qual, pids):
    return {"theta": np.asarray(qual["theta"], dtype=np.float64), "pid": np.asarray(pids),
            "n_ctx": np.asarray(len(qual["beta"])),
            "ctx_names": _arr(qual.get("ctx_names"), str),
            "arena_venue": _arr(qual.get("arena_venue"), str),
            "arena_season": _arr(qual.get("arena_season"), np.int64),
            "converged": np.asarray(qual["converged"]), "grad_norm": np.asarray(qual["grad_norm"]),
            "se_qcreate": qual["se_qcreate"]}


def _stage_conv(conv, pids):
    return {"theta": np.asarray(conv["theta"], dtype=np.float64), "pid": np.asarray(pids),
            "gid": np.asarray(conv["goalies"]),
            "keys": np.asarray(list(conv["a"].keys()), dtype=str),
            "n_ctx": np.asarray(len(conv["beta"])),
            "ctx_names": _arr(conv.get("ctx_names"), str),
            "converged": np.asarray(conv["converged"]), "grad_norm": np.asarray(conv["grad_norm"]),
            "se_fin": conv["se_fin"], "se_gsave": conv["se_gsave"]}


def ckpt_save(seasons, count_model, spg_scale, rates, qual, conv, players, path=None,
              ma_anchor_scale=1.0, ma_create_prior_sd=None, ma_def_prior_sd=None):
    pids = np.asarray(players)
    z = {"seasons": np.asarray([int(s) for s in seasons]), "count_model": np.asarray(count_model),
         "spg_scale": np.asarray(float(spg_scale)),
         "ma_anchor_scale": np.asarray(float(ma_anchor_scale)),
         "ma_create_prior_sd": np.asarray(float(ma_create_prior_sd)
                                          if ma_create_prior_sd else np.nan),
         "ma_def_prior_sd": np.asarray(float(ma_def_prior_sd) if ma_def_prior_sd else np.nan)}
    for key, rate in rates.items():
        for k, v in _stage_rate(rate, pids).items():
            z[f"{key}_{k}"] = v
    for k, v in _stage_qual(qual, pids).items():
        z[f"qual_{k}"] = v
    for k, v in _stage_conv(conv, pids).items():
        z[f"conv_{k}"] = v
    p = path or _ckpt_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **z)
    return p


def ckpt_load(path=None):
    p = path or _ckpt_path()
    if not p.exists():
        return None
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _ck_stage(ck, key):
    """Extract one stage's unprefixed block from a loaded checkpoint (None if absent)."""
    if ck is None or f"{key}_theta" not in ck:
        return None
    pre = key + "_"
    return {k[len(pre):]: v for k, v in ck.items() if k.startswith(pre)}


def _warm_rate_x0(x0, w, R, NU, PS, QI, has_a2, nb):
    """Best-effort map of a checkpoint's rate θ onto this fit's layout. Keys: (pid, season) for the
    three unit blocks, column name for ctx, (venue, season) for arenas. Unmatched entries keep the
    cold init. Returns the number of units carried."""
    u_pid, u_sea = _rate_unit_keys(R)
    oth = np.asarray(w["theta"], dtype=np.float64)
    onu, ok, onar = len(w["unit_pid"]), int(w["n_ctx"]), len(w["arena_venue"])
    ops = 1 + 3 * onu + ok                                   # old psi0 index
    old = {(str(p), int(s)): j for j, (p, s) in enumerate(zip(w["unit_pid"], w["unit_season"]))}
    x0[0] = oth[0]
    hits = 0
    for i, (p, s) in enumerate(zip(u_pid, u_sea)):
        j = old.get((str(p), int(s)))
        if j is not None:
            hits += 1
            for b in range(3):
                x0[1 + b * NU + i] = oth[1 + b * onu + j]
    oc = {str(nm): float(v) for nm, v in zip(w["ctx_names"], oth[1 + 3 * onu:ops])}
    for j, nm in enumerate(R.get("ctx_names") or []):
        if nm in oc:
            x0[1 + 3 * NU + j] = oc[nm]
    x0[PS] = oth[ops]                                        # psi0 (create_0)
    oa = {(str(v), int(s)): float(x) for v, s, x in
          zip(w["arena_venue"], w["arena_season"], oth[ops + 1:ops + 1 + onar])}
    for j, (v, s) in enumerate(zip(R.get("arena_venue") or [], R.get("arena_season") or [])):
        x0[PS + 1 + j] = oa.get((str(v), int(s)), 0.0)
    pos = ops + 1 + onar                                     # old tail: [q?] [log r?]
    if bool(w["has_a2"]):
        if has_a2:
            x0[QI] = oth[pos]
        pos += 1
    if bool(w["nb"]) and nb:
        x0[-1] = oth[pos]
    return hits


def _reuse_theta_rate(w, n_th, R, has_a2, nb):
    u_pid, u_sea = _rate_unit_keys(R)
    ok = (w is not None and len(np.asarray(w["theta"])) == n_th
          and np.array_equal(np.asarray(w["unit_pid"]), u_pid)
          and np.array_equal(np.asarray(w["unit_season"]), u_sea)
          and [str(x) for x in w["ctx_names"]] == [str(x) for x in (R.get("ctx_names") or [])]
          and [str(x) for x in w["arena_venue"]] == [str(x) for x in (R.get("arena_venue") or [])]
          and bool(w["has_a2"]) == bool(has_a2) and bool(w["nb"]) == bool(nb))
    if not ok:
        raise ValueError("reuse: checkpoint does not match this rate fit exactly — run a full fit")
    return np.asarray(w["theta"], dtype=np.float64)


def _warm_qual_x0(x0, w, pids, ctx_names, ar_v, ar_s, P, k):
    """Map a checkpoint's quality θ onto this fit's layout (players by id, ctx by name, arenas by
    (venue, season)). Returns the number of players carried."""
    oth = np.asarray(w["theta"], dtype=np.float64)
    oP, ok, onar = len(w["pid"]), int(w["n_ctx"]), len(w["arena_venue"])
    old = {str(p): j for j, p in enumerate(w["pid"])}
    x0[0] = oth[0]
    hits = 0
    for i, p in enumerate(pids):
        j = old.get(str(p))
        if j is not None:
            hits += 1
            x0[1 + i] = oth[1 + j]                           # qshoot
            x0[3 + P + i] = oth[3 + oP + j]                  # qdef
    x0[1 + P:3 + P] = oth[1 + oP:3 + oP]                     # qcreate position pair
    oc = {str(nm): float(v) for nm, v in zip(w["ctx_names"], oth[3 + 2 * oP:3 + 2 * oP + ok])}
    for j, nm in enumerate(ctx_names or []):
        if nm in oc:
            x0[3 + 2 * P + j] = oc[nm]
    oa = {(str(v), int(s)): float(x) for v, s, x in
          zip(w["arena_venue"], w["arena_season"], oth[3 + 2 * oP + ok:3 + 2 * oP + ok + onar])}
    for j, (v, s) in enumerate(zip(ar_v or [], ar_s or [])):
        x0[3 + 2 * P + k + j] = oa.get((str(v), int(s)), 0.0)
    return hits


def _reuse_theta_qual(w, n_th, pids, Q):
    ok = (w is not None and pids is not None and len(np.asarray(w["theta"])) == n_th
          and np.array_equal(np.asarray(w["pid"]), np.asarray(pids))
          and [str(x) for x in w["ctx_names"]] == [str(x) for x in (Q.get("ctx_names") or [])]
          and [str(x) for x in w["arena_venue"]] == [str(x) for x in (Q.get("arena_venue") or [])])
    if not ok:
        raise ValueError("reuse: checkpoint does not match this quality fit exactly — run a full fit")
    return np.asarray(w["theta"], dtype=np.float64)


def _warm_conv_x0(x0, w, pids, gids, keys, ctx_names, S, kc, P):
    """Map a checkpoint's conversion θ onto this fit's layout (a/b by strength key, ctx by name,
    fin by player id, gsave by goalie id). Returns the number of shooters carried."""
    oth = np.asarray(w["theta"], dtype=np.float64)
    okeys = [str(x) for x in w["keys"]]
    oS, oP, okc = len(okeys), len(w["pid"]), int(w["n_ctx"])
    for i, kk in enumerate(keys):
        if kk in okeys:
            j = okeys.index(kk)
            x0[i] = oth[j]
            x0[S + i] = oth[oS + j]
    oc = {str(nm): float(v) for nm, v in zip(w["ctx_names"], oth[2 * oS:2 * oS + okc])}
    for j, nm in enumerate(ctx_names or []):
        if nm in oc:
            x0[2 * S + j] = oc[nm]
    o, oo = 2 * S + kc, 2 * oS + okc
    oldp = {str(p): j for j, p in enumerate(w["pid"])}
    hits = 0
    for i, p in enumerate(pids):
        j = oldp.get(str(p))
        if j is not None:
            hits += 1
            x0[o + i] = oth[oo + j]
    oldg = {str(g): j for j, g in enumerate(w["gid"])}
    for i, g in enumerate(gids):
        j = oldg.get(str(g))
        if j is not None:
            x0[o + P + i] = oth[oo + oP + j]
    return hits


def _reuse_theta_conv(w, n_th, pids, Cr, keys):
    ok = (w is not None and pids is not None and len(np.asarray(w["theta"])) == n_th
          and np.array_equal(np.asarray(w["pid"]), np.asarray(pids))
          and np.array_equal(np.asarray(w["gid"]), np.asarray(Cr["goalies"]))
          and [str(x) for x in w["keys"]] == list(keys)
          and [str(x) for x in w["ctx_names"]] == [str(x) for x in (Cr.get("ctx_names") or [])])
    if not ok:
        raise ValueError("reuse: checkpoint does not match this conversion fit exactly — run a full fit")
    return np.asarray(w["theta"], dtype=np.float64)


def _last_view(v, lu):
    """Per-player view of a per-unit array at each player's last state (0 where never active)."""
    out = np.zeros(len(lu))
    m = lu >= 0
    out[m] = v[lu[m]]
    return out


def _rate_ses(H, M, NU, lu):
    """Per-player SEs at each player's LAST state for the three rate blocks. Dense inverse when the
    system is small enough; otherwise sparse splu column solves for just the reported columns.
    `create` (block 1) uses the sandwich Var_jj = h'Mh with h = H⁻¹e_j — the assist-credit's meat
    counts w² over the actual observed goals, so the up-weighted anchor adds curvature but not
    w× real information (F1). shoot/def use the plain (H⁻¹)_jj."""
    ncol = H.shape[0]
    P = len(lu)
    act = np.nonzero(lu >= 0)[0]
    cols = {b: (1 + b * NU) + lu[act] for b in range(3)}
    se = [np.zeros(P) for _ in range(3)]
    if ncol <= DENSE_H_MAX:
        Hinv = np.linalg.inv(H.toarray())
        dj = np.clip(np.diag(Hinv), 0.0, None)
        for b in (0, 2):
            se[b][act] = np.sqrt(dj[cols[b]])
        hc = Hinv[:, cols[1]]
        se[1][act] = np.sqrt(np.clip(np.einsum("ij,ij->j", hc, M @ hc), 0.0, None))
    else:
        slu = splu(H.tocsc())
        for b in range(3):
            cj = cols[b]
            for i0 in range(0, len(cj), 512):
                sel = cj[i0:i0 + 512]
                E = np.zeros((ncol, len(sel)))
                E[sel, np.arange(len(sel))] = 1.0
                Hc = slu.solve(E)
                var = (np.einsum("ij,ij->j", Hc, M @ Hc) if b == 1
                       else Hc[sel, np.arange(len(sel))])
                se[b][act[i0:i0 + 512]] = np.sqrt(np.clip(var, 0.0, None))
    return se


def fit_rate_create(R, g_team, g_cidx, shots_per_goal, count_model="poisson", a2=None,
                    warm=None, reuse=False, create_prior_sd=None, def_prior_sd=None):
    """UNIFIED CREATION over per-(player, season) STATES. One `create` parameter per unit that does
    double duty:
      (i) lifts teammates' shot rate in the Poisson/NB count layer, and
      (ii) is the creator-credit propensity in a conditional logit over the on-ice teammates,
           anchored to OBSERVED goal assisters (g_team = the 4 teammate unit indices per goal,
           g_cidx = the credited column: 0 = unassisted/create_0, 1..4 = teammate position).
    Fit jointly; `shots_per_goal` up-weights the sparse assist-credit — each observed goal-creator
    stands in for the ≈ shots/goal whose creator we never see (inverse-probability weighting) — so
    `create` blends 'I make my linemates shoot' with 'I'm the one credited with creating'.

    UNITS: when R carries the RW machinery (`states=True` rate rows), each block has one state per
    (player, active-season) pair, tied across seasons by the random-walk penalty (_rate_penalty):
    level ridge on the first state, N(prev, rw_sd²·gap) between consecutive states. Without it
    (the MA bucket, synthetic fixtures) units == players and the penalty is the old static ridge —
    the model degenerates exactly to the pre-curves behavior.

    SECONDARY ASSISTS (`a2` = (g2_team, g2_c1, g2_c2), unit-indexed like g_team): the (A1, A2) pair
    is a partial ranking of the teammates by creation involvement — an exploded-logit second stage
    over the 3 teammates left after masking A1's column. Because recorded A2s are part creation
    signal, part puck-touching noise, the second stage is a MIXTURE with a FITTED probability
    q = P(A2 reflects creation):  P(A2=c₂) = q·softmax(create_{T∖c₁})[c₂] + (1−q)/3.
    A bare weight on the A2 log-likelihood would be a pseudo-likelihood temperature (not MLE-
    fittable); q is a proper parameter, identified by A2's concordance with the create ordering
    already pinned by the counts + A1 anchor. The A2 term carries the same spg IPW weight, and its
    responsibility-weighted Fisher joins the create bread/meat.

    Returns per-unit shoot/create/def plus per-player `*_last` views (each player's most recent
    state — the "current skill" read) with SEs (`se_create_last` sandwich-corrected, F1).

    `warm` (a checkpoint stage block, see _ck_stage) initializes the optimizer at the previous
    solution mapped by key; `reuse=True` additionally SKIPS optimization and the SE Hessian,
    taking θ and SEs straight from the checkpoint — valid only for the exact same fit (enforced)."""
    P, k = len(R["players"]), R["Xctx"].shape[1]
    lsh = 1 / PRIOR_SD_SHOOT ** 2
    ldf = 1 / (def_prior_sd if def_prior_sd else PRIOR_SD_SHOOT) ** 2
    lcr = 1 / (create_prior_sd if create_prior_sd else PRIOR_SD_CREATE) ** 2
    nb = count_model == "nb"
    dmask = R.get("def_mask")
    if dmask is None:                                        # synthetic/legacy rows: all defenders real
        dmask = np.ones_like(R["def_idx"], dtype=np.float64)
    states = "unit_player" in R
    if states:
        NU = R["n_units"]
        sh_i, tm_i, df_i = R["shooter_unit"], R["team_unit"], R["def_unit"]
        up, fm = R["unit_player"], R["first_mask"]
        e_prev, e_next, e_gap = R["e_prev"], R["e_next"], R["e_gap"]
    else:
        NU = P
        sh_i, tm_i, df_i = R["shooter_idx"], R["team_idx"], R["def_idx"]
        up, fm = np.arange(P, dtype=np.int64), np.ones(P)
        e_prev = np.zeros(0, dtype=np.int64); e_next = np.zeros(0, dtype=np.int64); e_gap = np.ones(0)
    PS = 1 + 3 * NU + k                                      # index of create_0 in th
    w_sh, w_cr, w_df = 1 / RW_SD_SHOOT ** 2, 1 / RW_SD_CREATE ** 2, 1 / RW_SD_DEF ** 2
    has_rw = len(e_prev) > 0
    n_ar = int(R.get("n_arenas", 0))                         # arena offsets: [PS+1, PS+1+n_ar)
    acol = R.get("arena_col")
    if acol is None:
        acol = np.full(len(R["count"]), -1, dtype=np.int32)
    la_ar, lrw_ar = 1 / ARENA_SD ** 2, 1 / ARENA_RW_SD ** 2  # (venue, season)-state nuisance prior
    a_ep = jnp.asarray(R.get("arena_e_prev", np.zeros(0, dtype=np.int64)))
    a_en = jnp.asarray(R.get("arena_e_next", np.zeros(0, dtype=np.int64)))
    a_iw = jnp.asarray(1.0 / np.maximum(R.get("arena_e_gap", np.ones(0)), 1e-9))
    has_ar_rw = int(a_ep.shape[0]) > 0
    has_a2 = a2 is not None and len(a2[0]) > 0               # A2 mixture stage; q at index PS+1+n_ar
    if has_a2:
        g2_team, g2_c1, g2_c2 = (np.asarray(a2[0], dtype=np.int64),
                                 np.asarray(a2[1], dtype=np.int64),
                                 np.asarray(a2[2], dtype=np.int64))
    else:
        g2_team = np.zeros((0, 4), dtype=np.int64)
        g2_c1 = g2_c2 = np.zeros(0, dtype=np.int64)

    def split(th):
        return th[1:1 + NU], th[1 + NU:1 + 2 * NU], th[1 + 2 * NU:1 + 3 * NU], th[1 + 3 * NU:PS], th[PS]

    QI = PS + 1 + n_ar                                       # q's logit sits after the arena block

    def nll(th, sh_j, tm_j, df_j, dm, X, cnt, ofs, gt, gc, fmj, ep, en, iw, acl, g2t, g2a, g2b):
        sh, cr, df, b, psi0 = split(th)
        eta = th[0] + sh[sh_j] + jnp.sum(cr[tm_j], 1) + jnp.sum(df[df_j] * dm, 1) + X @ b
        if n_ar:                                             # venue recording-bias offset (−1 = ref)
            ar = th[PS + 1:PS + 1 + n_ar]
            eta = eta + jnp.where(acl >= 0, ar[jnp.clip(acl, 0, None)], 0.0)
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
        if has_a2:                                           # A2 mixture stage (A1's column masked)
            q = jax.nn.sigmoid(th[QI])
            lg2 = jnp.where(jnp.arange(4)[None, :] == g2a[:, None], -1e30, cr[g2t])
            logpi2 = lg2 - logsumexp(lg2, axis=1)[:, None]
            pl2 = jnp.exp(jnp.take_along_axis(logpi2, g2b[:, None], 1)[:, 0])
            credit = credit - jnp.sum(jnp.log(q * pl2 + (1.0 - q) / 3.0 + 1e-12))
        pen = 0.5 * (lsh * jnp.sum(fmj * sh ** 2) + lcr * jnp.sum(fmj * cr ** 2)
                     + ldf * jnp.sum(fmj * df ** 2))
        if has_rw:                                           # RW: θ_next ~ N(θ_prev, rw_sd²·gap)
            pen += 0.5 * (w_sh * jnp.sum(iw * (sh[en] - sh[ep]) ** 2)
                          + w_cr * jnp.sum(iw * (cr[en] - cr[ep]) ** 2)
                          + w_df * jnp.sum(iw * (df[en] - df[ep]) ** 2))
        if n_ar:                                             # arena nuisance prior: ridge + season RW
            pen += 0.5 * la_ar * jnp.sum(ar ** 2)
            if has_ar_rw:
                pen += 0.5 * lrw_ar * jnp.sum(a_iw * (ar[a_en] - ar[a_ep]) ** 2)
        return pois + shots_per_goal * credit + pen

    x0 = np.zeros(PS + 1 + n_ar + (1 if has_a2 else 0) + (1 if nb else 0))
    if nb:
        x0[-1] = 1.0                                         # q's logit starts at 0 ⇒ q = 0.5
    if reuse:
        th = _reuse_theta_rate(warm, len(x0), R, has_a2, nb)
        converged, gnorm, nit = bool(warm["converged"]), float(warm["grad_norm"]), 0
    else:
        if warm is not None:
            try:
                hits = _warm_rate_x0(x0, warm, R, NU, PS, QI, has_a2, nb)
                print(f"    warm start: {hits}/{NU} unit states carried from checkpoint")
            except Exception as e:                           # a stale checkpoint must never kill a fit
                print(f"    warm start skipped ({e}) — cold init")
                x0 = np.zeros_like(x0)
                if nb:
                    x0[-1] = 1.0
        res = _optimize(nll, x0, sh_i, tm_i, df_i, dmask, R["Xctx"], R["count"], R["offset"],
                        g_team, g_cidx, fm, e_prev, e_next, 1.0 / np.maximum(e_gap, 1e-9), acol,
                        g2_team, g2_c1, g2_c2)
        th, converged = res.x, bool(res.success)
        gnorm, nit = float(np.max(np.abs(res.jac))), int(res.nit)
    a2_q = float(_sigmoid(th[QI])) if has_a2 else None
    sh, cr, df = th[1:1 + NU], th[1 + NU:1 + 2 * NU], th[1 + 2 * NU:1 + 3 * NU]
    b, psi0 = th[1 + 3 * NU:PS], float(th[PS])
    ar_vec = th[PS + 1:PS + 1 + n_ar]
    r = float(np.exp(th[-1])) if nb else None

    lu = np.full(P, -1, dtype=np.int64)
    lu[up] = np.arange(NU)                                   # last unit per player (units season-sorted)
    if reuse:                                                # same fit ⇒ SEs come from the checkpoint
        se_sh = np.asarray(warm["se_shoot_last"], dtype=np.float64)
        se_cr = np.asarray(warm["se_create_last"], dtype=np.float64)
        se_df = np.asarray(warm["se_def_last"], dtype=np.float64)
    else:
        eta = th[0] + sh[sh_i] + cr[tm_i].sum(1) + (df[df_i] * dmask).sum(1) + R["Xctx"] @ b
        if n_ar:
            eta = eta + np.where(acol >= 0, ar_vec[np.clip(acol, 0, None)], 0.0)
        mu = np.exp(eta + R["offset"])
        W = mu * r / (r + mu) if nb else mu

        # bread H = XᵀWX + penalty + w·(credit Fisher); meat M = XᵀWX + w²·(credit Fisher)  [F1]
        Xs = _build_X(sh_i, tm_i, df_i, R["Xctx"], NU, dmask)
        XtWX = (Xs.T @ Xs.multiply(W[:, None])).tocsr()
        logit5 = np.concatenate([np.full((len(g_cidx), 1), psi0), cr[g_team]], 1)
        pi = np.exp(logit5 - logit5.max(1, keepdims=True)); pi /= pi.sum(1, keepdims=True)
        cinfo = np.zeros(NU)                                 # per-goal creator Fisher info (weight 1)
        for t in range(4):
            np.add.at(cinfo, g_team[:, t], pi[:, t + 1] * (1 - pi[:, t + 1]))
        if has_a2:                                           # A2 Fisher, responsibility-weighted by the
            lg2 = cr[g2_team].astype(np.float64)             # mixture (noise share carries no info)
            lg2[np.arange(4)[None, :] == g2_c1[:, None]] = -1e30
            pi2 = np.exp(lg2 - lg2.max(1, keepdims=True)); pi2 /= pi2.sum(1, keepdims=True)
            plc = pi2[np.arange(len(g2_c2)), g2_c2]
            resp = a2_q * plc / (a2_q * plc + (1.0 - a2_q) / 3.0 + 1e-12)
            for t in range(4):
                valid = (g2_c1 != t).astype(np.float64)
                np.add.at(cinfo, g2_team[:, t], resp * pi2[:, t] * (1 - pi2[:, t]) * valid)
        ci = np.arange(1 + NU, 1 + 2 * NU)
        cdiag = sparse.csr_matrix((cinfo, (ci, ci)), shape=(PS, PS))
        H = XtWX + _rate_penalty(NU, k, fm, e_prev, e_next, e_gap) + shots_per_goal * cdiag
        M = XtWX + shots_per_goal ** 2 * cdiag
        se_sh, se_cr, se_df = _rate_ses(H, M, NU, lu)
    out = {"intercept": float(th[0]), "beta": b, "psi0": psi0,
           "shoot": sh, "create": cr, "def": df,
           "unit_player": up, "unit_season": R.get("unit_season"), "last_unit": lu,
           "shoot_last": _last_view(sh, lu), "create_last": _last_view(cr, lu),
           "def_last": _last_view(df, lu),
           "se_shoot_last": se_sh, "se_create_last": se_cr, "se_def_last": se_df,
           "count_model": count_model, "r": r, "converged": converged,
           "grad_norm": gnorm, "nit": nit, "theta": np.asarray(th, dtype=np.float64).copy(),
           "ctx_names": R.get("ctx_names"),
           "arena_vec": ar_vec, "arena_venue": R.get("arena_venue"),
           "arena_season": R.get("arena_season"), "a2_q": a2_q, "n_a2": int(len(g2_c2))}
    if not states:                                           # static aliases (units == players)
        out["se_shoot"], out["se_create"], out["se_def"] = se_sh, se_cr, se_df
    return out


def fit_quality_creator(Q, P, creates, isD=None, warm=None, reuse=False, pids=None):
    """QUALITY fit, POOLED across strengths (EV+MA). Estimates one shared set of qshoot/qdef per player
    (danger is an intrinsic, strength-neutral skill — xG already encodes the man-advantage), with the
    strength environment absorbed by a `pp` context column (per-strength intercept). `qcreate` is a
    POSITION-LEVEL pair [F, D] (A1): per-player creation quality is not identifiable from ~20 observed
    setups each (docs §7), so the danger a creator adds is pooled to his position — every remaining
    parameter is one the data can actually estimate. The creator distribution
    pi = softmax([create_0, create[teammates]]) is FIXED from the rate fit and chosen PER ROW by that
    shot's strength: `creates` = {strength_label: (create_array, psi0[, team_cols])} where the optional
    `team_cols` (n,4) maps this strength's rows into `create_array` (the EV bucket passes per-season
    UNIT indices; without it Q["team_idx"] player indices are used). Creator is OBSERVED on goals
    (creator ≥ 0), MARGINALIZED over pi otherwise — including goals whose credited assister was not an
    on-ice teammate (creator = −1, F4). Defenders masked (≤5). SEs: diagonal Gauss-Newton."""
    n, k = len(Q["y"]), Q["Xctx"].shape[1]
    cre = Q["creator"]                                      # -1 latent, 0..3 teammate, 4 unassisted
    cidx_np = np.where(cre == 4, 0, np.clip(cre, 0, 3) + 1).astype(np.int64)  # col in [unassist, t0..t3]
    obs_np = (cre >= 0) & (Q["goal"] == 1)                  # rows with an OBSERVED creator label
    lqs, lqc, lqd = 1 / PRIOR_SD_QSHOOT ** 2, 1 / PRIOR_SD_QCREATE ** 2, 1 / PRIOR_SD_QSHOOT ** 2
    strength = Q.get("strength", np.zeros(n, dtype=np.int64))
    dmask = Q.get("def_mask")
    if dmask is None:
        dmask = np.ones_like(Q["def_idx"], dtype=np.float64)
    if isD is None:
        isD = np.zeros(P)
    tpos = isD[Q["team_idx"]].astype(np.int64)              # (n,4) teammate position: 0=F, 1=D
    n_ar = int(Q.get("n_arenas", 0))                        # arena offsets: last n_ar params
    acol = Q.get("arena_col")
    if acol is None:
        acol = np.full(n, -1, dtype=np.int32)
    la_ar, lrw_ar = 1 / ARENA_SD ** 2, 1 / ARENA_RW_SD ** 2
    a_ep = jnp.asarray(Q.get("arena_e_prev", np.zeros(0, dtype=np.int64)))
    a_en = jnp.asarray(Q.get("arena_e_next", np.zeros(0, dtype=np.int64)))
    a_iw = jnp.asarray(1.0 / np.maximum(Q.get("arena_e_gap", np.ones(0)), 1e-9))
    has_ar_rw = int(a_ep.shape[0]) > 0
    lg = np.zeros((n, 5))                                   # per-row creator logits by strength
    for st, spec in creates.items():
        cr_s, psi_s = spec[0], spec[1]
        tcols = spec[2] if len(spec) > 2 else Q["team_idx"]
        m = strength == st
        if m.any():
            lg[m] = np.concatenate([np.full((int(m.sum()), 1), psi_s), cr_s[tcols[m]]], 1)
    pi_np = np.exp(lg - lg.max(1, keepdims=True)); pi_np /= pi_np.sum(1, keepdims=True)

    def split(th):                                # [mq | qshoot(P) | qcreate(2) | qdef(P) | beta(k) | arena]
        return th[0], th[1:1 + P], th[1 + P:3 + P], th[3 + P:3 + 2 * P], th[3 + 2 * P:3 + 2 * P + k]

    def nll(th, sh_i, tm_i, tp_i, df_i, dm, X, xg, obs, cidx, pi, acl):
        mq, qs, qc, qd, b = split(th)
        base = mq + qs[sh_i] + jnp.sum(qd[df_i] * dm, 1) + X @ b
        if n_ar:                                            # venue location-bias offset (−1 = ref)
            arq = th[3 + 2 * P + k:3 + 2 * P + k + n_ar]
            base = base + jnp.where(acl >= 0, arq[jnp.clip(acl, 0, None)], 0.0)
        sig5 = jnp.concatenate([jax.nn.sigmoid(base)[:, None],
                                jax.nn.sigmoid(base[:, None] + qc[tp_i])], axis=1)   # (n,5) col0=unassisted

        def fb(p):
            return xg * jnp.log(p + EPS) + (1 - xg) * jnp.log(1 - p + EPS)
        gs = jnp.take_along_axis(sig5, cidx[:, None], 1)[:, 0]       # observed-creator quality
        marg = jnp.sum(pi * sig5, axis=1)                           # latent: marginalize over FIXED pi
        ll = jnp.where(obs, fb(gs), fb(marg))
        pen = 0.5 * (lqs * jnp.sum(qs ** 2) + lqc * jnp.sum(qc ** 2) + lqd * jnp.sum(qd ** 2))
        if n_ar:                                            # arena nuisance prior: ridge + season RW
            pen += 0.5 * la_ar * jnp.sum(arq ** 2)
            if has_ar_rw:
                pen += 0.5 * lrw_ar * jnp.sum(a_iw * (arq[a_en] - arq[a_ep]) ** 2)
        return -jnp.sum(ll) + pen

    x0 = np.zeros(3 + 2 * P + k + n_ar)
    if reuse:
        th = _reuse_theta_qual(warm, len(x0), pids, Q)
        converged, gnorm, nit = bool(warm["converged"]), float(warm["grad_norm"]), 0
    else:
        if warm is not None and pids is not None:
            try:
                hits = _warm_qual_x0(x0, warm, pids, Q.get("ctx_names"), Q.get("arena_venue"),
                                     Q.get("arena_season"), P, k)
                print(f"    quality warm start: {hits}/{P} players carried from checkpoint")
            except Exception as e:                           # a stale checkpoint must never kill a fit
                print(f"    quality warm start skipped ({e}) — cold init")
                x0 = np.zeros_like(x0)
        res = _optimize(nll, x0, Q["shooter_idx"], Q["team_idx"], tpos,
                        Q["def_idx"], dmask, Q["Xctx"], _sigmoid(Q["y"]), obs_np, cidx_np, pi_np, acol)
        th, converged = res.x, bool(res.success)
        gnorm, nit = float(np.max(np.abs(res.jac))), int(res.nit)
    mq, qs, qc, qd, b = (float(th[0]), th[1:1 + P], th[1 + P:3 + P],
                         th[3 + P:3 + 2 * P], th[3 + 2 * P:3 + 2 * P + k])
    arq_vec = th[3 + 2 * P + k:3 + 2 * P + k + n_ar]
    # Beta concentration from residual MSE around the fitted mean
    base = mq + qs[Q["shooter_idx"]] + (qd[Q["def_idx"]] * dmask).sum(1) + Q["Xctx"] @ b
    if n_ar:
        base = base + np.where(acol >= 0, arq_vec[np.clip(acol, 0, None)], 0.0)
    sig5 = np.concatenate([_sigmoid(base)[:, None], _sigmoid(base[:, None] + qc[tpos])], 1)
    ci = cidx_np
    m = (pi_np * sig5).sum(1)                                    # marginal mean, FIXED pi
    fitted = np.where(obs_np, sig5[np.arange(n), ci], m)
    mse = float(np.mean((_sigmoid(Q["y"]) - fitted) ** 2))
    s_conc = max(float(np.mean(fitted * (1 - fitted)) / max(mse, 1e-9) - 1.0), 1.0)

    # diagonal Gauss-Newton SE for the two qcreate params: observed-creator goals + latent marginal
    tm = Q["team_idx"]
    info_qc = np.zeros(2)
    tg, sg, cig, tpg = tm[obs_np], sig5[obs_np], ci[obs_np], tpos[obs_np]
    is_tm = cig >= 1
    cr_pl = tg[np.arange(len(tg)), np.clip(cig - 1, 0, 3)]       # creator player (for n_create)
    cr_pos = tpg[np.arange(len(tpg)), np.clip(cig - 1, 0, 3)]    # creator position (for info)
    sc = sg[np.arange(len(sg)), cig]
    np.add.at(info_qc, cr_pos[is_tm], (sc * (1 - sc))[is_tm])
    lat = ~obs_np
    pn, sn, mn, tpn = pi_np[lat], sig5[lat], m[lat], tpos[lat]
    den = np.clip(mn * (1 - mn), 1e-9, None)
    for pos in (0, 1):
        gpos = sum(pn[:, t + 1] * sn[:, t + 1] * (1 - sn[:, t + 1]) * (tpn[:, t] == pos)
                   for t in range(4))
        info_qc[pos] += float(np.sum(gpos ** 2 / den))
    se_qc = (np.asarray(warm["se_qcreate"], dtype=np.float64) if reuse
             else 1.0 / np.sqrt(info_qc + lqc))
    n_create = np.zeros(P, dtype=np.int64)
    np.add.at(n_create, cr_pl[is_tm], 1)
    # per-strength quality intercept: EV = mq; PP = mq + (pp-column coefficient, Xctx col 1)
    mu_qual = {"ev": mq, "ma": mq + (float(b[1]) if len(b) > 1 else 0.0)}
    return {"intercept": mq, "mu_qual": mu_qual, "qshoot": qs, "qcreate": qc, "qdef": qd, "beta": b,
            "beta_s": s_conc, "se_qcreate": se_qc, "n_create": n_create, "converged": converged,
            "grad_norm": gnorm, "nit": nit, "theta": np.asarray(th, dtype=np.float64).copy(),
            "ctx_names": Q.get("ctx_names"),
            "arena_vec": arq_vec, "arena_venue": Q.get("arena_venue"),
            "arena_season": Q.get("arena_season")}


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

def fit_conversion(Cr, P, warm=None, reuse=False, pids=None):
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
    Cctx = Cr.get("ctx")                                    # global block: season offsets + age curves
    kc = 0 if Cctx is None else Cctx.shape[1]
    if Cctx is None:
        Cctx = np.zeros((len(Cr["y"]), 0))

    def split(th):                                          # [a(S) | b(S) | c(kc) | fin(P) | gsave(G)]
        o = 2 * S + kc
        return th[0:S], th[S:2 * S], th[2 * S:o], th[o:o + P], th[o + P:o + P + G]

    def nll(th, si, sh_i, g_i, lxg, Cx, y):
        a, b, c, fin, gsave = split(th)
        eta = a[si] * lxg + b[si] + Cx @ c + fin[sh_i] + gsave[g_i]
        bce = jnp.sum(jax.nn.softplus(eta) - y * eta)       # = −Σ[y·log p + (1−y)·log(1−p)]
        pen = 0.5 * (lf * jnp.sum(fin ** 2) + lg * jnp.sum(gsave ** 2))
        return bce + pen                                    # a, b, c all unpenalized

    x0 = np.zeros(2 * S + kc + P + G)
    x0[0:S] = 1.0                                           # start each strength's slope at 1
    if reuse:
        th = _reuse_theta_conv(warm, len(x0), pids, Cr, keys)
        converged, gnorm, nit = bool(warm["converged"]), float(warm["grad_norm"]), 0
    else:
        if warm is not None and pids is not None:
            try:
                hits = _warm_conv_x0(x0, warm, pids, Cr["goalies"], keys, Cr.get("ctx_names"), S, kc, P)
                print(f"    conversion warm start: {hits}/{P} shooters carried from checkpoint")
            except Exception as e:                          # a stale checkpoint must never kill a fit
                print(f"    conversion warm start skipped ({e}) — cold init")
                x0 = np.zeros_like(x0); x0[0:S] = 1.0
        res = _optimize(nll, x0, sidx, Cr["shooter_idx"], Cr["goalie_idx"], Cr["logit_xg"], Cctx, Cr["y"])
        th, converged = res.x, bool(res.success)
        gnorm, nit = float(np.max(np.abs(res.jac))), int(res.nit)
    a_v, b_v, c_v, fin, gsave = (th[0:S], th[S:2 * S], th[2 * S:2 * S + kc],
                                 th[2 * S + kc:2 * S + kc + P], th[2 * S + kc + P:])

    eta = (a_v[sidx] * Cr["logit_xg"] + b_v[sidx] + Cctx @ c_v
           + fin[Cr["shooter_idx"]] + gsave[Cr["goalie_idx"]])
    p = _sigmoid(eta)
    o = 2 * S + kc
    if reuse:                                               # same fit ⇒ SEs come from the checkpoint
        se_fin = np.asarray(warm["se_fin"], dtype=np.float64)
        se_gsave = np.asarray(warm["se_gsave"], dtype=np.float64)
    else:
        W = p * (1 - p)
        X = _build_conv_X(Cr["logit_xg"], sidx, Cr["shooter_idx"], Cr["goalie_idx"], S, P, G,
                          Cctx if kc else None)
        pen = np.zeros(o + P + G); pen[o:o + P] = lf; pen[o + P:] = lg
        H = (X.T @ X.multiply(W[:, None])).toarray() + np.diag(pen)
        se = _se_from_hessian(H)
        se_fin, se_gsave = se[o:o + P], se[o + P:o + P + G]
    a = {k: float(v) for k, v in zip(keys, a_v)}
    b = {k: float(v) for k, v in zip(keys, b_v)}
    recon = "  ".join(f"{k}:Σp={p[sidx == i].sum():.0f}/Σy={Cr['y'][sidx == i].sum():.0f}"
                      for i, k in enumerate(keys))
    print(f"  conversion pre-calc EB prior SDs (per-fit): fin={prior_sd_fin:.3f} gsave={prior_sd_gsave:.3f} (logit)")
    print(f"  conversion (logit) fit: a={a} b={ {k: round(v,3) for k,v in b.items()} }  reconcile {recon}"
          f"  ({len(p):,} shots, {G} goalies, iters={nit})")
    out = {"a": a, "b": b, "fin": fin, "gsave": gsave, "goalies": Cr["goalies"],
           "beta": c_v, "ctx_names": Cr.get("ctx_names", []),
           "se_fin": se_fin, "se_gsave": se_gsave,
           "prior_sd_fin": prior_sd_fin, "prior_sd_gsave": prior_sd_gsave,
           "sum_p": float(p.sum()), "sum_y": float(Cr["y"].sum()), "n": int(len(p)),
           "converged": converged, "grad_norm": gnorm, "nit": nit,
           "theta": np.asarray(th, dtype=np.float64).copy()}
    if "season" in Cr:                                      # F6: per-season reconciliation (unpenalized
        out["recon_season"] = {int(s): [float(p[Cr["season"] == s].sum()),      # season offsets ⇒ exact)
                                        float(Cr["y"][Cr["season"] == s].sum())]
                               for s in np.unique(Cr["season"])}
    return out


# ── effective parameters: state + position offset + aging curve, per player ───────────────────────

def _coef_map(ctx_names, beta):
    """Named context coefficients (curve/offset extraction by column name; empty if unnamed)."""
    return {n: float(v) for n, v in zip(ctx_names or [], np.asarray(beta))}


def _block_curve(cm, blk, z, d, off_name=None):
    """One block's aging-curve + position-offset contribution at age-z and position d (0=F, 1=D):
    curve = F/D-split quadratic in z (coeffs `{blk}_zF … {blk}_z2D`), offset = `{blk}_D` (A2)."""
    w = [cm.get(f"{blk}_zF", 0.0), cm.get(f"{blk}_z2F", 0.0),
         cm.get(f"{blk}_zD", 0.0), cm.get(f"{blk}_z2D", 0.0)]
    return _curve_val(w, z, d) + d * cm.get(off_name or f"{blk}_D", 0.0)


def effective_params(rates, qual, conv, players, agepos, last_season, target=None):
    """Collapse the fitted model into per-player EFFECTIVE parameter arrays — last RW state (or
    static level) + position offset + aging curve — shaped exactly like the dicts player_values
    consumes. `target=None`: each player evaluated at his LAST active season ("current skill").
    `target=<season>`: ages advance to the target season while states stay at their last value (the
    RW mean) — the PROJECTION. Both use the reference-season environment (no season fixed-effects),
    so current and projected values are directly comparable. Missing birthdates stay at the curve
    reference (z = 0) and simply don't move under projection."""
    P = len(players)
    isD = agepos["isD"]
    slist = sorted(agepos["z"].keys())
    known = ~np.isnan(agepos["age"][slist[0]])
    z = np.zeros(P)
    for s in slist:
        m = last_season == s
        z[m] = agepos["z"][s][m]
    if target is not None:
        z = np.where(known & (last_season > 0), z + (target - last_season) / AGE_SCALE, z)
    rates_eff = {}
    for key, rate in rates.items():
        cm = _coef_map(rate.get("ctx_names"), rate["beta"])
        rates_eff[key] = {
            "intercept": rate["intercept"], "psi0": rate["psi0"],
            "value_env": rate.get("value_env"),
            "shoot": rate["shoot_last"] + _block_curve(cm, "shoot", z, isD),
            "create": rate["create_last"] + _block_curve(cm, "create", z, isD),
            "def": rate["def_last"] + _block_curve(cm, "def", z, isD),
        }
    qcm = _coef_map(qual.get("ctx_names"), qual["beta"])
    qc = np.asarray(qual["qcreate"])
    qual_eff = {"mu_qual": qual["mu_qual"], "beta_s": qual.get("beta_s"),
                "qshoot": qual["qshoot"] + isD * qcm.get("shooter_D", 0.0),
                "qcreate": qc[isD.astype(np.int64)] if qc.shape == (2,) else qc,   # A1: position pair
                "qcreate_pos": qc if qc.shape == (2,) else None,   # the [F, D] pair (creator classes)
                "qdef": qual["qdef"] + isD * qcm.get("def_D", 0.0)}
    ccm = _coef_map(conv.get("ctx_names"), conv.get("beta", []))
    conv_eff = {"a": conv["a"], "b": conv["b"],
                "fin": conv["fin"] + _block_curve(ccm, "fin", z, isD, off_name="shooter_D")}
    return rates_eff, qual_eff, conv_eff


def unit_effective(rate, agepos):
    """Per-unit (player, season) EFFECTIVE rate params — state + that season's curve value + D
    offset — the per-season skill trajectory behind the JSON `trend` block."""
    cm = _coef_map(rate.get("ctx_names"), rate["beta"])
    up, us = rate["unit_player"], rate["unit_season"]
    d = agepos["isD"][up]
    z = np.zeros(len(up))
    for s in sorted(agepos["z"].keys()):
        m = us == s
        z[m] = agepos["z"][s][up[m]]
    return {blk: rate[key] + _block_curve(cm, blk, z, d)
            for blk, key in (("shoot", "shoot"), ("create", "create"), ("def", "def"))}


# ── attribution: merge the rate (volume) and quality loadings into goal-scale numbers ──────────────

_GL_X, _GL_W = np.polynomial.legendre.leggauss(40)          # fixed nodes on (0,1) for the Beta marginal
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
    al, be = s * qbar, s * (1.0 - qbar)                      # (P,)
    x = np.clip(betaincinv(al[:, None], be[:, None], _GL_X[None, :]), EPS, 1 - EPS)   # (P, K)
    lx = np.log(x / (1.0 - x))
    f = _sigmoid(a * lx + b + np.atleast_1d(np.asarray(fin, dtype=np.float64))[:, None])
    return np.sum(_GL_W[None, :] * f, axis=1)


def value_environment(rate, season=None):
    """TOI-weighted mean ENVIRONMENT multiplier over the fit's own rows: everything in a row's
    log-rate beyond the intercept and the shooter's own effective shoot — teammates' create
    states, defenders' def states, the create/def age-position columns, game context, and season
    offsets. The value formulas multiply by this so card units are real-world per-60 rates: the
    reference cell (teammates at zero) runs ≈2× low because the league-average teammate's
    EFFECTIVE create is ≈ +0.17, not 0 (the 2026-07 Hyman-case audit). `season` restricts to that
    season's rows (the current-skill environment); arena states are excluded (≈0 mean by ridge)."""
    R = rate["R"]
    own = {"shoot_D", "shoot_zF", "shoot_z2F", "shoot_zD", "shoot_z2D"}
    beta = np.asarray(rate["beta"], dtype=np.float64).copy()
    for j, nm in enumerate(R.get("ctx_names") or []):
        if nm in own:
            beta[j] = 0.0
    cr, df = np.asarray(rate["create"]), np.asarray(rate["def"])
    dm = R.get("def_mask")
    if dm is None:
        dm = np.ones_like(R["def_idx"], dtype=np.float64)
    if "team_unit" in R:
        eta = cr[R["team_unit"]].sum(1) + (df[R["def_unit"]] * dm).sum(1)
    else:
        eta = cr[R["team_idx"]].sum(1) + (df[R["def_idx"]] * dm).sum(1)
    eta = eta + R["Xctx"] @ beta
    w = np.asarray(R["dur"], dtype=np.float64)
    if season is not None and "season_row" in R:
        m = np.asarray(R["season_row"]) == season
        if m.any():
            eta, w = eta[m], w[m]
    return float(np.sum(w * np.exp(eta)) / max(np.sum(w), 1e-9))


def creator_mix(psi0, isD, key):
    """Creator-class weights (unassisted, F-created, D-created) per shooter, in the model's own
    REFERENCE environment (teammate creates = 0): w0 = e^psi0/(e^psi0 + 4); the rest splits by the
    on-ice teammate position mix — EV 5v5 rosters are 3F+2D (an F shooter has 2F+2D teammates, a D
    has 3F+1D); the MA bucket uses the typical 4F+1D PP unit. A documented convention for the
    deployment-free VALUES; the WAR engine uses each stint's actual teammates instead."""
    isD = np.asarray(isD, dtype=np.float64)
    w0 = float(np.exp(psi0) / (np.exp(psi0) + 4.0))
    n_f = np.where(isD > 0, 3.0, 2.0) if key == "ev" else np.where(isD > 0, 4.0, 3.0)
    w_f = (1.0 - w0) * n_f / 4.0
    return w0, w_f, (1.0 - w0) - w_f


def player_values(rates, qual, conv, players, isD=None):
    """Per-strength deployment-free per-60 goal values. `rates` = {"ev": rate_fit, "ma": rate_fit}.
    For each strength (rate loadings are per-strength; quality/finishing loadings are POOLED, only the
    intercept splits):
       scoring(j)    = exp(mu_rate+shoot) · ḡ_j                                                   own shots, converted
       playmaking(p) = N_TM · exp(mu_rate) · (exp(create)−1) · sigmoid(mu_qual+qcreate)           teammate xG added
       defense(d)    = N_DEF · [exp(mu_rate)·sigmoid(mu_qual) − exp(mu_rate+def)·sigmoid(mu_qual+qdef)]  suppression
    ḡ_j is the model's own goals-per-shot: the conversion curve marginalized over BOTH the fitted
    shot-quality distribution (Beta, concentration beta_s — see marginal_goal_prob) AND the creator
    classes (unassisted / F-created / D-created; qcreate is parameterized with unassisted as the
    reference and created shots carry the negative position bumps, so skipping this marginalization
    prices every shot as unassisted ≈ +13% goals league-wide). Falls back to the point shortcut when
    beta_s / the qcreate pair / isD are unavailable (synthetic fixtures).
    EV → ev_scoring/playmaking/defense (N_DEF=5). MA → pp_scoring/pp_playmaking + pk_defense (N_DEF=4;
    the MA `def` loadings ARE the penalty-killers). All per 60; defense >0 = suppresses (good)."""
    out = {}
    qshoot, qcreate, qdef, fin = qual["qshoot"], qual["qcreate"], qual["qdef"], conv["fin"]
    s_conc = qual.get("beta_s")                              # Stage-2 Beta concentration
    qc_pos = qual.get("qcreate_pos")                         # the [F, D] creator-class pair
    if qc_pos is None and np.asarray(qcreate).shape == (2,):
        qc_pos = np.asarray(qcreate)
    for key, rate in rates.items():
        ml, mq = rate["intercept"], qual["mu_qual"][key]
        n_def = N_DEF_EV if key == "ev" else N_DEF_MA
        a, b = conv["a"][key], conv["b"][key]
        cr, shoot, defn = rate["create"], rate["shoot"], rate["def"]
        shots = np.exp(ml + shoot)
        q_own = np.clip(_sigmoid(mq + qshoot), EPS, 1 - EPS)
        if s_conc and qc_pos is not None and isD is not None:
            w0, wf, wd = creator_mix(rate["psi0"], isD, key)
            p_own = p_own0 = 0.0
            for w, qc in ((w0, 0.0), (wf, float(qc_pos[0])), (wd, float(qc_pos[1]))):
                qb = np.clip(_sigmoid(mq + qshoot + qc), EPS, 1 - EPS)
                p_own = p_own + w * marginal_goal_prob(qb, float(s_conc), a, b, fin)
                p_own0 = p_own0 + w * marginal_goal_prob(qb, float(s_conc), a, b)
        elif s_conc:                                         # Beta marginal, no creator classes
            p_own = marginal_goal_prob(q_own, float(s_conc), a, b, fin)
            p_own0 = marginal_goal_prob(q_own, float(s_conc), a, b)
        else:                                                # point shortcut (synthetic fixtures)
            lq = np.log(q_own / (1 - q_own))
            p_own = _sigmoid(a * lq + b + fin)
            p_own0 = _sigmoid(a * lq + b)
        base = np.exp(ml) * _sigmoid(mq)
        env = float(rate.get("value_env") or 1.0)            # average-environment factor (see
        out[key] = {                                         # value_environment; 1.0 = reference)
            "scoring": env * shots * p_own,                    # own shots, converted to goals
            "finishing": env * shots * (p_own - p_own0),       # goals above xG-implied conversion
            "own_xg": env * shots * _sigmoid(mq + qshoot), "own_shots": env * shots,
            "playmaking": env * N_TM * np.exp(ml) * (np.exp(cr) - 1.0) * _sigmoid(mq + qcreate),
            "defense": env * n_def * (base - np.exp(ml + defn) * _sigmoid(mq + qdef)),
            "creator_share": np.exp(cr) / (np.exp(rate["psi0"]) + np.exp(cr) + (N_TM - 1)),
        }
    return out


# ── simulation + posterior-predictive check ───────────────────────────────────────────────────────

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def ppc(R, rate, qual, conv, key, seed=0, agepos=None):
    """Forward-simulate every shooter-stint row of one strength bucket (`key` ∈ {'ev','ma'}) and compare
    to actuals: shot count (rate), shot quality (pooled quality at the strength intercept), total goals
    (conversion at the strength's a/b). Rate gathers use the per-season UNIT states when present; the
    quality context is mapped from the rate row — home, season indicators, position offsets (F5); the
    conversion side includes the season offsets and finishing curve (the goalie AGE curve is omitted —
    the per-goalie gsave still applies). Defenders masked (≤5)."""
    rng = np.random.default_rng(seed)
    dmask = R.get("def_mask")
    if dmask is None:
        dmask = np.ones_like(R["def_idx"], dtype=np.float64)
    sh_u = R.get("shooter_unit", R["shooter_idx"])
    tm_u = R.get("team_unit", R["team_idx"])
    df_u = R.get("def_unit", R["def_idx"])
    eta_r = (rate["intercept"] + rate["shoot"][sh_u] + rate["create"][tm_u].sum(1)
             + (rate["def"][df_u] * dmask).sum(1) + R["Xctx"] @ rate["beta"])
    acol = R.get("arena_col")
    if acol is not None and len(rate.get("arena_vec", [])):
        eta_r = eta_r + np.where(acol >= 0, rate["arena_vec"][np.clip(acol, 0, None)], 0.0)
    mu = np.exp(eta_r + R["offset"])
    if rate.get("count_model") == "nb" and rate.get("r"):
        r = rate["r"]
        N = rng.negative_binomial(r, r / (r + mu))
    else:
        N = rng.poisson(mu)

    # quality marginalized over the creator, at this strength's quality intercept (mu_qual[key]);
    # context mapped from the rate row: home + season indicators + position offsets (F5)
    qb = np.asarray(qual["beta"])
    nseas = R.get("n_season_cols", 0)
    qctx = qb[0] * R["Xctx"][:, 0]
    if nseas and len(qb) >= 2 + nseas:
        qctx = qctx + R["Xctx"][:, 5:5 + nseas] @ qb[2:2 + nseas]
    qcm = _coef_map(qual.get("ctx_names"), qb)
    isD = agepos["isD"] if agepos is not None else None
    if isD is not None:
        qctx = qctx + qcm.get("shooter_D", 0.0) * isD[R["shooter_idx"]] \
                    + qcm.get("def_D", 0.0) * (isD[R["def_idx"]] * dmask).sum(1)
    if acol is not None and len(qual.get("arena_vec", [])):
        qctx = qctx + np.where(acol >= 0, qual["arena_vec"][np.clip(acol, 0, None)], 0.0)
    base = (qual["mu_qual"][key] + qual["qshoot"][R["shooter_idx"]]
            + (qual["qdef"][R["def_idx"]] * dmask).sum(1) + qctx)
    qc2 = np.asarray(qual["qcreate"])
    if qc2.shape == (2,):                                   # A1: position-level creator bump
        qbump = qc2[(isD[R["team_idx"]] if isD is not None
                     else np.zeros_like(R["team_idx"], dtype=float)).astype(np.int64)]
    else:                                                   # legacy per-player array (synthetic tests)
        qbump = qc2[R["team_idx"]]
    sig5 = np.concatenate([_sigmoid(base)[:, None], _sigmoid(base[:, None] + qbump)], 1)
    lg = np.concatenate([np.full((len(base), 1), rate["psi0"]), rate["create"][tm_u]], 1)
    pi = np.exp(lg - lg.max(1, keepdims=True)); pi /= pi.sum(1, keepdims=True)
    qpred_marg = (pi * sig5).sum(1)
    gsave_map = {g: conv["gsave"][i] for i, g in enumerate(conv["goalies"])}
    fin_row = conv["fin"][R["shooter_idx"]]                                    # per shooter-stint row
    soff = np.zeros(len(base))
    ccm = _coef_map(conv.get("ctx_names"), conv.get("beta", []))
    if agepos is not None and "season_row" in R and ccm:
        zsh = np.zeros(len(base))
        for s in sorted(set(int(x) for x in np.unique(R["season_row"]))):
            m = R["season_row"] == s
            zsh[m] = agepos["z"][s][R["shooter_idx"][m]]
            soff[m] = ccm.get(f"season_{s}", 0.0)
        fin_row = fin_row + _block_curve(ccm, "fin", zsh, isD[R["shooter_idx"]], off_name="shooter_D")
    gsv_row = np.array([gsave_map.get(int(g), 0.0) if pd.notna(g) else 0.0 for g in R["def_goalie"]])

    rep = np.repeat(np.arange(len(N)), N)
    p_q = qpred_marg[rep]
    s = qual["beta_s"]
    q = rng.beta(s * p_q, s * (1 - p_q))                                      # this chance's drawn xG
    qc = np.clip(q, EPS, 1 - EPS)
    eta = (conv["a"][key] * np.log(qc / (1 - qc)) + conv["b"][key]
           + soff[rep] + fin_row[rep] + gsv_row[rep])
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


def _arena_report(fit, n=3, pct=True):
    """One-line summary of the fitted (venue, season) arena states: biggest/smallest venue means
    (rate: ≈% on shot counts; quality: logit-xG units) + the largest within-venue season drift."""
    vec = fit.get("arena_vec")
    if vec is None or not len(vec):
        return None
    g = pd.DataFrame({"v": fit["arena_venue"], "x": vec}).groupby("v").x
    m = g.mean().sort_values()
    fmt = (lambda x: f"{100 * x:+.1f}%") if pct else (lambda x: f"{x:+.3f}")
    lo = "  ".join(f"{v} {fmt(x)}" for v, x in m.head(n).items())
    hi = "  ".join(f"{v} {fmt(x)}" for v, x in m.tail(n)[::-1].items())
    return f"hi {hi}   lo {lo}   (max venue drift {fmt(float((g.max() - g.min()).max()))})"


def _arena_json(fit):
    """{venue: {season: coef}} from a fit's arena states (None when absent)."""
    vec = fit.get("arena_vec")
    if vec is None or not len(vec):
        return None
    out = {}
    for v, s, x in zip(fit["arena_venue"], fit["arena_season"], vec):
        out.setdefault(v, {})[int(s)] = round(float(x), 4)
    return out


def _curve_report(cm, blk):
    """One-line console summary of a fitted aging curve: curve value at 22/32 per position and the
    implied peak age (where the quadratic tops out, if it does)."""
    parts = []
    for pos in ("F", "D"):
        w1, w2 = cm.get(f"{blk}_z{pos}", 0.0), cm.get(f"{blk}_z2{pos}", 0.0)
        pk = AGE_PEAK - AGE_SCALE * w1 / (2 * w2) if w2 < -1e-9 else None
        v = {a: w1 * (a - AGE_PEAK) / AGE_SCALE + w2 * ((a - AGE_PEAK) / AGE_SCALE) ** 2
             for a in (22, 32)}
        parts.append(f"{pos}: 22y {v[22]:+.3f} 32y {v[32]:+.3f}"
                     + (f" peak {pk:.1f}" if pk is not None else ""))
    return "   ".join(parts)


def _curve_json(cm, blk, off_name=None):
    """JSON form of one fitted aging curve: coefficients, the D intercept offset, and the curve
    sampled over ages 18–40 per position (site-ready)."""
    return {"coef": {c: cm.get(f"{blk}_{c}", 0.0) for c in ("zF", "z2F", "zD", "z2D")},
            "d_offset": cm.get(off_name or f"{blk}_D", 0.0),
            "curve": {pos: {a: round(float(_curve_val(
                [cm.get(f"{blk}_zF", 0.0), cm.get(f"{blk}_z2F", 0.0),
                 cm.get(f"{blk}_zD", 0.0), cm.get(f"{blk}_z2D", 0.0)],
                (a - AGE_PEAK) / AGE_SCALE, d)), 4) for a in range(18, 41, 2)}
                      for pos, d in (("F", 0.0), ("D", 1.0))}}


def fit_all(seasons, count_model="poisson", spg_scale=1.0, warm=False, reexport=False,
            save_ckpt=False, ckpt_path=None, ma_anchor_scale=1.0, ma_create_prior_sd=None,
            ma_def_prior_sd=None):
    """Fit every stage on `seasons` and return the raw fit objects — the shared engine behind run()
    (which adds leaderboards, values, PPC, projection, JSON) and the held-out predictive harness
    (generative_holdout, which needs the fits without the reporting tail). Returns a dict with
    players/idx/agepos/Q/rates/spg/qual/conv/last_season.

    `warm` initializes each stage's optimizer from the θ̂ checkpoint (key-mapped; see ckpt_save) —
    exact refits converge in a handful of iterations, incremental refits (new season/players) in a
    fraction of a cold run. `reexport` skips optimization AND the SE Hessians entirely (requires a
    checkpoint from the exact same fit signature) — for regenerating reports/JSON after export-side
    changes. `save_ckpt` writes the checkpoint after fitting; `ckpt_path` overrides its location
    (run() uses the production path; the holdout harness keeps its own chain, so train-side fits
    never touch a solution that saw test data).

    `ma_anchor_scale` / `ma_create_prior_sd`: the MA (PP/PK) bucket's assist-anchor weight scale
    and create prior SD — hyperparameters selected by held-out validation (docs §7): on fixed PP
    units the shot counts pin only the unit's total creation, so the assist anchor alone splits
    it, and assists on the PP reflect ROLE (who touches the puck last), not creation skill. The EV
    bucket is untouched (linemate mixing identifies its split; validated at full weight)."""
    print(f"[generative_model:shooter-resolved] seasons {seasons} — count {count_model} — EV + PP/PK …")
    if ma_anchor_scale != 1.0 or ma_create_prior_sd or ma_def_prior_sd:
        print(f"  MA bucket: anchor×{ma_anchor_scale:g}"
              + (f", create prior sd {ma_create_prior_sd:g}" if ma_create_prior_sd else "")
              + (f", def prior sd {ma_def_prior_sd:g}" if ma_def_prior_sd else ""))
    ck = ckpt_load(ckpt_path) if (warm or reexport) else None
    if reexport:
        if ck is None:
            raise SystemExit("--reexport needs a θ̂ checkpoint from a prior full fit")
        ck_mas = float(ck.get("ma_anchor_scale", 1.0))
        ck_mcp = float(ck.get("ma_create_prior_sd", np.nan))
        if (list(map(int, ck["seasons"])) != [int(s) for s in seasons]
                or str(ck["count_model"]) != count_model
                or float(ck["spg_scale"]) != float(spg_scale)
                or ck_mas != float(ma_anchor_scale)
                or not (np.isnan(ck_mcp) and ma_create_prior_sd is None
                        or ck_mcp == float(ma_create_prior_sd or np.nan))):
            raise SystemExit("--reexport: checkpoint signature (seasons/count/spg-scale/MA hypers) "
                             "differs — run a full fit")
    elif ck is not None:
        print(f"  warm start ← checkpoint ({'+'.join(str(int(s)) for s in ck['seasons'])}, "
              f"{ck['count_model']})")
    players, idx = player_index(seasons)
    if not players:
        raise SystemExit("no stints")
    P = len(players)
    agepos = _age_position(players, seasons)
    print(f"  ages/positions: {P} players, {agepos['missing']} missing birthdates, "
          f"{int(agepos['isD'].sum())} D")

    # pooled quality rows (EV+MA); its goal subset (split by strength) anchors each bucket's `create`
    Q = quality_creator_rows(seasons, idx, ALL_STRENGTHS, agepos)
    cidx_all = np.where(Q["creator"] == 4, 0, np.clip(Q["creator"], 0, 3) + 1).astype(np.int64)

    rates, spg, creates = {}, {}, {}
    last_season = np.full(P, -1, dtype=np.int64)
    for key, strengths, dual in [("ev", EV_STRENGTHS, True), ("ma", MA_STRENGTHS, False)]:
        R = rate_rows(seasons, strengths, dual, players, idx, agepos, states=dual)
        if R is None or len(R["count"]) == 0:
            print(f"  [{key}] no stints — skipping"); continue
        slab = 0 if key == "ev" else 1
        gm = (Q["goal"] == 1) & (Q["strength"] == slab)
        ngoal = int(gm.sum()); nun = int(((Q["creator"] == 4) & (Q["strength"] == slab)).sum())
        aw = ma_anchor_scale if key == "ma" else 1.0         # per-bucket anchor weight (docs §7)
        spg[key] = aw * spg_scale * float(R["count"].sum()) / max(ngoal, 1)  # IPW weight (×A3 scale)
        anchor = gm & (Q["creator"] >= 0)                    # F4: off-ice-assister goals are latent
        gt, gc = Q["team_idx"][anchor], cidx_all[anchor]
        sord = {s: i for i, s in enumerate(R["seasons"])}
        # A2 stage: goals with an on-ice A1 (a column to mask) AND an on-ice A2
        anchor2 = (gm & (Q["creator"] >= 0) & (Q["creator"] <= 3)
                   & (Q["creator2"] >= 0) & (Q["creator2"] != Q["creator"]))
        gt2, g1c, gc2 = Q["team_idx"][anchor2], Q["creator"][anchor2], Q["creator2"][anchor2]
        if dual:                                             # anchor teammates → (player, season) units
            qs = np.array([sord[s] for s in Q["season"][anchor]], dtype=np.int64)
            gtu = R["unit_lut"][gt, qs[:, None]]
            ok = (gtu >= 0).all(1)
            if int((~ok).sum()):
                print(f"  [{key}] {int((~ok).sum())} anchor goals dropped (teammate w/o rate exposure)")
            gt, gc = gtu[ok], gc[ok]
            qs2 = np.array([sord[s] for s in Q["season"][anchor2]], dtype=np.int64)
            gtu2 = R["unit_lut"][gt2, qs2[:, None]]
            ok2 = (gtu2 >= 0).all(1)
            gt2, g1c, gc2 = gtu2[ok2], g1c[ok2], gc2[ok2]
        print(f"  [{key}] rate rows {len(R['count']):,}  shots {R['count'].sum():.0f}  goals {ngoal}  "
              f"spg {spg[key]:.1f}  anchored {len(gc)} ({100 * nun / max(ngoal, 1):.0f}% unassisted, "
              f"{len(gc2)} with A2)")
        rate = fit_rate_create(R, gt, gc, spg[key], count_model=count_model, a2=(gt2, g1c, gc2),
                               warm=_ck_stage(ck, key), reuse=reexport,
                               create_prior_sd=(ma_create_prior_sd if key == "ma" else None),
                               def_prior_sd=(ma_def_prior_sd if key == "ma" else None))
        rate["R"] = R
        rates[key] = rate
        last_season = np.maximum(last_season, R["last_season"])
        rdesc = f" r={rate['r']:.2f}" if rate.get("r") else ""
        nu = f" units={R['n_units']:,}" if dual and "n_units" in R else ""
        qd = f" a2_q={rate['a2_q']:.3f}" if rate.get("a2_q") is not None else ""
        print(f"    rate fit ({count_model}): converged={rate['converged']} |grad|={rate['grad_norm']:.1e}"
              f"{rdesc} create_0={rate['psi0']:+.3f}{nu}{qd} iters={rate['nit']}")
        cm = _coef_map(R["ctx_names"], rate["beta"])
        for blk in ("shoot", "create", "def"):
            print(f"    [{key}] {blk}-curve   {_curve_report(cm, blk)}")
        arep = _arena_report(rate)
        if arep:
            print(f"    [{key}] arena rate effects   {arep}")
        if dual:                                             # stage-2 pi via per-season unit states
            qs_all = np.array([sord[s] for s in Q["season"]], dtype=np.int64)
            tu_all = R["unit_lut"][Q["team_idx"], qs_all[:, None]]
            cre = np.append(rate["create"], 0.0)             # sentinel for unmapped teammates (no bump)
            tu_all = np.where(tu_all >= 0, tu_all, len(rate["create"]))
            creates[slab] = (cre, rate["psi0"], tu_all)
        else:
            creates[slab] = (rate["create"], rate["psi0"])
    if "ev" not in rates:
        raise SystemExit("no EV stints")

    # pooled quality fit (shared loadings; per-strength intercept; per-row creator dist by strength)
    qual = fit_quality_creator(Q, P, creates, isD=agepos["isD"], warm=_ck_stage(ck, "qual"),
                               reuse=reexport, pids=players)
    print(f"  quality fit (pooled EV+MA): converged={qual['converged']} |grad|={qual['grad_norm']:.1e} "
          f"beta_s={qual['beta_s']:.1f} mu_qual={ {k: round(v, 3) for k, v in qual['mu_qual'].items()} } "
          f"iters={qual['nit']}")
    qc, qse = qual["qcreate"], qual["se_qcreate"]
    print(f"  qcreate (position-level, A1): F {qc[0]:+.4f} ±{1.96 * qse[0]:.4f}   "
          f"D {qc[1]:+.4f} ±{1.96 * qse[1]:.4f}")
    qarep = _arena_report(qual, pct=False)
    if qarep:
        print(f"  arena quality (logit-xG) effects   {qarep}")

    # pooled conversion fit (shared fin/gsave; per-strength a/b; season offsets + curves in ctx)
    Cr = conversion_rows(seasons, idx, ALL_STRENGTHS, agepos)
    if Cr is None:
        raise SystemExit("no shots for the conversion fit")
    conv = fit_conversion(Cr, P, warm=_ck_stage(ck, "conv"), reuse=reexport, pids=players)
    ccm = _coef_map(conv.get("ctx_names"), conv.get("beta", []))
    if ccm:
        print(f"    fin-curve   {_curve_report(ccm, 'fin')}   goalie-age: z {ccm.get('g_z', 0.0):+.3f} "
              f"z² {ccm.get('g_z2', 0.0):+.3f}")
        if conv.get("recon_season"):
            rc = "  ".join(f"{s}:{v[0]:.0f}/{v[1]:.0f}" for s, v in sorted(conv["recon_season"].items()))
            print(f"    per-season Σp/Σy (F6): {rc}")
    if save_ckpt:
        cp = ckpt_save(seasons, count_model, spg_scale, rates, qual, conv, players, path=ckpt_path,
                       ma_anchor_scale=ma_anchor_scale, ma_create_prior_sd=ma_create_prior_sd,
                       ma_def_prior_sd=ma_def_prior_sd)
        print(f"  θ̂ checkpoint → {cp} (warm starts / --reexport)")
    return {"players": players, "idx": idx, "agepos": agepos, "Q": Q, "rates": rates, "spg": spg,
            "qual": qual, "conv": conv, "last_season": last_season}


def run(seasons, count_model="poisson", spg_scale=1.0, warm=True, reexport=False,
        ma_anchor_scale=1.0, ma_create_prior_sd=None, ma_def_prior_sd=None):
    names = roster_names(seasons)
    M = fit_all(seasons, count_model=count_model, spg_scale=spg_scale, warm=warm,
                reexport=reexport, save_ckpt=not reexport,
                ma_anchor_scale=ma_anchor_scale, ma_create_prior_sd=ma_create_prior_sd,
                ma_def_prior_sd=ma_def_prior_sd)
    players, agepos, Q = M["players"], M["agepos"], M["Q"]
    rates, spg, qual, conv, last_season = M["rates"], M["spg"], M["qual"], M["conv"], M["last_season"]

    # average-environment factor per bucket (card units = real-world per-60 rates)
    for key in rates:
        rates[key]["value_env"] = value_environment(rates[key], season=int(max(seasons)))
    print("  value environment: " + "  ".join(f"{k} ×{rates[k]['value_env']:.2f}" for k in rates))

    # effective per-player params (last state + position offset + curve) → values
    eff_rates, eff_qual, eff_conv = effective_params(rates, qual, conv, players, agepos, last_season)
    vals = player_values(eff_rates, eff_qual, eff_conv, players, isD=agepos["isD"])

    # pooled (shared) skill leaderboards — once. RAW player residuals, not effective params: the
    # position offsets / age curves are calibration terms (A2), and adding them back would turn a
    # talent board into a position board (e.g. the conversion shooter_D offset compensates the a>1
    # slope on low-xG point shots — every D would top "finishing").
    ev_toi = rates["ev"]["R"]["toi"]
    _board(names, players, qual["qshoot"], ev_toi, "TOP qshoot (own-shot danger above position baseline):")
    _board(names, players, conv["fin"], ev_toi, "TOP FINISHING (log-odds above position/age baseline):",
           conv["se_fin"])
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
        _board(names, players, rates[key]["create_last"], off,
               f"[{lbl}] TOP create (playmaking volume, last state):", rates[key]["se_create_last"],
               min_toi=gate)
        _board(names, players, vals[key]["scoring"], off, f"[{lbl}] TOP SCORING (goals/60, own shots):", min_toi=gate)
        _board(names, players, vals[key]["playmaking"], off, f"[{lbl}] TOP PLAYMAKING (xG/60):", min_toi=gate)
        _board(names, players, vals[key]["defense"], dfn, f"[{'EV' if key == 'ev' else 'PK'}] BEST DEFENSE (xG/60 suppressed):", min_toi=gate)
        pp = ppc(R, rates[key], qual, conv, key, agepos=agepos)
        slab = 0 if key == "ev" else 1
        pp["goals_actual"] = int(((Q["goal"] == 1) & (Q["strength"] == slab)).sum())
        ppcs[key] = pp
        print(f"  [{lbl}] PPC — shots a/s {pp['shots_actual']:,}/{pp['shots_sim']:,} "
              f"(model-exp {pp['shots_expected']:.0f})  goals a/s {pp['goals_actual']:,}/{pp['goals_sim']:,}")

    # projection: ages advance to the target season, states stay at their last value (RW mean)
    proj = None
    if len(set(seasons)) >= 2:
        target = int(max(seasons)) + 1
        pr, pq, pc = effective_params(rates, qual, conv, players, agepos, last_season, target=target)
        pvals = player_values(pr, pq, pc, players, isD=agepos["isD"])
        proj = {"season": target, "vals": pvals}
        print(f"\n──[projection → {target}] (last RW state + aging curve; reference environment) ──")
        _board(names, players, pvals["ev"]["scoring"], ev_toi, f"PROJECTED {target} EV SCORING (goals/60):")
        _board(names, players, pvals["ev"]["playmaking"], ev_toi, f"PROJECTED {target} EV PLAYMAKING (xG/60):")

    _save(seasons, players, rates, qual, conv, vals, ppcs, spg, agepos, last_season,
          (eff_rates, eff_qual, eff_conv), proj)


def _save(seasons, players, rates, qual, conv, vals, ppcs, spg, agepos, last_season, eff, proj):
    label = "+".join(map(str, seasons))
    ev, ma = rates.get("ev"), rates.get("ma")
    eff_rates, eff_qual, eff_conv = eff
    qc = np.asarray(qual["qcreate"])
    curves = {}
    for key, rate in rates.items():
        cm = _coef_map(rate.get("ctx_names"), rate["beta"])
        if cm:
            for blk in ("shoot", "create", "def"):
                curves[f"{key}_{blk}"] = _curve_json(cm, blk)
    ccm = _coef_map(conv.get("ctx_names"), conv.get("beta", []))
    if ccm:
        curves["fin"] = _curve_json(ccm, "fin", off_name="shooter_D")
        curves["gsave_age"] = {"z": ccm.get("g_z", 0.0), "z2": ccm.get("g_z2", 0.0)}
    out = {"model": "generative_model_ev_ma", "seasons": list(seasons),
           "count_model": ev.get("count_model"), "nb_r": ev.get("r"), "shots_per_goal": spg,
           "value_env": {k: r.get("value_env") for k, r in rates.items()},
           "quality_intercept": float(qual["intercept"]), "mu_qual": qual["mu_qual"],
           "beta_s": qual["beta_s"],
           "qcreate": {"F": float(qc[0]), "D": float(qc[1]),
                       "se_F": float(qual["se_qcreate"][0]), "se_D": float(qual["se_qcreate"][1])},
           "quality_ctx": {n: float(v) for n, v in zip(qual.get("ctx_names") or [], qual["beta"])},
           "age_curves": curves,
           "rw_sd": {"shoot": RW_SD_SHOOT, "create": RW_SD_CREATE, "def": RW_SD_DEF},
           "arena_effects": {"ev": _arena_json(ev) if ev else None,
                             "ma": _arena_json(ma) if ma else None,
                             "quality": _arena_json(qual),
                             "prior": {"sd": ARENA_SD, "rw_sd": ARENA_RW_SD}},
           "missing_birthdates": agepos["missing"],
           "conv": {"a": conv["a"], "b": conv["b"], "prior_sd_fin": conv["prior_sd_fin"],
                    "prior_sd_gsave": conv["prior_sd_gsave"], "sum_p": conv["sum_p"],
                    "sum_y": conv["sum_y"], "recon_season": conv.get("recon_season"),
                    "ctx": {n: float(v) for n, v in zip(conv.get("ctx_names", []), conv.get("beta", []))}},
           "strengths": {k: {"rate_intercept": float(rates[k]["intercept"]), "psi0": float(rates[k]["psi0"]),
                             "a2_q": rates[k].get("a2_q"), "n_a2": rates[k].get("n_a2"),
                             "rate_ctx": {n: float(v) for n, v in
                                          zip(rates[k].get("ctx_names") or [], rates[k]["beta"])},
                             "ppc": ppcs.get(k)} for k in rates},
           "players": [], "goalies": [{"id": int(conv["goalies"][j]), "gsave": float(conv["gsave"][j]),
                                       "gsave_se": float(conv["se_gsave"][j])}
                                      for j in range(len(conv["goalies"]))]}
    trend = {}
    if ev is not None and ev.get("unit_season") is not None:
        ue = unit_effective(ev, agepos)
        for u, (p, s) in enumerate(zip(ev["unit_player"], ev["unit_season"])):
            trend.setdefault(int(p), {})[int(s)] = {
                "shoot": round(float(ue["shoot"][u]), 4),
                "create": round(float(ue["create"][u]), 4),
                "def": round(float(ue["def"][u]), 4)}
    for i in range(len(players)):
        ls = int(last_season[i])
        age = agepos["age"].get(ls, np.full(len(players), np.nan))[i] if ls > 0 else np.nan
        rec = {"id": int(players[i]),
               "pos": "D" if agepos["isD"][i] else "F",
               "age": round(float(age), 1) if np.isfinite(age) else None,
               "last_season": ls if ls > 0 else None,
               "toi_ev": float(ev["R"]["toi_atk"][i]) if ev else 0.0,
               "toi_pp": float(ma["R"]["toi_atk"][i]) if ma else 0.0,
               "toi_pk": float(ma["R"]["toi_def"][i]) if ma else 0.0,
               "qshoot": float(qual["qshoot"][i]), "qdef": float(qual["qdef"][i]),
               "n_create": int(qual["n_create"][i]),
               "fin": float(conv["fin"][i]), "fin_se": float(conv["se_fin"][i]),
               "ev_defense": float(vals["ev"]["defense"][i]) if "ev" in vals else 0.0,
               "pk_defense": float(vals["ma"]["defense"][i]) if "ma" in vals else 0.0}
        for key, pre in [("ev", "ev"), ("ma", "pp")]:
            if key in rates:
                rec[f"{pre}_shoot"] = float(eff_rates[key]["shoot"][i])
                rec[f"{pre}_create"] = float(eff_rates[key]["create"][i])
                rec[f"{pre}_create_se"] = float(rates[key]["se_create_last"][i])
                rec[f"{pre}_def"] = float(eff_rates[key]["def"][i])
                rec[f"{pre}_scoring"] = float(vals[key]["scoring"][i])
                rec[f"{pre}_playmaking"] = float(vals[key]["playmaking"][i])
                rec[f"{pre}_finishing"] = float(vals[key]["finishing"][i])
        if i in trend:
            rec["trend"] = trend[i]
        out["players"].append(rec)
    if proj:
        pv = proj["vals"]
        pl = []
        for i in range(len(players)):
            rec = {"id": int(players[i]),
                   "ev_scoring": float(pv["ev"]["scoring"][i]),
                   "ev_playmaking": float(pv["ev"]["playmaking"][i]),
                   "ev_defense": float(pv["ev"]["defense"][i])}
            if "ma" in pv:
                rec.update(pp_scoring=float(pv["ma"]["scoring"][i]),
                           pp_playmaking=float(pv["ma"]["playmaking"][i]),
                           pk_defense=float(pv["ma"]["defense"][i]))
            pl.append(rec)
        out["projection"] = {"season": proj["season"], "players": pl,
                             "note": "last RW state + aging curve at target-season age; "
                                     "reference environment; uncertainty widens by rw_sd per block"}
    C.MODELS.mkdir(parents=True, exist_ok=True)
    (C.MODELS / f"generative_model_{label}.json").write_text(json.dumps(out))
    print(f"\n  -> data/models/generative_model_{label}.json")


def main(argv=None):
    p = argparse.ArgumentParser(description="Experimental shooter-resolved generative model — POC")
    p.add_argument("--season", type=int, default=None, help="one season (default: latest available)")
    p.add_argument("--pool", action="store_true", help="pool all available seasons")
    p.add_argument("--count", choices=["poisson", "nb"], default="poisson",
                   help="count layer: poisson (Var=μ) or nb (negative binomial, Var=μ+μ²/r)")
    p.add_argument("--spg-scale", type=float, default=1.0,
                   help="multiply the assist-credit weight (A3 sensitivity checks: 0.5 / 2.0)")
    p.add_argument("--ma-anchor-scale", type=float, default=0.25,
                   help="scale the PP/PK bucket's assist-anchor weight (default: the value the "
                        "2026-07 held-out calibration sweep selected — docs §7/§9; EV untouched)")
    p.add_argument("--ma-create-prior", type=float, default=0.04,
                   help="create prior SD for the PP/PK bucket (default: sweep-selected)")
    p.add_argument("--ma-def-prior", type=float, default=0.10,
                   help="def prior SD for the PP/PK bucket (default: sweep-selected)")
    p.add_argument("--cold", action="store_true",
                   help="ignore the θ̂ checkpoint (default: warm-start each stage from the last fit)")
    p.add_argument("--reexport", action="store_true",
                   help="skip optimization: reuse the checkpoint's θ̂/SEs (exact same fit signature "
                        "required) and just regenerate reports + JSON — for export-side changes")
    args = p.parse_args(argv)
    sd = C.PROCESSED / "shots_onice"
    avail = sorted(int(f.stem) for f in sd.glob("*.parquet")) if sd.exists() else []
    if not avail:
        raise SystemExit("no processed shots — run `make stints` first")
    seasons = avail if args.pool else ([args.season] if args.season else [avail[-1]])
    run(seasons, count_model=args.count, spg_scale=args.spg_scale,
        warm=not args.cold, reexport=args.reexport,
        ma_anchor_scale=args.ma_anchor_scale, ma_create_prior_sd=args.ma_create_prior,
        ma_def_prior_sd=args.ma_def_prior)


if __name__ == "__main__":
    main()
