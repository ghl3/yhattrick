# The Generative Player Model (shooter-resolved, unified-creation, player curves)

**Status:** experimental proof-of-concept. The implementation spans focused modules under
[`pipeline/src/yhattrick/models/`](../pipeline/src/yhattrick/models/): `generative_likelihood` (the
stage objectives, parameter layouts and math primitives), `generative_data` (fact/dimension tables →
design rows), `generative_model` (the fits + effective parameters), `generative_export` (values, PPC,
JSON, CLI) and `generative_features` (shared stint orientation). Run with
`uv run --group experimental python -m yhattrick.models.generative_model [--pool] [--count nb]`.
**Not wired into the site/export** — the production additive (RAPM) model remains primary. This model
is the *generative* counterpart: it specifies how a stint **produces** shots and goals, so you can both
fit it and simulate from it. Covers **even strength (5v5) and the man-advantage (5v4/4v5)**, pools
multiple seasons, and models **player skill as a function of time**: shared F/D aging curves in every
stage plus per-(player, season) random-walk states in the dense EV blocks — one pooled fit yields each
player's whole per-season trajectory and a **next-season projection**. See "Player curves" (§3).

This document is the reference for the model: its specification, inference, results, and open problems.

---

## 1. Why this model exists

The production model decomposes goals with a linear RAPM, then splits offense into scoring/playmaking
with a heuristic proportion `φ = own ixG / on-ice xGF`. That approach has two weaknesses this model is
designed to avoid:

1. **A baseline/delta split.** RAPM coefficients are deviations *from the league average*, so the
   average must be added back (`base/5`) to reconcile to real goals.
2. **Playmaking that is a relabel of total offense** (near-collinear with it), rather than a distinct
   skill.

The generative model instead attributes the *actual* chances/goals to the players who produced them,
and is **shooter-resolved**: it models *who shoots* and *who creates for teammates* separately, which is
what makes scoring vs playmaking identifiable without pass-tracking data.

A third weakness — shared with RAPM — was that pooling seasons assumed **one frozen skill level per
player**, which biases "current skill" over long windows and forbids projection. The player curves
(§3) remove that assumption: pooling more seasons now *adds* information instead of blurring it, so
the fitting window can grow past the current five seasons without retraining per year.

---

## 2. The model

A 5v5 stint is a **marked Poisson process with Bernoulli thinning**, in three **stages** — rate,
quality, conversion. Each stage has its own parameters and is **fit separately** (§5): the stages are
conditionally independent given the predictors, so the fits run in sequence rather than jointly. The one
parameter shared across stages is `create`/`create_0` — **estimated in Stage 1 and held fixed in Stage
2** (Stage 3 is independent of the other two). All notation is ASCII and defined in the glossary at the
end of this section; the logistic function is `sigmoid(z) = 1/(1+e^-z)` and
`softmax(v)_c = e^(v_c) / Σ_k e^(v_k)`.

An *attacking epoch* `e` is one (stint, attacking-team) pair — each 5v5 stint yields two (home-attacks,
away-attacks). An epoch has attackers `A` (≤5 skaters), defenders `B` (≤5), defending goalie `g`,
duration `t` (offset `o = t/3600`), and context vector `x`. For a focal shooter `j`, his teammates are
`T = A\{j}` (≤4).

### Strengths & seasons — "separate volume, shared skill"
Two strength buckets: **EV** = `{5v5}` (both sides attack → two epochs/stint) and **MA** = `{5v4, 4v5}`
(only the more-skaters side attacks → one epoch/stint; the 4-skater defenders are the penalty-killers).
The attacking side always has 5 skaters, so `T` is always 4 teammates; only the defender count varies
(5 at EV, 4 on the PK) — handled by a per-row **defender mask** (padded to width 5). The split follows
the production RAPM (`ev` vs `pp_pk`).

Because PP per-player samples are ~10× sparser than 5v5, we **share where skill lives and separate where
deployment lives**:
- **Rate (volume) is fit separately per strength** — `ev_shoot/create/def` and `pp_shoot/create`/`pk_def`
  are independent (each EB-ridged to its own strength). PP ice time is coach-driven, so EV volume is a
  poor prior for PP volume. The dense on-ice `create`/`def` give card-worthy `pp_playmaking`/`pk_defense`;
  the thin own-shot `shoot` makes `pp_scoring` lower-confidence.
- **Quality (`qshoot/qcreate/qdef`) and conversion (`fin/gsave`) loadings are POOLED** across strengths
  (one shared set), with only the *intercepts* split per strength (`mu_qual`, conversion `a`/`b`). This
  is sound because xG is **strength-aware** (its `strength_diff`/skater-count features already encode the
  man-advantage), so "danger above baseline" and "goals above xG" are strength-neutral skills — PP shots
  *add* to those samples instead of splitting them (as production pools `alpha`/`gamma`).
- **Seasons** are pooled by concatenation with **per-season indicator columns** in the rate/quality
  context and per-season intercept offsets in the conversion stage (first season = reference
  everywhere), so league-environment drift doesn't leak into player effects. On top of that, player
  effects themselves move over time — see "Player curves" (§3).

Notation below is written for a single strength; per-strength params carry the bucket (EV/PP) implicitly.

### Stage 1 — shot RATE (count), shooter-resolved
For **each attacker `j ∈ A` as the focal shooter** (teammates `T = A\{j}`), in season `s`:
```
rate_j = exp( mu_rate + shoot_j[s] + Σ_{p∈T} create_p[s] + Σ_{d∈B} def_d[s] + beta_rate·x )
N_j ~ Poisson( rate_j · o )        (or NegBin(mean = rate_j·o, size = r) with --count nb)
```
`rate_j` = j's expected Fenwick (unblocked) shots per hour in this lineup; `N_j` = his observed count.
A unit's shot generation is the **sum over its shooters**, each shooter's rate raised by the `create` of
his on-ice teammates and suppressed by the defenders. The `[s]` marks the per-season **states** of the
EV bucket (MA keeps one static level per player); the context `x` carries the **aging curves and
position offsets** for all three blocks — both defined in §3.

### Stage 2 — shot QUALITY (mark)
A shot's mean quality (xG) is a shooter/defense/context baseline plus a bump from the one player who
created the chance. On a goal the creator is observed (the primary assister); otherwise it is latent and
marginalized over the creator distribution `pi`:
```
base_i = mu_qual + qshoot_j + Σ_{d∈B} qdef_d + beta_qual·x + Σ_{t∈T} create_qual_t   # everything but the creator
qbar_i = sigmoid( base_i + qcreate_pos(c) )                 if GOAL with an on-ice creator label
       = Σ_{c∈{∅,T}} pi_c · sigmoid( base_i + qcreate_pos(c) )   otherwise — creator latent
   where  pi_c = softmax([ create_0 , create_T ])_c    and   qcreate_∅ = 0   (unassisted adds no bump)
```
The mark carries **two distinct creation-quality channels**, because a chance can be made more dangerous
in two ways the data separates cleanly:
- **`qcreate_pos(c)` — the credited-creator bump, position-level.** One scalar for F, one for D, keyed to
  the *creator* `c` via the anchor `pi`. It is *sparse*: a typical player is the observed primary creator
  on only ~20 goals even pooled, far too few for a per-player creator-danger offset (§8), so this bump is
  pooled to position.
- **`create_qual_t` — the on-ice creation quality, per-player.** Every on-ice teammate `t` lifts *every*
  shot's xG by his own `create_qual`, whether or not he is the credited creator (`Σ_{t∈T}` in `base_i`).
  This is a *dense* RAPM-on-quality effect: every teammate shot has an xG, ~16× the signal of goal-labeled
  assists, so it identifies a per-player quality skill that the sparse creator channel cannot (§7/§8). It is
  the quality analog of how volume `create` is identified from on-ice teammate shot *counts*, and carries a
  tight EB prior (`PRIOR_SD_QCREATE_PL`) since each player loads on 4× the rows.

Goals whose credited assister is *not* an on-ice teammate (data glitch, goalie assist) keep a latent
creator — they are marginalized like non-goals and excluded from the Stage-1 assist-credit anchor.

The fit maximizes a **fractional-Bernoulli quasi-likelihood** on the xG values
(`Σ xg·log(qbar) + (1−xg)·log(1−qbar)`) — consistent for the mean under a Beta(s·qbar, s·(1−qbar))
mark; the Beta concentration `s` is then estimated **post-hoc by method of moments** from the residual
MSE (it is needed only to *simulate* marks, not to rank players). `pi` reuses `create`/`create_0`,
which are **fixed values taken from Stage 1** (not re-fit here); the non-goal line is the mixture mean —
a mean-field plug-in (§5).

### Stage 3 — CONVERSION (goal | shot)
Fit **natively** inside this model (no shooting-model reuse) as a logistic recalibration of the shot's
**observed** xG, so the stage stays independent of Stages 1–2:
```
logit(p_goal_i) = a · logit(xg_i) + b + b_season(i) + fin_j + gsave_g + conv-curves(x_i)
y_i ~ Bernoulli( p_goal_i )
```
`a` (slope) recalibrates the xG→goal map (≈1 if xG is well-calibrated on 5v5); `b` (intercept) is the
reference-season baseline. Both are per-strength, **global and unpenalized**, so the MLE score equation
for `b` is `Σ(y_i − p_i) = 0`, i.e. **`Σ p = Σ goals` holds exactly**. `b_season` are unpenalized
per-season offsets (first season = reference): league finishing environments drift, and without these
that drift would leak into `fin`/`gsave` over long windows; their score equations make the
reconciliation hold **per season** too. `conv-curves` are the finishing aging curve + shooter position
offset and the goalie age curve (§3). `fin_j` (per shooter) and `gsave_g` (per goalie, `<0` = good) are
log-odds offsets with EB ridge priors, fit here alongside `a`,`b`; their prior SDs come from the
**pre-calculation stage** below (not hand-set). In simulation the *drawn* mark `q ~ Beta(s·qbar,·)`
stands in for `xg_i`; `sigmoid` is bounded, so only `logit(q)` needs an `EPS` clamp.

### Stage 0 — PRE-CALCULATION (data-calibrated conversion priors)
Before Stage 3 fits, one aggregate pass over the shots sets the ridge prior SDs for `fin` and `gsave`
from the data (empirical Bayes), so shrinkage is *measured*, not guessed — the logit-scale analogue of
`shooting_model`'s `_estimate_k`. For each shooter (and each goalie), aggregate `N` shots, expected
goals `Σxg`, actual goals, and `Σ xg(1−xg)`; the per-shot residual rate `r = (goals − Σxg)/N` has, across
high-volume entities, a volume-weighted spread of `(talent variance) + (mean sampling variance)`:
```
τ²_prob = weighted_Var(r) − mean(Σxg(1−xg) / N²)          # talent variance on the goals/shot scale
prior_sd = sqrt(τ²_prob) / v̄        where  v̄ = mean shot xg(1−xg)     # map goals/shot → logit scale
```
Estimated only from entities above a shot gate (`MIN_SHOTS_FIN_EST`, `MIN_SHOTS_GSAVE_EST`), floored at
`PRIOR_SD_FLOOR`, then applied to everyone and **held fixed** during the Stage-3 fit. It is *not* a fitted
model parameter and *not* in the Stage-3 objective — it is recomputed each fit and then frozen. When too
few entities clear the gate it falls back to the `PRIOR_SD_FIN`/`PRIOR_SD_GSAVE` constants. Because
finishing talent variance is genuinely tiny, `prior_sd_fin` comes out small ⇒ heavy shrinkage (honest);
goalies carry more signal ⇒ `prior_sd_gsave` larger ⇒ lighter shrinkage.

### Arena (venue) effects — scorer-bias nuisance states
NHL scorekeeping varies by building: some crews record more shots (count bias, hits the rate stage)
and record locations differently (shifts recorded xG — hits the quality stage). Players who play 41
home games in a biased rink would otherwise absorb that bias into `shoot`/`create`/`qshoot`. Both
stages therefore carry **per-(venue, season) offset states** `arena_{v,s}` on every row, keyed by the
actual building (so relocations and Arizona's moves split correctly):
```
prior:  arena_{v,s} ~ Normal(0, ARENA_SD²)            # nuisance ridge — biases are small; also
        arena_{v,s+g} ~ Normal(arena_{v,s}, ARENA_RW_SD²·g)   #   identifies the block (no reference venue)
```
Why venue×season with a random walk rather than one pooled offset per building: rink bias is a
*scorer-crew* phenomenon — persistent across adjacent seasons but movable when crews change — and a
per-cell free fit (~41 games) would be as noisy as the signal. Stable bias ⇒ the states glue into a
pooled effect; a crew change ⇒ the data can move that season. Venues with < `ARENA_MIN_GAMES` in the
window (outdoor/neutral sites) get no offset. These are nuisance parameters: they live outside the
SE Hessian (like `create_0`), are excluded from player values/effective params (reference
environment = no arena), and are reported per run + saved under `arena_effects` in the JSON. The
conversion stage carries none — location bias flows through the recorded xG itself.

**Indices & sets**
| symbol | meaning |
|---|---|
| `e` | attacking epoch = (stint, attacking team) |
| `j` | focal shooter (an attacker) |
| `p` | a teammate creator (index over `T`) |
| `d` | a defender (index over `B`) |
| `c` | the single creator of a shot: `∅` (self / unassisted) or a teammate in `T` |
| `g` | defending goalie |
| `i` | a shot |
| `s` | a season (2021 = 2021-22, …) |
| `A`, `T = A\{j}`, `B` | on-ice attackers (≤5), j's teammates (≤4), defenders (≤5) |

**Observed data**
| symbol | meaning |
|---|---|
| `N_j` | j's Fenwick shot count in the epoch |
| `t`, `o = t/3600` | epoch duration (s); hourly exposure offset |
| `x` | context vector (home, O/D-zone start, lead/trail, season indicators, position/age columns) |
| `xg_i`, `y_i` | shot i's upstream expected-goals value and goal indicator (0/1) |
| primary assister | the observed creator label on each goal (or `∅` = unassisted) |
| `age_{p,s}`, `pos_p` | p's age at mid-season s (Jan 1) and position (F/D), from the raw landing/roster data |

**Free parameters (estimated)** — per-skater unless noted
| symbol | stage | meaning |
|---|---|---|
| `shoot_j[s]` | rate | **own-shot rate** — how much j shoots when on the ice; per-season STATE at EV, static on MA |
| `create_p[s]` | rate **& quality** | **creation** — raises teammates' shot rate (Stage 1) *and* sets the creator distribution `pi` (used, fixed, in Stage 2); per-season STATE at EV |
| `create_0` | rate | **unassisted** creator propensity (the `∅` candidate in `pi`); a single global scalar |
| `def_d[s]` | rate | opponent shot-rate suppression (`<0` good); per-season STATE at EV |
| `qshoot_j` | quality | danger of j's own shots |
| `qcreate_{F,D}` | quality | danger a *credited creator* adds — **position-level**, sparse assist channel (2 params; per-player is unidentifiable, §8) |
| `create_qual_p` | quality | **on-ice creation quality** — danger p adds to *every* teammate shot he is on the ice for (per-player; dense teammate-xG channel, §7/§8) |
| `qdef_d` | quality | opponent danger suppression (`<0` good) |
| `fin_j` | conversion | finishing above xG on own shots (logit offset; fit natively) |
| `gsave_g` | conversion | goalie saves above expected (logit offset, `<0` good; per goalie `g`) |
| `a`, `b`, `b_season` | conversion | logit-conversion slope / intercept / per-season offsets (global, unpenalized; score eqns give Σp=Σgoals overall AND per season) |
| `q` | rate | P(a recorded secondary assist reflects creation) — the A2 anchor's mixture share (fitted; ~0 ⇒ A2s ignored as noise) |
| `arena_{v,s}` | rate & quality | per-(venue, season) scorer-bias offset (nuisance; ridge + season RW, no reference venue) |
| `mu_rate`, `mu_qual` | global | replacement-level log shot-rate / logit shot-quality |
| `beta_rate`, `beta_qual` | global | context coefficients (on `x`) — these INCLUDE the aging-curve and position-offset coefficients (§3) |
| `r` | global | NB dispersion (`--count nb`) |

**Derived quantities** (assembled from the free parameters, not fit directly)
| symbol | definition | meaning |
|---|---|---|
| `rate_j` | `exp(mu_rate + shoot_j[s] + Σ create_p[s] + Σ def_d[s] + beta_rate·x)` | j's Poisson shot rate (per hour) |
| `base_i` | `mu_qual + qshoot_j + Σ qdef_d + beta_qual·x` | creator-independent part of shot quality |
| `qbar_c` | `sigmoid(base_i + qcreate_pos(c))`; `qbar_∅ = sigmoid(base_i)` | mean xG of the shot if its creator is `c` |
| `qbar_i` | `qbar_c` (observed creator) or `Σ_c pi_c·qbar_c` (latent) | shot i's mean quality |
| `pi_c` | `softmax([create_0, create_T])_c` | **creator-identity distribution** — P(candidate `c` set it up) |
| `p_goal_i` | `sigmoid(a·logit(xg_i) + b + b_season + fin_j + gsave_g + curves)` | conversion probability |
| `s` (Beta conc.) | method-of-moments from residual MSE, post-fit | shot-to-shot xG spread around `qbar` (simulation only) |
| **effective param** | `state_last + curve(age) + pos-offset` | a player's card-ready skill read — see §3 |

**Functions & hyperparameters**
| symbol | meaning |
|---|---|
| `sigmoid(z)`, `softmax(v)_c` | logistic; `e^(v_c) / Σ_k e^(v_k)` |
| `spg` | up-weight on the assist-credit anchor — the bucket's shots-per-goal ratio computed from the data each run (≈16 at 5v5); `--spg-scale` multiplies it for sensitivity checks |
| `B(a) = [z, z²]`, `z=(a−27)/10` | the age basis; `B(27) = 0` so curves are deviations from peak age |
| `RW_SD_SHOOT/CREATE/DEF` | per-season random-walk SDs of the EV states (the drift "flexibility dial") |
| `PRIOR_SD_SHOOT/CREATE/QSHOOT/QCREATE` | hand-set ridge prior SDs per rate/quality block (level priors) |
| `prior_sd_fin`, `prior_sd_gsave` | conversion prior SDs — data-estimated each fit (Stage 0); `PRIOR_SD_FIN`/`PRIOR_SD_GSAVE` are only the fallbacks |
| `MIN_SHOTS_FIN_EST=200`, `MIN_SHOTS_GSAVE_EST=1000`, `PRIOR_SD_FLOOR` | shot gates + floor for the Stage-0 prior-SD estimate |
| `N_TM = 4`, `N_DEF = 5` | teammate / defender counts used in deployment-free attribution |
| `DENSE_H_MAX` | parameter-count cutoff between the dense Hessian inverse and sparse column solves |
| `ARENA_SD`, `ARENA_RW_SD`, `ARENA_MIN_GAMES` | arena-state nuisance prior (ridge + season RW) and the rare-venue gate |

**Code-symbol map** (this doc → code): `create_0`→`psi0`; `mu_rate/qual`→ each fit's
`intercept`; `a`/`b`→`conv["a"]`/`conv["b"]`; `b_season`+curves→`conv["beta"]` (named by
`conv["ctx_names"]`); `fin_j`→`conv["fin"]`; `gsave_g`→`conv["gsave"]`; `beta_rate/qual`→`beta` (named
by each row-builder's `ctx_names`); `qbar`→`sig5`; `pi_c`→`pi`; `rate_j`→`exp(eta)` in the Poisson NLL;
states→`shoot/create/def` per-unit arrays with `unit_player`/`unit_season`; last states→`*_last`;
effective params→`effective_params()`. The stage objectives, the parameter-vector layouts
(`RateLayout`/`QualLayout`/`ConvLayout` — one source of truth for how each fit's flat θ slices into
these blocks) and the ridge/random-walk precisions live in `generative_likelihood`; the fits and
`effective_params` in `generative_model`; the design-row builders in `generative_data`.

### Player-value attribution (deployment-free per-60, from EFFECTIVE params + intercepts)
All values are computed from each player's **effective parameters** — his last state (or static level)
+ his position offset + the aging curve at his last-season age (`effective_params()`), in the
reference-season environment:
```
q_own_j(c)    = sigmoid(mu_qual + qshoot_j + qcreate_pos(c))                    # mean xG per own shot, by CREATOR CLASS c
ḡ_j           = Σ_c π_c · E[ sigmoid(a·logit(x)+b+fin_j) ],  x ~ Beta(s·q_own_j(c), s·(1−q_own_j(c)))
scoring(j)    = exp(mu_rate+shoot_j) · ḡ_j                                                          # own shots, CONVERTED
playmaking(p) = N_TM · exp(mu_rate) · [ (exp(create_p) − 1) · sigmoid(mu_qual + qcreate_pos(p))          # teammate xG added:
              + exp(create_p) · (sigmoid(mu_qual+qcreate_pos(p)+create_qual_p) − sigmoid(mu_qual+qcreate_pos(p))) ]  #   VOLUME + QUALITY
defense(d)    = N_DEF · [ exp(mu_rate)·sigmoid(mu_qual) − exp(mu_rate+def_d)·sigmoid(mu_qual+qdef_d) ]  # opp xG suppressed
creator_share(p) = exp(create_p) / ( exp(create_0) + exp(create_p) + (N_TM−1) )                    # per-teammate-shot
```
**ḡ is the model's own goals-per-shot, marginalized twice** (`marginal_goal_prob` +
`creator_mix`): over the fitted Beta shot-quality distribution (concentration `beta_s` — the same
distribution the PPC draws from; quadrature in CDF space) and over the creator classes
c ∈ {unassisted, F-created, D-created} with weights π from `create_0` and a reference teammate
mix (the WAR engine instead uses each stint's actual teammates). Both marginalizations matter:
`qcreate` is parameterized with UNASSISTED as the reference and created shots carry the negative
position bumps, so evaluating the conversion at `sigmoid(mu_qual+qshoot)` alone prices every shot
as unassisted — the +13% league-wide goals overshoot the 2026-07 WAR audit traced. With both, the
engine's ΣE[GF] reconciles to actual goals within ~1% with NO correction factors (the audit
asserts this; a residual is a modeling finding, not a knob).
Computed **per strength** with that bucket's rate loadings + intercepts and the pooled
quality/finishing loadings: EV → `ev_scoring/playmaking/defense` (`N_DEF=5`); MA → `pp_scoring/
pp_playmaking` and `pk_defense` (`N_DEF=4`; the MA `def` loadings are the penalty-killers).
**Playmaking is the sum of two halves.** The VOLUME half prices the *extra* shots p causes his teammates
to take (`exp(create_p)−1`) at their danger; the QUALITY half prices p's on-ice `create_qual` lift applied
to *all* the teammate shots he is on the ice for (`exp(create_p)`). Two equal-volume creators now separate
by whether one sets up tap-ins and the other point shots. `player_values` also exposes the quality half
alone as `playmaking_quality` (0 when `create_qual` is off).
**Units note:** `scoring` is conversion-adjusted (goals/60) while `playmaking` and `defense` are xG/60 —
the same convention as the production cards; keep the units straight when comparing across cards.
Defense `>0` = suppresses (good).

---

## 3. Player curves — aging + drift + projection

The question this section answers: **how does a player's skill move over time, and what will it be
next season?** Two mechanisms, deliberately layered so each is identified by the data that can
actually support it:

### 3a. Shared aging curves (all stages, cheap, always on)
Every block that plausibly ages gets a **shared F/D-split quadratic in age** entering the design as
plain context columns with unpenalized global coefficients:

| block | curve | drift states |
|---|---|---|
| EV `shoot`, `create`, `def` | yes (F/D) | **yes** |
| MA `shoot`, `create`, `def` | yes (F/D, its own fit) | no — static level |
| `fin` (finishing) | yes (F/D) | no |
| `gsave` (goalies) | yes (no position split) | no |
| `qshoot`, `qdef`, `qcreate` | no — static | no |

The basis is `B(a) = [z, z²]` with `z = (age − 27)/10`, age measured at Jan 1 of the season
(mid-season), from the raw player landing JSONs; a missing birthdate pins `z = 0` — the player sits at
the curve's reference point and contributes no age signal instead of biasing the curve. In the rate
stage each row carries three curve column-groups: the focal shooter's basis (shoot curve), the *sum*
of his 4 teammates' bases (create curve — `create` is a per-teammate effect, so its curve accumulates
the same way), and the masked sum of the defenders' bases (def curve). Quality blocks get **no**
curves: the per-shot signal is weak and danger-above-baseline is the most age-stable skill — this is
exactly where added flexibility would buy noise, not insight. (Adding one later is two lines: the
columns already have a naming convention.)

**Position intercepts.** Alongside each curve group sits a **D-offset column** (`shoot_D`,
`create_D`, `def_D`, quality `shooter_D`/`def_D`, conversion `shooter_D`): ridge shrinkage pulls
player effects toward zero = the league mean, but defensemen shoot far less than forwards, so without
these offsets every D carries a systematically negative `shoot` fighting his prior. With them, player
effects mean "vs. positional baseline expressed at his age" and the shrinkage is unbiased by position.

### 3b. Per-player drift — random-walk states (EV rate blocks only)
The dense EV blocks (`shoot`, `create`, `def`) replace the single static coefficient with **one state
per (player, active season)**, tied together by a random walk:
```
θ_{p, first(p)} ~ Normal(0, PRIOR_SD_block²)                      # level prior, first state only
θ_{p, s+gap}    ~ Normal(θ_{p,s}, RW_SD_block² · gap)             # drift prior, consecutive active seasons
```
This is partial pooling **across time**, the same logic as the ridge across players: a season with
thin data collapses to its neighbors' value; a season with dense evidence of change moves. The
`RW_SD_*` constants are the flexibility dial — at 0 the states glue into the old static model, at ∞
each season fits independently. Defaults (`0.10` shoot/def, `0.05` create, per season on the
log/logit scale) are hand-set and deliberately conservative: a real breakout is smoothed by roughly a
season. An empirical-Bayes estimate of the within-player drift variance (the Stage-0 trick applied
across seasons) is the natural follow-up. Season gaps scale the variance (`gap` years between active
seasons); MA blocks, quality, and finishing stay static levels — their per-player-season data is too
thin for drift to be anything but noise.

**Why both layers don't fight:** the RW increments are penalized and the curve coefficients are not,
so league-typical aging loads on the shared curve and the states carry only *idiosyncratic* deviation
("aging worse than a normal 33-year-old"). The player's card-ready skill at any season is the
**effective parameter**: `state(s) + curve(age_{p,s}) + position offset`.

### 3c. Identifiability — the age–period–cohort caveat
`age = season − birth year`, so a linear-in-age trend is exactly collinear with (season fixed-effects
+ a cohort shift of the player levels). The resolution here: season FEs stay unpenalized (they absorb
genuine league drift), and the collinear direction is **softly identified through the player-level
ridge** — reallocating an age trend into the player levels costs cohort-constant penalty mass across
~2000 players, so the fit prefers loading the systematic cross-sectional age pattern onto the
unpenalized curve. Two caveats, stated honestly:
- **Survivor bias:** old players still in the league are disproportionately good, which flattens the
  cross-sectional old-age tail. The curve is a *league-composition* aging curve, not a within-player
  causal one.
- **Soft identification wobble:** with few seasons the age-linear vs season-linear split can move
  between fits. The run prints each fitted curve (values at 22/32 + implied peak age) as a
  diagnostic; if it wobbles, the documented fallback is season contrasts orthogonal to the linear
  trend (not currently implemented).

### 3d. Projection
With ≥2 fitted seasons the run always emits a projection for `last season + 1`:
- **states** hold at their last value (the RW mean — the walk's best forecast), with uncertainty
  widening by `RW_SD · gap` per block (reported qualitatively, not per player);
- **ages advance** to the target season, so each player slides along the fitted curve;
- **environment** stays the reference season's (no season FE / `b_season` applied) — projections and
  current values are therefore directly comparable skill reads, not league-environment forecasts.
Values are re-attributed from the projected effective params by the same `player_values` formulas, and
land in the output JSON under `projection` (per player: `ev_scoring/ev_playmaking/ev_defense`,
`pp_*`, `pk_defense`, `fin`). A player who missed the last season projects across the gap (his age
moves, his state doesn't — honest, with wider implied uncertainty).

---

## 4. The generative direction (simulate a stint)
Given fitted params and a stint `(A, B, g, t, x)` in season `s`: for each shooter `j`, draw
`N_j ~ Poisson(rate_j·o)` with `rate_j` built from that season's states and curve columns; for each
shot draw a creator `c ~ pi = softmax([create_0, create_T[s]])`, then
`xg ~ Beta(s_conc·qbar_c, s_conc·(1−qbar_c))`, then
`y ~ Bernoulli(sigmoid(a·logit(xg) + b + b_season + fin_j + gsave_g))`. Aggregating gives simulated
shots/xG/goals (used for the posterior-predictive check and counterfactual lineup swaps).

---

## 5. The inference direction (fit params from data)
The stages are conditionally independent given the predictors, so they are fit **in sequence** — each by
penalized MLE / empirical-Bayes (ridge priors) with **JAX autodiff gradients + scipy L-BFGS-B**; SEs via
the Hessian / Gauss-Newton (see below). Across strengths (§2): the **rate stage is fit once per
bucket** (EV with states, MA static), while **quality and conversion are single pooled fits** (shared
loadings, per-strength intercepts). The quantity shared across stages is `create`/`create_0`: **fit in
Stage 1, passed into Stage 2 as a fixed constant** (per-row by the shot's strength *and season* — the
EV `pi` uses that season's create states). Stage 3 uses neither, so it is independent of 1–2.

- **Stage 1 — rate (`fit_rate_create`, run per strength bucket)** — fits `mu_rate, shoot, create, def,
  beta_rate, create_0 (, r)` on the Poisson/NB count NLL (one row per (epoch, shooter); EV ~4.4M
  rows/season, MA ~0.3M; defenders masked to the bucket's count). At EV the player blocks are
  per-(player, season) states with the RW penalty (§3b); statically (MA, single season) the penalty
  reduces exactly to the old ridge. Added to the NLL is the **assist-credit anchor**: `spg ×` a
  conditional logit scoring each goal's observed primary assister under
  `pi = softmax([create_0, create_T])`. We only observe a shot's creator on the ~1-in-16 Fenwick shots
  that are goals, so weighting each observed goal-creator by `spg` (≈ shots per goal ≈ 1/goal-rate,
  computed per bucket from the data) makes it stand in for the shots whose creator we never see — an
  **inverse-probability weight** (`--spg-scale` exists to *check* IPW sensitivity, A3). On top of the
  IPW, a held-out-selected **anchor weight** (EV and PP both ×0.25, §7) discounts the assist evidence
  because a credited assist reflects puck-role as much as creation — the A1 analogue of the A2 mixture
  share `q`. Without any anchor the dense counts identify `create` only as a possession/volume effect;
  with it `create` is pulled toward the players actually credited with setups, but only as far as
  held-out prediction supports (over-anchoring overfits role, §7). Goals whose assister is not an on-ice
  teammate are excluded from the anchor. **Secondary assists** join as an exploded-logit second
  ranking stage: with A1's column masked, the recorded A2 is scored as
  `P(A2=c₂) = q·softmax(create_{T\c₁})[c₂] + (1−q)/3` — a MIXTURE whose share
  `q = P(A2 reflects creation)` is a fitted parameter (a bare weight on the A2 term would be a
  pseudo-likelihood temperature, not MLE-fittable; the mixture is). A2s add ~1.8× labeled events at
  a fit-determined discount; measured priors: A1/60 repeats year-over-year at 0.73 vs A2/60 at 0.55.
  A2 deliberately does NOT enter Stage 2 (a shot's xG is set by the LAST pass — adding `qcreate` for
  the A2 would double-count the causal path). The count NLL and both anchor stages share
  `create`/`create_0` and are optimized **together** in this one stage.
- **Stage 2 — quality (`fit_quality_creator`)** — a single POOLED fit (EV+MA) of `qshoot,
  qcreate_{F,D}, qdef, beta_qual` plus a per-strength intercept (`mu_qual` via a `pp` context column),
  with `create`/`create_0` **taken as fixed values from the per-strength Stage-1 fits** (per row by
  that shot's strength and season): creator observed on eligible goals, marginalized over `pi`
  otherwise. Defenders are masked (≤5). The Beta concentration `s` is method-of-moments post-fit.
  Nothing feeds back to Stage 1.
- **Stage 0 — conversion pre-calc (`estimate_conversion_prior_sds`)** — one aggregate pass sets the
  `fin`/`gsave` ridge prior SDs from the data (empirical Bayes, §2 Stage 0), recomputed each fit and
  then frozen. Not in any objective; the logit-scale analogue of `shooting_model._estimate_k`.
- **Stage 3 — conversion (`fit_conversion`)** — fits `a, b, b_season, curves, fin, gsave` by
  penalized-MLE on the Bernoulli goal likelihood, natively. `a`, `b`, `b_season`, and the curve
  coefficients are unpenalized (so Σp=Σgoals holds exactly, overall and per season); `fin`/`gsave`
  carry the Stage-0 EB ridge priors. Independent of Stages 1–2: it keys off the upstream **observed**
  xG, never `create`, `qcreate`, or `qbar`.

**Approximations (where inference departs from one exact joint marginalization).**
1. **Volume uses no latent creator.** In Stage 1 `create` enters as a deterministic sum over *all*
   teammates, so a non-goal shot's creation credit is spread across the lineup; only Stage 2 has a
   single creator to marginalize.
2. **Two-stage plug-in, not joint.** `create`/`pi` are pinned first (dense Poisson + goals-only credit
   logit) and then **frozen** for the quality fit, rather than estimating `create` and `qcreate` jointly
   by marginalizing the creator once over both factors.
3. **Mean-field marginal.** On latent-creator shots the creator sum sits *inside* the mark mean
   (`E[xg] = Σ pi_c·qbar_c`) rather than wrapping the likelihood (`Σ pi_c·Beta(xg|qbar_c)`). Equal to
   first order and exact as `qcreate→0`; the plug-in slightly overstates the log-likelihood (Jensen).
4. **Goal-selection in Stage 2 (open).** Creator labels exist only on goals — a high-xG-biased
   subsample (selection on `y` depends on `xg`) — and Stage 2 mixes `P(xg | goal, creator)` and
   `P(xg | non-goal)` rows without conditioning on that selection. Position-pooling `qcreate` (A1)
   shrinks the damage surface, but a principled fix would model the joint `(xg, y)` or reweight goal
   rows by `1/p(goal|xg)`.

**Why `create` is identified.** The rate response is *shooter-specific* (`N_j`), so `shoot_j` loads on
rows where j shoots and `create_p` loads on rows where p is a *teammate* of the shooter — estimated
from how much the players around p out-shoot when p is on the ice (lineup variation, **no pass data
needed**), then *lightly anchored* by the assist-credit so it leans toward creation rather than mere
possession. The anchor weight is calibrated (EV ×0.25, §7): the dense lineup channel is the primary
identifier and over-anchoring overfits assist-role, but the anchor is not zero — removing it entirely
un-identifies `create` (its held-out block flips sign).

**Standard errors.** The rate stage assembles the exact curvature `H = XᵀWX + K` (K = level ridge + RW
tridiagonal precision) sparsely, plus the credit Fisher on the create diagonal. When the parameter
count fits (`≤ DENSE_H_MAX`) it is inverted densely; beyond that (multi-season state expansions) a
sparse `splu` factorization solves only the **reported** columns — each player's *last* state per
block. `se_create` is **sandwich-corrected**: the anchor enters the objective at weight `spg`, but `spg`
pseudo-replications of one observed goal are not `spg` independent observations — the sandwich
`Var = H⁻¹ M H⁻¹` puts the credit in the meat `M` at `spg²` over the *actual* goals, so the anchor adds
curvature without manufacturing information. Quality `qcreate_{F,D}` and conversion get diagonal
Gauss-Newton SEs. `z = estimate/se`; low-confidence players are greyed out (`⚠`) in the leaderboards.

**Performance note.** Data arrays are passed as **arguments** to the jitted `value_and_grad`, not
closed over — otherwise `jax.jit` bakes the ~3 GB of index arrays into the compiled program as
captured constants (slow compile, double memory). Unit gathers are int32. See `_optimize(nll, x0, *data)`.

---

## 6. Data
- **Shots:** `processed/shots_onice/<season>.parquet` — per Fenwick shot: `shooter_id`, `xg`, `goal`,
  `strength`, on-ice `home_skaters`/`away_skaters`, `home_goalie`/`away_goalie`, `event_idx`. Used for
  `strength ∈ {5v5, 5v4, 4v5}` here; `xg` is present for all of these (only empty-net shots lack it, and
  those are excluded). MA keeps only PP-side shots (shooter on the 5-skater side).
- **Stints:** `processed/stints/<season>.parquet` — on-ice skaters, `strength` (=`"{home_n}v{away_n}"`),
  duration, Fenwick counts, context, goalies.
- **Goal assists:** raw play-by-play (`raw/nhl/pbp/<game>.json`) — `assist1PlayerId` per goal, joined to
  shots via `shots_onice.event_idx == pbp play sortOrder`. Creator labels come only from these recorded
  assists (passing is otherwise unrecorded); the *volume* half of `create` is what recovers the
  unobserved passing creation on non-goal shots.
- **Ages & positions:** birthdates from the raw player landing JSONs (`raw/nhl/players/<id>.json`,
  `birthDate` — goalies too); positions from `interim/roster/<season>.parquet` (F/D; unknowns count
  as F). Missing birthdates are reported per run and pinned to the curve reference (z = 0).
- **Venues (arena effects):** each game's building from raw pbp `venue.default`, lazily cached to
  `interim/game_venue/<season>.parquet`. Sponsor renames are aliased to the current name
  (`_VENUE_ALIAS` — one RW chain per physical building); real building moves stay split.

---

## 7. Results

Pooled 2021–2025 fit (NB counts, 1,481 players, 4,739 EV player-season states, 21.4M EV rate rows):

- **`create` is well-identified and forward-weighted** — even with the sandwich-corrected SEs (F1),
  the last-state leaderboard runs z up to ~11 (Kucherov +0.48): Crosby, Kucherov, Hagel, Scheifele,
  MacKinnon, McCann — recognized setup men, no defenseman over-representation.
  The dense volume signal (how much a player's linemates shoot with him on the ice) is the primary
  identifier; the assist anchor at its calibrated EV weight (EV sweep above) sharpens the within-unit
  ordering without overfitting who-gets-credited. Pure distributors top the board and shooter-
  playmakers like Pastrnak settle just below them — an ordering that predicts held-out teammate shots
  better than the old assist-heavy one.
- **Finishing identifies under pooling + position/age baselines.** 39/1040 eligible clear `|z|>2`
  (single-season: ~2/633), and the leaderboard is the canonical elite-finisher list — Tage Thompson,
  Draisaitl, Panarin, Marner, Forsberg, Ovechkin (z 2.5–5.1). The A2 position offsets matter here:
  without them, the D calibration offset (compensating the a>1 slope on low-xG point shots) was
  absorbed into individual defensemen's `fin`.
- **`qcreate` (position-level, A1) is significant and interpretable:** F −0.097 ±0.015, D −0.401
  ±0.041 — assisted chances are slightly less dangerous than unassisted ones when set up by a forward,
  and much less dangerous when set up by a defenseman (point-shot/perimeter feeds). This is priced
  into every player's playmaking value by position.
- **`create_qual` (per-player on-ice creation quality) identifies from the dense channel and predicts
  out of sample.** Spread F std 0.020 / D std 0.013 (logit-xG). The leaderboard is the recognized
  chance-creation elite — Kucherov, MacKinnon, McDavid, Pastrnak (F); Lane Hutson, Karlsson, Bouchard
  (D) — distinct from volume `create` (corr ≈ +0.31, so it is a *separate* skill, not a relabel). The
  held-out teammate-xG gate (`generative_holdout.quality_gate`) is the go/no-go: the per-player lift
  beats the position pair at predicting the next season's teammate xG-per-shot on **two** independent
  target years — corr 0.692→0.760 (2025) and 0.728→0.793 (2024), both bootstrap P(win)=1.000 — with an
  own-cq calibration corr of **+0.40 in both years** and a centered slope ≈ 1 (the spread predicts at
  face value, and beats naive last-season teammate-xG). It flows into playmaking value (the QUALITY
  half, §2) and into WAR (each shooter's goals-per-shot absorbs his teammates' Σ`create_qual`): the WAR
  gain tracks `cq` (corr +0.83), lifting the elite creators (MacKinnon +1.35, Kucherov +1.06, McDavid
  +1.05 WAR — the only three regulars moving >1.0) while league WAR rises modestly (1080→1119) as a
  previously-unattributed value channel; κ, value scale, calibration, and the reconciliation synergy
  are all undisturbed.
- **Trajectories read like careers, not noise.** Per-season effective states move 0.02–0.2/season
  under the RW prior: Pastrnak's `create` rises across 2021–25 to +0.38 (his shooter→dual-threat
  evolution), Bedard jumps in year 3, Suzuki rises steadily, Crosby holds near +0.44 into his late
  30s, Ovechkin's `shoot` is flat at 40.
- **Aging curves are gentle once position and level are controlled** — a few % per decade-z on the
  log scale. The finishing curve peaks young (~23 F) and declines, consistent with the shooting-talent
  literature; the EV shoot curve peaks later (~28) than raw-data curves suggest — the survivor-bias
  flattening documented in §3c. Read the curves as league-composition aging, not within-player decline.
- **Goals and shots reconcile.** `Σp=Σgoals` exact overall and per season (26,809 EV; per-season
  6,599–6,994 all exact); PPC: EV shots within 0.08%, EV goals within 0.5%, PP goals +1.4% (sim high —
  watch item). The NB count model matches the per-row overdispersion (r≈1.4 EV, 20 MA).
- **Projections are face-valid:** the 2026 board is the current elite with small age adjustments —
  young players tick up (Guenther, Gauthier), older players tick down, no wild extrapolations (the RW
  mean holds states; only ages move).
- **Watch items:** the PK-defense *leaderboard* gate (`SNIFF_MIN_TOI_MA` = 40 min, matching
  production) lets tiny-TOI players top the board — consumers should gate harder; PP playmaking has a
  large top-2-vs-field scale cliff (McDavid/Kucherov ~5.3 xG/60 vs ~2.1 next) via the `exp(create)−1`
  nonlinearity at the high PP shot baseline; PP-goal PPC runs ~2.5% hot.

### Held-out validation (the predictive harness)

`generative_holdout.py` fits the model on seasons ≤ T and scores the T+1 projection on that
season's REAL stint rows against simpler reads of the same fit (league-avg floor, the static
pooled-mean read, the last drift state without aging) and the naive last-season-raw-rates bar —
row-level Poisson deviance (deployment identical across candidates) + TOI-weighted player
own-shots/60 correlation/MAE. First run, 2021–24 → 2025 (677 eligible players ≥200 EV min):

Own shots (rate corr/MAE) and teammate shots while on ice (tm — the create-side observable);
current model (A2 mixture + arena states):

| candidate | row-dev/1k | Σμ/ΣN | rate corr | MAE/60 | tm corr | tm MAE |
|---|---|---|---|---|---|---|
| league-avg (floor) | 128.45 | 1.116 | 0.639 | 2.749 | 0.440 | 4.801 |
| pooled-mean (static read) | **126.75** | 0.992 | 0.837 | 1.182 | **0.657** | 3.408 |
| last-state (drift, no aging) | 126.93 | 0.997 | 0.838 | **1.173** | 0.630 | 3.565 |
| projection (drift + aging) | 126.94 | 1.006 | 0.838 | 1.174 | 0.629 | 3.600 |
| naive last-season raw rate | — | — | **0.864** | 1.087 | 0.612 | 2.803 |

The A2 mixture's fitted share saturated at **q̂ = 1.00** — recorded secondary assists are fully
concordant with the create ordering (more creation signal than the 0.75 repeatability prior
suggested; repeatability includes deployment noise, concordance doesn't). A2 vs the no-A2 arm:
tm-corr +0.008 (pooled read) / +0.003–0.004 (last/proj), own-shot metrics unchanged — modest,
consistent, free. Note the **model beats the naive bar on teammate rates** (0.657 vs 0.612): the
first metric where lineup-aware decomposition wins outright — naive raw rates can't know the
target season's linemates. (naive tm MAE is smaller because raw on-ice rates carry persistent
team effects the deployment-free read deliberately strips; corr is the like-for-like column.)

Honest reading:
- **The model's skill content is large and well-calibrated**: floor → skill reads moves corr
  0.64→0.84, halves MAE, and Σμ/ΣN sits at 1.00 (the projection runs ~1% hot — the aging deltas
  slightly overshoot in TOI-weighted aggregate).
- **Drift and aging are a wash for one-season-ahead own-shot rates**: the three model reads sit
  within ~0.2% of each other. Their value is in unbiased current-skill reads, trajectories, and
  long windows — not in beating a static read at this one-step horizon, on this one metric.
- **The naive bar stands**: last season's raw rate is still the best single predictor of next
  season's raw rate for high-TOI regulars (unshrunk, and it bakes in residual role effects that
  the decomposition spreads across linemates). The model's purpose is decomposition, attribution,
  and simulation — not raw-rate nowcasting — but this row is the bar improvements (secondary
  assists, arena effects, RW tuning) should now be measured against.
Follow-ups: RW_SD grid (one training fit per value; the training side caches to
`holdout_fit_<T>.npz` and `--rescore` re-scores in minutes) and extending scoring beyond the rate
stage (on-ice xGF, goals) where the joint structure should differentiate.

### Calibration slopes (γ) + the MA identification sweep (2026-07)

The harness also estimates, per parameter block, an out-of-sample **calibration slope** γ: the
1-D Poisson MLE of μ = exp(base + γ·term) on the held-out rows, where `term` is that block's
fitted contribution and everything else is held at training values. γ = 1 ⇔ the block's fitted
spread predicts next season at face value. γ < 1 has two causes and only one is a defect:
(a) genuine year-over-year skill drift attenuates any one-step forecast (the RW_SD values imply a
ceiling ≈ 0.9 for EV shoot/create, lower for def), and (b) within-season **misallocation** —
spread the data never identified. The EV blocks sit near their drift ceilings (shoot/create/def
0.95 / 0.94 / 0.81 at the selected anchor + position-mean prior);
PP create at the default settings sat at **0.31** — the fixed-unit residual-sink problem (§8):
on five-man units that never change, shot counts pin only the unit's creation SUM, the assist
anchor splits it, and PP assists are ROLE (who touches the puck last), not creation. Assist-light
net-front players absorb the negative residual of their anchored teammates.

The fix is in the fit: the MA bucket's anchor weight and create/def priors became hyperparameters
selected BY this harness (γ toward the EV ceiling, subject to held-out MA deviance not degrading).
The EV anchor is selected the same way (next subsection). The sweep (train ≤2024 → score 2025;
warm-started via the holdout θ̂ chain, ~15–45 min per candidate):

| MA anchor | create prior | def prior | γ shoot | γ create | γ def | MA dev/1k | Σμ/ΣN |
|---|---|---|---|---|---|---|---|
| ×1.0 (old default) | 0.12 | 0.30 | 0.77 | 0.31 | 0.64 | 233.8 | 1.011 |
| ×0.5 | 0.12 | 0.30 | 0.80 | 0.40 | 0.58 | 232.0 | 1.000 |
| ×0.25 | 0.12 | 0.30 | 0.82 | 0.51 | 0.53 | 231.1 | 0.993 |
| ×0.1 | 0.12 | 0.30 | 0.83 | 0.60 | 0.50 | 230.8 | 0.987 |
| ×0.25 | 0.06 | 0.30 | 0.82 | 0.63 | 0.49 | 230.7 | 0.990 |
| ×0.25 | 0.04 | 0.30 | 0.82 | 0.75 | 0.48 | 230.6 | 0.989 |
| ×0.1 | 0.06 | 0.30 | 0.83 | 0.74 | 0.47 | 230.6 | 0.987 |
| ×0.25 | 0.04 | 0.15 | 0.81 | 0.77 | 0.64 | 230.2 | 0.993 |
| **×0.25 (selected)** | **0.04** | **0.10** | 0.81 | **0.78** | **0.74** | **230.0** | 0.994 |

Three lessons the table encodes: (i) down-weighting the anchor monotonically improves held-out PP
prediction — the model predicts power plays better when it stops over-trusting PP assists;
(ii) misallocation migrates to the loosest block (γ_def slid 0.64 → 0.48 as create tightened,
recovering only when def got the same prior treatment); (iii) the selected setting is best on
every axis simultaneously — honesty and prediction were not in tension. Escalation if a future
sweep can't reach the ceiling: a separate per-player assist-style offset in the MA anchor
(decoupling "last passer" from "chance creator"). Slopes land in
`data/models/holdout_calibration*.json`; the WAR audit reports them alongside the card numbers.

### EV anchor sweep — the same lever, EV bucket (2026-07)

γ alone hid an EV problem the ranking metric exposes. At full anchor weight EV `create` sat right
on its calibration ceiling (γ_create ≈ 0.90), which reads as healthy — but the held-out **teammate-
shots correlation** (does a player's `create` predict how much his linemates actually shoot next
season) was only 0.657, barely above the naive last-season-rates bar (0.612), and on the 2024 target
the full-weight model (0.620) *underperformed* naive counting (0.639). γ measures whether the fitted
scale is calibrated; it does not measure whether the ranking is right. The assist anchor at full
weight was overfitting assist-ROLE (who gets credited) into `create`, inflating its spread with
information that does not generalize — the same defect as the PP bucket, milder because EV linemate
mixing gives the volume channel real independent signal. The EV anchor is selected by the harness
just like MA (train ≤2024 → score 2025; the create-side teammate-shots track is the criterion):

| EV anchor | tm-corr | tm-MAE | own corr | dev/1k | γ create |
|---|---|---|---|---|---|
| ×1.0 (old default) | 0.657 | 3.41 | 0.837 | 126.75 | 0.90 |
| ×0.5 | 0.679 | 2.89 | 0.852 | 126.43 | 0.89 |
| **×0.25 (selected)** | **0.690** | 2.67 | 0.859 | 126.27 | 0.83 |
| ×0.1 | 0.687 | 2.57 | 0.860 | 126.18 | 0.58 |
| ×0.0 (anchor off) | 0.646 | 2.74 | 0.852 | 126.24 | 0.98 |

tm-corr peaks at ×0.25 and the improvement holds on the independent 2024 target (×1.0 → ×0.25 lifts
tm-corr 0.620 → 0.687). Turning the anchor fully off is worse: below ~0.1 `create` loses its grip and
its held-out block contribution flips sign — the anchor carries real signal, it was just massively
over-weighted. ×0.25 keeps near-peak prediction with a healthy γ (0.83, above the accepted MA 0.78)
and is the shipped EV default. The population effect: forward `create` spread tightens (std 0.13 →
0.08) and negative-`create` forwards drop from 2.6% to 0.8% — the compression is shed overfitting,
not lost signal, since it *raises* held-out prediction. It also relaxes the Kapanen-class drag
(§5e / roadmap): a finisher glued to an elite distributor moves toward replacement rather than being
dragged below it.

### Position-mean `create` prior (2026-07)

The level ridge on the first `create` state shrinks toward the F/D **position mean** (forwards +0.174,
defensemen −0.308), not toward 0 — so a weakly-identified forward defaults to the forward baseline
rather than to "suppresses offense". Set by `create_prior_center=position-mean` (the default), computed
two-pass (fit at 0, measure the raw-state means, refit centered). The center is a fixed constant, so the
level stays pinned and the SE curvature is unchanged. Because the F/D means differ, the shift is
non-uniform and data-identified (a uniform re-center would be a gauge no-op). Held-out it improves
every axis on two target years — tm-corr 0.690 → 0.719 (2025) / 0.687 → 0.705 (2024) and **ev:create γ
0.83 → 0.94** — and in-sample it is a clean prior change: well-identified forwards are invariant
(corr 0.9989), spread preserved, negative-`create` forwards drop to 0.2%. It closes the residual
Kapanen-class drag (Kapanen −0.059 → −0.007, playmaking ≈ 0). The correct validation here is
well-identified invariance (a prior change should leave likelihood-pinned players put), the *opposite*
of the anchor-weight check above (a likelihood change) — a distinction worth keeping straight.

---

## 8. Per-player creation quality — the sparse channel vs the dense channel

Per-player creation quality lives on two different data channels, and only one of them identifies it.

**The sparse channel (credited creator) does NOT identify per-player.** Earlier versions carried a
per-player `qcreate_p` (danger a player adds *when he is the credited creator*). It **did not identify**:
essentially no player cleared `|z|>2`, SEs sat at the prior SD, and the estimate shrank to the prior for
everyone. The cause is structural — a typical player is the observed primary setup man on only
**~17–26 goals even across several pooled seasons**, far too few events for a per-player offset; non-goal
shots contribute little because the creator is latent there. So the credited-creator bump `qcreate` is a
**position-level pair** (A1), the danger an F/D creator adds.

**The dense channel (on-ice teammate xG) DOES identify per-player — `create_qual`.** Every on-ice
teammate shot has an xG, ~16× the signal of goal-labeled assists. Adding a per-player on-ice xG lift
`create_qual_p` to the mark's `base` (§2) — the quality analog of how volume `create` is identified from
on-ice shot *counts* — recovers a per-player creation-quality skill that the sparse channel cannot. It is
held-out validated: on two independent target years, the per-player lift beats the position pair at
predicting held-out teammate xG-per-shot (corr 0.692→0.760 and 0.728→0.793, both bootstrap-certain), with
an own-cq calibration corr of +0.40 in *both* years (§7). So per-player creation quality is now a real,
shipped skill — sourced from the dense channel, not the creator-latent mixture the sparse route offered.

**Open threads.** (a) The Stage-2 goal-selection bias (approximation 4, §5). (b) Whether the non-goal
marginal can be made more informative (weighting it when `pi` concentrates). (c) Whether the sparse
credited-creator danger is separable from shooter + location at all on this data. (d) EB estimation of the
RW drift SDs. (e) Card-level CIs by parametric bootstrap through `player_values` (sample from the Laplace
posteriors — the natural payoff of a generative model).

**The full ranked improvement backlog** — including unused data sources (secondary assists,
blocked-shot credits, penalties), the held-out-season predictive harness, arena recording-bias
intercepts, and the 2016–2020 window extension — lives in
[`generative_model_roadmap.md`](generative_model_roadmap.md).

---

## 9. Implementation map & how to run
- **Modules** (`pipeline/src/yhattrick/models/`):
  - `generative_features` — `load_stints(seasons, strengths)` and the attacking-team orientation
    (`base_ctx`, `side_rows`) shared by the fit and the WAR consumers.
  - `generative_likelihood` — the three stage objectives (`make_rate_nll`/`make_quality_nll`/
    `make_conversion_nll`), the parameter layouts (`RateLayout`/`QualLayout`/`ConvLayout`), the sparse
    design + prior-precision matrices (`_build_X`, `_rate_penalty`, …), and the math primitives
    (`marginal_goal_prob`, `creator_mix`, `_sigmoid`) with the prior constants.
  - `generative_data` — the fact/dimension tables → design rows: `player_index`, `_age_position`/
    `_birthdates` (ages + F/D), `_arena_index` (the (venue, season) arena-state machinery),
    `rate_rows` (+ `_unit_machinery` for the per-season states), `quality_creator_rows`,
    `conversion_rows`, `estimate_conversion_prior_sds`.
  - `generative_model` — the fits `fit_rate_create` (per-strength rate + credit; RW penalty via
    `_rate_penalty`, SEs via `_rate_ses`), `fit_quality_creator` (pooled quality, position-level
    qcreate), `fit_conversion` (pooled fin/gsave, per-strength `a`/`b`, season offsets, curves); the
    checkpoint/warm-start machinery and `fit_all`; and the effective-parameter interpretation
    `effective_params` (state + curve + offset → card-ready arrays; `target=` for projections),
    `unit_effective` (per-season trajectories), `value_environment`.
  - `generative_export` — `player_values` (attribution), `ppc(R, rate, qual, conv, key, agepos=)`, the
    leaderboards, and `run`/`_save` (loop over EV + MA buckets, projection, JSON) + the CLI, reached
    through a thin shim in `generative_model`.
- **Tests:** `pipeline/tests/test_generative_{model,likelihood,data,cards,holdout,features}.py`
  (synthetic recovery — incl. RW drift, aging-curve + projection math, sparse/dense SE parity,
  per-season reconciliation, the parameter layouts and each stage objective — plus a data-gated smoke
  test).
- **Run:** `make generative-model` (single latest season) or
  `uv run --group experimental python -m yhattrick.models.generative_model --pool --count nb`
  (all seasons — multi-season ⇒ RW states + projection). `--spg-scale 0.5|2.0` for the assist-credit
  sensitivity check (A3).
- **θ̂ checkpoint / warm starts:** every full fit dumps its raw solution vectors + SEs to
  `data/models/generative_ckpt.npz`, keyed by (player, season) unit, ctx-column name, and
  (venue, season) arena state. The next fit **warm-starts** from it by key (default; `--cold`
  disables): identical refits converge in a handful of iterations, incremental refits (new
  season/players — those init cold) in a fraction of a cold run. The optimum is unchanged — a warm
  start only shortens the path to it. `--reexport` goes further: with a checkpoint from the *exact
  same* fit signature (seasons + count model + spg-scale, enforced key-by-key), it skips
  optimization **and** the SE Hessians and just regenerates reports + JSON — export-side tweaks
  rerun in minutes instead of hours. The holdout harness never warm-starts (its train fits must not
  touch a solution that saw test data), and a stale/mismatched checkpoint can never break a fit
  (mapping failures fall back to cold init; `--reexport` refuses loudly).
- **Output:** `data/models/generative_model_<seasons>.json` —
  - per-strength blocks (`strengths.ev/ma` with intercepts + PPC), `conv` (per-strength `a`/`b`,
    season offsets + curve coefficients under `ctx`, per-season reconciliation, EB prior SDs);
  - `qcreate` (position pair + SEs), `age_curves` (per block: coefficients, D offset, curve sampled
    over ages 18–40 per position — site-ready), `rw_sd`, `arena_effects` ({venue: {season: coef}}
    per stage + the nuisance prior), `missing_birthdates`;
  - `players[]`: `pos`, `age`, `last_season`, per-strength TOI, **effective** `ev_/pp_` shoot/create/
    def (+ SEs at the last state), `scoring/playmaking/finishing`, `ev_defense`/`pk_defense`,
    `qshoot`/`qdef`/`fin` (+ `fin_se`), `n_create`, and `trend` — the per-season effective
    shoot/create/def trajectory (multi-season fits). **Two deliberate semantics:** the rate fields
    (`ev_shoot` …) are *effective* — state + position offset + curve — i.e. production expectations
    ("how much does he actually shoot"); the quality/conversion fields (`qshoot`, `qdef`, `fin`) are
    *raw residuals above the position/age baseline* — i.e. talent reads (a D's `fin` is his skill, not
    the calibration offset that compensates the a>1 slope on low-xG point shots). Leaderboards follow
    the same rule. The offsets/curves needed to convert between the two live in `age_curves`;
  - `projection` (target season + per-player projected values, §3d);
  - `goalies[]` (gsave + SE).
- **Strength config:** `EV_STRENGTHS`, `MA_STRENGTHS` (extensible — e.g. add `5v3/3v5` to MA); the
  defender mask (`MAX_DEF`) handles the varying PK size. All tunable constants: see the
  **hyperparameter reference** below.

### Hyperparameter reference — every assumed value, what it controls, where it came from

Provenance legend — **rules**: fixed by how hockey works; **data**: re-estimated from the data at
every fit (not assumed); **hand-set**: chosen at design time with the stated rationale, then
retrospectively supported by the held-out harness; **validated**: selected BY the held-out
calibration sweep (§7) — the value is evidence, not judgment; **numerics**: computational only,
no statistical content.

**Structure (rules)**

| Constant | Value | Controls |
|---|---|---|
| `EV_STRENGTHS` / `MA_STRENGTHS` | 5v5 / {5v4, 4v5} | which stints feed each bucket |
| `N_TM`, `N_DEF_EV`, `N_DEF_MA` | 4, 5, 4 | teammates per shooter; defenders per side |

**Anchor weight w — the assist↔create tie (data × validated)**

`w` multiplies the assist-credit likelihood term (A1 + A2): "one credited goal stands in for w
shots whose creator is unobserved" (inverse-probability weighting).

| Bucket | Value | Provenance |
|---|---|---|
| EV | shots-per-goal × **0.25** (≈ 4.1) | **validated** (2026-07 sweep, §7): at full weight the anchor overfit assist-role — held-out teammate-shots corr 0.657, barely over naive 0.612 (and under it in 2024); ×0.25 lifts it to 0.690 on two target years. Below ×0.1 create loses identification |
| PP/PK | shots-per-goal × **0.25** (≈ 2.6) | **validated** (2026-07 sweep): PP assists are role-censored, so each carries far less creation information; γ_create 0.31 → 0.75 with held-out PP deviance improving |
| `--spg-scale` | 1.0 | CLI multiplier for A3 sensitivity checks only |

**Level priors — ridge SD per parameter block (how much individual talent can differ)**

| Constant | Value | Scale | Controls | Provenance |
|---|---|---|---|---|
| `PRIOR_SD_SHOOT` | 0.30 | log shot rate | `shoot_j` and (EV) `def_d` spread | hand-set; EV γ_shoot 0.94 ≈ ceiling |
| `PRIOR_SD_CREATE` | 0.12 | log rate (per-teammate) | EV `create_p` spread | hand-set; EV γ_create 0.94 with the position-mean prior (§7) |
| EV create prior CENTER | **position-mean** | log rate | ridge target for EV `create` — F +0.174 / D −0.308, not 0 | **validated** (2026-07, §7): held-out tm-corr + γ improve on two years; well-identified players invariant |
| MA create prior | **0.04** | log rate | PP `create_p` spread | **validated** (sweep; fixed units can't identify a wider individual spread) |
| MA def prior | **0.10** | log rate | PK `def_d` spread | **validated** (sweep: restores γ_def to 0.74 ≈ the EV-def ceiling after the create cap pushed unit residual into def; best held-out PP deviance) |
| `PRIOR_SD_QSHOOT` | 0.20 | logit xG | `qshoot_j` / `qdef_d` spread | hand-set |
| `PRIOR_SD_QCREATE` | 0.25 | logit xG | `qcreate` (position pair, credited creator) spread | hand-set (identification resolved at position level, §8) |
| `PRIOR_SD_QCREATE_PL` | 0.08 | logit xG | per-player `create_qual` (on-ice creation quality) spread | hand-set tight (dense channel — each player loads on 4× the rows); held-out gate confirms the spread predicts at face value (§7/§8) |

**Conversion priors (data — estimated every fit, then held fixed during it)**

| Constant | Value | Controls | Provenance |
|---|---|---|---|
| fin / gsave prior SDs | estimated (2021-25 fit: 0.157 / 0.056) | finishing & goalie talent spread on the logit-conversion scale | data (empirical Bayes, Stage 0) |
| `PRIOR_SD_FIN` / `PRIOR_SD_GSAVE` | 0.20 / 0.20 | fallback if the EB estimate is unavailable | hand-set, rarely used |
| `PRIOR_SD_FLOOR` | 0.02 | floor on the EB estimate (keeps the ridge finite) | numerics |
| `MIN_SHOTS_FIN_EST` / `MIN_SHOTS_GSAVE_EST` | 200 / 1000 | who enters the talent-SD estimate | hand-set gates |

**Drift & aging (the player-curve machinery, §3)**

| Constant | Value | Controls | Provenance |
|---|---|---|---|
| `RW_SD_SHOOT` / `RW_SD_CREATE` / `RW_SD_DEF` | 0.10 / 0.05 / 0.10 | per-season random-walk SD of each EV state (per √season gap) — how fast skill can drift | hand-set; holdout shows drift/aging a wash one step ahead and γ ceilings consistent with these values |
| `AGE_PEAK` / `AGE_SCALE` | 27 / 10 | aging basis z = (age−27)/10; curve = deviation from peak age | convention (hockey aging literature); curve shapes are fitted, only the basis is assumed |

**Arena nuisance states (§2)**

| Constant | Value | Controls | Provenance |
|---|---|---|---|
| `ARENA_SD` | 0.05 | ridge SD per (venue, season) recording-bias state | hand-set (biases are small) |
| `ARENA_RW_SD` | 0.03 | random-walk SD between a venue's consecutive seasons (scorer crews persist) | hand-set |
| `ARENA_MIN_GAMES` | 20 | venues below this map to no offset (outdoor/neutral) | hand-set gate |

**Numerics & reporting (no statistical content)**

| Constant | Value | Controls |
|---|---|---|
| `--count` | nb (production) | count layer: Poisson or negative binomial (r fitted) |
| L-BFGS-B `maxiter/ftol/gtol` | 1500 / 1e-10 / 1e-7 | optimizer stopping |
| `DENSE_H_MAX` | 8000 | dense vs sparse SE path |
| quadrature nodes | 40 (CDF space) | the ḡ Beta marginal (§2) — exact to MC noise |
| `EPS` | 1e-6 | clipping |
| `SNIFF_MIN_TOI` / `SNIFF_MIN_TOI_MA` | 24000 s / 2400 s | leaderboard eligibility (reporting only) |

Fitted quantities are deliberately NOT here: `beta_s` (shot-quality concentration), the A2 mixture
`q`, the NB dispersion `r`, aging-curve shapes, arena states, and all player parameters are
estimated from data every fit. Card-layer knobs (replacement band, goals-per-win, card TOI gates)
live in `docs/metrics.md`.
- **Longer windows:** the model is now correct for arbitrarily long windows (drift states + season
  effects in every stage). The practical gate is data (pre-2021 fetch/process is a separate task) and
  two Python-loop hotspots that will grow linearly (`_shooter_counts`, the per-game PBP assist reads) —
  cache assists to a parquet when the window grows.
- **Player cards (Cards v2):** `generative_cards.py` consumes the pooled fit JSON + stints and
  writes `data/models/gen_cards.json` — current-skill attributes, GA/60 vs a position-baseline
  team, per-season trajectory values with a league age-reference, projections, and stint-
  counterfactual WAR. `export_players.py` merges it into the site JSONs (`gen`/`gen_meta` blocks).
  Reader-facing definitions: docs/metrics.md. Run: `make generative-cards` then `make players`.
