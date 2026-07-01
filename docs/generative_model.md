# The Generative Player Model (shooter-resolved, unified-creation)

**Status:** experimental proof-of-concept. Lives in
[`pipeline/src/yhattrick/models/generative_model.py`](../pipeline/src/yhattrick/models/generative_model.py),
run with `uv run --group experimental python -m yhattrick.models.generative_model [--pool] [--count nb]`.
**Not wired into the site/export** — the production additive (RAPM) model remains primary. This model
is the *generative* counterpart: it specifies how a stint **produces** shots and goals, so you can both
fit it and simulate from it. Covers **even strength (5v5) and the man-advantage (5v4/4v5)**, and pools
multiple seasons (with a season fixed-effect). See "Strengths & seasons" below.

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

---

## 2. The model

A 5v5 stint is a **marked Poisson process with Bernoulli thinning**, in three **stages** — rate,
quality, conversion. Each stage has its own parameters and is **fit separately** (§4): the stages are
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
- **Seasons** are pooled by concatenation (one coefficient set per player) with **per-season indicator
  columns** added to the rate/quality context `x` (first season = reference), so league-environment
  drift doesn't leak into player effects.

Notation below is written for a single strength; per-strength params carry the bucket (EV/PP) implicitly.

### Stage 1 — shot RATE (count), shooter-resolved
For **each attacker `j ∈ A` as the focal shooter** (teammates `T = A\{j}`):
```
rate_j = exp( mu_rate + shoot_j + Σ_{p∈T} create_p + Σ_{d∈B} def_d + beta_rate·x )
N_j ~ Poisson( rate_j · o )        (or NegBin(mean = rate_j·o, size = r) with --count nb)
```
`rate_j` = j's expected Fenwick (unblocked) shots per hour in this lineup; `N_j` = his observed count.
A unit's shot generation is the **sum over its shooters**, each shooter's rate raised by the `create` of
his on-ice teammates and suppressed by the defenders.

### Stage 2 — shot QUALITY (mark)
A shot's mean quality (xG) is a shooter/defense/context baseline plus a bump from the one player who
created the chance. On a goal the creator is observed (the primary assister); otherwise it is latent and
marginalized over the creator distribution `pi`:
```
base_i = mu_qual + qshoot_j + Σ_{d∈B} qdef_d + beta_qual·x                 # everything but the creator
qbar_i = sigmoid( base_i + qcreate_c )                      if GOAL      — c = observed primary assister
       = Σ_{c∈{∅,T}} pi_c · sigmoid( base_i + qcreate_c )    if NON-GOAL  — creator latent
   where  pi_c = softmax([ create_0 , create_T ])_c    and   qcreate_∅ = 0   (unassisted adds no bump)
xg_i ~ Beta( s·qbar_i , s·(1−qbar_i) )
```
Estimated here: `qshoot, qcreate, qdef` (and `mu_qual, beta_qual, s`); `pi` reuses `create`/`create_0`,
which are **fixed values taken from Stage 1** (not re-fit here). (The non-goal line is the mixture
mean — a mean-field plug-in; see §4.)

### Stage 3 — CONVERSION (goal | shot)
Fit **natively** inside this model (no shooting-model reuse) as a logistic recalibration of the shot's
**observed** xG, so the stage stays independent of Stages 1–2:
```
logit(p_goal_i) = a · logit(xg_i) + b + fin_j + gsave_g
y_i ~ Bernoulli( p_goal_i )
```
`a` (slope) recalibrates the xG→goal map (≈1 if xG is well-calibrated on 5v5); `b` (intercept) replaces
the old `mu_conv`. Both are **global and unpenalized**, so the MLE score equation for `b` is
`Σ(y_i − p_i) = 0`, i.e. **`Σ p = Σ goals` holds exactly** — deterministic expected goals reconcile as a
natural first-order condition, not a hand-solved constant. `fin_j` (per shooter) and `gsave_g` (per
goalie, `<0` = good) are log-odds offsets with EB ridge priors, fit here alongside `a`,`b`; their prior
SDs come from the **pre-calculation stage** below (not hand-set). In simulation the *drawn* mark
`q ~ Beta(s·qbar,·)` stands in for `xg_i`; `sigmoid` is bounded, so the old `clip(…,0,1)` is gone (only
`logit(q)` needs an `EPS` clamp).

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

### Glossary (used consistently throughout)

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
| `A`, `T = A\{j}`, `B` | on-ice attackers (≤5), j's teammates (≤4), defenders (≤5) |

**Observed data**
| symbol | meaning |
|---|---|
| `N_j` | j's Fenwick shot count in the epoch |
| `t`, `o = t/3600` | epoch duration (s); hourly exposure offset |
| `x` | context vector (home, O/D-zone start, lead/trail) |
| `xg_i`, `y_i` | shot i's upstream expected-goals value and goal indicator (0/1) |
| primary assister | the observed creator label on each goal (or `∅` = unassisted) |

**Free parameters (estimated)** — per-skater unless noted
| symbol | stage | meaning |
|---|---|---|
| `shoot_j` | rate | **own-shot rate** — how much j shoots when on the ice |
| `create_p` | rate **& quality** | **creation** — raises teammates' shot rate (Stage 1) *and* sets the creator distribution `pi` (used, fixed, in Stage 2) |
| `create_0` | quality | **unassisted** creator propensity (the `∅` candidate in `pi`); a single global scalar |
| `def_d` | rate | opponent shot-rate suppression (`<0` good) |
| `qshoot_j` | quality | danger of j's own shots |
| `qcreate_c` | quality | danger a player adds **when he is the creator** *(currently unidentifiable — see §7)* |
| `qdef_d` | quality | opponent danger suppression (`<0` good) |
| `fin_j` | conversion | finishing above xG on own shots (logit offset; fit natively) |
| `gsave_g` | conversion | goalie saves above expected (logit offset, `<0` good; per goalie `g`) |
| `a`, `b` | conversion | logit-conversion slope / intercept (global, unpenalized; `b` gives `Σp=Σgoals`) |
| `mu_rate`, `mu_qual` | global | replacement-level log shot-rate / logit shot-quality |
| `beta_rate`, `beta_qual` | global | context coefficients (on `x`) for rate / quality |
| `r`, `s` | global | NB dispersion (`--count nb`); Beta concentration for the xG mark |

**Derived quantities** (intermediate predictors — assembled from the free parameters, not fit directly)
| symbol | definition | meaning |
|---|---|---|
| `rate_j` | `exp(mu_rate + shoot_j + Σ create_p + Σ def_d + beta_rate·x)` | j's Poisson shot rate (per hour) |
| `base_i` | `mu_qual + qshoot_j + Σ qdef_d + beta_qual·x` | creator-independent part of shot quality |
| `qbar_c` | `sigmoid(base_i + qcreate_c)`; `qbar_∅ = sigmoid(base_i)` | mean xG of the shot if its creator is `c` |
| `qbar_i` | `qbar_c` (goal) or `Σ_c pi_c·qbar_c` (non-goal) | shot i's mean quality — creator known vs. latent |
| `pi_c` | `softmax([create_0, create_T])_c` | **creator-identity distribution** — P(candidate `c` set it up); observed on goals, latent (marginalized) on non-goals |
| `p_goal_i` | `sigmoid(a·logit(xg_i) + b + fin_j + gsave_g)` | conversion probability (logit; observed `xg`) |

**Functions & hyperparameters**
| symbol | meaning |
|---|---|
| `sigmoid(z)` | logistic, `1/(1+e^-z)` |
| `softmax(v)_c` | `e^(v_c) / Σ_k e^(v_k)` |
| `SHOTS_PER_GOAL` | up-weight on the assist-credit anchor when fitting `create`; ≈ Fenwick shots per goal (inverse goal rate), a fixed integer (~16) |
| `PRIOR_SD_SHOOT/CREATE/QSHOOT/QCREATE` | hand-set ridge prior SDs per rate/quality block |
| `prior_sd_fin`, `prior_sd_gsave` | conversion prior SDs — data-estimated each fit (Stage 0); `PRIOR_SD_FIN`/`PRIOR_SD_GSAVE` are only the fallbacks |
| `MIN_SHOTS_FIN_EST=200`, `MIN_SHOTS_GSAVE_EST=1000`, `PRIOR_SD_FLOOR` | shot gates + floor for the Stage-0 prior-SD estimate |
| `N_TM = 4`, `N_DEF = 5` | teammate / defender counts used in deployment-free attribution |

**Code-symbol map** (this doc → `generative_model.py`): `create_0`→`psi0`; `mu_rate/qual`→ each fit's
`intercept`; `a`/`b`→`conv["a"]`/`conv["b"]`; `fin_j`→`conv["fin"]`; `gsave_g`→`conv["gsave"]`;
`beta_rate/qual`→`beta`; `qbar`→`sig5`; `pi_c`→`pi`; `rate_j`→`exp(eta)` in the Poisson NLL.

### Player-value attribution (deployment-free per-60, from fitted params + intercepts)
```
q_own_j       = sigmoid(mu_qual+qshoot_j)                                                          # own-shot mean xG
scoring(j)    = exp(mu_rate+shoot_j) · sigmoid(a·logit(q_own_j) + b + fin_j)                        # own shots, CONVERTED
playmaking(p) = N_TM · exp(mu_rate) · (exp(create_p) − 1) · sigmoid(mu_qual + qcreate_p)           # teammate xG added
defense(d)    = N_DEF · [ exp(mu_rate)·sigmoid(mu_qual) − exp(mu_rate+def_d)·sigmoid(mu_qual+qdef_d) ]  # opp xG suppressed
creator_share(p) = exp(create_p) / ( exp(create_0) + exp(create_p) + (N_TM−1) )                    # per-teammate-shot
```
Computed **per strength** with that bucket's rate loadings + intercepts and the pooled quality/finishing
loadings: EV → `ev_scoring/playmaking/defense` (`N_DEF=5`); MA → `pp_scoring/pp_playmaking` and
`pk_defense` (`N_DEF=4`; the MA `def` loadings are the penalty-killers). `mu_qual`, `a`, `b` are the
strength's intercepts.
`N_TM = 4`, `N_DEF = 5` (EV) or `4` (PK). Playmaking = the volume of teammate shots p adds (`create`) × their danger
(`qcreate`). Defense `>0` = suppresses (good).

---

## 3. The generative direction (simulate a stint)
Given fitted params and a stint `(A, B, g, t, x)`: for each shooter `j`, draw `N_j ~ Poisson(rate_j·o)`;
for each shot draw a creator `c ~ pi = softmax([create_0, create_T])`, then `xg ~ Beta(s·qbar_c, s·(1−qbar_c))`,
then `y ~ Bernoulli(sigmoid(a·logit(xg) + b + fin_j + gsave_g))`. Aggregating gives simulated
shots/xG/goals (used for the posterior-predictive check and counterfactual lineup swaps).

---

## 4. The inference direction (fit params from data)
The stages are conditionally independent given the predictors, so they are fit **in sequence** — each by
penalized MLE / empirical-Bayes (ridge priors) with **JAX autodiff gradients + scipy L-BFGS-B**; SEs via
the Hessian / Gauss-Newton diagonal. Across strengths (§2, "Strengths & seasons"): the **rate stage is
fit once per bucket** (EV, MA — giving separate `ev_*` and `pp_*`/`pk_def` volume params), while
**quality and conversion are single pooled fits** (shared loadings, per-strength intercepts). The
quantity shared across stages is `create`/`create_0`: **fit in Stage 1, passed into Stage 2 as a fixed
constant** (per-row by the shot's strength). Stage 3 uses neither, so it is independent of 1–2.

- **Stage 1 — rate (`fit_rate_create`, run per strength bucket)** — fits `mu_rate, shoot, create, def,
  beta_rate, create_0 (, r)` on the Poisson/NB count NLL (one row per (epoch, shooter); EV ~4.4M
  rows/season, MA ~0.3M; defenders masked to the bucket's count). Added to
  it is an **assist-credit anchor**: `SHOTS_PER_GOAL ×` a conditional logit scoring each goal's observed
  primary assister under `pi = softmax([create_0, create_T])`. We only observe a shot's creator on the
  ~1-in-16 Fenwick shots that are goals, so weighting each observed goal-creator by `SHOTS_PER_GOAL`
  (≈ shots per goal ≈ 1/goal-rate) makes it stand in for the shots whose creator we never see — an
  **inverse-probability weight, not a free knob**. Without it the dense counts identify `create` only as
  a possession/volume effect; with it `create` is pulled toward the players actually credited with
  setups. (Goals are a danger-biased subsample of shots, so this is an approximate IPW.) The count NLL
  and the anchor share `create`/`create_0` and are optimized **together** in this one stage.
- **Stage 2 — quality (`fit_quality_creator`)** — a single POOLED fit (EV+MA) of `qshoot, qcreate, qdef,
  beta_qual, s` plus a per-strength intercept (`mu_qual` via a `pp` context column), with `create`/
  `create_0` **taken as fixed values from the per-strength Stage-1 fits** (so `pi` is a constant, chosen
  per row by that shot's strength): creator observed on goals, marginalized over `pi` on non-goals.
  Defenders are masked (≤5). Nothing feeds back to Stage 1.
- **Stage 0 — conversion pre-calc (`estimate_conversion_prior_sds`)** — one aggregate pass sets the
  `fin`/`gsave` ridge prior SDs from the data (empirical Bayes, §2 Stage 0), recomputed each fit and
  then frozen. Not in any objective; the logit-scale analogue of `shooting_model._estimate_k`.
- **Stage 3 — conversion (`fit_conversion`)** — fits `a, b, fin, gsave` by penalized-MLE (JAX + L-BFGS)
  on a Bernoulli goal likelihood, `logit(p) = a·logit(xg) + b + fin_j + gsave_g`, **natively** — no
  shooting-model reuse. `a, b` are unpenalized (so `Σp=Σgoals` holds exactly); `fin`/`gsave` carry the
  Stage-0 EB ridge priors (data-calibrated, held fixed here). Independent of Stages 1–2: it keys off the
  upstream **observed** xG, never `create`, `qcreate`, or `qbar`. Fitting `fin`/`gsave` on the 5v5-only
  subset is low-power for these weak-signal skills (like `qcreate`, §7): the EB prior shrinks them
  honestly and the run reports the `fin` `|z|>2` share so this stays visible.

**Approximations (where inference departs from one exact joint marginalization; none is *why* `qcreate`
fails — that is label sparsity, §7).**
1. **Volume uses no latent creator.** In Stage 1 `create` enters as a deterministic sum over *all*
   teammates, so a non-goal shot's creation credit is spread across the lineup; only Stage 2 has a
   single creator to marginalize.
2. **Two-stage plug-in, not joint.** `create`/`pi` are pinned first (dense Poisson + goals-only credit
   logit) and then **frozen** for the quality fit, rather than estimating `create` and `qcreate` jointly
   by marginalizing the creator once over both factors.
3. **Mean-field marginal.** On non-goals the creator sum sits *inside* the mark mean
   (`E[xg] = Σ pi_c·qbar_c`) rather than wrapping the likelihood (`Σ pi_c·Beta(xg|qbar_c)`). Equal to
   first order and exact as `qcreate→0`; the plug-in slightly overstates the log-likelihood (Jensen).

**Why `create` is identified.** The rate response is *shooter-specific* (`N_j`), so `shoot_j` loads on
rows where j shoots and `create_p` loads on rows where p is a *teammate* of the shooter — estimated
from how much the players around p out-shoot when p is on the ice (lineup variation, **no pass data
needed**), then *anchored* by the assist-credit so it means creation, not mere possession.

**Standard errors.** Diagonal Gauss-Newton: the rate Hessian `XᵀWX` for `shoot/create/def`, with the
credit Fisher added to the `create` diagonal; a diagonal info for `qcreate` from observed-creator goal
quality + the non-goal marginal. `z = estimate/se`; low-confidence players are greyed out (`⚠`).

**Performance note.** Data arrays are passed as **arguments** to the jitted `value_and_grad`, not
closed over — otherwise `jax.jit` bakes the ~3 GB of index arrays into the compiled program as
captured constants (slow compile, double memory). See `_optimize(nll, x0, *data)`.

---

## 5. Data
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

---

## 6. Results

Fit on a single 5v5 season (NB counts), eligible = on-ice ≥ ~400 5v5 minutes:

- **`create` is well-identified and forward-weighted.** ~77% of eligible players clear `|z|>2`
  (`z` up to ~14); `create` tracks primary assists among forwards; the metric is forward-led (top of the
  leaderboard is recognized setup men, no defenseman over-representation). Both halves matter: the dense
  volume signal captures the unobserved passing the assists miss, and the credit anchor keeps the metric
  pointed at genuine creation rather than raw possession.
- **Goals and shots reconcile.** Deterministic expected goals reconcile exactly (`Σp=Σgoals` falls out
  of the unpenalized conversion intercept `b`); a single-seed posterior-predictive simulation lands
  within ~1% on goals and on total shots, and the NB count model matches the per-row shot-count
  overdispersion.
- **Scoring is clean.** `scoring` correlates strongly with goals; the `shoot` leaderboard is snipers and
  the `create` leaderboard is recognized setup men, with the two loadings near-uncorrelated (own-shot
  volume and teammate-lift are separately identified).
- **Playmaking is driven by `create`.** The `playmaking` value = teammate shot-volume added (`create`)
  × danger per chance (`qcreate`); because `qcreate` is not per-player identifiable (§7), it is held near
  a league constant and `create` does the ranking. A few high-volume two-way centers rank up through the
  volume half — correct behavior (they do drive teammate chances), and adjustable via `SHOTS_PER_GOAL`.

---

## 7. Open problem — `qcreate` is not identifiable

`qcreate_p` (the danger a player adds *when he is the creator*) is the one parameter that **does not
identify**, even when seasons are pooled: essentially no player clears `|z|>2`, and `qcreate_se`
sits at roughly its prior SD, so the estimate shrinks to the prior for everyone. The point estimates
still order sensibly (elite creators drift to the top), but none reach significance.

**Why:** `qcreate` is informed almost entirely by goals on which the player is the *observed* primary
creator — and a typical player is the primary setup man on only **~17–26 goals even across several
pooled seasons**. That is far too few events to estimate a per-player offset on the danger (logit-xG) of
his setups above the prior. Non-goal shots contribute little, because there the creator is latent and
enters only through the soft, `create`-weighted marginal.

**Consequence for the metric:** playmaking is driven by `create` (volume, well-identified); creation
*quality* is best treated as a **league- or position-level constant**, not a per-player rating — or
always shown with its CI and never ranked on. The *volume × quality* symmetry with scoring is therefore
**half-realized**: volume works, quality is data-limited.

**Open threads.** (a) Is it purely the event-count limit, or is the danger signal genuinely weak
conditional on the creator (i.e. is there any per-player variance to recover)? (b) Would a
hierarchical/position prior — pooling `qcreate` toward a role-level mean — help? (c) Can the non-goal
marginal be made more informative (e.g. weighting it only when `create`/`pi` concentrates the creator)?
(d) Does the unassisted/observed split introduce bias? (e) Is "danger per setup" even a separable skill
on this data, or is a shot's danger mostly the shooter's (`qshoot`) and the location's?

---

## 8. Implementation map & how to run
- **File:** `pipeline/src/yhattrick/models/generative_model.py`. Tests:
  `pipeline/tests/test_generative_model.py` (synthetic recovery + a data-gated smoke test).
- **Key functions:** `player_index` (shared index), `_load_stints(seasons, strengths)`,
  `rate_rows(seasons, strengths, dual, players, idx)`, `quality_creator_rows(seasons, idx, strengths)`,
  `conversion_rows(seasons, idx, strengths)` (data); `fit_rate_create` (per-strength rate + credit, with
  the defender mask), `fit_quality_creator` (pooled quality, per-strength intercept + per-row creator
  dist), `estimate_conversion_prior_sds` + `fit_conversion` (pooled fin/gsave, per-strength `a`/`b`);
  `player_values` (per-strength attribution); `ppc(R, rate, qual, conv, key)`; `run`/`_save` (loop over
  EV + MA buckets).
- **Run:** `make generative-model` (single latest season) or
  `uv run --group experimental python -m yhattrick.models.generative_model --pool --count nb`. Output:
  `data/models/generative_model_<seasons>.json` — per-strength blocks (`strengths.ev/ma` with intercepts
  + PPC), per-player `ev_*`/`pp_*`/`pk_defense` values plus pooled `qshoot/qcreate/fin` and per-strength
  TOI (`toi_ev/toi_pp/toi_pk`), `conv` (per-strength `a`/`b`, EB prior SDs), and goalies.
- **Strength config:** `EV_STRENGTHS`, `MA_STRENGTHS` (extensible — e.g. add `5v3/3v5` to MA); the
  defender mask (`MAX_DEF`) handles the varying PK size. **Tunables:** per-strength `SHOTS_PER_GOAL`
  (auto-computed as shots/goal per bucket), `PRIOR_SD_*` (conversion priors are data-estimated each fit;
  these are fallbacks), `MIN_SHOTS_*`, `SNIFF_MIN_TOI`/`SNIFF_MIN_TOI_MA` (leaderboard gates), `--count
  poisson|nb`.
