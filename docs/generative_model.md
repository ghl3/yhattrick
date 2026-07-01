# The Generative Player Model (shooter-resolved, unified-creation)

**Status:** experimental proof-of-concept. Lives in
[`pipeline/src/yhattrick/models/generative_model.py`](../pipeline/src/yhattrick/models/generative_model.py),
run with `uv run --group experimental python -m yhattrick.models.generative_model [--pool] [--count nb]`.
**Not wired into the site/export** — the production additive (RAPM) model remains primary. This model
is the *generative* counterpart: it specifies how a stint **produces** shots and goals, so you can both
fit it and simulate from it. 5v5 only.

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
Reused from the production shooting model (finishing + goaltending), with the intercept recalibrated to
this model's 5v5-Fenwick universe:
```
p_goal_i = clip( qbar_i + mu_conv + fin_j + gsave_g , 0, 1 )
y_i ~ Bernoulli( p_goal_i )
mu_conv = ( Σ goals − Σ(xg + fin + gsave) ) / N     # so simulated goals reconcile to actual exactly
```

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
| `fin_j` | conversion | finishing above xG on own shots |
| `gsave_g` | conversion | goalie saves above expected (per goalie `g`) |
| `mu_rate`, `mu_qual`, `mu_conv` | global | replacement-level log shot-rate / logit shot-quality / conversion intercept |
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
| `p_goal_i` | `clip(qbar_i + mu_conv + fin_j + gsave_g, 0, 1)` | conversion probability |

**Functions & hyperparameters**
| symbol | meaning |
|---|---|
| `sigmoid(z)` | logistic, `1/(1+e^-z)` |
| `softmax(v)_c` | `e^(v_c) / Σ_k e^(v_k)` |
| `SHOTS_PER_GOAL` | up-weight on the assist-credit anchor when fitting `create`; ≈ Fenwick shots per goal (inverse goal rate), a fixed integer (~16) |
| `PRIOR_SD_*` | ridge / empirical-Bayes prior SDs per parameter block |
| `N_TM = 4`, `N_DEF = 5` | teammate / defender counts used in deployment-free attribution |

**Code-symbol map** (this doc → `generative_model.py`): `create_0`→`psi0`; `mu_rate/qual/conv`→ each
fit's `intercept`; `beta_rate/qual`→`beta`; `qbar`→`sig5`; `pi_c`→`pi`; `rate_j`→`exp(eta)` in the
Poisson NLL.

### Player-value attribution (deployment-free per-60, from fitted params + intercepts)
```
scoring(j)    = exp(mu_rate+shoot_j)·sigmoid(mu_qual+qshoot_j) + exp(mu_rate+shoot_j)·fin_j        # own shots
playmaking(p) = N_TM · exp(mu_rate) · (exp(create_p) − 1) · sigmoid(mu_qual + qcreate_p)           # teammate xG added
defense(d)    = N_DEF · [ exp(mu_rate)·sigmoid(mu_qual) − exp(mu_rate+def_d)·sigmoid(mu_qual+qdef_d) ]  # opp xG suppressed
creator_share(p) = exp(create_p) / ( exp(create_0) + exp(create_p) + (N_TM−1) )                    # per-teammate-shot
```
`N_TM = 4`, `N_DEF = 5`. Playmaking = the volume of teammate shots p adds (`create`) × their danger
(`qcreate`). Defense `>0` = suppresses (good).

---

## 3. The generative direction (simulate a stint)
Given fitted params and a stint `(A, B, g, t, x)`: for each shooter `j`, draw `N_j ~ Poisson(rate_j·o)`;
for each shot draw a creator `c ~ pi = softmax([create_0, create_T])`, then `xg ~ Beta(s·qbar_c, s·(1−qbar_c))`,
then `y ~ Bernoulli(clip(xg + mu_conv + fin_j + gsave_g, 0, 1))`. Aggregating gives simulated
shots/xG/goals (used for the posterior-predictive check and counterfactual lineup swaps).

---

## 4. The inference direction (fit params from data)
The three stages are conditionally independent given the predictors, so they are fit **in sequence** —
each by penalized MLE / empirical-Bayes (ridge priors) with **JAX autodiff gradients + scipy L-BFGS-B**;
SEs via the Hessian / Gauss-Newton diagonal. The only quantity shared across stages is
`create`/`create_0`: **fit in Stage 1, then passed into Stage 2 as a fixed constant** (a plug-in, not a
joint fit — see the approximations below). Stage 3 uses neither, so it is fully independent.

- **Stage 1 — rate (`fit_rate_create`)** — fits `mu_rate, shoot, create, def, beta_rate, create_0 (, r)`
  on the Poisson/NB count NLL (one row per (epoch, shooter); ~4.4M rows/season, ~21M pooled). Added to
  it is an **assist-credit anchor**: `SHOTS_PER_GOAL ×` a conditional logit scoring each goal's observed
  primary assister under `pi = softmax([create_0, create_T])`. We only observe a shot's creator on the
  ~1-in-16 Fenwick shots that are goals, so weighting each observed goal-creator by `SHOTS_PER_GOAL`
  (≈ shots per goal ≈ 1/goal-rate) makes it stand in for the shots whose creator we never see — an
  **inverse-probability weight, not a free knob**. Without it the dense counts identify `create` only as
  a possession/volume effect; with it `create` is pulled toward the players actually credited with
  setups. (Goals are a danger-biased subsample of shots, so this is an approximate IPW.) The count NLL
  and the anchor share `create`/`create_0` and are optimized **together** in this one stage.
- **Stage 2 — quality (`fit_quality_creator`)** — fits `mu_qual, qshoot, qcreate, qdef, beta_qual, s`
  with `create`/`create_0` **taken as fixed values from Stage 1** (so `pi` is a constant here): the
  creator is observed on goals, marginalized over `pi` on non-goals. Nothing in this stage feeds back to
  Stage 1.
- **Stage 3 — conversion (`conversion_params`)** — finishing/goalie from the production shooting model,
  with `mu_conv` recalibrated to the 5v5-Fenwick universe. Independent of Stages 1–2: it uses the
  upstream observed xG, not `create` or `qcreate`.

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
- **Shots:** `processed/shots_onice/<season>.parquet` — per 5v5 Fenwick shot: `shooter_id`, `xg`,
  `goal`, on-ice `home_skaters`/`away_skaters`, `event_idx`.
- **Stints:** `processed/stints/<season>.parquet` — on-ice skaters, duration, Fenwick counts, context.
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
- **Goals and shots reconcile.** Deterministic expected goals reconcile exactly (by the `mu_conv`
  calibration); a single-seed posterior-predictive simulation lands within ~1% on goals and on total
  shots, and the NB count model matches the per-row shot-count overdispersion.
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
- **Key functions:** `rate_rows`, `quality_creator_rows` (data); `fit_rate_create` (unified rate +
  credit), `fit_quality_creator` (quality given fixed creator dist), `conversion_params`;
  `player_values` (attribution); `ppc` (posterior-predictive check); `run`/`_save`.
- **Run:** `make generative-model` (single latest season) or
  `uv run --group experimental python -m yhattrick.models.generative_model --pool --count nb`. Output:
  `data/models/generative_model_<seasons>.json` (per-player params, SEs, `n_create`, attributed values,
  PPC).
- **Tunables:** `SHOTS_PER_GOAL` (≈ shots per goal; up-weights the assist-credit anchor), the
  `PRIOR_SD_*` ridge priors, `--count
  poisson|nb`.
