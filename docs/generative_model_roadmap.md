# Generative Model — Improvement Roadmap

Ranked backlog for the shooter-resolved generative model
([`generative_model.py`](../pipeline/src/yhattrick/models/generative_model.py), spec in
[`generative_model.md`](generative_model.md)). Each item is written to be implementable
independently by whoever picks it up; file/function references are current as of the player-curves
change (July 2026). Rank reflects value-per-effort toward the product goal: **interpretable
per-player features that power player cards**.

**Design principle (user-set, applies to every item):** player cards are **broad latent qualities we
isolate** (Scoring, Playmaking, Defense, Finishing, …), not restatements of countable stats. Observed
events are *evidence that grounds a quality*, never a card: assists ground Playmaking, blocked shots
ground Defense, tracking measurables (if ever available) would sharpen priors. If a proposed feature
is directly computable from the data, it belongs in descriptive stats or inside a quality — not on an
attribute card.

**Baseline context.** The model already has: three-stage fit (rate/quality/conversion), per-(player,
season) RW drift states on the EV rate blocks, shared F/D aging curves in every stage, position
intercepts, position-level `qcreate`, per-season conversion offsets, sandwich-corrected `se_create`,
next-season projections in the output JSON, and per-season trajectory (`trend`) blocks. Raw pbp +
shiftcharts for 2016–2020 are downloaded (unprocessed).

---

## Tier 1 — do next (each unlocks or de-risks others)

### 1. Held-out-season predictive harness — ✅ IMPLEMENTED (July 2026)
Landed as `yhattrick/models/generative_holdout.py` (built on the `fit_all()` refactor of `run()`):
fits seasons ≤ `--train-through`, scores the projection on the target season's REAL stint rows
against three same-fit reads — league-avg (floor), pooled-mean (the static-pooled read),
last-state (drift, no aging) — plus the naive last-season-raw-rates bar. Metrics: row-level Poisson
deviance (deployment-identical across candidates) and TOI-weighted player own-shots/60 corr/MAE.
Run: `uv run --group experimental python -m yhattrick.models.generative_holdout
[--train-through 2024]` → tables + `data/models/holdout_<target>.json`. Results live in the model
doc §7. Still open from this item: **RW_SD grid tuning** (rerun the harness under different
`RW_SD_*` constants and keep the held-out minimizer) and extending scoring beyond the rate stage
(on-ice xGF, goals). Rookie handling is honest-but-blunt (unseen players sit at the prior mean).

### 2. Secondary assists in the credit anchor — ✅ IMPLEMENTED (July 2026)
Landed as designed below (mixture-q, fitted). Results: **q̂ saturated at 1.00** (recorded A2s fully
concordant with the create ordering); held-out create-side tm-corr +0.008 (pooled) / +0.004
(last/proj) vs the no-A2 arm, own-shot metrics unchanged; the model now **beats the naive bar on
teammate rates** (0.657 vs 0.612). Harness also extended with the create-side (teammate-shots)
scoring track. Original design (kept for reference):
`quality_creator_rows` reads `assist1PlayerId` and discards `assist2PlayerId` from the same pbp
JSON. Design: treat (A1, A2) as a PARTIAL RANKING of the on-ice teammates by creation involvement —
an exploded-logit second stage reusing the same `create` parameters:
`credit = log softmax([create_0, create_T])[c1] + A2 term` (A2 stage: A1's column masked out, no
unassisted option — an A2 implies a second passer existed). Both terms keep the `spg` IPW weight;
the F1 sandwich extends mechanically.
**Setting the A1-vs-A2 balance:** a bare weight λ on the A2 log-likelihood is NOT fittable by MLE
(a pseudo-likelihood temperature — degenerate). Instead model A2 label noise as a MIXTURE with a
proper MLE parameter `q` = P(recorded A2 reflects creation):
`P(A2=c₂|c₁,T) = q·softmax(create_{T\c₁})[c₂] + (1−q)/|T\c₁|` — `q` is identified by A2's
concordance with the create ordering already pinned by counts + A1; the effective down-weight
emerges from the fit. Sandwich uses q-weighted responsibilities.
**Measured priors (July 2026, our exports, 2,166 consecutive-season pairs ≥400 min):** YoY
repeatability A1/60 = 0.73, A2/60 = 0.55 (ratio ≈ 0.75 — an upper bound for `q`, since
repeatability includes deployment persistence); volumes A1 0.62/60 vs A2 0.49/60 ⇒ ~1.8× labeled
events. Validate with ONE harness A/B (no-A2 vs mixture-A2) against the recorded bars
(naive corr 0.864; row-dev 126.66).
**Deliberately NOT in Stage 2 (`qcreate`/`qbar`):** a shot's xG is set by its final geometry — the
LAST pass — so A2's danger effect flows through A1; adding it would double-count the causal path.
Stage 2 still benefits indirectly: better-anchored `create` sharpens the latent-creator `pi` on
every non-goal shot.
**Touch points:** `quality_creator_rows` (emit `creator2`, same F4 not-on-ice rule), `run()` (second
anchor arrays, unit-mapped), `fit_rate_create` (masked-softmax credit term + bread/meat).
**Effort:** small.

### 3. Arena/scorer recording-bias states — ✅ IMPLEMENTED (July 2026)
Landed as per-(venue, season) nuisance states in the rate AND quality stages: ridge to zero
(`ARENA_SD`, identifies the block — no reference venue) + random-walk smoothing across seasons
(`ARENA_RW_SD` — crews persist but change), venues keyed by physical building via pbp
`venue.default` with a rename alias map (`_VENUE_ALIAS`; real building moves stay split), rare
venues (outdoor/neutral, < `ARENA_MIN_GAMES`) unadjusted. Outside the SE Hessian (nuisance, like
`create_0`); excluded from player values; reported per run + saved as `arena_effects` in the JSON.
Still open from this item: **port the same correction to the production RAPM** (same confounder).

### 4. Blocked shots as evidence INSIDE Defense (not a card)
Design principle (user-set): cards are broad latent qualities; observed events are evidence that
grounds them, never cards themselves — assists ground Playmaking, blocks should ground Defense.
Structural symmetry: `create` = dense volume + sparse observed credit (assists); `def` currently has
only the dense on-ice half, and blocked shots are entirely invisible (the model is Fenwick-only).
The clean integration, consistent with the model's marked-Poisson-plus-thinning structure: model
ATTEMPTS (Corsi) in the rate stage and add a **block-thinning stage** — an attempt survives to an
unblocked shot unless blocked, with each block credited to a defender via a conditional logit over
the 5 on-ice defenders (`blockingPlayerId` is recorded on every blocked shot, ~15/game — far denser
than assists; mirror the assist-anchor code path incl. the sandwich SE treatment). Blocking then
folds into the single Defense value as "suppress attempts + thin attempts", individually anchored —
and the attempt/thinning decomposition absorbs the classic confound that raw block counts partly
measure being hemmed in. Output stays ONE Defense card (optionally sub-split shot- vs
chance-suppression internally).
**Effort:** medium-large — response variable moves Fenwick→Corsi in the rate stage + a new thinning
likelihood; do the design in the doc first.

### 5. Card-level CIs via parametric bootstrap
Sample parameters from their Laplace posteriors (per-player SEs already computed; diagonal
approximation is acceptable), push through `player_values`, report percentile CIs on
scoring/playmaking/defense (+ projections, GA/60, and **WAR** — Cards v2 ships them without CIs).
This is what makes the weak-tier features (`fin`, `pp_shoot`) honestly displayable on cards.
**Effort:** small — pure post-processing in `run()`/`_save`/`generative_cards`.

### 5b. Cards v2 — ✅ SHIPPED (July 2026); refinements open
`generative_cards.py` + site UI (GA/60 baseline-team rate, stint-counterfactual WAR, trajectory
chart with projection + league age-reference). Open refinements, in priority order: bootstrap CIs
(#5); penalties into WAR (#8); season-estimated goals-per-win (replaces the 6.0 constant);
replacement-level sensitivity study (the 8th–12th GA/60 percentile band is a documented knob);
exact defender-quality treatment in the WAR swap (currently a multiplicative approximation);
uncertainty band on the trajectory chart (needs per-season state SEs exported).

## Tier 2 — valuable, after Tier 1

### 6. Stage-2 goal-selection reweighting
Creator labels exist only on goals — a high-xG-biased subsample (selection on `y` depends on `xg`).
Importance-weight the observed-creator rows by `1/p(goal|xg)` using Stage 3's own fitted map (fit
order would become 0/3 before 2, or use the previous fit's `a`/`b`). Cleans `qshoot`/`qcreate_{F,D}`.
Documented as approximation 4 in the model doc §5.

### 7. Process the 2016–2020 backfill
Raw pbp/shiftcharts are on disk (fetched July 2026). Run `clean-data`/`xg`/`stints` for those
seasons, then validate: shiftchart completeness per season (the empty-feed problem was late-2024-25+,
but verify), stints golden-equivalence checks, xG calibration per era (rule/equipment changes), and
`_age_position` coverage for retired players (their landing JSONs may need fetching). Then extend
`--pool`. The curves/drift machinery is already built for it; watch `splu` runtime and memory at
~10 seasons (`DENSE_H_MAX` path) and do item 12 first if slow.

### 8. Penalties drawn/taken stage
The one production card family the generative model doesn't cover. Penalty draw/take are per-player
Poisson rates in the same stint framework (labels in pbp). Completes the model as a full card
generator and gives penalties an aging curve.

### 9. Rebound/rush creator semantics
On rebounds the "creator" is usually the original shooter, but the creator model sees rebounds as
mostly unassisted — misattributed creation. The rebound/rush flags already exist as xG features
(`expected_goal_model.py`); thread them into `shots_onice` rows, credit the prior shooter in the
anchor for rebounds, and consider a rush-creation split later (card-worthy distinction).

### 10. A3 sensitivity run (document it)
Run `--pool --spg-scale 0.5` and `2.0`, compute the Spearman correlation of `create_last` against the
1.0 run, put the numbers in the doc §7. Converts "the anchor weight is an IPW, not a knob" from a
claim into a measurement. Trivial effort — two runs + one correlation.

### 11. Context parity: period + score magnitude
Production RAPM's design has 2nd/3rd-period columns; the generative rate context has only
home/zone/lead-trail-binary (`RATE_CTX`). Add period indicators and score-differential magnitude.
Cheap context hygiene protecting the player parameters.

### 12. Scale-readiness: cache assists + vectorize `_shooter_counts`
Two Python-loop hotspots grow linearly with the window: per-game pbp JSON reads in
`quality_creator_rows` (cache assist labels to a per-season parquet on first build) and the groupby
loop in `_shooter_counts`. Do before/with item 7.

## Tier 3 — opportunistic

- **13. Feature-based `qcreate`** — the identifiable middle ground between the position pair and a
  (hopeless) per-player parameter: `qcreate(c) = position_c + γ·x_c` with 2–3 GLOBAL coefficients
  on observable creator features (e.g., mean xG of the shots he's credited with assisting, or his
  setup-location profile). Every labeled goal informs γ, so it fits today's data; creation quality
  then varies across players exactly as far as evidence supports. Rationale: per-player `qcreate`
  needs ~20 labels/player ⇒ SE ≈ 0.5 vs plausible talent spread ≤ 0.1 — a data ceiling that
  sharper `create` (A2) does not lift (Stage-2 labels are A1-only by design). The full per-player
  slot stays reserved for pass-tracking data (#18).
- **14. Lineup-diversity reliability diagnostic** — entropy of each player's teammate distribution;
  `create` is identified by lineup variation, so low-entropy players get a "context-dependent" flag
  on cards.
- **15. Playoffs** — fetched but filtered out everywhere (regular-season gate in `_load_stints`).
  ~5% more games; needs a playoff environment flag. Decide whether cards should include them at all.
- **16. Handedness / off-wing** — already fetched (`fetch-handedness`); could refine position
  offsets (LD/RD, off-wing one-timers). Marginal until the bigger signals land.
- **17. Empty-net / extra-attacker bucket** — 6v5/5v6 currently excluded; could become a third
  strength bucket like MA. Small.
- **18. Pass-tracking / NHL EDGE data (external)** — the only route to a real per-player
  creation-quality (`qcreate`), i.e. upgrading Playmaking's quality half. Mechanism: the last pass
  before EVERY shot is an "assist on a shot that wasn't a goal" — ~16× more creator labels — and the
  shot's xG is the outcome that identifies per-player setup danger. Everything upgrades in place:
  labels replace the latent-creator marginalization row-by-row (the `obs` mask in
  `fit_quality_creator` already supports mixed labeled/unlabeled rows, so PARTIAL game coverage —
  e.g. manually-tracked samples — works with zero structural change); the assist-credit anchor drops
  its `spg` IPW up-weight (and the sandwich concern) because labels are no longer goal-conditioned,
  which also eliminates roadmap #6; pass origin/type additionally improves the xG model itself
  (cross-ice/one-timer danger). Per the cards-are-qualities principle: tracking measurables (speed,
  shot speed) are EVIDENCE that sharpens latent qualities, not cards — don't ship a "Speed"
  attribute. Watch for public access (NHL EDGE, AllThreeZones-style tracked samples).

## Not recommended (considered, rejected for now)

- **Full Bayesian port (NumPyro/HMC):** likelihoods are already JAX, so it's a small port if exact
  posteriors are ever needed — but Laplace + EB gives ~90% at ~5% of the runtime. Revisit only with a
  concrete need (e.g. hierarchical qcreate with pass data).
- **Player embeddings / learned representations:** contradicts the product goal — parameters must
  stay interpretable.
- **Per-player parametric aging curves:** dominated by the RW drift states, which capture atypical
  trajectories without assuming they're age-shaped.
- **Per-player `qcreate` on current data:** structurally unidentifiable (~20 observed setups/player);
  resolved as position-level (A1). Only pass/tracking data (item 18) reopens this — or, partially,
  the feature-based γ compromise (item 13).

---

*Origin: model review + player-curves implementation session, July 2026. The review's report-only
findings (goal-selection bias, units asymmetry, optimizer budget) are folded into the items above or
documented in `generative_model.md`.*
