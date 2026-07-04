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
replacement-level sensitivity study (the 8th–12th percentile band is a documented knob);
exact defender-quality treatment in the WAR swap (currently a multiplicative approximation);
uncertainty band on the trajectory chart (needs per-season state SEs exported).

### 5c. WAR audit fixes — ✅ SHIPPED (July 2026)
The 2026-07 WAR audit (now a permanent harness: `generative_war_audit.py`) found and fixed three
defects, all in the model / its honest evaluation — no post-processing:
- **Marginal conversion**: values + WAR convert shots at the model's own goals-per-shot —
  marginalized over the fitted Beta shot-quality distribution AND the creator classes (qcreate's
  reference is unassisted; skipping creator marginalization priced every shot as unassisted =
  +13% league E[GF]). Audited residual: +0.8%, zero correction factors.
- **Context-matched replacement**: PP/PK archetypes from the PP/PK GA/60 bands among PP/PK
  regulars (the EV-band players' shrunk-to-zero PP params had made replacement = league-average).
- **PP identification (the residual-sink bug)**: on fixed units, counts pin only the unit
  creation SUM; role-driven assists split it, sinking net-front players (Hyman pp_create −0.64,
  −23 PP wins). Fixed IN the fit: the MA anchor weight + create/def priors are now selected by
  the held-out calibration-slope sweep (γ_create 0.31 → 0.78, γ_def restored to 0.74, held-out
  PP deviance improved throughout) — model doc §7/§9. **Escalation if a future sweep stalls**: a
  per-player assist-style offset in the MA anchor (decouples "last passer" from "chance
  creator"), with its own validation round.

### 5d. The Hyman case (2026-07) — three TODOs from one player page
Zach Hyman 2025-26: 31 goals (= 31.3 xG, zero shooting luck) yet WAR −0.07 at the 1st percentile
and GA/60 −0.50. Investigating the clash surfaced three distinct issues:

- **TODO(1) Value-environment alignment — IN PROGRESS.** The value formulas evaluate players in
  the REFERENCE environment (teammates at effective create = 0), but the league-average teammate's
  effective create ≈ +0.17, so absolute card units run ~2× low (model F-mean scoring 0.30 vs
  actual ~0.66 goals/60). Fix in `player_values`: multiply by the model's own average-environment
  factor, computed from the fit's last-season rows (TOI-weighted mean of exp(Σcreate + Σdef +
  non-own ctx)); export as `value_env`; cards read it; audit asserts league model scoring ≈
  actual per-60. Percentiles, WAR, and κ unchanged — this is a units fix.
- **TODO(2) WAR percentile among season regulars — IN PROGRESS.** The WAR percentile pool
  includes ~370 part-timers whose WAR ≈ 0 by TOI alone, so a full-timer at −0.07 reads "1st
  percentile in hockey". Gate the WAR percentile on latest-season EV TOI (≥200 min); card copy
  says "among regulars".
- **TODO(3) Situational pricing layer — design AGREED 2026-07-03, not started.** League-wide
  15.5% of goals are unpriced (Hyman 2025-26: 8 of 31; true SHG ≈ 200/season). Transparency is
  SHIPPED (per-player `unpriced_goals` on the WAR card). The extension design, per the case-study
  evidence (docs/notes/2026-07-03): NOT a new skill bucket — a situational pricing layer that
  REUSES trusted skills, because the tail can't identify per-player parameters (SH volume repeats
  at only 0.33 y/y; 3v3 samples are tiny):
  1. **Skills reused, none added**: SH attack / 4v4 / 3v3 / extra-attacker → the player's EV
     shoot/create/def; 5v3 → his PP skills. Each situation class gets one fitted GLOBAL
     environment offset (a handful of well-identified scalars).
  2. **Conversion per situation**: real-goalie classes go through the existing quality/conversion
     stages; EMPTY-NET shots stay out of Stages 2–3 (no goalie ⇒ xG semantics break) and convert
     at an empirically measured league EN rate — the one deliberately "basic" piece, because an
     ENG is deployment + opportunity, not a modelable skill.
  3. **WAR gains a fourth component** (EV | PP | PK | Situational) via the same stint-swap
     engine, with per-class ΣE-vs-actual audit lines (zero-correction-factor standard). GA/60
     stays a 5v5 skill rate. Note: this prices EN value at EXPECTED value (league rate ×
     deployment), not realized goals — consistent with the rest of WAR; the transparency note
     stays so realized-vs-expected is always visible.
  Phases by risk: (1) 5v3/4v3 attack → MA bucket context flag (defender mask already handles
  short defenses — cheap); (2) the ATTACKER mask (3v3 has 2 teammates, extra-attacker has 5) —
  load-bearing: row builders + rate fit + variable-size Plackett–Luce anchor masking; synthetic
  fixtures then a holdout-validated refit; (3) WAR rows + audit + card component. ~1 focused day
  + one warm-started refit. Extends #17.
- **TODO(4) Shot `strength` labels are HOME-oriented — consumer hazard (model pools VERIFIED
  clean).** Found 2026-07-03: `shots_onice.strength` is "{home}v{away}", NOT shooter-relative
  (Draisaitl's away-PP goals read "4v5"). Verified: the model's quality/conversion pools are
  UNAFFECTED — `quality_creator_rows` requires the shooter's actual side to have 5 skaters (pool
  count 76,296 ≈ true-PP 75,763), and the rate buckets pick sides from skater counts. The hazard
  is downstream consumers classifying by raw label (the first unpriced-goals pass did exactly
  this). Documented in metrics.md + the data-sources memory; consider renaming the column or
  adding a shooter-relative `sit` column at the next stints regeneration.

### 5e. `create` identifiability for low-linemate-variation players (the Kapanen class)
**Status: open, root-cause understood, first experiment specified.** Motivating case: Oliver Kapanen
(MTL, 8482775) reads worst-in-league WAR (−1.32) despite 22 goals; a four-model comparison agreed his
defense is bad but split hard on EV-offense credit (E-H +6.5/+7.0 GAR and actual WAR +0.1; us and
hockeyviz harsh). His EV offense collapses because his `create` is fit at −0.109 — below both the
forward mean (+0.28) and replacement (+0.088). See `docs/notes/2026-07-03-assist-role-negative-result.md`
(the reverted fix) and `docs/notes/2026-07-03-war-case-studies.md`.

**Why `create` goes far-negative for these players (mechanism, confirmed).** `create` does double duty:
(i) a multiplicative lift on teammates' shot RATE (identified only by seeing a player with *varied*
linemates), and (ii) the credited-assister propensity in the conditional-logit anchor
`softmax([psi0, create[teammates]])` (`generative_model.py` ~1146). For a finisher glued to elite
distributors (Kapanen: 754 of 1102 5v5 min with Demidov), the rate channel carries almost no
independent info about him, so his WAR-relevant `create` is pinned almost entirely by the anchor —
i.e. by his **assist share within the unit**, which is puck-role, not chance creation. On his line's
55 on-ice 5v5 goals he is credited on 13 (24%; 5 A1 + 8 A2) while Demidov gets 21, Slafkovský 15,
Hutson 11. The anchor reads "few assists" as "low creation" and drags `create` below his linemates;
because WAR reads `create` absolutely in the stint swap, that within-unit relativity becomes an
absolute "suppresses offense" penalty. The with/without shift data confirms it is NOT real
suppression: his FORWARD linemates shoot the same or more with him (Demidov +0.38, Slafkovský −0.01
F/60); the only "shoots more without Kapanen" signal is his defensemen, and it scales with how much of
their time is spent AWAY from him (Dobson/Matheson ~88% away) — a deployment artifact, not a Kapanen
effect.

**The gauge finding (this is the key correction — read before re-trying anything).** The overall
`create` LEVEL is *exactly unidentified* by the likelihood, so **re-centering the ridge prior uniformly
is a mathematical no-op** (this matches the earlier "+0.28 shift across all groups" experiment):
- Rate term `eta = intercept + shoot + Σ create[teammates] + …` (~1135): at 5v5 there are exactly 4
  teammates per shot, so shifting *all* `create` by a constant c adds exactly 4c to every eta, which
  the strength intercept absorbs. Exactly flat.
- Anchor `softmax([psi0, create[4 teammates]])` (~1146): shift all four creates AND `psi0` by c and the
  softmax is exactly invariant. **`psi0` is a FREE fitted parameter** (`th[PS]`, split at ~1129), so it
  completes the gauge rather than breaking it.
- The ridge `Σ create²` centered at C (~1155) is the ONLY thing that fixes the level, setting
  mean(create)=C. Re-centering C just translates the whole cloud along the flat direction.
- WAR swaps player→replacement and reads `exp(Σ create)`; the swap depends on
  (create_player − create_replacement), a DIFFERENCE, invariant to the level. So a uniform re-center
  cannot change any player's WAR.

**Position-mean re-centering is NOT the same as uniform (and was never run).** Re-centering forwards
toward +0.28 and defensemen toward their (lower) mean is *not* a uniform shift, and the single global
gauge cannot absorb it: lines mix F and D in varying ratios, so an F-vs-D differential shift changes
different lines' etas differently and IS data-identified. So position-mean re-centering does change the
solution — but whether its non-gauge component lifts a collinear forward enough to overcome the
anchor's within-line pull is **unproven**; it needs a refit. The note's prescription ("re-center on the
position mean") is therefore *uniform-dead, position-untested*, not a guaranteed fix.

**Why the reverted `assist_role` attempt cannot simply be re-tuned.** The reverted implementation is
NOT in git history (it was `git checkout`-reverted); it survives as **dangling blob `d426cbe4`** (the
model) and `7bb4df89` (the tests) — recover with `git cat-file -p <blob>`. Verified spec:
`crx = cr + assist_role[player]` used ONLY in the anchor (~1157), rate term keeps bare `cr`; role prior
= unconstrained ridge to 0 (~1170), `sd` via `--assist-role-sd` (failed refit at 0.30, looser than
create's 0.12); the create prior stayed centered at 0. Documented failure modes: (1) didn't fire for
Kapanen (create −0.109→−0.098, role≈0) because create-prior-at-0 makes "low create" cheaper than
"average create + negative role"; (2) well-identified players moved too much and create's spread
collapsed 0.142→0.077 because the role prior was *looser* than create's and stole shared signal;
(3) fringe/low-sample players over-lifted from noise. **Self-defeating tension:** a *tighter* role prior
(the note's own fix for modes 2/3) makes negative role MORE expensive, worsening mode 1. With create
centered at 0 there is no single global `role_sd` that both helps Kapanen and stays stable — and
re-centering (the escape) is uniform-dead per above. So re-running this family is not expected to work.

**Candidate levers, ranked (only the first is not a re-run):**
1. **Global EV `anchor_scale` (do this first — cheap, and the one genuinely different lever).** Reuse
   the existing per-bucket anchor-weight plumbing (`ma_anchor_scale`, applied at ~1932) but expose it
   for the EV bucket, selected by held-out validation like the MA hyperparameters (item 5c / §7).
   Lowering it loosens the anchor's grip so a collinear finisher's relative `create` relaxes toward the
   population center instead of being dragged below by assist share. It does not add a per-player
   parameter and does not depend on the prior center. Cost is real: it discards genuine assist
   information for players who need it, so it is a global bias/variance trade, not a free win — decide
   by held-out likelihood, not by whether it fixes Kapanen. **First experiment:** sweep EV
   `anchor_scale ∈ {1.0, 0.5, 0.25}`; success = Kapanen moves toward average AND well-identified
   forwards stay put (corr(create old,new) ~0.99, spread preserved) AND held-out create-side tm-corr
   does not regress. The θ̂ checkpoint chain makes the sweep cheap.
2. **Better grounding data (item 18, pass-tracking) — the real fix.** The anchor is sparse (goals only)
   AND role-biased (passer credited, finisher not). Grounding `create` on the last pass before EVERY
   shot (~16× more labels, not goal/role-conditioned) removes both defects and drops the `spg` IPW.
   Data-limited: not in the NHL public feed. This is the principled resolution; everything else is a
   workaround.
3. **Position-mean create prior + a sum-to-zero / constrained role term.** Untested and not a pure
   no-op (unlike uniform), but carries the self-defeating tension above; only a refit settles it. Lower
   priority than (1).
4. **Honest fallback (item 14 + #5 bootstrap CIs).** Flag low-linemate-variation players
   ("context-dependent") and widen their WAR CI so an unidentified number is not shown as precise. Does
   not fix the point estimate; it stops overstating it. Cheapest correct thing if (1) and (2) stall.

**Calibration anchor for any fix.** Counterfactuals (holding all else at production): Kapanen create
+0.088 (replacement) → WAR −0.46; +0.199 (average) → WAR +0.12 ≈ E-H's +0.1. So the −1.32 rests
entirely on the create artifact, and any principled relaxation toward average lands near the
third-party consensus — a good sanity target, not a tuning objective. Validate every candidate on a
synthetic that checks LEVEL and SPREAD stability and well-identified invariance (the metric that masked
the first failure was focal-minus-population, which cancels exactly the level degeneracy above), plus
the real two-failure-mode harness (`scratchpad/validate_assist_role.py` pattern). Related: items 13
(feature-based `qcreate`), 14 (lineup-diversity diagnostic), 18 (pass tracking), and the 5c PP escalation
note (same anchor split, PP flavor).

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
~10 seasons (`DENSE_H_MAX` path) and do item 12 first if slow. **Before shipping a 10-season fit,
re-run the holdout + γ sweep on the extended window** (add a 2016–19 → 2020 target): the MA
hyperparameters were selected on the 2021–25 era and must be re-validated, not assumed (decided
2026-07 — the WAR-fix refit deliberately stayed on 2021–25 for this reason). The θ̂ checkpoint
chain makes the sweep iterations cheap.

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
