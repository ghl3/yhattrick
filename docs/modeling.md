# 06 — Modeling

The goal is a from-scratch **Wins Above Replacement (WAR)** rating for skaters (and goalies),
assembled from isolated components that **add up to goals**. Built on `data/processed/`. Authoritative
design: `../PLAN.md`.

## The additive theory of goals (organizing principle)

Every goal is **chance × conversion**. We hold the whole modeling system together with one
**generative model of a stint**:

```
1.  A stint = fixed time, with off players + def players + context (home, zone, score, …)
2.  → that lineup generates a shot process: some number of shots, each with an xG
        (chance quality)                                          ← STAGE 1: creation
3.  Each shot's xG is adjusted by the shooter (finishing ↑) and the goalie (saving ↓):
        goal = xG + μ + finishing + goalie + noise                ← STAGE 2: conversion
4.  Sum over shots → goals.
```

This yields one **fully additive identity** that reconciles to actual goals:

```
Goals_for ≈  Σ xG          +   μ        +  finishing        −  opponent goalie saves
             └ Stage 1 ────┘   league      └ Stage 2: shooters & goalies split (goal − xG) ┘
             on-ice SKATERS    baseline
             (offense creates,
              defense suppresses)
```

- **Stage 1 — creation & suppression** distributes each stint's `Σ xG` across the on-ice skaters:
  forwards and defensemen, offense (`ev_off`) and defense (`ev_def`). This is the RAPM model
  (`player_onice_model.py`), unchanged. **Defensemen live here** — their job is to shrink the
  opponent's `Σ xG allowed` before a shot is taken.
- **Stage 2 — conversion** splits the per-shot residual `(goal − xG)` between the two actors present
  at the shot: the **shooter** (finishing) and the **goalie** (saving). This is the shooting model
  (`shooting_model.py`).
- **μ** is a tiny league baseline (see below), attributed to no player.

**Two readings, same model.** Plug in *expected* shots → expected goals for any lineup. Plug in the
*realized* shots → the accounting identity that reconciles to the goals that actually happened
(`goal_accounting.py` checks this).

**What we fit vs. the theory.** We do **not** fit one joint hierarchical likelihood. We estimate the
two stages **separately** — Stage 1 is a per-stint *rate* model, Stage 2 a per-shot *conversion*
model (different sampling units) — each consuming the same calibrated per-shot xG. The generative
model above is the theory the two fits *compose into*, not the estimator we run.

| Actor | Stage | Mechanism | Module |
|---|---|---|---|
| Forwards & D — offense | 1 (creation) | raise on-ice xG **for** (`ev_off`) | `player_onice_model` |
| Defensemen & all skaters — defense | 1 (suppression) | lower opp on-ice xG **against** (`ev_def`) | `player_onice_model` |
| Shooter | 2 (conversion) | finishing — convert above/below xG | `shooting_model` |
| Goalie | 2 (conversion) | saving — stop above/below xG | `shooting_model` |

## Stage 1 — isolated-impact model (`player_onice_model.py`)

A regularized adjusted plus-minus that isolates each player's per-60 effect on the expected-goal
rate, adjusted for linemates and competition, fit separately per strength state. This is the
**creation/suppression** layer of the additive theory: it decomposes `Σ xG` among on-ice skaters.

- **Observations**: each stint contributes a team-attacking row — response = that team's xG per
  60 min; predictors = an OFFENCE indicator for each on-ice attacker and a DEFENCE indicator for
  each defender. Even strength (5v5) emits both perspectives; special teams (5v4) emits only the
  power-play team attacking, so PP offence and PK defence stay pure. Regular season only; stints
  flagged `overload` and sub-10s line-change stints are dropped.
- **Context covariates** (shared, non-per-player, oriented to the attacking team): home ice,
  offensive/defensive **zone start**, trailing/leading **score state**, per-**season** indicators
  (era drift), and **period**. These soak up deployment/score/era bias so the player coefficients
  better reflect skill; they're fit but not reported.
- **Fit**: penalized weighted normal equations `β = (ZᵀWZ + diag(pen))⁻¹ ZᵀWy` by Cholesky.
  Weights normalized so `Σw = n`; only player columns penalized (intercept + covariates free);
  λ grid anchored to the median player-column curvature, chosen by game-grouped CV.
- **Response family** (`--family`): `gaussian` (per-60 rate, default) or `tweedie` (log link,
  compound Poisson–Gamma) linearized back to a per-60 xGF delta.
- **Parameters** per player (each with a standard error and role TOI):

  | Strength | Offence | Defence |
  |---|---|---|
  | 5v5 | `ev_off` (xG/60 added) | `ev_def` (xG/60 allowed) |
  | 5v4 | `pp_off` (power-play offence) | `pk_def` (penalty-kill xG allowed) |

- **Outputs**: `data/models/{ev,pp_pk}_<seasons>[_<family>].parquet`; full λ-sweep + diagnostics in
  `logs/model/`.

```bash
uv run python -m yhattrick.models.player_onice_model --pool                  # both models, pooled
uv run python -m yhattrick.models.player_onice_model --pool --family tweedie # Tweedie GLM
```

Known limitation: always-together pairs are still hard to separate; pooling seasons helps.

## The expected-goals baseline (`expected_goal_model.py`)

The probability an unblocked shot becomes a goal, trained on the NHL play-by-play event stream
(`interim/events`). Every feature describes the **play itself** (geometry + pre-shot context + game
state), **never who is on the ice**: player/team/goalie identity is deliberately excluded so xG is a
pure chance-quality measure and isn't double-counted by the RAPM / shooting layers.

Features: distance & angle (oriented to the attacking net), shot type, zone; pre-shot context
(last-event type/coords, time/distance/speed since last event, rebound, rush, angle change,
royal-road cross-slot, since-faceoff, 15s shot flurry); **state — strength differential, skater
counts (so power plays / 3v3 / odd-man are priced in), score margin, period, game time, home/away**;
and off-wing (handedness × shot side). Empty-net (goalie absent) and <3-skater situations are
excluded (`xg = NaN`), so they never enter finishing/GSAx.

Model: XGBoost (`binary:logistic`), GroupKFold-by-game out-of-fold predictions, then **isotonic
recalibration** so the league total xG equals goals. This `xg` is the canonical expected-goals value
across the pipeline — the on-ice xGF the RAPM regresses on, the baseline the shooting model splits,
and the per-shot value in the game timelines. **Because strength is an xG feature, the conversion
layer can pool all situations** (xG already removed the man-advantage's effect on chance quality;
see Stage 2).

Outputs `data/processed/xg/<season>.parquet` (joined downstream on `nhl_game_id`+`event_idx`),
`data/models/xg_booster.json` + `xg_isotonic.json`, and `web/public/data/xg_model.json` (the `/xg`
page). Run: `uv run python -m yhattrick.models.expected_goal_model --pool` (after `clean-data`,
before `stints`; needs `make fetch-handedness` for off-wing).

## Stage 2 — the shooting model (`shooting_model.py`)

A shot is a duel between the **shooter** (who can finish above the league baseline) and the **goalie**
(who can save below it). The shooting model attributes the per-shot residual `(goal − xG)` to the two
of them **jointly**, in one penalized **crossed ridge** regression:

```
goal_i − xG_i  =  μ  +  α_shooter(i)  +  γ_goalie(i)  +  e_i          (ridge on α, γ)
```

`α` is finishing (goals/shot above expected, α>0 = good finisher); `γ` is the goalie (goals/shot
above expected *allowed*, γ<0 = good goalie). This **replaces** the old separate `finishing.py` and
`goalie.py`: each of those modelled `(goal − xG)` ignoring the other actor, so the same residual was
**double-counted** (full credit to the shooter *and* full blame to the goalie). Fitting them together
**splits** it — finishing is now adjusted for the goalie faced and vice-versa.

**Why linear, not logistic.** The model is **additive on the goals scale** by construction: the
fitted residual is `μ + α + γ`, so `finishing_i = α` and `goalie_i = γ` sum to it exactly, the chance
baseline stays *raw xG* (so it ties straight to Stage 1), and with a free intercept the least-squares
residuals sum to zero — giving the exact identity `goals = ΣxG + Σμ + Σfinishing + Σgoalie`. A
logistic offset model is a proper binary model but its nonlinearity (Jensen) recalibrates the
baseline ~1.5% off raw xG and inflates the finishing scale; we chose exact additivity instead.

**Shrinkage = the same empirical Bayes.** The ridge penalty *is* shrinkage toward zero (a
league-average shooter / goalie — the natural null). For an unweighted dummy with `n` shots, ridge
`λ` shrinks the effect by `n/(n+λ)`; we set `λ = k` (the EB shot-count constant `k = σ̄²/τ²`,
method-of-moments, exactly as the old models), so a player with **no crossing reduces exactly** to
the old shrunk metric `(G − ixG)·n/(n+k)`. We kept the trusted shrinkage and changed only the (now
joint, mutually adjusted) estimation.

**The intercept μ — why it exists, why it isn't zero, where it goes.** The design is rank-deficient
(every shot has one shooter and one goalie, so the intercept column = Σ shooter columns = Σ goalie
columns); a *free* intercept resolves that and its normal equation forces `Σ(goal − xG − μ − α − γ)
= 0`, which is **what makes the identity reconcile to actual goals to the penny**. μ is *not* zero
even though xG is calibrated to ~0.1%: shrinkage pulls every α, γ toward zero, so the shrunk effects
no longer sum to the raw residual, and μ absorbs exactly that **shrinkage leakage** (plus a
second-order selection effect — xG is calibrated to the shot-weighted population, in which good
finishers shoot more, so an average *player* sits a hair below xG). So μ is the *accounting footprint
of the shrinkage*, not a calibration patch — drop the regularization and μ → ~0, but the per-player
estimates become overfit noise. μ is tiny (≈ −0.0007 goals/shot, −0.07 / 100 shots) and is
**attributed to no player** — a league-baseline line in the identity, deliberately not split into
finishing/goalie (it's not skill, and an even split would be arbitrary and change no ranking).

**Crossed ≠ interaction.** "Crossed" means both a shooter term *and* a goalie term coexist and are
fit together (the model is purely additive `μ + α + γ`, no α·γ product). The dense bipartite overlap
(every goalie faces ~every shooter) is what makes α and γ *separately identifiable*. We do **not**
include a per-pair shooter×goalie **interaction** `δ_{s,g}`: ~2 shots per pair → pure noise, not
repeatable, and not part of an additive player rating.

**Power plays / odd-man.** Handled in **xG**, not here — strength is an xG feature, so a 5v4 shot
already carries higher xG. The shooting model works on the residual, which is therefore already
strength-adjusted, so α/γ are pooled across all situations (one finishing skill, one save skill).
This is the clean asymmetry with Stage 1: *creation* must be split by strength (generation rates
differ enormously), *conversion* need not be (xG normalized the chance). (Descriptive EV/PK/PP
splits still exist on the goalie side in `goalie_aggregates.py`.)

**Outputs** (same columns the site already read, so export is a drop-in swap):
- finishing: `data/models/shooting_finishing_<seasons>.parquet` — `fin_per100` (= α·100), `fin_goals`
  (= α·shots), SEs, plus `shots/ixg/goals`.
- goalie: `data/models/shooting_goalie_<seasons>.parquet` — `gsax_per100` (= −γ·100), `gsax_saved`
  (= −γ·sa), SEs, plus `sa/xga/ga`.
Run: `uv run python -m yhattrick.models.shooting_model --pool`.

## Reconciliation — goal accounting (`goal_accounting.py`)

Rolls the per-shot decomposition `goal ≈ xG + μ + finishing + goalie` up to **team-season** and
**league** and checks it. League-wide the identity is exact (`goals = Σxg + Σμ + Σfinishing +
Σgoalie` to rounding). At team-season level the gap `actual − reconstructed` is the team's
**unmodeled finishing / goaltending luck** — a single season's deviation from the players' multi-year
effects (≈ 5% on average, which is real shooting variance, not model error). The chance term is *raw*
xG — the same quantity Stage 1 distributes among the on-ice skaters — so this closes the loop between
the two stages. Output `data/models/goal_accounting_<seasons>.parquet` + league summary in
`logs/model/`. Run: `uv run python -m yhattrick.models.goal_accounting`.

## The three per-player metric families (site)

- **Isolated impact** — value *adjusted* for linemates & competition (Stage-1 RAPM coefficients).
- **On-ice rates** — the team's rates *while the player is on the ice*, unadjusted (xGF/60, xGA/60,
  xGF%, Corsi), from `aggregates.py`.
- **Individual rates** — the player's *own* on-puck production (all situations): Shots/60, xG/shot,
  ixG/60, **Finishing/100** (now goalie-adjusted, from the shooting model), Goals/60, Assists/60,
  Penalties drawn/taken per 60. Note `Shots/60 × xG/shot = ixG/60`.

Goalies get their own page: shrunk **GSAx/100** (shooter-adjusted, from the shooting model) plus
descriptive Sv%/GAA and danger/situation/shot-type splits (`goalie_aggregates.py`).

## Player value — goals attributed (`export_players.value_table`)

The headline value layer synthesizes the component fits into **goals attributed**: the goals we credit
each player with creating, allowing, finishing, and drawing/taking, such that the shares reconcile to
actual goals leaguewide. It is built **only** from the existing linear fits — no new model.

The key move is **absorbing the RAPM intercept**. Stage 1 fits `xGF/60 = intercept + Σ ev_off + Σ ev_def
+ ctx` with the intercept *free*, so the exported `ev_off`/`ev_def` are *deviations* from the league
baseline (they read as "vs. average"). Folding each player's share of that baseline back in —
`baseline ÷ (on-ice skaters) + coef` — turns the deviation into an **absolute attributed share** (≥0)
that sums, across the on-ice skaters, back to the stint's xG:

```
5v5 created /60:  create60    = ev_off_base/5 + ev_off      his share of 5v5 xG created
5v5 allowed /60:  allow60     = ev_off_base/5 + ev_def      his share of 5v5 xG allowed (lower better)
5v5 NET /60:      ev5_net60   = create60 + fin5 − allow60   = ev_off + fin5 − ev_def (baseline cancels)
PP created /60:   pp_create60 = pp_off_base/5 + pp_off      (5 PP skaters share the PP baseline)
PK allowed /60:   pk_allow60  = pp_off_base/4 + pk_def      (4 PK skaters share it; lower better)
```

`ev_off_base`/`pp_off_base` are the model intercepts (`baseline_xgf60`), carried on the coefficient
frame. The baseline is split equally among that side's on-ice skaters; this is exactly the split that
makes the shares reconcile to Σ xG (the per-player deviations already encode F-vs-D differences).

On the site, each offense share is **presented split** into *scoring* (his own shots) and *playmaking*
(creation for teammates) by a **proportional partition**: `φ = own ixG ÷ team on-ice xGF`, then
`scoring = φ·create60 + finishing`, `playmaking = (1−φ)·create60`. Both are ≥0 and sum to
`create60 + finishing`, so the net is unchanged. (Subtracting raw own ixG instead would be zero-sum and
go negative for volume shooters — see [`metrics.md`](metrics.md).)

**Actual goals** (season totals): scale each per-60 share by *real* role-TOI and sum across situations:

```
g_created   = create60·(T5/60) + pp_create60·(Tpp/60)
g_allowed   = allow60·(T5/60)  + pk_allow60·(Tpk/60)
g_fin       = fin_goals                                   finishing (goals − xG on his shots)
g_pen       = drawn·V − taken·V                           net penalty goals
g_net       = g_created + g_fin − g_allowed + g_pen
gnet_pg     = g_net / GP        ← TOP-LINE METRIC: Net Goals Added per game
```

At 5v5 the baseline cancels in `g_created − g_allowed` (same slice on offense and defense, equal TOI), so
the 5v5 contribution to the net is the pure marginal differential `ev_off − ev_def`; only the
specialist-role special-teams shares carry a baseline into the net. `V` ≈ 0.14 is the net goal value of a
drawn minor, derived from our own data (`penalty_value`: league 5v4 GF − 4v5 GA per drawn penalty).

**Why no double-count.** `create60` is *on-ice* xG — it already contains the goal-value of his own shot
volume × quality (his individual xG). So `ixG`/`shots60`/`xg_per_shot` are **not** added; only
**finishing** (the goals-above-xG residual) is new. For his own shots, `create60` credits the xG and
finishing credits (goals − xG) → exactly full goal credit, no more. (There is deliberately no combined
"scoring = ixG + finishing" metric — it would double-count the own-shot xG already inside `create60`.)

**Reader-facing definitions** — what each player-page card means, the worked baseline example, and why a
net centered at 0 is *not* "vs average" — live in **[`docs/metrics.md`](metrics.md)**.

**Caveats.** Attribution is approximate per-stint (ridge shrinkage, tiny `μ`) but calibrated in
aggregate; a roster's `g_net` does **not** sum to the team's goal differential. **Goalie analog:** no
offense; `g_net = g_prevented = gsax_saved` (shrunk GSAx above expected), `gprev60 = GSAx/60`.

## Alternative approach (experimental): generative model (the `generative_*` modules)

The additive linear model above is our **main model** — and deliberately so (see "Why the linear model
stays primary" below). The generative model (the `models/generative_*` modules) is an **experimental
alternative** we explored: a **marked Poisson process with Bernoulli thinning** that, given a stint's
lineup/context/length, *draws* the number of chances, each chance's xG, and each outcome, so it can be
**simulated** as well as fit. It is **not wired into the site**; it's a proof of concept. The full
likelihood (all parameters labelled) is specified in [`generative_model.md`](generative_model.md) and
realised in `models/generative_likelihood.py`.

Three layers, fit independently (the likelihood factorizes — counts, qualities, and outcomes are all
observed, with disjoint parameter blocks):
- **rate** — shots per stint side ~ Poisson(λ·t), `log λ = μ_λ + Σ offense(a_p) + Σ defense(d_p) + ctx`;
- **quality** — each shot's xG via a mean-calibrated **fractional logistic** `E[xG] = σ(μ_q + Σ u_p + Σ w_p + ctx)`
  (simulated from a Beta with that mean);
- **conversion** — the production shooting model (`α`, `γ`, `μ_c`), reused.

Fit by **MAP / penalized MLE in JAX** (autodiff gradients + scipy L-BFGS-B; ridge = the EB prior),
**not** MCMC — point estimates with **Hessian-based (Laplace) ±95% CIs** (`SE = sqrt(diag(H⁻¹))`,
`H = XᵀWX + penalty`). Notably it splits what RAPM estimates as one xG/60 number into **volume** (`a_p`)
× **quality** (`u_p`), distinguishing high-volume from high-danger creators.

A posterior-predictive check (re-simulate every real stint) confirms shots and goals reconcile to actuals.
It also surfaced that **real shot counts are overdispersed vs Poisson** (variance > mean), so the count
layer is **configurable** (`--count poisson|nb`): the **negative-binomial** option adds one global
dispersion `r` (the per-stint Gamma rate-multiplier is integrated out, so the three layers still fit
independently) and matches the data — e.g. on 2024 it lifts the simulated per-side count variance from
0.119 (Poisson) to 0.139 (≈ actual 0.136), fixes the zero-fraction, and tightens the goals reconciliation
(+0.7% vs +3.5%). The only cost is that Poisson's *exact* Σμ = Σcount becomes approximate (off by ~2 shots
in 89k here). Going further — a Cox process with a *shared* latent intensity driving both volume and
quality — would couple the rate and quality fits (the separability tradeoff discussed above). Run:
`uv run --group experimental python -m yhattrick.models.generative_model --count nb` (needs the JAX
`experimental` dep group; `make generative-model`).

**Why the linear model stays primary.** The generative model is more *faithful* (goals are produced
multiplicatively — rate × quality × conversion — and it yields full predictive distributions and a
volume-vs-quality split), but that multiplicative composition is exactly what makes **per-player
attribution harder**: a player's goals/60 then depends on his linemates and context, so there is no single
context-free number — you have to average over deployment or Shapley-decompose the product into additive
shares. The **additive linear model is additive by construction**: each player's contribution is one
context-free number and they *sum* to the team total (`goals = ΣxG + μ + finishing + goalie`). For the
ultimate goal — clean, independent, summable per-player WAR components — that additivity is the feature we
want, so the linear model is the workhorse. The generative model's value is as a complementary lens
(realism, simulation/uncertainty, the volume/quality decomposition) rather than the attribution engine.

## Next (not yet built)

- **Above replacement** — the value layer above is *goals attributed* (net centered at break-even);
  subtract a replacement-level baseline (a constant per-60 shift by position) for goals-above-replacement.
- **GAR → WAR** — convert the additive goal components (creation, finishing, goalie saves) to goals
  over replacement, divide by goals-per-win; then percentiles. The additive identity makes the
  components directly summable into a single goals-based value.
- **Arena coordinate adjustment** — correct rink-scorer bias in shot x/y before distance/angle.
- A possible future **joint hierarchical** fit (Stages 1+2 in one likelihood) and a shooter×strength
  interaction, if the pooled assumptions ever prove too coarse.
