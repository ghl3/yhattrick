# 06 — Modeling

The goal is a from-scratch **Wins Above Replacement (WAR)** rating for skaters, assembled from
isolated, per-60 impact components. Built on `data/processed/`. Authoritative design: `../PLAN.md`.

## Implemented: isolated-impact model (`player_onice_model.py`)

A regularized adjusted plus-minus that isolates each player's per-60 effect on the expected-goal
rate, adjusted for linemates and competition, fit separately per strength state.

- **Observations**: each stint contributes a team-attacking row — response = that team's xG per
  60 min; predictors = an OFFENCE indicator for each on-ice attacker and a DEFENCE indicator for
  each defender. Even strength (5v5) emits both perspectives; special teams (5v4) emits only the
  power-play team attacking, so PP offence and PK defence stay pure. Regular season only; stints
  flagged `overload` and sub-10s line-change stints are dropped.
- **Context covariates** (shared, non-per-player, oriented to the attacking team): home ice,
  offensive/defensive **zone start**, trailing/leading **score state**, per-**season** indicators
  (era drift), and **period**. These soak up deployment/score/era bias so the player coefficients
  better reflect skill; they're fit but not reported. (Competition and teammate quality are not
  added — the opponent/teammate player columns already adjust for them.)
- **Fit**: we solve the *penalized weighted normal equations* directly,
  `β = (ZᵀWZ + diag(pen))⁻¹ ZᵀWy`, by Cholesky (exact — no iterative-solver convergence doubt).
  Three choices make the regularization behave (they fixed an earlier "λ pins at the grid max,
  nothing shrinks, low-minute players top the leaderboard" failure):
  - **Weights normalized** so `Σw = n` (`w = dur/mean(dur)`), decoupling λ from absolute seconds
    and sample size — otherwise the weighted data term (∝ total seconds, ~10⁹) dwarfs any λ in a
    sane grid and no shrinkage happens.
  - **Only player columns are penalized**; the intercept and covariates are free (`pen = 0`), so
    context bias is absorbed instead of leaking into player coefficients.
  - **λ grid anchored to the data scale**: `λ = multiplier × median player-column curvature`
    (median nonzero diagonal of `ZᵀWZ`); a player shrinks ~`curv/(curv+λ)`, so the grid spans
    no-shrinkage → heavy-shrinkage regardless of units. Chosen by game-grouped CV (`GroupKFold`).
- **Response family** (`--family`): `gaussian` (per-60 rate, identity link — the default) or
  `tweedie` (log link, compound Poisson–Gamma — the correct likelihood for the zero-inflated,
  non-negative xG response, removing the per-60 variance blow-up of short stints). Tweedie
  coefficients (log-rate) are linearized to a per-60 xGF delta so the reported units match.
- **Parameters** per player (each with a standard error and role TOI):

  | Strength | Offence | Defence |
  |---|---|---|
  | 5v5 | `ev_off` (xG/60 added) | `ev_def` (xG/60 allowed) |
  | 5v4 | `pp_off` (power-play offence) | `pk_def` (penalty-kill xG allowed) |

- **Uncertainty**: analytic penalized-GLM standard errors (`*_se`) from the sandwich
  `A⁻¹(ZᵀWZ)A⁻¹·dispersion` — intervals widen with low ice time and with collinearity (linemates
  who never separate).
- **Full fit logging**: every fit writes `logs/model/<key>_<seasons>[_<family>].meta.json` (and
  appends to `logs/model_fits.jsonl`) with config/provenance, the **whole λ sweep** (per-fold CV,
  effective degrees of freedom, train fit, explained deviance), chosen-fit quality, numerical
  conditioning, and residual summary — enough to know and debug a fit after the fact.
- **Outputs**: `data/models/{ev,pp_pk}_<seasons>[_<family>].parquet`. Runs on whatever seasons are
  processed, per-season or pooled (`--pool`).

```bash
uv run python -m yhattrick.player_onice_model --season 2021             # one season, both models
uv run python -m yhattrick.player_onice_model --pool --model pp_pk       # special teams, pooled
uv run python -m yhattrick.player_onice_model --pool --family tweedie    # Tweedie GLM, pooled
```

Known limitation: always-together pairs are still hard to separate (an elite D pair can have its
offence assigned to one partner). Pooling seasons helps; a future hierarchical variant (PP↔EV
pooling, player overall component) would help more.

## Implemented: finishing (`finishing.py`)

Isolated on-ice xGF measures team chance *volume*, not the player's own shooting, so it can't credit
elite scorers. **Finishing** fills that gap: goals scored above the expected-goals value of the
player's own shots, `G − ixG` (ixG = Σ`xGoal` over his unblocked shots). It's low-repeatability, so
it's **regressed toward zero** by empirical-Bayes shrinkage — estimate the population spread of true
per-shot finishing talent τ² (method of moments) and shrink each player by `k = σ̄²/τ²` shots:

  `fin_per100 = (G−ixG)/(F+k)·100` (headline)   `fin_goals = (G−ixG)·F/(F+k)` (WAR-ready total)

Pooled over all in-play situations (regular season), excluding empty-net and shootout; analytic SE
→ a 95% CI. Output `data/models/finishing_<seasons>.parquet`; full meta (k, τ², σ̄², league G/ixG)
in `logs/model/`. Run: `uv run python -m yhattrick.finishing --pool`.

## Implemented: expected goals (`xg.py`)

The probability an unblocked shot becomes a goal, trained on the NHL play-by-play event stream
(`interim/events`). Every feature describes the **play itself** (geometry + pre-shot context + game
state), never who is on the ice: player/team/goalie quality and deployment are deliberately excluded
so the xG stays a pure chance-quality measure and isn't double-counted by the downstream RAPM /
finishing layer.

Features: distance & angle (oriented to the attacking net via `homeTeamDefendingSide`), shot type,
zone; pre-shot context (last-event type/coords, time/distance/speed since last event, time since the
last shot attempt, rebound, rush, angle change = goalie's lateral sweep, royal-road cross-slot,
possession continuity, since-faceoff, 15-second shot-attempt flurry); state (strength differential,
skater counts for 3v3/4v4/OT, score margin, period, game time, home/away); and **off-wing**
(handedness × shot side, from the NHL player-landing endpoint — a geometry term, not skill).

Model: XGBoost (`binary:logistic`), GroupKFold-by-game out-of-fold predictions, then an **isotonic
recalibration** on the OOF probabilities — this flattens the per-bin bias and makes the league total
xG equal goals. Validation (AUC / log-loss / Brier / reliability) is on the OOF predictions; the fit
also reports a head-to-head with MoneyPuck's `xGoal` on the same shots. Pooled over all seasons it is
calibrated to the goal total and on par on discrimination, using only play-level, double-count-safe
inputs.

Outputs `data/processed/xg/<season>.parquet` (per-shot predictions), `data/models/xg_booster.json` +
`xg_isotonic.json`, full meta in `logs/model/`, and `web/public/data/xg_model.json` for the
exploration page (`/xg`: shot-danger heatmap, calibration vs MoneyPuck, feature importances).
Run: `uv run python -m yhattrick.xg --pool` (needs `make fetch-handedness` for off-wing).

## The three per-player metric families (site)

Each player is described by three parallel families, shown as separate card sections and an index
view-toggle:

- **Isolated impact** — value *adjusted* for linemates & competition (the RAPM coefficients above).
- **On-ice rates** — the team's rates *while the player is on the ice*, unadjusted (xGF/60, xGA/60,
  xGF%, Corsi CF/60, CA/60, CF%), from `aggregates.py`. A big gap vs isolated impact means
  linemates/usage drove the on-ice number.
- **Individual rates** — the player's *own* on-puck production (all situations): Shots/60,
  xG/shot (shot quality), ixG/60, **Finishing/100**, Goals/60, Assists/60, Penalties drawn/taken
  per 60. Note `Shots/60 × xG/shot = ixG/60`.

## Next (not yet built)

- **Penalties value** — (drawn − taken) × a goal value (rates already shown).
- **GAR → WAR** — convert the impact coefficients to goals over ice time, subtract a
  position-specific replacement baseline, divide by goals-per-win; then percentiles.
- **Swap in our xG downstream** — our calibrated `xg` (above) now exists per shot; the remaining step
  is to make it the canonical value the stints → RAPM, finishing, on-ice rates, and game/player
  exports consume (currently still the borrowed MoneyPuck `xGoal`), then drop the MoneyPuck shots
  dependency entirely. This fixes the S-shaped per-bin bias that loads into `xG/shot`, finishing, and
  the RAPM.
- **Arena coordinate adjustment** — correct rink-scorer bias in shot x/y before computing distance/
  angle (its own per-rink calibration project).
- A leaderboard + player-card route on the website surfacing these with their confidence.
