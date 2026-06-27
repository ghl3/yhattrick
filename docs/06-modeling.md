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

**Cross-check on the site**: each modeled coefficient is shown next to the player's *raw* on-ice
per-60 rate (xGF/60 for offence, xGA/60 for defence), computed in `aggregates.py` from the stint
table. A large gap between raw and isolated means linemates/usage — not the player — drove the
on-ice number.

Known limitation: always-together pairs are still hard to separate (an elite D pair can have its
offence assigned to one partner). Pooling seasons helps; a future hierarchical variant (PP↔EV
pooling, player overall component) would help more.

## Next (not yet built)

- **Finishing** — shrunk goals-minus-xG residual per player-season.
- **Penalties** — (drawn − taken) × a goal value.
- **GAR → WAR** — convert the impact coefficients to goals over ice time, subtract a
  position-specific replacement baseline, divide by goals-per-win; then percentiles.
- **Our own xG** — XGBoost on the shot features, validated against the borrowed `xGoal`, swapped
  in under the whole stack.
- A leaderboard + player-card route on the website surfacing these with their confidence.
