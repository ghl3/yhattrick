# 06 — Modeling

The goal is a from-scratch **Wins Above Replacement (WAR)** rating for skaters, assembled from
isolated, per-60 impact components. Built on `data/processed/`. Authoritative design: `../PLAN.md`.

## Implemented: isolated-impact model (`impact.py`)

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
- **Fit**: weighted ridge (TOI weights), strength chosen by game-grouped cross-validation
  (`scikit-learn` `Ridge` + `GroupKFold`; sparse design via `scipy.sparse`).
- **Parameters** per player (each with a standard error and role TOI):

  | Strength | Offence | Defence |
  |---|---|---|
  | 5v5 | `ev_off` (xG/60 added) | `ev_def` (xG/60 allowed) |
  | 5v4 | `pp_off` (power-play offence) | `pk_def` (penalty-kill xG allowed) |

- **Uncertainty**: analytic ridge standard errors (`*_se`) from the coefficient covariance —
  intervals widen with low ice time and with collinearity (linemates who never separate).
- **Outputs**: `data/models/{ev,pp_pk}_<seasons>.parquet`. Runs on whatever seasons are processed,
  per-season or pooled (`--pool`).

```bash
uv run python -m hockeywar.player_onice_model --season 2021          # one season, both models
uv run python -m hockeywar.player_onice_model --pool --model pp_pk    # special teams, pooled seasons
```

Known limitation (single-season): always-together pairs are hard to separate (e.g. an elite D
pair can have its offence assigned to one partner). Pooling seasons and a future hierarchical
variant address this.

## Next (not yet built)

- **Finishing** — shrunk goals-minus-xG residual per player-season.
- **Penalties** — (drawn − taken) × a goal value.
- **GAR → WAR** — convert the impact coefficients to goals over ice time, subtract a
  position-specific replacement baseline, divide by goals-per-win; then percentiles.
- **Our own xG** — XGBoost on the shot features, validated against the borrowed `xGoal`, swapped
  in under the whole stack.
- A leaderboard + player-card route on the website surfacing these with their confidence.
