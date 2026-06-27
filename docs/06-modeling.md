# 06 — Modeling (deferred)

> Status: **not yet implemented.** Phase 1 builds the clean data layer and the inspection site;
> modeling comes next, on top of `data/processed/`. This doc records the intended approach; the
> authoritative design is `../PLAN.md`.

The goal is a from-scratch **Wins Above Replacement (WAR)** rating for skaters over five seasons,
assembled from isolated, per-60 impact components.

## Planned stages

1. **xG** — start on the **borrowed** MoneyPuck `xGoal` (already attached to every shot) to get
   the whole pipeline working end-to-end, then train our own calibrated XGBoost model on the
   shot features and swap it in, validating against the borrowed values.
2. **RAPM** — ridge regression over stints (the `processed/stints` table) with dual
   offense/defense player encodings, weighted by stint TOI, run per strength state, yielding
   per-player per-60 coefficients (EV offense/defense, PP, PK). Stints flagged `overload` or with
   `large` on-ice disagreement are dropped/downweighted.
3. **Finishing** — shrunk goals-minus-xG residual per player-season.
4. **Penalties** — (drawn − taken) × a goal value.
5. **GAR → WAR** — convert coefficients to goals over ice time, subtract a position-specific
   replacement baseline (roster-depth cutoff), divide by goals-per-win; then percentiles.

A later v2 explores a hierarchical/shared-effect RAPM to stabilize small special-teams samples.

## What modeling will consume

Everything it needs already exists in `data/processed/`:
- `stints/<season>.parquet` — the RAPM design substrate (on-ice players, strength, xGF/xGA, TOI).
- `shots_onice/<season>.parquet` — per-shot xG with on-ice context (xG model + finishing).
- `interim/events` + MoneyPuck `skaters` — penalties drawn/taken and validation totals.

The website will then gain leaderboard and player-card routes alongside the existing game view.
