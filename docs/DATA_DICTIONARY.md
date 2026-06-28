# Data dictionary

Column-level reference for every stage. Seasons are named by starting year (`2021` = 2021-22).
All times are **game-elapsed seconds** unless noted. See [04-processing.md](04-processing.md)
for how each table is produced.

## raw/ (immutable, as downloaded)

- `moneypuck/shots_<season>.zip` — MoneyPuck per-shot CSV (137 cols; we keep the subset below).
- `moneypuck/skaters_<season>.csv` — MoneyPuck season box-score per player.
- `nhl/shiftcharts/<gameId>.json` — NHL shiftcharts API response (`data[]` of shifts).
- `nhl/pbp/<gameId>.json` — NHL play-by-play API response (`plays[]`, `rosterSpots[]`, teams).
- `nhl/players/<playerId>.json` — NHL player-landing API response; we use `shootsCatches` (handedness, for off-wing).

## interim/shots/&lt;season&gt;.parquet — one row per shot attempt

| Column | Type | Meaning |
|---|---|---|
| `shotID` | int | MoneyPuck shot id |
| `mp_game_id`, `nhl_game_id` | int | MoneyPuck (5-digit) and NHL (10-digit) game ids |
| `season`, `period` | int | season start year; period number |
| `game_seconds` | int | game-elapsed seconds of the shot |
| `event` | str | `SHOT` / `MISS` / `GOAL` |
| `goal` | 0/1 | was a goal |
| `isHomeTeam` | 0/1 | shooting team is home |
| `teamCode`, `homeTeamCode`, `awayTeamCode` | str | team abbreviations |
| `shooterPlayerId`, `shooterName` | int/str | shooter (rarely null in source data) |
| `goalieIdForShot`, `goalieNameForShot` | int/str | goalie faced |
| `xCord`, `yCord`, `xCordAdjusted`, `yCordAdjusted` | int | shot coordinates |
| `shotDistance`, `shotAngle` | float | geometry |
| `shotType` | str | WRIST / SNAP / SLAP / … |
| `homeSkatersOnIce`, `awaySkatersOnIce` | int | skater counts (no goalie) — the join check |
| `homeEmptyNet`, `awayEmptyNet` | 0/1 | empty net flags |
| `homeTeamGoals`, `awayTeamGoals` | int | score state before the shot |
| `shotRebound`, `shotRush` | 0/1 | context flags |
| `xGoal` | float | **borrowed** expected-goals value |

## interim/shifts/&lt;season&gt;.parquet — one row per merged shift interval

`nhl_game_id`, `player_id`, `player_name`, `team`, `team_id`, `period`, `start_g`, `end_g`,
`shift_number` (re-numbered after merge), `duration_s`.

## interim/events/&lt;season&gt;.parquet — one row per pbp event

`nhl_game_id`, `event_idx` (order), `period`, `time_g`, `type` (e.g. `faceoff`, `shot-on-goal`,
`goal`, `penalty`, `hit`, `giveaway`, `takeaway`, `stoppage`), `team`, `is_home`, `x`, `y`,
`zone`, `situation_code` (NHL strength code), `home_defending_side` (`left`/`right`, for shot
orientation), `primary_player_id` (scorer/shooter/hitter/…), `assist1_player_id`,
`assist2_player_id` (goals only), `shot_type`, `detail_key` (penalty type).

## interim/roster/&lt;season&gt;.parquet — one row per player-season

`player_id`, `season`, `player_name`, `position` (`C/L/R/D/G`), `number`, `team`.

## interim/box/&lt;season&gt;.parquet — one row per player-season-`game_type`

Descriptive box score from our own pbp + shifts (`game_type` ∈ {`regular`, `playoff`}, kept
separate): `player_id`, `name`, `pos`, `team`, `teams` (list), `gp`, `toi_s`, `toi_min`, and the
counting stats `g`, `a1`, `a2`, `points`, `sog`, `icf` (Corsi), `blocks`, `hits`, `takeaways`,
`giveaways`, `fo_won`, `fo_lost`, `pen_taken`, `pen_drawn`, `so_g`, `so_att` (shootout, separate).

Raw on-ice sums (regular season, same stint filter as the model — non-overload, ≥`MIN_STINT_S` —
so rates share a TOI base with the isolated coefficients; descriptive metrics in their own right
and a cross-check on the model): `ev_xgf_on`/`ev_xga_on` (5v5 xG for/against),
`ev_cf_on`/`ev_ca_on` (5v5 Corsi/shot-attempts for/against), `ev_onice_s` (5v5 on-ice seconds),
`pp_xgf_on` + `pp_onice_s` (power-play offence), `pk_xga_on` + `pk_onice_s` (penalty-kill xG
allowed). Stored as **sums** so seasons pool additively; `export_players` divides into per-60
rates (`ev_xgf60`, `ev_cf60`, …) and shares (`ev_xgshare`, `ev_cfshare`), each with a
within-position percentile.

## data/models/finishing_&lt;seasons&gt;.parquet — one row per shooter

Finishing (goals above expected, regressed), pooled over all in-play regular-season unblocked
shots, empty-net + shootout excluded: `player_id`, `name`, `pos`, `shots` (F, unblocked),
`ixg` (Σ xGoal), `goals` (G), `fin_per100` (= (G−ixG)/(F+k)·100, the shrunk headline),
`fin_per100_se`, `fin_goals` (= (G−ixG)·F/(F+k), shrunk total for WAR), `fin_goals_se`. Shrinkage
`k` and its EB diagnostics are logged to `logs/model/finishing_<seasons>.meta.json`.

## processed/xg/&lt;season&gt;.parquet — one row per modeled shot

Per-shot expected-goals predictions from `xg.py` (regular-season, goalie-present, unblocked shots):
`nhl_game_id`, `event_idx`, `time_g`, `shooter_id`, `goal`, `distance`, `abs_angle`, `shot_type`,
`strength_diff`, `rebound`, `rush`, and `xg` (calibrated goal probability). Model artifacts live in
`data/models/xg_booster.json` + `xg_isotonic.json`; metrics/reliability/importances and the MoneyPuck
head-to-head are in `logs/model/xg_<seasons>.meta.json`.

## processed/stints/&lt;season&gt;.parquet — one row per stint

Core: `nhl_game_id`, `stint_idx`, `start_g`, `end_g`, `duration_s`, `home_skaters` (list[int]),
`away_skaters` (list[int]), `home_goalie`, `away_goalie` (int|null), `home_n`, `away_n` (skater
counts), `strength` (e.g. `5v5`), `home_xgf`, `away_xgf` (summed borrowed xG), `overload`
(bool — illegal >6-skater stint, a source shift-timing artifact).

Context & volume (all from our own pbp, not MoneyPuck; stored for current and future modeling):
- `home_corsi`/`away_corsi` — shot attempts (SOG + missed + blocked + goal) by the shooting team.
- `home_fen`/`away_fen` — unblocked attempts (Fenwick); `home_sog`/`away_sog` — shots on goal.
- `home_lead` — home goals − away goals **before** the stint (score state).
- `start_zone`/`end_zone` — home-perspective faceoff zone (`O`/`D`/`N`, null if not a faceoff).
- `start_type` — `faceoff` or `fly` (on-the-fly change).

## processed/shots_onice/&lt;season&gt;.parquet — one row per shot, with on-ice context

`nhl_game_id`, `shotID`, `game_seconds`, `period`, `stint_idx`, `strength`, `shooter_id`,
`shooter`, `is_home`, `xGoal`, `event` (SHOT/MISS/GOAL), `goal`, `shot_type`, `distance`,
`angle`, `rebound`, `rush`, `x`, `y`, `home_skaters`, `away_skaters`, `home_goalie`,
`away_goalie`, `mp_home_n`, `mp_away_n` (MoneyPuck counts), `onice_match` ∈ {`exact`,
`within1`, `large`}.

## data/games/ (site JSON; synced to web/public/data/)

**`games.json`** — array of index rows: `game_id`, `season`, `date`, `home`, `away`,
`home_score`, `away_score`, `n_stints`, `n_shots`, `n_events`, `onice_exact` (fraction),
`large_mismatch`, `overload_stints`.

**`game/<id>.json`**:
- top: `game_id`, `date`, `home`, `away`, `home_score`, `away_score`.
- `totals`: `home_xgf`, `away_xgf`, `home_shots`, `away_shots`, scores.
- `players[]`: `id`, `name`, `pos`, `number`, `team`, `side`, `shifts`, `toi_s`, `g`, `a1`,
  `a2`, `pts`, `shots`, `xg`.
- `stints[]`: `idx`, `start`, `end`, `clock_start`, `clock_end`, `duration_s`, `strength`,
  `overload`, `home_skaters[]`/`away_skaters[]` (arrays of **player ids**),
  `home_goalie`/`away_goalie` (player id or `null` if pulled), `home_xgf`, `away_xgf`, and
  `events[]`: `t`, `clock`, `type`, `team`, `x`, `y`, `zone`, `player?`, `detail?`, and on shot
  events `xGoal?`, `onice_match?`, `shot_type?`, `distance?`, `angle?`, `rebound?`, `rush?`.
  On-ice ids are resolved to name/pos/number via the file's `players[]` (normalized to keep the
  per-game files small enough to serve from object storage).

## web/public/data/xg_model.json — xG model exploration page (`/xg`)

Written by `xg.py`. `seasons`, `n_shots`, `n_goals`; `metrics` (out-of-fold `auc`, `logloss`,
`brier`, `total_xg`, `total_goals`, `n`); `reliability[]` (`p_lo`, `p_hi`, `pred`, `obs`, `n`);
`importances[]` (`feature`, `gain`); `comparison` (`match_rate`, `n_matched`, and `ours`/`moneypuck`,
each a metrics block + `reliability[]`, on shots both models scored); `heatmap` (`x`, `y` grid axes,
`shot_types`, `strengths`, and `combos` keyed `"${shot_type}|${rebound}|${rush}|${strength}"` →
`[y][x]` predicted-xG grid for the offensive zone).
