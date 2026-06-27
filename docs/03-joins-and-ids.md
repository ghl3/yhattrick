# 03 — IDs and the join contract

Three sources have to be stitched into one per-event, on-ice-aware view. This is the trickiest
part of the data layer; it was verified before any code was written (game 2021020009 and
2024020500) and is re-checked every run by the on-ice asserts.

## game_id mapping

MoneyPuck uses a short 5-digit `game_id`; the NHL APIs use the 10-digit `gameId`. They relate
by inserting a `0` between the 4-digit season and the 5-digit MoneyPuck id:

```
nhl_id = int(f"{season}0{mp_game_id:05d}")     # MoneyPuck 20500 + 2024 -> 2024020500
mp_game_id = int(str(nhl_id)[5:])              # 2024020500 -> 20500
```

The MoneyPuck leading digit encodes game type (`2` = regular, `3` = playoffs). Implemented in
`config.mp_to_nhl_game_id` / `config.nhl_to_mp_game_id`. The unique `game_id`s in a season's
MoneyPuck shots file therefore enumerate every game to fetch from the NHL APIs.

## Time

- MoneyPuck `time` is **game-elapsed seconds** (0…3600 in regulation).
- NHL shiftcharts `startTime`/`endTime` and pbp `timeInPeriod` are **period-elapsed `MM:SS`**.
- Everything is converted to game-elapsed seconds: `game_sec = (period-1)*1200 + mm:ss`
  (`clean.game_sec`). This single clock is what shots, shifts, and events are joined on.

## On-ice reconstruction

MoneyPuck shots carry only skater *counts*, not identities. We recover identities from the
shiftcharts: a player is on the ice at game-second `t` for a team iff one of their shift
intervals satisfies `start_g <= t < end_g`. A **stint** is a maximal interval between
consecutive shift boundaries, over which the on-ice set is constant (see
[04-processing.md](04-processing.md)).

### Two data quirks handled

1. **Duplicate shift rows.** NHL shiftcharts sometimes emit a player's shift twice (e.g.
   shift #21 and #22 with identical times), which would double-count them on the ice. `clean.py`
   merges each player's overlapping/duplicate intervals (`_merge_player_intervals`) before any
   stint is built. This took the rate of illegal (>6-skater) stints from ~0.8% to ~0.01%.
2. **Shot-at-the-whistle.** A shot that ends a play shares its game-second with the following
   faceoff (which starts a new stint). `build_games` attaches such pre-whistle events
   (`shot/goal/missed/blocked/hit`) to the *prior* stint so a shooter always appears on the ice
   in the stint their shot is listed under.

## Health gate (every run)

`stints.py` compares each shot's reconstructed on-ice skater counts to MoneyPuck's
`home/awaySkatersOnIce` and reports:

- **exact** — counts match (~97%).
- **within ±1** — off by one skater on one side (~99.96% cumulative); inherent ambiguity at a
  line-change second, tolerated.
- **large** — disagreement >1 (~0.04%); should be ~0, investigated if it climbs.
- **overload** — a stint with >6 skaters (~0.01%, durations ~1–8s); flagged on the stint
  (`overload`) and surfaced in the site, safe to drop/downweight in modeling.

Per-shot match category (`exact`/`within1`/`large`) is stored on every shot for inspection.
