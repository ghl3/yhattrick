# 03 — IDs and the join contract

The NHL pbp, shiftcharts, and our xG have to be stitched into one per-event, on-ice-aware view.
This is the trickiest part of the data layer and is re-checked every run by the on-ice asserts.

## Game ids

The NHL 10-digit `gameId` is the only id used. Its game-type digits encode the type
(`…02…` = regular season, `…03…` = playoffs); `config.is_regular_season` reads them. The game list
for a season comes from the NHL schedule (the season's teams via the standings endpoint, then each
club's `club-schedule-season`, keeping `gameType == 2`). Within a game, a shot is identified by its
pbp `event_idx` — the join key from the shot table to the xG model.

## Time

- NHL shiftcharts `startTime`/`endTime` and pbp `timeInPeriod` are **period-elapsed `MM:SS`**.
- Everything is converted to game-elapsed seconds: `game_sec = (period-1)*1200 + mm:ss`
  (`clean.game_sec`). This single clock is what shots, shifts, and events are joined on — and since
  shots and shift boundaries now share it, boundary handling matters (see below).

## On-ice reconstruction

Shots come from the pbp (each carries its coordinates and `situationCode`). We recover the on-ice
identities from the shiftcharts: a player is on the ice at game-second `t` for a team iff one of
their shift intervals satisfies `start_g <= t < end_g`. A **stint** is a maximal interval between
consecutive shift boundaries, over which the on-ice set is constant (see
[04-processing.md](04-processing.md)).

**Shift source.** The JSON shiftcharts feed (`api.nhle.com/stats/rest`) stopped being populated
~spring 2025 (returns 0 rows for late-2024-25 + 2025-26). For those games `clean_shifts` falls back
to the **HTML TOI reports** (`TH`=home, `TV`=away; `html_shifts.py`), which list shifts by sweater
**number** + name — resolved to `playerId` via that game's pbp `rosterSpots` (exact per game, so
trades/number changes are handled). Both paths produce the identical shift schema. The parser is
validated by diffing it against the JSON feed on games that have both (matches to the second).

### Two data quirks handled

1. **Duplicate shift rows.** NHL shiftcharts sometimes emit a player's shift twice (e.g.
   shift #21 and #22 with identical times), which would double-count them on the ice. `clean.py`
   merges each player's overlapping/duplicate intervals (`_merge_player_intervals`) before any
   stint is built. This took the rate of illegal (>6-skater) stints from ~0.8% to ~0.01%.
2. **Events on a stint boundary.** A shot can share its game-second with the shift change that ends
   the stint. We use a single half-open rule everywhere — an event at time `t` belongs to the stint
   `[t0, t1)` that contains it (`bisect_right(starts, t) - 1`) — applied identically in `stints.py`
   (xGF attribution) and `export_games.py` (timeline display), so a shot's xG is always summed into
   the same stint it's shown under.

## Health gate (every run)

`stints.py` compares each shot's reconstructed on-ice skater counts to the pbp `situationCode`
skater counts and reports:

- **exact** — counts match.
- **within ±1** — off by one skater on one side; inherent ambiguity at a line-change second, tolerated.
- **large** — disagreement >1; should be ~0, investigated if it climbs (floor `ONICE_MATCH_FLOOR`).
- **overload** — a stint with >6 skaters (~0.01%, durations ~1–8s); flagged on the stint
  (`overload`) and surfaced in the site, safe to drop/downweight in modeling.

Per-shot match category (`exact`/`within1`/`large`) is stored on every shot for inspection. (This is
now a shifts-vs-pbp consistency check, both NHL sources, rather than an independent cross-source one.)
