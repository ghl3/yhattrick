# 04 — Processing stages

Four pipeline stages, each idempotent, moving data left→right through the `data/` tree. Run them
with `make pipeline` (or individually); `make fetch` populates `raw/` first.

```
raw/ ──clean.py──▶ interim/ ──stints.py──▶ processed/ ──build_games.py──▶ data/games/ ──▶ web/public/data/
```

## clean.py — raw → interim (parse + type, per source)

Reads the MoneyPuck shots zip and the per-game NHL JSON; writes one tidy parquet per source per
season. No joining. Key work: select/type the MoneyPuck columns and resolve `nhl_game_id`;
convert shift/event clocks to game-seconds; **merge duplicate shift intervals** per player;
pull a roster (id → name/pos/number) and the assist ids from goal events.

| Output | Grain | Notes |
|---|---|---|
| `interim/shots/<season>.parquet` | shot | MoneyPuck subset + `nhl_game_id`, `game_seconds`, borrowed `xGoal` |
| `interim/shifts/<season>.parquet` | shift interval | merged, sec-precise `start_g`/`end_g`, `duration_s` |
| `interim/events/<season>.parquet` | pbp event | type, team, players, coords, assists, `situation_code` |
| `interim/roster/<season>.parquet` | player-season | name, position, number, team |

## stints.py — interim → processed (the join)

Builds stints per game from shift boundaries, attributes shots (and their borrowed `xGoal`) to
stints, and runs the on-ice health asserts (see [03-joins-and-ids.md](03-joins-and-ids.md)).

| Output | Grain | Notes |
|---|---|---|
| `processed/stints/<season>.parquet` | stint | on-ice skaters/goalies per side, `strength`, `home_xgf`/`away_xgf`, `overload` flag |
| `processed/shots_onice/<season>.parquet` | shot | on-ice players at the shot, `onice_match` ∈ {exact, within1, large} |

## build_games.py — processed + interim → data/games (site JSON)

Assembles the per-game timeline and aggregates, writes canonical JSON to `data/games/`, then
**syncs** a copy to `web/public/data/` (the only place the website reads). See
[05-website.md](05-website.md) for the shapes.

| Output | Contents |
|---|---|
| `data/games/games.json` | one index row per game (teams, score, counts, on-ice match %) |
| `data/games/game/<id>.json` | totals, per-player aggregates (shifts/TOI/G/A1/A2/Pts/shots/xG), ordered stints with on-ice + interleaved events |

Column-level detail for every file is in [DATA_DICTIONARY.md](DATA_DICTIONARY.md).

## Reproducibility

- `raw/` is immutable; re-running a later stage never touches it.
- Every stage is partitioned by season and safe to re-run; outputs are overwritten in place.
- `--season` / `--limit` flags scope a run for development. A bare `make pipeline` does all
  seasons present in `interim/`.
