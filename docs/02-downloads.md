# 02 — Downloads (the fetch stage)

Data fetching is a **first-class, repeatable pipeline step** — not something done by hand or to
a hidden location. Everything lands in the organized `pipeline/data/` tree and re-running only
fetches what's missing.

## One command

```bash
cd pipeline
make fetch                      # MoneyPuck zips/skaters + NHL shiftcharts+pbp for all seasons
# or, granularly:
make fetch-moneypuck            # just the MoneyPuck season files
make fetch-season SEASON=2024   # NHL per-game data for one season
```

Under the hood these call `python -m hockeywar.download {moneypuck|games|all}` (also exposed as
the `hockeywar-fetch` console script). Minimal parameters by design: seasons come from
`config.SEASONS`, and the **game list is derived from the data itself** (the unique `game_id`s
in each MoneyPuck shots file), so there is nothing to configure per game.

## Where the data goes (the data directory)

Bulk data lives at the **repo top level** (`/data`). The website does not read this tree; it
reads only the JSON the export step writes into `web/public/data/`.

```
data/                                   # top-level, shared
  raw/                                  # immutable; exactly as downloaded
    moneypuck/
      shots_<season>.zip                # ~20 MB each, per-shot rows + borrowed xGoal
      skaters_<season>.csv              # season box-score / validation totals
    nhl/
      shiftcharts/<gameId>.json         # on-ice shifts per game (~270 KB)
      pbp/<gameId>.json                 # play-by-play events per game (~130 KB)
  interim/                              # typed per-source tables (see 04-processing.md)
  processed/                            # joined/derived tables
```

`raw/` is **never mutated** by later stages — parsing/joining writes to `interim/` and
`processed/` (see [04-processing.md](04-processing.md)). One file per game keeps the fetch
resumable and lets us re-pull a single game if needed.

## Resumability & politeness

- Every fetch checks for the destination file first; an interrupted run resumes cleanly and a
  full re-run is nearly free.
- NHL API calls are throttled (`config.THROTTLE_SECONDS`, default 0.4 s) with retries
  (`config.MAX_RETRIES`). A clean `404` (e.g. a game with no data) is recorded and skipped, not
  retried forever.
- Volume: ~1,300 games/season × 2 files ≈ ~2,600 files/season; ~100 MB MoneyPuck zips total.

## Sources

See [01-data-sources.md](01-data-sources.md) for the exact URLs and why each source is used,
and [03-joins-and-ids.md](03-joins-and-ids.md) for how the MoneyPuck `game_id` maps to the NHL
`gameId` that names the per-game files.
