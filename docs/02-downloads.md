# 02 — Downloads (the fetch stage)

Data fetching is a **first-class, repeatable pipeline step** — not something done by hand or to
a hidden location. Everything lands in the organized `pipeline/data/` tree and re-running only
fetches what's missing.

## One command

```bash
cd pipeline
make fetch                      # NHL shiftcharts+pbp for all seasons + player handedness
# or, granularly:
make fetch-season SEASON=2024   # NHL per-game data for one season
make fetch-handedness           # player landing json (handedness)
make fetch-htmlshifts           # HTML TOI reports where the JSON shift feed is empty
```

Under the hood these call `python -m yhattrick.download {games|handedness|all}` (also exposed as
the `yhattrick-fetch` console script). Minimal parameters by design: seasons come from
`config.SEASONS`, and the **game list is derived from the NHL schedule** (the season's teams from
the standings endpoint, then each club's `club-schedule-season`, regular-season games only), so
there is nothing to configure per game.

## Where the data goes (the data directory)

Bulk data lives at the **repo top level** (`/data`). The website does not read this tree; it
reads only the JSON the export step writes into `web/public/data/`.

```
data/                                   # top-level, shared
  raw/                                  # immutable; exactly as downloaded
    nhl/
      shiftcharts/<gameId>.json         # on-ice shifts per game (~270 KB; empty for late-2024-25+)
      htmlshifts/T{H,V}<game6>.HTM      # HTML TOI reports — shift fallback where the JSON is empty
      pbp/<gameId>.json                 # play-by-play events per game (~130 KB)
      players/<playerId>.json           # player landing (handedness)
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
- Volume: ~1,300 games/season × 2 files ≈ ~2,600 files/season, plus ~1,700 player files.

## Sources

See [01-data-sources.md](01-data-sources.md) for the exact URLs and why each source is used,
and [03-joins-and-ids.md](03-joins-and-ids.md) for the game-id / clock conventions.
