# 04 — Processing stages

Pipeline stages, each idempotent, moving data through the `data/` tree. Run them with `make all`
(or individually); `make fetch` populates `raw/` first. The xG model fits between `clean` and
`stints` so its per-shot `xg` is available for the join.

The data splits into three kinds of table:

- **Raw** (`data/raw/`) — immutable NHL API dumps, one JSON per game / per player.
- **Facts** (`data/interim/`, `data/processed/`) — one row per *event we train on* (a shot, a
  stint): stable **keys** (`nhl_game_id`, `event_idx`, player ids, `venue_id`) plus **measures**
  (`xg`, `goal`, `duration_s`, on-ice lists, `assist1_id`). These are the tables the models read.
- **Dimensions** (`data/dimensions/`) — one row per *entity* (a player, a game, a venue): descriptive
  **attributes** (birthdate, position, game date, canonical venue name) stored **once** per entity and
  joined into the facts by key at model time, so an attribute like a birthdate lives in exactly one
  place.

Placement test: a property of *this event* → fact column; a property of an *entity* → dimension row.

```
RAW  data/raw/ (immutable — NHL APIs)
  nhl/pbp/<gameId>.json · nhl/shiftcharts/<gameId>.json · nhl/players/<id>.json
        │                                                        │
        │  clean.py  (parse + type per source; shot table from   │  dimensions.py
        ▼            pbp; roster + goal-assist ids from events)   ▼  (bio ← players JSON;
INTERIM  data/interim/                                       DIM  data/dimensions/   per-season ← roster;
  events · shifts · roster · shots ──────────────────┐          players · player_season   game facts +
        │                                            │          games · arenas          venues ← pbp)
        │  expected_goal_model.py                    │                │
        ▼  (XGBoost + isotonic)                      │                │  ── DIMENSIONS: one row per
  processed/xg  (per-shot xg)                        │                │     entity, joined by key ──
        │                                            │                │
        │  stints.py   shots ⋈ processed/xg (game, event_idx)         │
        │              + shifts + events + goal-assist ids            │
        ▼                                                            │
PROCESSED  data/processed/  ── FACTS: one row per event we train on ─┤
  stints · shots_onice                                              │
        ├─ aggregates.py ─────────▶ interim/box                     │  (descriptive box score)
        ├─ player_onice_model.py ─▶ data/models/{ev, pp_pk}         │  (RAPM isolated impact)
        ├─ shooting_model.py ─────▶ data/models/shooting_{finishing,goalie}
        ├─ goal_accounting.py ────▶ data/models/goal_accounting     │  (goals = xG + μ + fin + goalie)
        ├─ export_games.py ───────▶ data/games/ ──▶ web/public/data/ (games.json, game/<id>.json)
        │                                                            │
        └─ generative_model.py    FACTS (stints, shots_onice) ⋈ DIM (players, player_season,
              │                    games, arenas) by key ──▶ data/models/generative_model_*.json
              └─ generative_cards.py ──▶ data/models/gen_cards.json  (WAR + card qualities)

export_players.py  ⟵ interim/box + data/models/{ev, pp_pk, shooting_finishing, gen_cards}
        ──▶ web/public/data/   (players.json, player/<id>.json)      → /players, /player, /teams
```

## clean.py — raw → interim (parse + type, per source)

Reads the per-game NHL JSON; writes one tidy parquet per source per season. No joining. Key work:
convert shift/event clocks to game-seconds; **merge duplicate shift intervals** per player; pull a
roster (id → name/pos/number/team) and the assist ids from goal events; and build the **shot table**
from pbp shot events (oriented geometry, rebound/rush, `situationCode` strength) — `event_idx` is the
per-shot id.

| Output | Grain | Notes |
|---|---|---|
| `interim/shots/<season>.parquet` | shot | unblocked shots from pbp: `event_idx`, `time_g`, geometry, `home_n`/`away_n`, `empty_net` |
| `interim/shifts/<season>.parquet` | shift interval | merged, sec-precise `start_g`/`end_g`, `duration_s` |
| `interim/events/<season>.parquet` | pbp event | type, team, players, coords, **`assist1_player_id`/`assist2_player_id`**, `situation_code`, `home_defending_side` |
| `interim/roster/<season>.parquet` | player-season | name, position, number, team |

## dimensions.py — raw + interim → dimensions (`make dims`)

Builds the descriptive **dimension** tables; the models join their attributes into the facts by key.
Depends on the `RAW_PLAYERS` landing JSONs, `RAW_PBP` (game facts), and `interim/roster`. Covers the
seasons present in `interim/roster`.

| Output | Grain | Notes |
|---|---|---|
| `dimensions/players.parquet` | player | **static bio**: `name`, `birthdate`, `shoots`, `height_in`, `weight_lb`, `draft_year`, `draft_overall`. Source: `raw/nhl/players/<id>.json`. `birthdate` feeds the aging curves. |
| `dimensions/player_season.parquet` | player-season | `position`, `number`. **No team** — see below. Source: `interim/roster`. |
| `dimensions/games.parquet` | game | `date`, `season`, `game_type` (2 = regular, 3 = playoff), `home_team`, `away_team`, `venue_id` (FK → arenas). `game_type == 2` selects the regular season. |
| `dimensions/arenas.parquet` | venue | `venue_id`, `canonical_name`, `raw_names[]`, `active_seasons[]`. Holds the venue alias map: sponsor renames (e.g. FLA Live → Amerant Bank) collapse to one building so its scorer-bias states stay one chain, while real building moves stay split. |

**Trades are first-class, not a special case.** A player's team is a property of **(player, game)** —
his team in game *G* is that game's `home_team`/`away_team` depending on which on-ice side he is on
(already a fact in the stint's on-ice lists). So team-per-game is a **join** (`games` + the player's
side), and a mid-season trade falls out automatically (games 1–40 → team A, 41–82 → team B). A single
`player_season.team` would be *wrong* for a traded player, so team deliberately does **not** live there.
"Current team" = his latest game; "teams played" = distinct over his games.

## stints.py — interim → processed (the join)

Builds stints per game from shift boundaries, attributes shots (and their model `xg`, joined from
`processed/xg` on `nhl_game_id`+`event_idx`) to stints, and runs the on-ice health asserts (see
[joins-and-ids.md](joins-and-ids.md)). Goal rows also carry the credited **assist ids** (`assist1_id`,
`assist2_id`, from `interim/events`), which ground the creator/playmaking anchor.

| Output | Grain | Notes |
|---|---|---|
| `processed/stints/<season>.parquet` | stint | on-ice skaters/goalies per side, `strength`, `home_xgf`/`away_xgf`, `overload` flag |
| `processed/shots_onice/<season>.parquet` | shot | on-ice players at the shot, model `xg`, `assist1_id`/`assist2_id` (goals), `onice_match` ∈ {exact, within1, large} |

## export_games.py — processed + interim → data/games (site JSON)

Assembles the per-game timeline and aggregates, writes canonical JSON to `data/games/`, then
**syncs** a copy to `web/public/data/` (the only place the website reads). See
[website.md](website.md) for the shapes.

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
- Dimensions are cheap to rebuild (`make dims`, a few seconds); rerun after `clean-data` or when new
  `RAW_PLAYERS` / games land.
