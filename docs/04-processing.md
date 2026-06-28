# 04 — Processing stages

Pipeline stages, each idempotent, moving data through the `data/` tree. Run them with `make all`
(or individually); `make fetch` populates `raw/` first. The xG model fits between `clean` and
`stints` so its per-shot `xg` is available for the join.

```
RAW  data/raw/ (immutable — NHL APIs)
  nhl/pbp/<gameId>.json · nhl/shiftcharts/<gameId>.json · nhl/players/<id>.json
        │
        │  clean.py   (parse + type per source; build the shot table from pbp)
        ▼
INTERIM  data/interim/
  events · shifts · roster · shots   (shots = unblocked, oriented geometry, situationCode strength)
        │
        │  xg.py   (features from the event stream; XGBoost + isotonic)
        ▼
  processed/xg   (per-shot xg, keyed by nhl_game_id + event_idx)
        ├──▶ data/models/xg_booster.json, xg_isotonic.json
        └──▶ web/public/data/xg_model.json          → /xg page
        │
        │   stints.py   joins  shots ⋈ processed/xg (game, event_idx)  +  shifts + events
        ▼
PROCESSED  data/processed/
  stints · shots_onice
        ├─ aggregates.py ──────────▶ interim/box                     (descriptive box score)
        ├─ player_onice_model.py ──▶ data/models/{ev, pp_pk}         (RAPM isolated impact)
        ├─ finishing.py  ⟵ processed/xg ──▶ data/models/finishing    (goals above expected)
        └─ export_games.py ────────▶ data/games/ ──▶ web/public/data/  (games.json, game/<id>.json)

export_players.py   ⟵ interim/box + data/models/{ev, pp_pk, finishing}
        ──▶ web/public/data/   (players.json, player/<id>.json)      → /players, /player, /teams
```

## clean.py — raw → interim (parse + type, per source)

Reads the per-game NHL JSON; writes one tidy parquet per source per season. No joining. Key work:
convert shift/event clocks to game-seconds; **merge duplicate shift intervals** per player; pull a
roster (id → name/pos/number) and the assist ids from goal events; and build the **shot table** from
pbp shot events (oriented geometry, rebound/rush, `situationCode` strength) — `event_idx` is the
per-shot id.

| Output | Grain | Notes |
|---|---|---|
| `interim/shots/<season>.parquet` | shot | unblocked shots from pbp: `event_idx`, `time_g`, geometry, `home_n`/`away_n`, `empty_net` |
| `interim/shifts/<season>.parquet` | shift interval | merged, sec-precise `start_g`/`end_g`, `duration_s` |
| `interim/events/<season>.parquet` | pbp event | type, team, players, coords, assists, `situation_code`, `home_defending_side` |
| `interim/roster/<season>.parquet` | player-season | name, position, number, team |

## stints.py — interim → processed (the join)

Builds stints per game from shift boundaries, attributes shots (and their model `xg`, joined from
`processed/xg` on `nhl_game_id`+`event_idx`) to stints, and runs the on-ice health asserts (see
[03-joins-and-ids.md](03-joins-and-ids.md)).

| Output | Grain | Notes |
|---|---|---|
| `processed/stints/<season>.parquet` | stint | on-ice skaters/goalies per side, `strength`, `home_xgf`/`away_xgf`, `overload` flag |
| `processed/shots_onice/<season>.parquet` | shot | on-ice players at the shot, model `xg`, `onice_match` ∈ {exact, within1, large} |

## export_games.py — processed + interim → data/games (site JSON)

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
