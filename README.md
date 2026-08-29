# Hockey WAR

A from-scratch NHL **Wins Above Replacement** model and a data-inspection website. Two halves:

- **`pipeline/`** — a Python (`uv`) data + modeling pipeline that downloads NHL data, joins it
  into clean tables, and (in a later phase) fits the WAR model.
- **`web/`** — a React + Vite + TypeScript site for inspecting the data, centered on a per-game
  stint & event timeline.

Bulk data lives under **`data/`** (`raw → interim → processed → games`); the website reads the
small JSON the pipeline writes to `web/public/data/`.

## Status

**Phase 1 — clean data layer + inspection site.** Modeling (xG → RAPM → GAR → WAR) is the next
phase; see [`PLAN.md`](PLAN.md) for the full design and [`docs/`](docs/) for how the
implementation actually works.

## Quick start

```bash
# 1. data  (MoneyPuck shots + NHL shiftcharts/pbp -> data/raw, resumable)
make fetch

# 2. process  (raw -> interim -> processed -> data/games -> web/public/data)
make pipeline

# 3. site
make web-dev          # http://localhost:5173

# in season: one-shot daily refresh (new games -> models -> site); see docs/updating.md
make update
```

Requirements: `uv`, Node 22+. Data sources and the join contract are documented in
[`docs/data-sources.md`](docs/data-sources.md) and
[`docs/joins-and-ids.md`](docs/joins-and-ids.md).
