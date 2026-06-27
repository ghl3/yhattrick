# Project Documentation

A from-scratch NHL **Wins Above Replacement (WAR)** model and an inspection website. The
project has two halves: a **Python data+modeling pipeline** (`pipeline/`) that turns raw NHL
data into clean tables and, eventually, player ratings; and a **React website** (`web/`) that
visualizes the data — starting with a per-game inspection view.

## Documentation map

| Doc | Covers |
|---|---|
| [01-data-sources.md](01-data-sources.md) | Where the data comes from, why, and the verified URLs |
| [02-downloads.md](02-downloads.md) | The download stage (`download.py`): what is fetched, caching, resumability |
| [03-joins-and-ids.md](03-joins-and-ids.md) | The game_id mapping and the shots↔shifts↔events join contract |
| [04-processing.md](04-processing.md) | The clean → interim → processed stages and the data layout |
| [05-website.md](05-website.md) | The inspection site: games index + game view |
| [06-modeling.md](06-modeling.md) | (Deferred) the WAR model: xG → RAPM → finishing → GAR → WAR |

The authoritative high-level design is `../PLAN.md`. This `docs/` folder documents the
**implementation as it actually exists**, stage by stage; it is updated as each stage lands.

## Current status

**Phase 1 — data + game-view site (modeling deferred).** Goal: a clean, well-organized data
layout and a site to manually verify the joins are correct before any modeling.
