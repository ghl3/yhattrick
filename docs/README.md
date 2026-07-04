# Project Documentation

A from-scratch NHL **Wins Above Replacement (WAR)** model and an inspection website. The
project has two halves: a **Python data+modeling pipeline** (`pipeline/`) that turns raw NHL
data into clean tables and, eventually, player ratings; and a **React website** (`web/`) that
visualizes the data — starting with a per-game inspection view.

## Documentation map

| Doc | Covers |
|---|---|
| [data-sources.md](data-sources.md) | Where the data comes from, why, and the verified URLs |
| [downloads.md](downloads.md) | The download stage (`download.py`): what is fetched, caching, resumability |
| [joins-and-ids.md](joins-and-ids.md) | The game_id mapping and the shots↔shifts↔events join contract |
| [processing.md](processing.md) | The clean → interim → dimensions → processed stages, the fact/dimension split, and the data layout |
| [website.md](website.md) | The inspection site: games index + game view |
| [modeling.md](modeling.md) | The additive theory of goals: xG → RAPM (creation) + shooting model (finishing + GSAx) → reconciliation |

The authoritative high-level design is `../PLAN.md`. This `docs/` folder documents the
**implementation as it actually exists**, stage by stage; it is updated as each stage lands.

## Current status

**Phase 1 — data + game-view site (modeling deferred).** Goal: a clean, well-organized data
layout and a site to manually verify the joins are correct before any modeling.
