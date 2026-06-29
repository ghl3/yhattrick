# 01 — Data sources

We build everything from the public NHL APIs — no third-party data. The original plan relied on
the `hockeyR-data` GitHub dump, but that is **dead** (every download path returns 404). An earlier
iteration borrowed MoneyPuck's shots file (and its `xGoal`); we now derive shots, geometry, and our
own xG entirely from NHL play-by-play.

| What we need | Source | URL pattern |
|---|---|---|
| The game list for a season | **NHL standings + club schedule** | `…/v1/standings/<date>`, `…/v1/club-schedule-season/<team>/<season8>` |
| Play-by-play **events** (shots w/ coords, goals, penalties, faceoffs), strength, rosters | **NHL play-by-play API** | `https://api-web.nhle.com/v1/gamecenter/<id>/play-by-play` |
| **On-ice player identities + time-on-ice** (→ stints) | **NHL shiftcharts JSON**, with the **HTML TOI reports** as fallback | `https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId=<id>` · `https://www.nhl.com/scores/htmlreports/<season8>/T{H,V}<game6>.HTM` |
| Shooter **handedness** (→ off-wing feature) | **NHL player landing** | `https://api-web.nhle.com/v1/player/<id>/landing` |

Seasons covered: `2021`–`2025`, i.e. 2021-22 through 2025-26 (a "season" is named by its
starting year).

## Why this split

- **NHL play-by-play** is the spine: every shot with its coordinates, type, strength
  (`situationCode`) and attacking side (`homeTeamDefendingSide`) — from which we build the shot
  table, the geometry/pre-shot features, and our xG — plus goals, penalties (strength changes) and
  faceoffs (zone starts) for the rest of the event timeline.
- **NHL shiftcharts** supply who was on the ice and for how long, as real shift intervals: the
  substrate for stints and accurate TOI, and how we reconstruct the on-ice five (plus goalie) at
  the instant of every shot. The JSON feed (`stats/rest`) stopped being populated ~spring 2025, so
  for games where it returns 0 rows we parse the **HTML TOI reports** (`TH`=home, `TV`=away) instead
  (`html_shifts.py`), resolving sweater number → playerId via that game's pbp rosterSpots. Same shift
  schema either way. See `docs/joins-and-ids.md`.
- **NHL player landing** gives shooter handedness, the one player attribute the pbp lacks, used
  only for the off-wing shot-geometry feature.

## Fallbacks (not currently used)

- `hockey_scraper` 1.40.3 (PyPI, current) scrapes NHL pbp + shifts into per-event on-ice player
  IDs directly — our backup if the shiftcharts↔shots join ever proves insufficient.
- `nhlpy` 0.3.0 is a thin NHL API wrapper.

## Environment notes

The NHL hosts (`api-web.nhle.com`, `api.nhle.com`) work directly. Python's `urllib` may fail TLS
verification for lack of system certs — the pipeline uses `requests`.
