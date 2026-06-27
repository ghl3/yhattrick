# 01 — Data sources

We build everything from three public sources. The original plan relied on the `hockeyR-data`
GitHub dump, but that is **dead** (the repo lists its play-by-play files but every download path
— raw, raw.githubusercontent, the single-file API, and LFS media — returns 404, with no
releases; it is a defunct ~25 GB LFS repo). We re-sourced and verified working replacements.

| What we need | Source | URL pattern |
|---|---|---|
| Per-shot rows + a **borrowed xG** (`xGoal`) + rich shot features | **MoneyPuck** (via mirror — moneypuck.com itself is Cloudflare-blocked) | `https://peter-tanner.com/moneypuck/downloads/shots_<season>.zip` |
| Season box-score totals (for validation, penalties) | MoneyPuck | `https://moneypuck.com/moneypuck/playerData/seasonSummary/<season>/regular/skaters.csv` |
| **On-ice player identities + time-on-ice** (→ stints) | **NHL shiftcharts API** | `https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId=<id>` |
| Play-by-play **events** (goals, penalties, faceoffs), strength, rosters | **NHL play-by-play API** | `https://api-web.nhle.com/v1/gamecenter/<id>/play-by-play` |

Seasons covered: `2021`–`2025`, i.e. 2021-22 through 2025-26 (a "season" is named by its
starting year).

## Why this split

- **MoneyPuck shots** give us a clean, single-download-per-season table of every unblocked shot
  attempt with 137 columns: coordinates, distance/angle, shot type, rush/rebound flags, score
  and strength context, the shooter/goalie, and a pre-computed **`xGoal`**. This is both our
  *borrowed xG baseline* (so we can build the whole WAR pipeline before training our own model)
  and the feature set we'll later train our own xG on. It does **not** contain on-ice player
  identities — only skater *counts*.
- **NHL shiftcharts** supply exactly what MoneyPuck lacks: who was on the ice and for how long,
  as real shift intervals. This is the substrate for stints and accurate TOI, and it lets us
  reconstruct the on-ice five (plus goalie) at the instant of every shot.
- **NHL play-by-play** rounds out the event timeline (penalties explain strength changes,
  faceoffs mark zone starts, goals confirm on-ice attribution) and is the source for the
  penalty and zone-start inputs the model will need later.

## Fallbacks (not currently used)

- `hockey_scraper` 1.40.3 (PyPI, current) scrapes NHL pbp + shifts into per-event on-ice player
  IDs directly — our backup if the shiftcharts↔shots join ever proves insufficient.
- `nhlpy` 0.3.0 is a thin NHL API wrapper.

## Environment notes

Large GitHub-raw and moneypuck.com requests can return HTML/404 behind restrictive network
sandboxes; the mirror and NHL hosts work directly. Python's `urllib` may fail TLS verification
for lack of system certs — the pipeline uses `requests`.
