# 05 — Website

A minimal React + Vite + TypeScript app (`web/`) for **inspecting** the data — verifying that
on-ice reconstruction, strength, and shot attribution are correct before any modeling. Light-blue
theme. It reads only `web/public/data/` (synced from `data/games/` by the pipeline).

```bash
make web-dev          # or: cd web && npm run dev   -> http://localhost:5173
```

Stack: `react-router-dom` (routes), `@tanstack/react-table` (the games index). Types mirroring
the JSON live in `src/lib/types.ts`.

## Routes

### `/` — games index (`routes/GamesIndex.tsx`)

A sortable, searchable TanStack table over `games.json`: date, season, matchup, score, stint /
shot / event counts, and the **on-ice match %** (the at-a-glance data-quality signal). Each row
links to the game view.

### `/game/:gameId` — game view (`routes/GameView.tsx`)

The inspection centerpiece, reading `game/<id>.json`:

- **Header**: scoreline, date, and a stat grid (xGF away/home, shots, stint count).
- **Player aggregates** (collapsible): two sortable tables (away / home) — shifts, TOI, G, A1,
  A2, Points, shots, xG.
- **Timeline**: a vertical, period-grouped flow of **stints** (game start at top). Each stint is
  a shaded block with a header (clock range, strength badge — special teams tinted, duration,
  per-stint xGF) and an always-visible **line-change diff** (green `+on` / red `−off` per team)
  so it's clear why each stint is distinct. Expanding a stint shows the full on-ice lines
  (skaters + goalie) and the **events** that occurred during it — faceoffs, shots (with model
  xG and a `⚠on-ice` mark if the count disagreed), goals, penalties, hits, etc. Period headers
  are sticky and show the running score; stints with an illegal skater count carry an `overload`
  flag.
- **Event cards**: any event with coordinates is clickable and expands to a card with a subtle
  full-rink plot of the location plus its metadata (`components/EventCard.tsx`). Shots show the
  full xG-model inputs — type, distance, angle, rush/rebound, model xG, on-ice match — and the
  dot is shaded by xG; other events (faceoff, hit, give/takeaway, penalty) are color-coded by type.

## Data flow

`export_games.py` writes `data/games/` and copies it to `web/public/data/`. Vite serves
`public/` at the site root, so the app fetches `/data/games.json` and `/data/game/<id>.json`.
Production build: `cd web && npm run build` → static bundle in `web/dist/`.
