# ŷHatTrick — web

The ŷHatTrick site: a Next.js (App Router) front end for browsing NHL games
shift-by-shift and inspecting isolated player-impact ratings.

It reads static JSON written by the pipeline to `public/data/` (see the project
root `Makefile`: `make games` and `make players`).

## Develop

```bash
npm install
npm run dev      # http://localhost:3000
```

## Build

```bash
npm run build
npm run start
```

## Routes

- `/` — games index
- `/game/[gameId]` — stint-by-stint timeline + event/rink cards
- `/players` — player impact leaderboard
- `/player/[id]` — player card (impact, per-season trends, linemates)
- `/about` — what the project is and how the model works

Deployed on Vercel.
