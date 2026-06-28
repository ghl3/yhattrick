// Where the site loads its JSON from.
//
// Small data (the games index, the player ratings + cards) always ships inside the app's own
// /public/data, so it's available with zero config — including local dev.
//
// The heavy per-game timelines (~thousands of files) are too big to bundle into the Vercel
// deploy, so in production they live in object storage (Cloudflare R2). The base URL is read
// from NEXT_PUBLIC_GAME_DATA_BASE:
//   - LOCAL / unset  -> "/data/game"            (served from web/public/data by `make games`)
//   - PRODUCTION     -> "https://<r2-host>/game" (set as a Vercel env var at deploy time)
//
// So you never need Cloudflare for local testing — only the deployed app reads from R2.

const GAME_BASE = process.env.NEXT_PUBLIC_GAME_DATA_BASE ?? "/data/game";

export const gameDetailUrl = (id: string | number) => `${GAME_BASE}/${id}.json`;

// Index + player JSON stay local to the deploy (small); kept as helpers for one source of truth.
export const gamesIndexUrl = () => `/data/games.json`;
export const playersIndexUrl = () => `/data/players.json`;
export const goaliesIndexUrl = () => `/data/goalies.json`;
// goalie detail ships in the same player/<id>.json file (discriminated by `kind`)
export const playerDetailUrl = (id: string | number) => `/data/player/${id}.json`;
