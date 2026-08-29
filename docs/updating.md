# 03 — In-season updates (the update stage)

One command keeps the site current while a season is underway:

```bash
make update            # fetch new final games -> rebuild -> models -> export -> validate -> publish -> deploy
make update-dry        # report what an update would do (read-only schedule check)
```

Under the hood this runs `python -m yhattrick.update` (see its module docstring for every flag).
The updater is **stateless and idempotent**: every decision derives from the NHL schedule plus
what is on disk, so an interrupted or failed run is simply re-run. A file lock
(`logs/update.lock`) makes overlapping runs impossible, so it is safe to invoke from cron. Each
run appends a summary line to `logs/update_runs.jsonl` and tees console output to
`logs/update.log`.

## What a run does

1. **Schedule check** — the season's schedule (with per-game `gameState`) from the NHL API.
   Only *finished* games (state `OFF`/`FINAL`) are fetchable: for unplayed games the pbp
   endpoint answers 200 with zero plays, and caching that would poison the resumable
   exists-check. A previously cached empty pbp for a now-finished game is re-fetched
   automatically (repair).
2. **Fetch** — pbp + shiftchart JSON for newly-finished games, then HTML TOI reports for games
   whose JSON shift feed is empty (the feed is dead for current seasons; a shiftchart 404 is
   recorded as `{"data":[]}` so the game still reaches clean and the HTML fallback carries the
   shifts).
3. **Short-circuit** — with no new raw files and current interim tables, the run stops here
   ("up to date"). `--force` overrides.
4. **Rebuild** — `clean --season`, handedness for new players, then the season's
   stints/box/gamelog tables.
5. **xG: score, don't refit** — daily runs score the season's shots with the **frozen booster**
   from the last full fit (`--score-only`). Identical shots score identically day over day, so
   per-game JSON for already-exported games stays byte-identical and the R2 publish uploads
   only genuinely new games (the CDN objects are cached immutable). `--refit-xg` runs a full
   pooled refit instead — do that deliberately, not on a schedule.
6. **Models** — pooled RAPM (`model`), the joint shooter×goalie fit (`shooting`), and the
   goal-accounting reconciliation, all on the full season window. Per-season fits refresh
   inside the export via the mtime-checked fit cache.
7. **Export + validate** — games (per-season index merge keeps the other seasons' rows),
   players, goalies. `validate --strict` then gates everything outward-facing: an EXACT
   invariant failure stops the run before anything is published.
8. **Publish + deploy** — per-game JSON to R2 (incremental, size-skip; credentials from
   `pipeline/.env`), then `npx vercel --prod` in `web/`. `--no-publish` / `--no-deploy` keep a
   run local.

The generative model (JAX, experimental dep group) is intentionally outside the daily loop:
run `make generative-model` + `make generative-cards` + `make players` deliberately; exports
merge the latest `gen_cards.json` either way.

## Season-start checklist

When a new season is about to begin (e.g. 2026-27 = season `2026`):

1. Add the season to `SEASONS` in `pipeline/src/yhattrick/config.py` (deliberate one-line
   change: model pools, artifact names, and exports all key off it).
2. Run `make xg` once — a full pooled refit also (re)writes the frozen-scoring artifacts
   (`xg_booster.json`, `xg_isotonic.json`, `xg_categories.json`) that daily `--score-only`
   needs.
3. `make update-dry`, then `make update`. Before the first puck drops it reports every game as
   pending; from the first final game onward it does real work.

Notes for early season:

- Player/goalie exports include a season once it has ≥200 processed games
  (`FULL_MIN_GAMES` in `export_players.py`), roughly mid-November; game pages appear
  immediately. Lower that threshold deliberately if the season should surface sooner.
- The standings endpoint has no rows until the season starts; the team list falls back to the
  previous season's standings automatically (only matters in an expansion/relocation year).

## Cron, when wanted

The tool is one-shot by design; scheduling is a separate decision. A morning run (all games
final, league-wide) looks like:

```
30 6 * * * cd ~/Projects/yhattrick && make update >> logs/cron.log 2>&1
```

The lock makes an overlapping manual run a clean no-op, and an "up to date" run costs ~35
read-only schedule requests.
