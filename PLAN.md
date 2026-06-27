# Build Our Own Hockey WAR Model (JFresh/Evolving-Hockey style)

## Context

Across a long discussion we deconstructed JFresh's WAR player cards down to their
machinery: a per-shot **xG** model → **RAPM** ridge regression that isolates each
player's per-60 impact (EV Offence, EV Defence, PP, PK) → **regressed Finishing** →
combine into **GAR** → rescale to **WAR** → age/weight → **percentiles** → visual card.

We confirmed MoneyPuck (already downloaded to `skaters_2025.csv`) is the *pre-isolation,
pre-shrinkage* layer — it has individual xG but no linemate-adjusted coefficients and no
on-ice player identities, so it **cannot** support RAPM alone. The goal now is to build
the real pipeline ourselves over the **last 5 seasons (2021-22 → 2025-26)** and surface
the results as a browsable website.

**Decisions locked (from user):**
- **Full pipeline** — all four RAPM coefficients + Finishing + Penalties → GAR → WAR → percentiles.
- **Build our own xG** (XGBoost), validated against hockeyR's bundled `xg`.
- **Static website**: sortable table of all players → click a player → detail page (our "card").
- **Modern packaging**: `uv` (v0.11.8 present) — project-local venv, **no global installs**.
- **Existing package for data, not raw scraping.**

## Data source (verified)

`hockeyR-data` publishes pre-scraped NHL play-by-play, one file per season, with the
on-ice info RAPM needs. **This is the substrate** — no live scraping required.

- Format: `play_by_play_YYYY_YY.csv.gz`, Python-readable via `pd.read_csv(url, compression='gzip')`.
- Each event row already contains, in one place:
  - `xg` (bundled per-shot expected goals — our validation baseline)
  - on-ice identities: `home_on_1..7`, `away_on_1..7`, `home_goalie`, `away_goalie`
  - shot features: `x, y, x_fixed, y_fixed, shot_distance, shot_angle, event_type`, shot type
  - context: period, time, strength state, score, zone, event team
- Fallback if a season file is missing/stale: `hockey_scraper` (Python, Harry Shomer,
  v≥1.40, post-2023 API compatible) — `scrape_seasons()` → `{'pbp','shifts'}` with
  `homePlayer1..6_id` / `awayPlayer1..6_id`.

We keep the MoneyPuck files we already have for cross-validation of our xG and finishing.

## Tech stack

Two cleanly separated halves: a **Python pipeline** (data + model → JSON) and a
**React frontend** (presentation). Python never renders HTML.

- **Python** (`uv`, v0.11.8 present): `pyproject.toml`, deps via `uv add` — `pandas`,
  `numpy`, `scipy` (sparse matrices), `scikit-learn` (Ridge), `xgboost`, `pyarrow`
  (parquet), `matplotlib` (only for validation/diagnostic plots). Pipeline's final step
  writes **JSON** for the frontend, not HTML.
- **Frontend** (Node 22 + npm 10 present): **React + Vite + TypeScript** via npm.
  `react-router` (`/` table, `/player/:id` card), **TanStack Table** (sort/filter/search),
  **Recharts** (per-player trend charts). Consumes the pipeline's JSON. `npm run dev`
  to view locally; `npm run build` for a static bundle.

## Data layers (one download, the rest derived)

Only **play-by-play is downloaded**; the per-shot and per-stint sets are things we build.

| Layer | Grain | Origin | Used for |
|---|---|---|---|
| play-by-play | per **event** | downloaded (hockeyR-data) | substrate: `xg`, on-ice players, coords |
| per-shot | per **unblocked shot attempt** | *filtered* from PBP | training the xG model |
| per-stint | per **stint** (constant on-ice personnel within a strength state) | *aggregated* from PBP | the RAPM regression |
| design matrix | per **stint × team-perspective**, cols = players | *encoded* from stints (sparse) | fed to ridge |

Flow: **PBP → (filter) per-shot → train xG → attach xG back to PBP shots → (aggregate)
per-stint → (encode) design matrix → RAPM.** A stint boundary = any change in the on-ice
set or strength state; between boundaries accumulate elapsed time + each team's summed
`xg`. On-ice columns (`home_on_1..7`/`away_on_1..7`) define who's out there; `hockey_scraper`
shift data is the exact fallback if event-to-event derivation proves too coarse.

## Project layout

```
hockey/
  pipeline/                      # Python (uv project)
    pyproject.toml
    src/hockeywar/
      config.py                  # seasons, constants (goals/win≈6, penalty≈0.17, k, λ), paths
      download.py                # fetch 5 seasons of hockeyR-data pbp -> data/raw/
      clean.py                   # parse/filter events, normalize on-ice cols -> tidy events
      xg.py                      # train XGBoost xG; attach xg_own; validate vs bundled xg + MoneyPuck
      stints.py                  # build per-stint table + sparse RAPM design matrix per strength state
      rapm.py                    # ridge regressions -> per-60 coefficients (EVO/EVD/PP/PK)
      rapm_hier.py               # v2: hierarchical shared-effect model (see Modeling alternatives)
      finishing.py               # regressed finishing residual (shrunk actual-xG)
      penalties.py               # (drawn - taken) * penalty goal value
      gar.py                     # assemble GAR -> WAR -> aging/recency weight -> percentiles
      export_json.py             # write players.json + player/<id>.json for the frontend
    data/raw/  data/processed/   # csv.gz inputs; parquet intermediates
    output/                      # coefficients.parquet, player_war.csv, json/
  web/                           # React + Vite + TS app
    src/{routes,components,lib}/ # index table route, player card route, chart components
    public/data/                 # players.json + player/<id>.json (copied from pipeline output)
  README.md
```

## Modeling techniques (per stage)

**Stage 0 — Download & clean** (`download.py`, `clean.py`)
- Pull `play_by_play_<season>.csv.gz` for the 5 seasons → `data/raw/`.
- Filter to usable events; standardize player-name/id columns; map each event to a
  (game, strength_state, score_state, zone) context. Persist as parquet in `data/processed/`.

**Stage 1 — xG** (`xg.py`)
- Model: XGBoost binary classifier on **unblocked shot attempts**, label = goal.
- Features: `shot_distance`, `shot_angle`, coords, shot type (one-hot), `is_rebound`
  (shot shortly after a save), `is_rush` (time since last event in another zone), strength
  state, score state, is_home. Calibrate probabilities (isotonic/Platt).
- Output: `xg_own` attached to every shot event (same rows as on-ice players).
- Validate: calibration curve + correlation vs bundled `xg` and vs MoneyPuck season totals.
  Decision point in code: use `xg_own` downstream (default) with bundled `xg` as fallback.

**Stage 2 — RAPM** (`stints.py`, `rapm.py`) — the load-bearing step
- **Stints**: split each game into segments of constant on-ice personnel within one
  strength state. Aggregate `xGF` over the stint (from Stage-1 xg), response = `xGF/60`.
- **Dual encoding**: emit two rows per stint (one per team's perspective). Each player gets
  an **offense** column (+1 when on the attacking team) and a **defense** column (+1 when on
  the defending team). Covariates: home, zone-start mix, score state.
- **Fit (v1, baseline)**: weighted **ridge** (L2) — `sklearn` Ridge / sparse `scipy` CSR —
  weights = stint TOI. λ by cross-validation. Run **separately per strength state**
  (5v5 → EV Off/Def, 5v4 → PP, 4v5 → PK), each shrinking toward **zero**.
- Output: per-player, per-season **per-60 coefficients** (ice-time-independent rates):
  `EVO, EVD, PP, PK`. Persist to `coefficients.parquet`.

**Stage 2b — Hierarchical / shared-effect model** (`rapm_hier.py`, v2 — addresses
"an overall player parameter in all regressions")
- Motivation: PP/PK samples are small and noisy; the 4 independent fits shrink them toward
  zero, discarding the fact that a player's special-teams signal is correlated with his EV
  signal. Instead, give each player a **shared offensive ability** and a **shared defensive
  ability**, and let each strength state be a deviation from it:
  `effect[j,s] = α_phase[j] + δ[j,s]`, where phase ∈ {offense, defense}.
- So the noisy PP estimate shrinks toward the player's **own EV-offense ability**, not zero
  (borrowing strength) — what JFresh meant by PP being "lightly composited."
- **Key design choice**: pool **within phase across strength states** (EV-off ↔ PP-off are
  highly correlated; EV-def ↔ PK-def likewise), but **do NOT** share a single term across
  offense and defense — they're ~uncorrelated at the player level (McCurdy), so one "overall
  good player" term would wrongly drag specialists toward the middle on both.
- Implementation: one **joint** regression with a player phase-main-effect + player×state
  interactions (ridge with grouped penalties), or a mixed-effects/Bayesian multilevel fit.
- Deliverable: compare v1 vs v2 coefficients; report whether shared effects stabilize the
  special-teams numbers (expect tighter PP/PK estimates, especially for low-TOI players).
- Note: because the response is **xG** (pre-save), goalies don't contaminate defensive
  coefficients — a free benefit of the xG choice. (Other v-next options, not in scope now:
  box-score-prior RAPM shrinking toward an SPM prior; blending GF/60 + xGF/60 responses.)

**Stage 3 — Finishing** (`finishing.py`)
- Per player-season: raw residual = (own goals − Σ own `xg`).
- **Shrink**: `w = shots / (shots + k)`; `finishing = w * residual`. Estimate `k` empirically
  via split-half reliability of finishing-per-shot. Pool the 5 seasons with recency weights
  to raise effective sample (true snipers survive; hot streaks regress out).

**Stage 4 — Penalties** (`penalties.py`)
- `(penaltiesDrawn − penaltiesTaken) * ~0.17` goals (era-calibrated power-play value).

**Stage 5 — GAR → WAR → percentiles** (`gar.py`)
- Convert each per-60 coefficient to goals: `coef × (TOI_in_that_state / 60)`.
- Sum EVO+EVD+PP+PK goals + Finishing + Penalties.
- Subtract **replacement** baseline (empirical per-role rate × TOI; F vs D separate).
- `GAR = sum − replacement`; `WAR = GAR / 6`.
- **Projected** WAR: 3-year recency-weighted average + simple aging curve (peak ~24-27).
- **Percentiles**: rank within position group (F and D separately, matching the cards) for
  WAR and each component. Persist `player_war.csv` (raw + GAR + WAR + every percentile).

## Output & visualization (Python `export_json.py` → React `web/`)

Python writes JSON; React renders it.
- **`export_json.py`** emits `players.json` (index rows: id, name, team, pos, GP, TOI, WAR,
  WAR %ile, component %iles) and `player/<id>.json` (full detail: per-season coefficients,
  GAR/WAR, all percentiles, descriptive box-score, competition/teammates context). Copied
  into `web/public/data/`.
- **React app (`web/`)**:
  - `/` — leaderboard: **TanStack Table** over `players.json`, sortable/filterable/searchable;
    each row links to the player route.
  - `/player/:id` — our "card": header (name, team, pos, age, projected WAR %ile), color-coded
    percentile boxes for EVO/EVD/PP/PK/Finishing (JFresh palette), descriptive row, and
    **Recharts** trend lines of WAR + components across the 5 seasons.
  - When v2 exists, the card shows v1 vs v2 (independent vs hierarchical) side by side.
- View: `cd web && npm run dev`; ship via `npm run build` (static bundle).

## Build order (full pipeline, incremental milestones)

1. `uv` env + `download.py` + `clean.py` → tidy events for 5 seasons.
2. `xg.py` → `xg_own` + validation report. (Checkpoint: calibration sane vs bundled xg.)
3. `stints.py` + `rapm.py` 5v5 only → EVO/EVD. (Checkpoint: McDavid/MacKinnon top EVO; depth
   grinders NOT topping EVD — i.e. isolation removed the usage artifact we saw in MoneyPuck.)
4. Add PP/PK regressions; `finishing.py`; `penalties.py`.
5. `gar.py` → GAR/WAR/percentiles. (Checkpoint: leaderboard face-validity vs JFresh names.)
6. `export_json.py` → JSON; scaffold `web/` (Vite+React+TS) → leaderboard + player card.
7. `rapm_hier.py` (v2 hierarchical) → compare vs v1; surface both in the card.

## Verification

- **xG**: calibration plot (predicted vs actual goal rate by bin); season-total xG correlates
  with MoneyPuck per player (expect r > ~0.9).
- **RAPM sniff test**: 5v5 EVO leaders are genuine drivers (McDavid/MacKinnon/Draisaitl);
  EVD leaders are NOT sheltered 4th-liners (the failure mode we saw in MoneyPuck's raw on-ice
  numbers) — confirms isolation works.
- **Finishing**: Caufield/Dorofeyev land high after shrinkage; reconstructs toward (not equal
  to) raw goals; year-to-year stability higher than raw residual.
- **GAR/WAR**: top-of-leaderboard names broadly match JFresh/EH; F and D percentile
  distributions ~uniform by construction.
- **Hierarchical v2**: shared-effect model yields tighter PP/PK estimates for low-TOI players
  than v1, without collapsing genuine specialists; EV coefficients stay close to v1.
- **Site**: `npm run dev` serves the leaderboard; sort/filter works; clicking a player opens
  their card; Recharts trends render for a known player (e.g. Dorofeyev) and tell the "elite
  finisher, average engine" story.
- Spot-check Dorofeyev end-to-end against the JFresh card + HockeyViz Magnus read we discussed.
```
