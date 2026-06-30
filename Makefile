# Top-level entry points for the whole project. Python steps run via uv inside pipeline/.
# Every stage tees its output to logs/<stage>.log; the model also writes logs/model/.
#
#   make                            # = make all: full local build (clean -> site + model)
#   make all                        # clean-data stints box model games players
#   make fetch                      # download all NHL seasons + handedness -> data/raw/
#   make fetch-season SEASON=2024   # NHL shiftcharts+pbp for one season
#   make fetch-handedness           # NHL player handedness (for off-wing)
#   make fetch-htmlshifts           # HTML TOI reports where the JSON shift feed is empty
#   make clean-data                 # parse raw -> data/interim/        (idempotent)
#   make xg                         # fit the expected-goals model       -> data/processed/xg + web JSON
#   make stints                     # join      -> data/processed/      (idempotent)
#   make box                        # per-player box score (our pbp)    -> data/interim/box
#   make model                      # fit isolated-impact models        -> data/models + logs/model
#   make model FAMILY=tweedie       # same, but the Tweedie-GLM response (default: gaussian)
#   make shooting                   # joint shooter×goalie fit: finishing + GSAx -> data/models + logs/model
#   make goal-accounting            # reconcile goals = xG + μ + finishing + goalie -> data/models + logs
#   make games                      # per-game timelines (site JSON)    -> data/games (+ web sync)
#   make gamelog                    # per-player game log                -> data/processed/gamelog
#   make players                    # player ratings + box (site JSON)  -> web/public/data
#   make goalie-box                 # per-goalie box + splits           -> data/processed/goalie_box
#   make goalie-gamelog             # per-goalie game log               -> data/processed/goalie_gamelog
#   make goalies                    # goalie ratings + detail (site JSON) -> web/public/data
#   make web-dev                    # run the website dev server
#   make web-build                  # production build of the website
#
# Raw/interim/processed data lives under ./data; run logs under ./logs; site JSON under ./data/games
# (synced to ./web/public/data). `make all` does NOT fetch — run `make fetch` first if raw data is missing.

SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -ec

LOGDIR := $(CURDIR)/logs
RUN = mkdir -p $(LOGDIR) && cd pipeline && uv run python -m
SEASON ?=
FAMILY ?= gaussian        # response family for `make model`: gaussian | tweedie

.DEFAULT_GOAL := all
.PHONY: all fetch fetch-season fetch-handedness fetch-htmlshifts clean-data xg stints box model shooting goal-accounting games gamelog players goalie-box goalie-gamelog goalies generative-model publish-data validate test pipeline web-dev web-build

all: clean-data xg stints box model shooting goal-accounting games gamelog players goalie-gamelog goalie-box goalies

fetch:
	$(RUN) yhattrick.data.download all 2>&1 | tee $(LOGDIR)/download.log

fetch-season:
	$(RUN) yhattrick.data.download games --season $(SEASON) 2>&1 | tee $(LOGDIR)/download.log

fetch-handedness:
	$(RUN) yhattrick.data.download handedness 2>&1 | tee $(LOGDIR)/download.log

fetch-htmlshifts:
	$(RUN) yhattrick.data.download htmlshifts 2>&1 | tee $(LOGDIR)/download.log

clean-data:
	$(RUN) yhattrick.data.clean 2>&1 | tee $(LOGDIR)/clean.log

stints:
	$(RUN) yhattrick.data.stints 2>&1 | tee $(LOGDIR)/stints.log

box:
	$(RUN) yhattrick.data.aggregates 2>&1 | tee $(LOGDIR)/box.log

model:
	$(RUN) yhattrick.models.player_onice_model --pool --family $(FAMILY) 2>&1 | tee $(LOGDIR)/model.log

# joint shooter×goalie shot-conversion fit: finishing (shooters) + GSAx (goalies) in one model
shooting:
	$(RUN) yhattrick.models.shooting_model --pool 2>&1 | tee $(LOGDIR)/shooting.log

# reconcile the additive identity goals = xG + μ + finishing + goalie (league + team-season QC)
goal-accounting:
	$(RUN) yhattrick.models.goal_accounting 2>&1 | tee $(LOGDIR)/goal_accounting.log

xg:
	$(RUN) yhattrick.models.expected_goal_model --pool 2>&1 | tee $(LOGDIR)/xg.log

games:
	$(RUN) yhattrick.export.export_games 2>&1 | tee $(LOGDIR)/games.log

gamelog:
	$(RUN) yhattrick.data.gamelog 2>&1 | tee $(LOGDIR)/gamelog.log

players:
	$(RUN) yhattrick.export.export_players 2>&1 | tee $(LOGDIR)/players.log

# goalie descriptive box/splits/gamelog -> site JSON (the modeled GSAx now lives in `make shooting`)
goalie-gamelog:
	$(RUN) yhattrick.data.goalie_gamelog 2>&1 | tee $(LOGDIR)/goalie_gamelog.log

goalie-box:
	$(RUN) yhattrick.data.goalie_aggregates 2>&1 | tee $(LOGDIR)/goalie_box.log

goalies:
	$(RUN) yhattrick.export.export_goalies 2>&1 | tee $(LOGDIR)/goalies.log

# EXPERIMENTAL: generative (Poisson marked-process) proof-of-concept model. Needs the experimental
# dep group (JAX); NOT part of `all` and not wired into the site. See docs/modeling.md.
generative-model:
	mkdir -p $(LOGDIR) && cd pipeline && uv run --group experimental python -m yhattrick.models.generative_model --count nb 2>&1 | tee $(LOGDIR)/generative_model.log

# final validation: read the exported artifacts and confirm the inference-stage invariants hold
# (EXACT identities + APPROX at-scale bands). Report-only; never blocks `all`. Add --strict for CI.
validate:
	$(RUN) yhattrick.validate 2>&1 | tee $(LOGDIR)/validate.log

# python unit tests (processing / modeling / export)
test:
	cd pipeline && uv run pytest -q

# upload the heavy per-game JSON to Cloudflare R2 (needs R2_* env vars; see yhattrick.export.publish)
publish-data:
	$(RUN) yhattrick.export.publish 2>&1 | tee $(LOGDIR)/publish.log

# alias kept for muscle memory
pipeline: all

web-dev:
	cd web && npm run dev

web-build:
	cd web && npm run build
