# Top-level entry points for the whole project. Python steps run via uv inside pipeline/.
# Every stage tees its output to logs/<stage>.log; the model also writes logs/model/.
#
#   make                            # = make all: full local build (clean -> site + model)
#   make all                        # clean-data stints box model games players
#   make fetch                      # download MoneyPuck + all NHL seasons -> data/raw/
#   make fetch-moneypuck            # just the MoneyPuck shots/skaters files
#   make fetch-season SEASON=2024   # NHL shiftcharts+pbp for one season
#   make clean-data                 # parse raw -> data/interim/        (idempotent)
#   make stints                     # join      -> data/processed/      (idempotent)
#   make box                        # per-player box score (our pbp)    -> data/interim/box
#   make model                      # fit isolated-impact models        -> data/models + logs/model
#   make model FAMILY=tweedie       # same, but the Tweedie-GLM response (default: gaussian)
#   make finishing                  # fit finishing (goals above expected) -> data/models + logs/model
#   make games                      # per-game timelines (site JSON)    -> data/games (+ web sync)
#   make players                    # player ratings + box (site JSON)  -> web/public/data
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
.PHONY: all fetch fetch-moneypuck fetch-season clean-data stints box model finishing games players publish-data pipeline web-dev web-build

all: clean-data stints box model finishing games players

fetch:
	$(RUN) yhattrick.download all 2>&1 | tee $(LOGDIR)/download.log

fetch-moneypuck:
	$(RUN) yhattrick.download moneypuck 2>&1 | tee $(LOGDIR)/download.log

fetch-season:
	$(RUN) yhattrick.download games --season $(SEASON) 2>&1 | tee $(LOGDIR)/download.log

clean-data:
	$(RUN) yhattrick.clean 2>&1 | tee $(LOGDIR)/clean.log

stints:
	$(RUN) yhattrick.stints 2>&1 | tee $(LOGDIR)/stints.log

box:
	$(RUN) yhattrick.aggregates 2>&1 | tee $(LOGDIR)/box.log

model:
	$(RUN) yhattrick.player_onice_model --pool --family $(FAMILY) 2>&1 | tee $(LOGDIR)/model.log

finishing:
	$(RUN) yhattrick.finishing --pool 2>&1 | tee $(LOGDIR)/finishing.log

games:
	$(RUN) yhattrick.export_games 2>&1 | tee $(LOGDIR)/games.log

players:
	$(RUN) yhattrick.export_players 2>&1 | tee $(LOGDIR)/players.log

# upload the heavy per-game JSON to Cloudflare R2 (needs R2_* env vars; see yhattrick.publish)
publish-data:
	$(RUN) yhattrick.publish 2>&1 | tee $(LOGDIR)/publish.log

# alias kept for muscle memory
pipeline: all

web-dev:
	cd web && npm run dev

web-build:
	cd web && npm run build
