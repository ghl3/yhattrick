# Top-level entry points for the whole project. Python steps run via uv inside pipeline/.
#
#   make fetch                      # download MoneyPuck + all NHL seasons -> data/raw/
#   make fetch-moneypuck            # just the MoneyPuck shots/skaters files
#   make fetch-season SEASON=2024   # NHL shiftcharts+pbp for one season
#   make clean-data                 # parse raw -> data/interim/        (idempotent)
#   make stints                     # join      -> data/processed/      (idempotent)
#   make box                        # per-player box score (our pbp)    -> data/interim/box
#   make games                      # per-game timelines (site JSON)    -> web/public/data
#   make players                    # player ratings + box (site JSON)  -> web/public/data
#   make model                     # fit/cache the isolated-impact models
#   make pipeline                   # clean-data + stints + box + games + players
#   make web-dev                    # run the website dev server
#
# Raw/interim/processed data lives under ./data; site JSON is written to ./web/public/data.

PIPELINE := cd pipeline && uv run python -m
SEASON ?=

.PHONY: fetch fetch-moneypuck fetch-season clean-data stints box model games players pipeline web-dev

fetch:
	$(PIPELINE) hockeywar.download all

fetch-moneypuck:
	$(PIPELINE) hockeywar.download moneypuck

fetch-season:
	$(PIPELINE) hockeywar.download games --season $(SEASON)

clean-data:
	$(PIPELINE) hockeywar.clean

stints:
	$(PIPELINE) hockeywar.stints

box:
	$(PIPELINE) hockeywar.aggregates

model:
	$(PIPELINE) hockeywar.player_onice_model

games:
	$(PIPELINE) hockeywar.export_games

players:
	$(PIPELINE) hockeywar.export_players

pipeline: clean-data stints box games players

web-dev:
	cd web && npm run dev
