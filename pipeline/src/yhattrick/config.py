"""Central config: seasons, paths, network constants, and the game_id mapping.

Everything else imports paths and helpers from here so the data layout stays consistent.
"""
from __future__ import annotations

from pathlib import Path

# --- seasons -----------------------------------------------------------------
# A "season" is named by its starting year: 2021 == 2021-22 ... 2025 == 2025-26.
SEASONS: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025)


def season_label(season: int) -> str:
    """2024 -> '2024-25'."""
    return f"{season}-{str(season + 1)[-2:]}"


def nhl_season8(season: int) -> str:
    """2024 -> '20242025' (the 8-digit season id NHL schedule endpoints expect)."""
    return f"{season}{season + 1}"


# --- paths -------------------------------------------------------------------
PKG_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = PKG_DIR.parent.parent          # .../hockey/pipeline
REPO_ROOT = PIPELINE_DIR.parent               # .../hockey
DATA = REPO_ROOT / "data"

RAW = DATA / "raw"
RAW_SHIFTS = RAW / "nhl" / "shiftcharts"
RAW_HTMLSHIFTS = RAW / "nhl" / "htmlshifts"   # TH/TV HTML TOI reports (fallback when JSON is empty)
RAW_PBP = RAW / "nhl" / "pbp"
RAW_PLAYERS = RAW / "nhl" / "players"   # per-player landing json (handedness, bio)

INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
MODELS = DATA / "models"          # model outputs (impact coefficients, WAR, ...)

# Run logs for every stage (parse, join, fits, exports). The model-fit log lives here too.
LOGS = REPO_ROOT / "logs"
LOGS_MODEL = LOGS / "model"       # per-fit metadata snapshots + the model_fits.jsonl history

# Canonical site-facing JSON lives in the data tree; build_games syncs a copy to the web app.
SITE_JSON = DATA / "games"
WEB_DIR = REPO_ROOT / "web"
WEB_DATA = WEB_DIR / "public" / "data"

_ALL_DIRS = (RAW_SHIFTS, RAW_HTMLSHIFTS, RAW_PBP, RAW_PLAYERS, INTERIM, PROCESSED, SITE_JSON, LOGS, LOGS_MODEL)


def ensure_dirs() -> None:
    for d in _ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# --- remote sources (all NHL) ------------------------------------------------
NHL_SHIFTCHARTS_URL = "https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={game_id}"
NHL_PBP_URL = "https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
NHL_PLAYER_URL = "https://api-web.nhle.com/v1/player/{player_id}/landing"
# HTML time-on-ice reports (shift fallback when the JSON shiftcharts feed is empty); hv = H or V
NHL_HTML_SHIFTS_URL = "https://www.nhl.com/scores/htmlreports/{season8}/T{hv}{game6}.HTM"
# the season's game list: standings -> the season's teams, then each club's full schedule
NHL_STANDINGS_URL = "https://api-web.nhle.com/v1/standings/{date}"
NHL_CLUB_SCHEDULE_URL = "https://api-web.nhle.com/v1/club-schedule-season/{team}/{season8}"

USER_AGENT = "Mozilla/5.0 (yhattrick data pipeline; research/personal use)"
REQUEST_TIMEOUT = 30           # seconds per request
THROTTLE_SECONDS = 0.4         # polite delay between NHL API calls
MAX_RETRIES = 3

# --- constants ---------------------------------------------------------------
PERIOD_SECONDS = 1200          # 20:00 regulation period


# --- game_id helpers ---------------------------------------------------------
def is_regular_season(nhl_game_id: int) -> bool:
    """NHL gameId game-type digits are 02 for regular season, 03 for playoffs."""
    return (nhl_game_id // 10000) % 100 == 2


def game6(nhl_game_id: int) -> str:
    """2024020500 -> '020500' (game-type + number, the 6 digits the HTML reports use)."""
    return str(nhl_game_id)[4:]
