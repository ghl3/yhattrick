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


# --- paths -------------------------------------------------------------------
PKG_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = PKG_DIR.parent.parent          # .../hockey/pipeline
REPO_ROOT = PIPELINE_DIR.parent               # .../hockey
DATA = REPO_ROOT / "data"

RAW = DATA / "raw"
RAW_MONEYPUCK = RAW / "moneypuck"
RAW_SHIFTS = RAW / "nhl" / "shiftcharts"
RAW_PBP = RAW / "nhl" / "pbp"

INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
MODELS = DATA / "models"          # model outputs (impact coefficients, WAR, ...)

# Canonical site-facing JSON lives in the data tree; build_games syncs a copy to the web app.
SITE_JSON = DATA / "games"
WEB_DIR = REPO_ROOT / "web"
WEB_DATA = WEB_DIR / "public" / "data"

_ALL_DIRS = (RAW_MONEYPUCK, RAW_SHIFTS, RAW_PBP, INTERIM, PROCESSED, SITE_JSON)


def ensure_dirs() -> None:
    for d in _ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# --- remote sources ----------------------------------------------------------
# moneypuck.com is Cloudflare-blocked; the peter-tanner mirror serves the same zips.
MONEYPUCK_SHOTS_URL = "https://peter-tanner.com/moneypuck/downloads/shots_{season}.zip"
MONEYPUCK_SKATERS_URL = "https://moneypuck.com/moneypuck/playerData/seasonSummary/{season}/regular/skaters.csv"
NHL_SHIFTCHARTS_URL = "https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={game_id}"
NHL_PBP_URL = "https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"

USER_AGENT = "Mozilla/5.0 (hockeywar data pipeline; research/personal use)"
REQUEST_TIMEOUT = 30           # seconds per request
THROTTLE_SECONDS = 0.4         # polite delay between NHL API calls
MAX_RETRIES = 3

# --- constants ---------------------------------------------------------------
PERIOD_SECONDS = 1200          # 20:00 regulation period


# --- game_id mapping (verified on 2024020500; see data-sources memory) --------
def mp_to_nhl_game_id(mp_game_id: int, season: int) -> int:
    """MoneyPuck game_id (e.g. 20500) + season (2024) -> NHL gameId 2024020500.

    MoneyPuck game_id is 5 digits: leading 2=regular, 3=playoffs; the NHL id inserts a 0
    between the 4-digit season and the 5-digit MoneyPuck id.
    """
    return int(f"{season}0{int(mp_game_id):05d}")


def nhl_to_mp_game_id(nhl_game_id: int) -> int:
    """2024020500 -> 20500 (drop the 4-digit season + the inserted 0)."""
    return int(str(nhl_game_id)[5:])


def is_regular_season(nhl_game_id: int) -> bool:
    """NHL gameId game-type digits are 02 for regular season, 03 for playoffs."""
    return (nhl_game_id // 10000) % 100 == 2
