"""Dimension tables — descriptive attributes stored ONCE per entity and joined into the fact tables
(stints, shots_onice) by stable keys at model time, so an attribute like a birthdate lives in exactly
one place rather than across millions of fact rows.

Built in the clean/parse stage (`make dims`). Output lives in `data/dimensions/`:

  players.parquet        one row per player_id — STATIC bio: name, birthdate, shoots, height, weight,
                         draft_year, draft_overall. Source: the per-player landing JSONs
                         (`RAW_PLAYERS/*.json`); `birthdate` drives the aging curves.
  player_season.parquet  one row per (player_id, season) — position, number. Source: interim/roster.
                         Deliberately NO team: a player's team is a property of (player, GAME), because
                         a mid-season trade splits his games across two teams, so team is a games-join
                         (games.home/away_team + the player's on-ice side), never a stored season
                         attribute. A single season "team" would be wrong for a traded player.

Placement test (fact vs dimension): a property of THIS event → fact column (stints/shots); a property
of an ENTITY (player, game, venue) → dimension row joined by key.
"""

from __future__ import annotations

import json

import pandas as pd

from .. import config as C

# Same building, new sponsor name → alias to the CURRENT name so a venue's arena-bias states stay one
# random-walk chain across the rename (the scorer crew doesn't change with the signage). Real building
# moves stay split on purpose: Gila River → Mullett → Delta Center, Joe Louis → Little Caesars,
# Nassau/Barclays → UBS are different rinks with different crews. Covers 2016+ for the backfill.
VENUE_ALIAS = {
    "STAPLES Center": "Crypto.com Arena",
    "FLA Live Arena": "Amerant Bank Arena",
    "BB&T Center": "Amerant Bank Arena",
    "Amalie Arena": "Benchmark International Arena",
    "PNC Arena": "Lenovo Center",
    "Wells Fargo Center": "Xfinity Mobile Arena",
    "Xcel Energy Center": "Grand Casino Arena",
    "Pepsi Center": "Ball Arena",
    "Scottrade Center": "Enterprise Center",
    "Verizon Center": "Capital One Arena",
    "Air Canada Centre": "Scotiabank Arena",
    "Bell MTS Place": "Canada Life Centre",
    "MTS Centre": "Canada Life Centre",
    "First Niagara Center": "KeyBank Center",
    "Consol Energy Center": "PPG Paints Arena",
}


def _default(v):
    """NHL API multilang fields are {'default': ..., 'cs': ...}; pull the default."""
    return (v or {}).get("default") if isinstance(v, dict) else v


def _full_name(d: dict) -> str | None:
    parts = [_default(d.get("firstName")), _default(d.get("lastName"))]
    return " ".join(p for p in parts if p) or None


def build_players() -> pd.DataFrame:
    """One row per downloaded landing JSON. A player without a landing file has no row here, so a
    lookup for that id returns NaT/None."""
    rows = []
    for f in sorted(C.RAW_PLAYERS.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        draft = d.get("draftDetails") or {}
        rows.append(
            {
                "player_id": int(d.get("playerId") or f.stem),
                "name": _full_name(d),
                "birthdate": d.get("birthDate"),
                "shoots": d.get("shootsCatches"),
                "height_in": d.get("heightInInches"),
                "weight_lb": d.get("weightInPounds"),
                "draft_year": draft.get("year"),
                "draft_overall": draft.get("overallPick"),
            }
        )
    df = pd.DataFrame(
        rows,
        columns=[
            "player_id",
            "name",
            "birthdate",
            "shoots",
            "height_in",
            "weight_lb",
            "draft_year",
            "draft_overall",
        ],
    )
    df = df.drop_duplicates("player_id").sort_values("player_id").reset_index(drop=True)
    df["birthdate"] = pd.to_datetime(df["birthdate"], errors="coerce")
    return df


def build_player_season() -> pd.DataFrame:
    """Per-(player, season) stable attributes from the season rosters, MINUS team (see module doc).
    A traded player has one roster row per team in a season; we keep his position/number (invariant to
    the trade) and dedup to one row per (player_id, season)."""
    frames = []
    for p in sorted((C.INTERIM / "roster").glob("*.parquet")):
        frames.append(pd.read_parquet(p, columns=["player_id", "season", "position", "number"]))
    if not frames:
        return pd.DataFrame(columns=["player_id", "season", "position", "number"])
    df = pd.concat(frames, ignore_index=True)
    df = (
        df.drop_duplicates(["player_id", "season"])
        .sort_values(["season", "player_id"])
        .reset_index(drop=True)
    )
    return df


def _scan_games(seasons) -> pd.DataFrame:
    """Parse each game's pbp ONCE for the game-level facts. The pbp is the only source of game DATE;
    everything else (season, type, teams, venue) is also top-level so we take it here in one pass. The
    RAW venue name is carried through — canonicalisation + venue_id happen in build_arenas/build_games.
    """
    rows = []
    for s in sorted(set(int(x) for x in seasons)):
        for pf in sorted(C.RAW_PBP.glob(f"{s}0*.json")):
            try:
                d = json.loads(pf.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            gid = int(d.get("id") or pf.stem)
            rows.append(
                {
                    "nhl_game_id": gid,
                    "date": d.get("gameDate"),
                    "season": int(str(gid)[:4]),
                    "game_type": d.get("gameType"),
                    "home_team": (d.get("homeTeam") or {}).get("abbrev"),
                    "away_team": (d.get("awayTeam") or {}).get("abbrev"),
                    "raw_venue": _default(d.get("venue"))
                    or (d.get("homeTeam") or {}).get("abbrev"),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "nhl_game_id",
            "date",
            "season",
            "game_type",
            "home_team",
            "away_team",
            "raw_venue",
        ],
    )


def build_arenas(games_raw: pd.DataFrame) -> pd.DataFrame:
    """Canonical venue table: one row per physical building (after applying VENUE_ALIAS), the raw names
    that map to it, and the seasons it hosted games. venue_id = stable index by canonical name. Rare
    outdoor/neutral sites are kept — the model's arena-bias fit decides which get a scorer-bias offset
    by a game-count threshold, not this table."""
    g = games_raw.dropna(subset=["raw_venue"]).copy()
    g["canonical"] = g["raw_venue"].map(lambda v: VENUE_ALIAS.get(v, v))
    rows = []
    for canon, sub in g.groupby("canonical"):
        rows.append(
            {
                "canonical_name": canon,
                "raw_names": sorted(sub.raw_venue.unique().tolist()),
                "active_seasons": sorted(int(x) for x in sub.season.unique()),
            }
        )
    ar = pd.DataFrame(rows, columns=["canonical_name", "raw_names", "active_seasons"])
    ar = ar.sort_values("canonical_name").reset_index(drop=True)
    ar.insert(0, "venue_id", ar.index.astype(int))
    return ar


def build_games(games_raw: pd.DataFrame, arenas: pd.DataFrame) -> pd.DataFrame:
    """One row per game: keys + game-level facts + a venue_id FK into arenas. Team-per-game (→ trades)
    is a JOIN of these home/away teams with a player's on-ice side, never a stored season attribute;
    game_type == 2 selects the regular season."""
    raw2id = {raw: r.venue_id for r in arenas.itertuples(index=False) for raw in r.raw_names}
    g = games_raw.copy()
    g["venue_id"] = g["raw_venue"].map(lambda v: raw2id.get(v, -1)).astype(int)
    return (
        g[["nhl_game_id", "date", "season", "game_type", "home_team", "away_team", "venue_id"]]
        .sort_values("nhl_game_id")
        .reset_index(drop=True)
    )


def _model_seasons() -> list[int]:
    """Seasons with processed interim data (the window the dims cover — matches player_season)."""
    return sorted(int(p.stem) for p in (C.INTERIM / "roster").glob("*.parquet"))


def main() -> None:
    C.DIM.mkdir(parents=True, exist_ok=True)

    players = build_players()
    players.to_parquet(C.DIM / "players.parquet", index=False)
    print(
        f"[dimensions] players: {len(players):,} rows "
        f"({int(players.birthdate.notna().sum()):,} with birthdate, "
        f"{int(players.draft_overall.notna().sum()):,} with draft)"
    )

    ps = build_player_season()
    ps.to_parquet(C.DIM / "player_season.parquet", index=False)
    nseas = ps.season.nunique() if len(ps) else 0
    print(
        f"[dimensions] player_season: {len(ps):,} rows "
        f"({ps.player_id.nunique():,} players × {nseas} seasons)"
    )

    games_raw = _scan_games(_model_seasons())
    arenas = build_arenas(games_raw)
    arenas.to_parquet(C.DIM / "arenas.parquet", index=False)
    print(
        f"[dimensions] arenas: {len(arenas):,} venues "
        f"({int((arenas.canonical_name.isin(games_raw.raw_venue) == False).sum())} aliased)"
    )

    games = build_games(games_raw, arenas)
    games.to_parquet(C.DIM / "games.parquet", index=False)
    reg = int((games.game_type == 2).sum())
    print(
        f"[dimensions] games: {len(games):,} rows ({reg:,} regular-season, "
        f"{len(games) - reg:,} other) across {games.season.nunique()} seasons"
    )


if __name__ == "__main__":
    main()
