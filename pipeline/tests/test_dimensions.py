"""Tests for the dimension-table builders (yhattrick.data.dimensions).

The novel logic worth locking is in the venue/games builders: `build_arenas` must collapse
sponsor-rename aliases (via VENUE_ALIAS) onto ONE physical building so its scorer-bias states stay a
single random-walk chain, while real building moves stay split; `build_games` must resolve each
game's raw venue to that building's `venue_id` and expose the `game_type == 2` regular-season test.
These are pure functions over a small synthetic pbp-scan frame, so no data build is needed.
"""

import pandas as pd

from yhattrick.data import dimensions as D


def _scan_frame():
    # Two names for the same Florida building (aliased) + a distinct building; a reg game and a playoff.
    return pd.DataFrame(
        [
            {
                "nhl_game_id": 2021020001,
                "date": "2021-10-12",
                "season": 2021,
                "game_type": 2,
                "home_team": "FLA",
                "away_team": "TBL",
                "raw_venue": "FLA Live Arena",
            },
            {
                "nhl_game_id": 2022020002,
                "date": "2022-10-13",
                "season": 2022,
                "game_type": 2,
                "home_team": "FLA",
                "away_team": "BOS",
                "raw_venue": "Amerant Bank Arena",
            },
            {
                "nhl_game_id": 2022030003,
                "date": "2023-05-01",
                "season": 2022,
                "game_type": 3,
                "home_team": "TOR",
                "away_team": "MTL",
                "raw_venue": "Scotiabank Arena",
            },
        ],
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


def test_build_arenas_collapses_sponsor_aliases():
    arenas = D.build_arenas(_scan_frame())
    # FLA Live Arena aliases to Amerant Bank Arena → one building carrying both raw names.
    amerant = arenas[arenas.canonical_name == "Amerant Bank Arena"]
    assert len(amerant) == 1
    assert set(amerant.iloc[0].raw_names) == {"FLA Live Arena", "Amerant Bank Arena"}
    assert sorted(amerant.iloc[0].active_seasons) == [2021, 2022]
    # Distinct building stays separate; venue_id is unique and stable.
    assert (arenas.canonical_name == "Scotiabank Arena").sum() == 1
    assert arenas.venue_id.is_unique


def test_build_games_resolves_venue_fk_and_regular_season_test():
    scan = _scan_frame()
    arenas = D.build_arenas(scan)
    games = D.build_games(scan, arenas)
    amerant_id = arenas.loc[arenas.canonical_name == "Amerant Bank Arena", "venue_id"].iloc[0]
    by_game = dict(zip(games.nhl_game_id, games.venue_id))
    # Both Florida games (different raw names) resolve to the SAME building.
    assert by_game[2021020001] == amerant_id
    assert by_game[2022020002] == amerant_id
    assert (games.venue_id == -1).sum() == 0  # every game mapped
    # game_type == 2 is the regular-season test.
    assert set(games.loc[games.game_type == 2, "nhl_game_id"]) == {2021020001, 2022020002}
    assert set(games.loc[games.game_type == 3, "nhl_game_id"]) == {2022030003}


def test_player_season_carries_no_team():
    # The dimension deliberately omits team (a (player, game) property — trades). Guard the schema.
    cols = set(D.build_player_season().columns)
    assert "team" not in cols
    assert {"player_id", "season", "position"} <= cols
