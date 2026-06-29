"""Tests for stint construction + shot→stint attribution (yhattrick.stints).

Background — what a stint is. ``build_stints_for_game`` splits a game at every instant a player
hops on or off (``bounds`` = the union of shift start/end times). Each consecutive pair ``(t0, t1)``
is a stint: a maximal interval of constant on-ice personnel. For each stint we record the personnel
+ strength, the model xGF (summed from the shots table), and pbp shot-attempt volume
(Corsi / Fenwick / SOG counted from the events table), plus the score state and zone start.

Boundary semantics — which events fall on which side of a boundary B. Game-seconds are elapsed
*play* time, so a stoppage freezes the clock and several events share the same second. A line change
happens at a stoppage, so at a boundary B the natural ordering is:
    live-play event (shot, goal, hit, block) → stoppage → line change (the boundary) → faceoff.
So at B:
  * a **shot / goal** happened during the live play that just ended → it belongs to the stint
    *ending* at B (the personnel and strength on the ice when it was taken);
  * the **faceoff** that resumes play belongs to the stint *starting* at B.

This is why a *goal* (which stops play, and whose penalty expiry starts a fresh even-strength stint
at the same second) must attribute to the ending stint. The original ``[t0, t1)`` / ``bisect_right``
rule put it in the post-goal stint instead. Real-data impact: ~26% of all goals (the PP-goal share)
had their stint strength disagree with the pbp situationCode — e.g. game ``2024020001``, Paul
Cotter's goal at t=3448s, filed ``5v5`` though the situationCode was ``6v5``. Non-goal shots
(which don't stop play) were unaffected (~0.25%).

The fix uses ``(t0, t1]`` for both the xGF sum and the Corsi/Fenwick/SOG volume, and
``shot_stint_index`` (``bisect_left``) for the per-shot on-ice table — so all shot-attempt
attribution agrees — while non-shot events (faceoffs) keep opening the next stint.
"""
from __future__ import annotations

import bisect

import numpy as np
import pandas as pd
import pytest

from yhattrick import config as C
from yhattrick.data.stints import (
    assign_other, assign_shotlike, build_stints_for_game, classify_onice_match,
    drop_foreign_shifts, other_stint_index, personnel_per_stint, shot_stint_index, stint_bounds,
    teams_reconcile,
)


# --- the pure attribution helper -------------------------------------------------------------

def test_shot_stint_index_interior():
    # a shot strictly inside a stint maps to that stint, regardless of tie-break rule
    starts = [0, 100, 200]
    assert shot_stint_index(starts, 50) == 0
    assert shot_stint_index(starts, 150) == 1
    assert shot_stint_index(starts, 250) == 2


def test_shot_stint_index_on_boundary_belongs_to_ending_stint():
    # the regression: a shot exactly on a boundary belongs to the stint ENDING there (the one being
    # played when it was taken), not the stint starting there.
    starts = [0, 100, 200]
    assert shot_stint_index(starts, 100) == 0     # ends stint 0
    assert shot_stint_index(starts, 200) == 1     # ends stint 1
    # the old, buggy rule would have returned the *next* stint:
    assert bisect.bisect_right(starts, 100) - 1 == 1   # documents the prior behavior


def test_shot_stint_index_before_first_stint_is_guarded():
    # a second before the first stint has no owner (no shot occurs at the opening faceoff)
    assert shot_stint_index([100, 200], 100) == -1
    assert shot_stint_index([100, 200], 50) == -1


# --- fixtures -------------------------------------------------------------------------------

def _pp_goal_game():
    """HOME on a 5-on-4 power play from 0–100s; the penalized AWAY skater (24) returns at t=100,
    creating a 5-on-5 stint, and HOME scores at t=100 — the canonical 'PP goal ends the penalty'
    boundary case. Returns (shifts, shots, events, goalie_ids, home_team)."""
    home, away = "AAA", "BBB"
    rows = []
    for pid in [1, 10, 11, 12, 13, 14]:          # HOME goalie + 5 skaters
        rows.append((0, 200, pid, home))
    for pid in [2, 20, 21, 22, 23]:              # AWAY goalie + 4 skaters (down a man)
        rows.append((0, 200, pid, away))
    rows.append((100, 200, 24, away))            # AWAY penalized skater returns at t=100 -> 5v5
    shifts = pd.DataFrame(rows, columns=["start_g", "end_g", "player_id", "team"])
    shots = pd.DataFrame({"time_g": [100], "is_home": [1], "xg": [0.6]})   # HOME goal at the boundary
    events = pd.DataFrame({"time_g": [100, 100], "type": ["goal", "faceoff"],
                           "is_home": [True, True], "zone": [None, "O"]})
    return shifts, shots, events, {1, 2}, home


def _event_types_game():
    """Same 5v4→5v5 frame, but with one of every shot-attempt event type so we can pin down which
    side of the boundary each lands on:
      t=50  HOME shot-on-goal        (interior, stint 0)
      t=80  blocked-shot owned by AWAY (the blocker) -> HOME is the shooter (interior, stint 0)
      t=100 HOME goal                (ON the boundary -> ending stint 0, the 5v4)
      t=100 faceoff won by HOME, O   (ON the boundary -> starting stint 1, the 5v5)
      t=150 AWAY missed-shot         (interior, stint 1)
    """
    shifts, _, _, goalies, home = _pp_goal_game()
    events = pd.DataFrame({
        "time_g": [50, 80, 100, 100, 150],
        "type": ["shot-on-goal", "blocked-shot", "goal", "faceoff", "missed-shot"],
        "is_home": [True, False, True, True, False],   # blocked-shot is owned by the blocking team
        "zone": [None, None, None, "O", None],
    })
    # the shots table is Fenwick only (no blocked shots); carries the model xG
    shots = pd.DataFrame({"time_g": [50, 100, 150], "is_home": [1, 1, 0], "xg": [0.1, 0.6, 0.2]})
    return shifts, shots, events, goalies, home


def _by_strength(stints):
    return {s["strength"]: s for s in stints}


# --- xGF attribution across the boundary ----------------------------------------------------

def test_pp_goal_xg_credited_to_powerplay_stint():
    shifts, shots, events, goalies, home = _pp_goal_game()
    st = _by_strength(build_stints_for_game(shifts, shots, events, goalies, home))
    assert set(st) == {"5v4", "5v5"}
    assert st["5v4"]["home_xgf"] == pytest.approx(0.6)   # scored on the PP -> PP stint
    assert st["5v5"]["home_xgf"] == pytest.approx(0.0)   # not the post-goal even-strength stint


def test_no_xg_double_count_across_the_boundary():
    shifts, shots, events, goalies, home = _pp_goal_game()
    stints = build_stints_for_game(shifts, shots, events, goalies, home)
    assert sum(s["home_xgf"] for s in stints) == pytest.approx(0.6)
    assert sum(s["away_xgf"] for s in stints) == pytest.approx(0.0)


# --- Corsi / Fenwick / SOG volume by event type and side of boundary ------------------------

def test_goal_volume_counts_in_the_ending_stint():
    # the goal at t=100 must count toward the 5v4 stint's Corsi/Fenwick/SOG, not the 5v5 stint's
    shifts, shots, events, goalies, home = _event_types_game()
    st = _by_strength(build_stints_for_game(shifts, shots, events, goalies, home))
    pp, ev = st["5v4"], st["5v5"]
    # 5v4: sog@50 + block@80 + goal@100 = 3 Corsi; sog@50 + goal@100 = 2 Fenwick = 2 SOG
    assert (pp["home_corsi"], pp["home_fen"], pp["home_sog"]) == (3, 2, 2)
    # the goal does NOT leak into the post-goal even-strength stint
    assert (ev["home_corsi"], ev["home_fen"], ev["home_sog"]) == (0, 0, 0)


def test_blocked_shot_attributed_to_the_shooting_team():
    # a blocked shot is owned (in the pbp) by the blocking team; the shooter is the opponent, so it
    # is the SHOOTING team's Corsi, and it is Corsi-only (not Fenwick / not a shot on goal)
    shifts, shots, events, goalies, home = _event_types_game()
    pp = _by_strength(build_stints_for_game(shifts, shots, events, goalies, home))["5v4"]
    assert pp["home_corsi"] == 3 and pp["away_corsi"] == 0
    assert pp["home_fen"] == 2          # block excluded from Fenwick
    assert pp["home_sog"] == 2          # and from shots on goal


def test_missed_shot_is_fenwick_but_not_sog():
    shifts, shots, events, goalies, home = _event_types_game()
    ev = _by_strength(build_stints_for_game(shifts, shots, events, goalies, home))["5v5"]
    assert (ev["away_corsi"], ev["away_fen"], ev["away_sog"]) == (1, 1, 0)


def test_volume_partition_is_complete_and_not_double_counted():
    # every shot-attempt event lands in exactly one stint
    shifts, shots, events, goalies, home = _event_types_game()
    stints = build_stints_for_game(shifts, shots, events, goalies, home)
    corsi = sum(s["home_corsi"] + s["away_corsi"] for s in stints)
    fen = sum(s["home_fen"] + s["away_fen"] for s in stints)
    sog = sum(s["home_sog"] + s["away_sog"] for s in stints)
    xgf = sum(s["home_xgf"] + s["away_xgf"] for s in stints)
    assert (corsi, fen, sog) == (4, 3, 2)            # sog+block+goal+missed; minus block; minus missed/block
    assert xgf == pytest.approx(0.9)                 # 0.1 + 0.6 + 0.2


# --- non-shot events: the faceoff opens the next stint --------------------------------------

def test_faceoff_at_boundary_opens_the_new_stint():
    shifts, shots, events, goalies, home = _event_types_game()
    ev = _by_strength(build_stints_for_game(shifts, shots, events, goalies, home))["5v5"]
    assert ev["start_g"] == 100
    assert ev["start_type"] == "faceoff"             # the faceoff belongs to the stint it opens
    assert ev["start_zone"] == "O"                   # offensive zone (home perspective)


def test_powerplay_stint_starts_on_the_fly_not_a_faceoff():
    shifts, shots, events, goalies, home = _event_types_game()
    pp = _by_strength(build_stints_for_game(shifts, shots, events, goalies, home))["5v4"]
    assert pp["start_g"] == 0 and pp["start_type"] == "fly"


# --- score state across a goal boundary -----------------------------------------------------

def test_score_state_reflects_the_boundary_goal():
    # the stint that begins right after the goal is played at the post-goal score
    shifts, shots, events, goalies, home = _event_types_game()
    st = _by_strength(build_stints_for_game(shifts, shots, events, goalies, home))
    assert st["5v4"]["home_lead"] == 0               # before the goal
    assert st["5v5"]["home_lead"] == 1               # after the HOME goal at t=100


# --- vectorized attribution vs. scalar reference (no real data needed) ----------------------

def test_assign_shotlike_matches_scalar_oracle():
    rng = np.random.default_rng(0)
    for _ in range(300):
        k = int(rng.integers(2, 8))
        bounds = np.sort(rng.choice(np.arange(0, 60), size=k, replace=False)).astype(np.int64)
        starts = bounds[:-1].tolist()
        ts = rng.integers(-3, int(bounds[-1]) + 4, size=24)
        got = assign_shotlike(bounds, ts)
        for t, g in zip(ts, got):
            if t > bounds[-1]:
                assert g == -1                                   # upper guard
            else:
                assert g == shot_stint_index(starts, int(t))      # equals the scalar oracle


def test_assign_other_matches_scalar_oracle():
    rng = np.random.default_rng(1)
    for _ in range(300):
        k = int(rng.integers(2, 8))
        bounds = np.sort(rng.choice(np.arange(0, 60), size=k, replace=False)).astype(np.int64)
        starts = bounds[:-1].tolist()
        ts = rng.integers(0, int(bounds[-1]) + 1, size=24)
        got = assign_other(bounds, ts)
        for t, g in zip(ts, got):
            assert g == other_stint_index(starts, int(t))


def test_assign_shotlike_upper_and_lower_guards():
    bounds = np.array([0, 100, 200])
    idx = assign_shotlike(bounds, [250, 200, 100, 50, 0, -5])
    assert list(idx) == [-1, 1, 0, 0, -1, -1]   # 250 past end -> drop; 200 ends stint1; 0/-5 pre-first


# --- personnel (pure) -----------------------------------------------------------------------

def _personnel(rows, goalies, home="AAA"):
    sh = pd.DataFrame(rows, columns=["start_g", "end_g", "player_id", "team"])
    return personnel_per_stint(sh, stint_bounds(sh), goalies, home)


def test_personnel_boundary_off_by_one():
    # A's shift ENDS at 100 -> off stint(100,200); B's shift STARTS at 100 -> off stint(0,100)
    rows = [(0, 200, 1, "AAA"), (0, 200, 2, "BBB"),       # goalies, full game
            (0, 100, 10, "AAA"), (100, 200, 11, "AAA"),   # A then B
            (0, 200, 20, "BBB")]
    hs, as_, hg, ag = _personnel(rows, {1, 2})
    assert hs == [[10], [11]]
    assert hg == [1, 1] and ag == [2, 2]


def test_personnel_strengths_3v3_and_pulled_goalie():
    three = ([(0, 100, 1, "AAA"), (0, 100, 2, "BBB")]
             + [(0, 100, p, "AAA") for p in (10, 11, 12)]
             + [(0, 100, p, "BBB") for p in (20, 21, 22)])
    hs, as_, hg, ag = _personnel(three, {1, 2})
    assert (len(hs[0]), len(as_[0])) == (3, 3) and hg[0] == 1 and ag[0] == 2

    pulled = ([(0, 100, 2, "BBB")]                                   # only AWAY has a goalie
              + [(0, 100, p, "AAA") for p in range(10, 16)]          # HOME pulled: 6 skaters
              + [(0, 100, p, "BBB") for p in range(20, 25)])         # AWAY: 5 skaters
    hs, as_, hg, ag = _personnel(pulled, {1, 2})
    assert (len(hs[0]), hg[0], len(as_[0]), ag[0]) == (6, None, 5, 2)


def test_teams_reconcile():
    ok = pd.DataFrame({"team": ["AAA", "AAA", "BBB"]})
    assert teams_reconcile(ok, "AAA", "BBB")
    assert teams_reconcile(ok, "AAA", None)                          # check disabled
    assert not teams_reconcile(pd.DataFrame({"team": ["AAA", "VGK"]}), "AAA", "BBB")  # foreign team
    assert not teams_reconcile(pd.DataFrame({"team": ["AAA"]}), "BBB", "CCC")          # home absent


def test_drop_foreign_shifts_is_general():
    # generic: keep exactly the two teams the pbp says are playing, drop anything else — no game- or
    # abbrev-specific logic, any number of spliced-in foreign teams.
    sh = pd.DataFrame({"team": ["NJD", "BUF", "VGK", "SJS", "VGK"], "player_id": [1, 2, 3, 4, 5],
                       "start_g": [0] * 5, "end_g": [10] * 5})
    cleaned, foreign = drop_foreign_shifts(sh, "NJD", "BUF")
    assert foreign == {"VGK", "SJS"} and set(cleaned.team) == {"NJD", "BUF"} and len(cleaned) == 2
    # a clean game is a no-op (same object, no foreign)
    clean = sh.iloc[:2]
    out, f = drop_foreign_shifts(clean, "NJD", "BUF")
    assert f == set() and len(out) == 2
    # unknown away team -> can't tell what's foreign, leave untouched
    out2, f2 = drop_foreign_shifts(sh, "NJD", None)
    assert f2 == set() and len(out2) == 5


@pytest.mark.parametrize("hn,an,sh,sa,exp", [
    (5, 5, 5, 5, "exact"), (5, 4, 5, 4, "exact"),
    (5, 4, 5, 5, "within1"), (4, 5, 5, 5, "within1"), (4, 4, 5, 5, "within1"),
    (5, 3, 5, 5, "large"), (3, 5, 5, 5, "large"),
    (5, 5, None, 5, "large"), (5, 5, 5, None, "large"),
])
def test_classify_onice_match(hn, an, sh, sa, exp):
    assert classify_onice_match(hn, an, sh, sa) == exp


# --- whole-build invariants -----------------------------------------------------------------

def test_stint_idx_is_contiguous():
    shifts, shots, events, goalies, home = _event_types_game()
    stints = build_stints_for_game(shifts, shots, events, goalies, home)
    assert [s["stint_idx"] for s in stints] == list(range(len(stints)))


def test_shootout_goal_excluded_from_score_state():
    # a period-5 (shootout) goal far in the future must not move any stint's home_lead
    shifts, shots, _, goalies, home = _pp_goal_game()
    events = pd.DataFrame({
        "time_g": [50, 5000], "type": ["goal", "goal"], "is_home": [True, True],
        "zone": [None, None], "period": [1, 5],            # the 2nd is a shootout decider
    })
    st = _by_strength(build_stints_for_game(shifts, shots, events, goalies, home))
    # the regulation goal at t=50 (stint 0) lifts only the stint(s) that start at/after it
    assert st["5v4"]["home_lead"] == 0 and st["5v5"]["home_lead"] == 1   # SO goal adds nothing


def test_empty_events_game_builds():
    shifts, _, _, goalies, home = _pp_goal_game()
    empty_ev = pd.DataFrame(columns=["time_g", "type", "is_home", "zone", "period"])
    no_shots = pd.DataFrame(columns=["time_g", "is_home", "xg"])
    stints = build_stints_for_game(shifts, no_shots, empty_ev, goalies, home)
    assert len(stints) == 2
    for s in stints:
        assert s["home_xgf"] == 0.0 and s["home_corsi"] == 0 and s["away_corsi"] == 0
        assert s["start_type"] == "fly" and s["home_lead"] == 0


# --- data-motivated regression on the real processed output ---------------------------------

def _processed_seasons():
    d = C.PROCESSED / "shots_onice"
    return sorted(int(p.stem) for p in d.glob("*.parquet")) if d.exists() else []


@pytest.mark.parametrize("season", _processed_seasons())
def test_real_goals_strength_matches_situationcode(season):
    """On real data, a goal's reconstructed stint strength (even vs. special teams) must agree with
    its play-by-play situationCode. The boundary bug broke this for ~26% of goals; after the fix the
    disagreement should be negligible (a handful of genuinely ambiguous shift-timing cases)."""
    df = pd.read_parquet(C.PROCESSED / "shots_onice" / f"{season}.parquet",
                         columns=["event", "goal", "strength", "sit_home_n", "sit_away_n"])
    g = df[(df.event == "goal") & df.sit_home_n.notna() & df.sit_away_n.notna()].copy()
    parts = g.strength.str.split("v", expand=True)
    stint_even = pd.to_numeric(parts[0]) == pd.to_numeric(parts[1])
    sit_even = g.sit_home_n == g.sit_away_n
    mismatch = (stint_even != sit_even).mean()
    assert mismatch < 0.03, f"{season}: {mismatch:.1%} of goals misfiled by strength (boundary bug?)"
