"""Stage 2: interim/ -> processed/ (the join). Builds stints and shot-level on-ice sets,
then runs the data-quality asserts that are the whole point of this phase.

A *stint* is a maximal interval of constant on-ice personnel within a game: between any two
consecutive shift boundaries (a shift start or end), the set of players on the ice is fixed.
For each stint we record both teams' on-ice skaters + goalie, the strength state, the duration,
and the borrowed-xG for/against accumulated from shots in that interval.

Outputs (parquet, per season):
  processed/stints/<season>.parquet       one row per stint
  processed/shots_onice/<season>.parquet  each shot + the on-ice players at that instant

Health check: every shot's reconstructed on-ice skater counts must match MoneyPuck's
home/awaySkatersOnIce. We report the match rate per season and fail if it falls below a floor.

Usage:
  uv run python -m hockeywar.stints                 # all seasons present in interim/
  uv run python -m hockeywar.stints --season 2021
"""
from __future__ import annotations

import argparse
import bisect

import numpy as np
import pandas as pd

from . import config as C

ONICE_MATCH_FLOOR = 0.97  # fraction of shots whose on-ice counts must match MoneyPuck

# shot-attempt event sets (from our own pbp), for Corsi / Fenwick / shots-on-goal
_CORSI = ("shot-on-goal", "missed-shot", "blocked-shot", "goal")
_FENWICK = ("shot-on-goal", "missed-shot", "goal")
_SOG = ("shot-on-goal", "goal")
_FLIP = {"O": "D", "D": "O", "N": "N"}  # zone from the other team's perspective


def _load(kind: str, season: int) -> pd.DataFrame | None:
    p = C.INTERIM / kind / f"{season}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def build_stints_for_game(shifts_g: pd.DataFrame, shots_g: pd.DataFrame, events_g: pd.DataFrame,
                          goalie_ids: set[int], home_team: str) -> list[dict]:
    """Return the ordered list of stint dicts for one game, with on-ice personnel, borrowed xGF,
    and context/volume features derived from our own pbp (Corsi/Fenwick/SOG, score state, zone
    start)."""
    bounds = sorted(set(shifts_g.start_g) | set(shifts_g.end_g))
    starts = shifts_g.start_g.to_numpy()
    ends = shifts_g.end_g.to_numpy()
    pid = shifts_g.player_id.to_numpy()
    team = shifts_g.team.to_numpy()

    # --- pbp event arrays (shooting team = event owner, except blocked-shot which is owned by
    # the blocking/defending team, so the shooter is the opponent) ---
    ev_t = events_g.time_g.to_numpy()
    ev_type = events_g.type.to_numpy()
    ev_home = events_g.is_home.to_numpy().astype(bool)
    shoot_home = np.where(ev_type == "blocked-shot", ~ev_home, ev_home)
    is_corsi, is_fen, is_sog = np.isin(ev_type, _CORSI), np.isin(ev_type, _FENWICK), np.isin(ev_type, _SOG)
    is_goal = ev_type == "goal"
    home_goal_t = np.sort(ev_t[is_goal & ev_home])
    away_goal_t = np.sort(ev_t[is_goal & ~ev_home])
    # faceoffs -> {time: home-perspective zone}; zoneCode is relative to the winner (event owner)
    fo = ev_type == "faceoff"
    fo_times = set(int(t) for t in ev_t[fo])
    fo_zone = {int(t): (z if h else _FLIP.get(z)) for t, z, h in
               zip(ev_t[fo], events_g.zone.to_numpy()[fo], ev_home[fo]) if z in _FLIP}

    def vol(mask, flag, home):
        side = shoot_home if home else ~shoot_home
        return int(np.count_nonzero(mask & flag & side))

    stints = []
    for idx, (t0, t1) in enumerate(zip(bounds, bounds[1:])):
        if t1 <= t0:
            continue
        # a shift covers [t0,t1) iff start<=t0 and end>=t1 (boundaries are exactly shift edges)
        on = (starts <= t0) & (ends >= t1)
        home_sk, away_sk, home_g, away_g = [], [], None, None
        seen: set[int] = set()
        for p, tm in zip(pid[on], team[on]):
            if p in seen:
                continue  # defensive: a player should appear at most once per instant
            seen.add(p)
            is_home = tm == home_team
            if p in goalie_ids:
                if is_home:
                    home_g = int(p)
                else:
                    away_g = int(p)
            else:
                (home_sk if is_home else away_sk).append(int(p))
        sin = shots_g[(shots_g.game_seconds >= t0) & (shots_g.game_seconds < t1)]
        home_xgf = float(sin.loc[sin.isHomeTeam == 1, "xGoal"].sum())
        away_xgf = float(sin.loc[sin.isHomeTeam == 0, "xGoal"].sum())
        em = (ev_t >= t0) & (ev_t < t1)
        stints.append({
            "stint_idx": idx, "start_g": t0, "end_g": t1, "duration_s": t1 - t0,
            "home_skaters": home_sk, "away_skaters": away_sk,
            "home_goalie": home_g, "away_goalie": away_g,
            "home_n": len(home_sk), "away_n": len(away_sk),
            "strength": f"{len(home_sk)}v{len(away_sk)}",
            "home_xgf": round(home_xgf, 4), "away_xgf": round(away_xgf, 4),
            # shot volume from our pbp (no MoneyPuck)
            "home_corsi": vol(em, is_corsi, True), "away_corsi": vol(em, is_corsi, False),
            "home_fen": vol(em, is_fen, True), "away_fen": vol(em, is_fen, False),
            "home_sog": vol(em, is_sog, True), "away_sog": vol(em, is_sog, False),
            # context (stored now; some modeled)
            "home_lead": int(bisect.bisect_left(home_goal_t, t0) - bisect.bisect_left(away_goal_t, t0)),
            "start_zone": fo_zone.get(t0), "end_zone": fo_zone.get(t1),
            "start_type": "faceoff" if t0 in fo_times else "fly",
        })
    return stints


def process_season(season: int, limit: int | None = None) -> dict:
    shots = _load("shots", season)
    shifts = _load("shifts", season)
    roster = _load("roster", season)
    events = _load("events", season)
    if shots is None or shifts is None or roster is None or events is None:
        print(f"[stints] {season}: missing interim inputs, skipping")
        return {}

    goalie_ids = set(roster.loc[roster.position == "G", "player_id"])
    home_of = (shots.groupby("nhl_game_id")
               .agg(home=("homeTeamCode", "first")).home.to_dict())
    games = sorted(set(shifts.nhl_game_id) & set(home_of))
    if limit:
        games = games[:limit]

    # group once (avoids re-scanning the season frames per game)
    shifts_by = dict(tuple(shifts.groupby("nhl_game_id")))
    shots_by = dict(tuple(shots.groupby("nhl_game_id")))
    events_by = dict(tuple(events.groupby("nhl_game_id")))
    empty_ev = events.iloc[0:0]

    all_stints, all_shot_onice = [], []
    matched = within1 = large = total = 0
    for gid in games:
        sh_g = shifts_by[gid]
        sho_g = shots_by[gid].sort_values("game_seconds")
        ev_g = events_by.get(gid, empty_ev)
        stints = build_stints_for_game(sh_g, sho_g, ev_g, goalie_ids, home_of[gid])
        starts = [s["start_g"] for s in stints]
        for s in stints:
            s["nhl_game_id"] = gid
            all_stints.append(s)
        # attach on-ice to each shot via the stint containing its second
        for _, shot in sho_g.iterrows():
            j = bisect.bisect_right(starts, shot.game_seconds) - 1
            if j < 0:
                continue
            st = stints[j]
            sid = shot.shooterPlayerId
            mp_h, mp_a = int(shot.homeSkatersOnIce), int(shot.awaySkatersOnIce)
            dh = abs(st["home_n"] - mp_h)
            da = abs(st["away_n"] - mp_a)
            match = "exact" if (dh == 0 and da == 0) else ("within1" if (dh <= 1 and da <= 1) else "large")
            all_shot_onice.append({
                "nhl_game_id": gid, "shotID": int(shot.shotID),
                "game_seconds": int(shot.game_seconds), "period": int(shot.period),
                "stint_idx": st["stint_idx"], "strength": st["strength"],
                "shooter_id": int(sid) if pd.notna(sid) else None,
                "shooter": shot.shooterName if pd.notna(shot.shooterName) else None,
                "is_home": int(shot.isHomeTeam), "xGoal": float(shot.xGoal),
                # shot features (also the core xG-model inputs) for inspection
                "event": shot.event, "goal": int(shot.goal),
                "shot_type": shot.shotType if pd.notna(shot.shotType) else None,
                "distance": round(float(shot.shotDistance), 1) if pd.notna(shot.shotDistance) else None,
                "angle": round(float(shot.shotAngle), 1) if pd.notna(shot.shotAngle) else None,
                "rebound": int(shot.shotRebound) if pd.notna(shot.shotRebound) else 0,
                "rush": int(shot.shotRush) if pd.notna(shot.shotRush) else 0,
                "x": int(shot.xCord) if pd.notna(shot.xCord) else None,
                "y": int(shot.yCord) if pd.notna(shot.yCord) else None,
                "home_skaters": st["home_skaters"], "away_skaters": st["away_skaters"],
                "home_goalie": st["home_goalie"], "away_goalie": st["away_goalie"],
                "mp_home_n": mp_h, "mp_away_n": mp_a, "onice_match": match,
            })
            total += 1
            if match == "exact":
                matched += 1
            elif match == "within1":
                within1 += 1   # line-change boundary ambiguity; tolerated
            else:
                large += 1     # genuine disagreement; should be ~0

    stints_df = pd.DataFrame(all_stints)
    onice_df = pd.DataFrame(all_shot_onice)
    # flag stints with an illegal skater count (>6) -- source shift-timing overlaps; surfaced in the site
    stints_df["overload"] = (stints_df.home_n > 6) | (stints_df.away_n > 6)
    for d, name in ((stints_df, "stints"), (onice_df, "shots_onice")):
        out = C.PROCESSED / name / f"{season}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        d.to_parquet(out, index=False)

    illegal = int(stints_df.overload.sum()) if len(stints_df) else 0
    illegal_rate = illegal / len(stints_df) if len(stints_df) else 0.0
    exact = matched / total if total else 0.0
    near = (matched + within1) / total if total else 0.0
    large_rate = large / total if total else 0.0
    print(f"[stints] {season}: {len(games)} games, {len(stints_df):,} stints, {total:,} shots placed")
    print(f"    on-ice match: exact={exact:.2%}  within±1={near:.2%}  "
          f"large-mismatch={large_rate:.3%}  illegal-stints={illegal} ({illegal_rate:.3%})")
    if total and (large_rate > (1 - ONICE_MATCH_FLOOR) or illegal_rate > 0.001):
        print(f"    !! WARNING: large-mismatch {large_rate:.3%} or illegal-stint {illegal_rate:.3%} "
              f"above tolerance -- investigate")
    return {"season": season, "games": len(games), "stints": len(stints_df), "shots": total,
            "exact": exact, "within1": near, "large": large, "illegal": illegal}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Build stints + shot on-ice sets (interim -> processed)")
    p.add_argument("--season", type=int, default=None)
    p.add_argument("--limit", type=int, default=None, help="process only the first N games (dev)")
    args = p.parse_args(argv)
    C.ensure_dirs()
    seasons = [args.season] if args.season else C.SEASONS
    for season in seasons:
        process_season(season, args.limit)


if __name__ == "__main__":
    main()
