"""Stage 2: interim/ -> processed/ (the join). Builds stints and shot-level on-ice sets,
then runs the data-quality asserts that are the whole point of this phase.

A *stint* is a maximal interval of constant on-ice personnel within a game: between any two
consecutive shift boundaries (a shift start or end), the set of players on the ice is fixed.
For each stint we record both teams' on-ice skaters + goalie, the strength state, the duration,
and the model xG for/against accumulated from shots in that interval.

Boundary attribution (the load-bearing rule, defined once here and reused everywhere — including
export_games): stints partition the game into half-open intervals. A *shot/goal* on a boundary
belongs to the stint ENDING there (the personnel on the ice when it was taken) — `(t0, t1]`,
`assign_shotlike` / `shot_stint_index`. A goal ends play and a fresh (often even-strength) stint
starts at the same second, so the `[t0, t1)` rule would misfile the goal's xG into the post-goal
personnel. Non-shot events (a faceoff that opens the next stint) use `[t0, t1)` — `assign_other`.

Outputs (parquet, per season):
  processed/stints/<season>.parquet       one row per stint
  processed/shots_onice/<season>.parquet  each shot + the on-ice players at that instant

Health check: every shot's reconstructed on-ice skater counts must match the pbp situationCode
skater counts. We report the match rate per season and fail if it falls below a floor.

Usage:
  uv run python -m yhattrick.stints                 # all seasons present in interim/
  uv run python -m yhattrick.stints --season 2021
"""
from __future__ import annotations

import argparse
import bisect

import numpy as np
import pandas as pd

from . import config as C

ONICE_MATCH_FLOOR = 0.97  # fraction of shots whose on-ice counts must match the pbp situationCode

# shot-attempt event sets (from our own pbp), for Corsi / Fenwick / shots-on-goal
_CORSI = ("shot-on-goal", "missed-shot", "blocked-shot", "goal")
_FENWICK = ("shot-on-goal", "missed-shot", "goal")
_SOG = ("shot-on-goal", "goal")
# every shot attempt is attributed with the (t0,t1] "ending stint" rule; other events open the next
_SHOTLIKE = frozenset(_CORSI)
_FLIP = {"O": "D", "D": "O", "N": "N"}  # zone from the other team's perspective


# --- boundary attribution: ONE source of truth -----------------------------------------------
# Scalar forms (used by export_games and as test oracles); vectorized forms used in the hot paths.

def shot_stint_index(starts: list[int], t: int) -> int:
    """Stint index for a shot/goal at second ``t``: the stint ENDING at ``t`` if ``t`` is on a
    boundary (``(t0, t1]``). ``starts`` is the ascending list of stint start times. -1 before the
    first stint."""
    return bisect.bisect_left(starts, t) - 1


def other_stint_index(starts: list[int], t: int) -> int:
    """Stint index for a non-shot event at second ``t``: the stint STARTING at ``t`` if on a
    boundary (``[t0, t1)``). -1 before the first stint."""
    return bisect.bisect_right(starts, t) - 1


def assign_shotlike(bounds: np.ndarray, t) -> np.ndarray:
    """Vectorized `shot_stint_index` over stint `bounds` (the full edge list, len = nstints+1).
    Returns -1 before the first stint AND after the last stint's end (guards shots past the last
    recorded shift — e.g. gappy 2025 shift data — from latching onto the final stint)."""
    t = np.asarray(t)
    n = len(bounds) - 1
    idx = np.minimum(np.searchsorted(bounds, t, side="left") - 1, n - 1)
    idx[t > bounds[-1]] = -1
    return idx


def assign_other(bounds: np.ndarray, t) -> np.ndarray:
    """Vectorized `other_stint_index` (the `[t0, t1)` rule for non-shot events). The upper clamp maps
    t == bounds[-1] to the last stint (matching `bisect_right` over the stint-start list), since
    searchsorted over the full edge list would otherwise return an out-of-range index."""
    t = np.asarray(t)
    n = len(bounds) - 1
    idx = np.minimum(np.searchsorted(bounds, t, side="right") - 1, n - 1)
    idx[t > bounds[-1]] = -1
    return idx


# --- pure building blocks (unit-testable) ----------------------------------------------------

def stint_bounds(shifts_g: pd.DataFrame) -> np.ndarray:
    """Sorted unique shift-edge times. Stint i is the open interval (bounds[i], bounds[i+1])."""
    return np.array(sorted(set(shifts_g.start_g) | set(shifts_g.end_g)), dtype=np.int64)


def teams_reconcile(shifts_g: pd.DataFrame, home_team: str, away_team: str | None) -> bool:
    """True iff the shift table's team abbrevs are exactly this game's two teams. A mismatch means a
    corrupt shiftchart (e.g. a second game's shifts spliced in under the same gameId), which would
    silently mis-attribute `is_home`. None away_team => unknown, treated as reconciled."""
    if away_team is None:
        return True
    teams = set(shifts_g.team.unique())
    return home_team in teams and teams <= {home_team, away_team}


def drop_foreign_shifts(shifts_g: pd.DataFrame, home_team: str,
                        away_team: str | None) -> tuple[pd.DataFrame, set]:
    """Recover a corrupt shiftchart by keeping only the two teams the pbp says are playing.

    Some NHL shift feeds splice a second game's shifts in under one gameId (e.g. game 2025020565 had
    VGK/SJS shifts mixed into a BUF@NJD game). The pbp (events/shots) is unaffected, so we keep the
    home+away shifts and drop the foreign ones — recovering a clean game instead of discarding it.
    Returns (cleaned_shifts, foreign_team_abbrevs). Away unknown => no-op (can't tell what's foreign)."""
    if away_team is None:
        return shifts_g, set()
    foreign = set(shifts_g.team.unique()) - {home_team, away_team}
    if not foreign:
        return shifts_g, set()
    return shifts_g[shifts_g.team.isin({home_team, away_team})], foreign


def personnel_per_stint(shifts_g: pd.DataFrame, bounds: np.ndarray, goalie_ids: set[int],
                        home_team: str) -> tuple[list, list, list, list]:
    """On-ice skaters + goalie for every stint. A shift [s,e] covers stint i iff s<=bounds[i] and
    e>=bounds[i+1]. Skater lists follow shift-table order (player_id-ascending), deduped."""
    starts = shifts_g.start_g.to_numpy()
    ends = shifts_g.end_g.to_numpy()
    pid = shifts_g.player_id.to_numpy()
    team = shifts_g.team.to_numpy()
    n = len(bounds) - 1
    home_sk: list = [None] * n
    away_sk: list = [None] * n
    home_g: list = [None] * n
    away_g: list = [None] * n
    for i in range(n):
        t0, t1 = bounds[i], bounds[i + 1]
        on = (starts <= t0) & (ends >= t1)
        hs: list[int] = []
        a_s: list[int] = []
        hg = ag = None
        seen: set[int] = set()
        for p, tm in zip(pid[on], team[on]):
            if p in seen:
                continue  # defensive: a player should appear at most once per instant
            seen.add(p)
            is_home = tm == home_team
            if p in goalie_ids:
                if is_home:
                    hg = int(p)
                else:
                    ag = int(p)
            else:
                (hs if is_home else a_s).append(int(p))
        home_sk[i], away_sk[i], home_g[i], away_g[i] = hs, a_s, hg, ag
    return home_sk, away_sk, home_g, away_g


def xgf_per_stint(bounds: np.ndarray, shots_g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Model xGF accumulated per stint, by side (NaN xg from empty-net/penalty shots contributes 0,
    matching a NaN-skipping sum)."""
    n = len(bounds) - 1
    if not len(shots_g):
        return np.zeros(n), np.zeros(n)
    sidx = assign_shotlike(bounds, shots_g.time_g.to_numpy())
    home = shots_g.is_home.to_numpy() == 1
    xg = np.nan_to_num(shots_g.xg.to_numpy(dtype=float))
    v = sidx >= 0
    hxg = np.bincount(sidx[v & home], weights=xg[v & home], minlength=n)
    axg = np.bincount(sidx[v & ~home], weights=xg[v & ~home], minlength=n)
    return hxg, axg


def volume_per_stint(bounds: np.ndarray, events_g: pd.DataFrame) -> dict[str, np.ndarray]:
    """Corsi/Fenwick/SOG shot-attempt counts per stint, by shooting side (blocked shots are owned in
    the pbp by the blocking team, so their shooter is the opponent)."""
    n = len(bounds) - 1
    keys = ("home_corsi", "away_corsi", "home_fen", "away_fen", "home_sog", "away_sog")
    if not len(events_g):
        return {k: np.zeros(n, dtype=np.int64) for k in keys}
    ev_t = events_g.time_g.to_numpy()
    ev_type = events_g.type.to_numpy()
    ev_home = events_g.is_home.to_numpy().astype(bool)
    shoot_home = np.where(ev_type == "blocked-shot", ~ev_home, ev_home)
    sidx = assign_shotlike(bounds, ev_t)
    valid = sidx >= 0

    def counts(flag: np.ndarray, home: bool) -> np.ndarray:
        m = valid & flag & (shoot_home if home else ~shoot_home)
        return np.bincount(sidx[m], minlength=n)

    corsi, fen, sog = (np.isin(ev_type, _CORSI), np.isin(ev_type, _FENWICK), np.isin(ev_type, _SOG))
    return {
        "home_corsi": counts(corsi, True), "away_corsi": counts(corsi, False),
        "home_fen": counts(fen, True), "away_fen": counts(fen, False),
        "home_sog": counts(sog, True), "away_sog": counts(sog, False),
    }


def context_per_stint(bounds: np.ndarray, events_g: pd.DataFrame) -> tuple[np.ndarray, list, list, list]:
    """Score state (home_lead) + faceoff zone/start_type per stint. home_lead holds for the whole
    stint: a goal at t0 ended the prior stint, so this stint plays at the post-goal score
    (searchsorted 'right' counts a goal exactly at t0; a goal at t1 ends this stint, excluded).
    Shootout (period >= 5) goals are excluded from the score state."""
    n = len(bounds) - 1
    if not len(events_g):
        return np.zeros(n, dtype=np.int64), [None] * n, [None] * n, ["fly"] * n
    ev_t = events_g.time_g.to_numpy()
    ev_type = events_g.type.to_numpy()
    ev_home = events_g.is_home.to_numpy().astype(bool)
    period = (events_g.period.to_numpy() if "period" in events_g.columns
              else np.zeros(len(events_g), dtype=int))   # real events always carry period
    is_goal = (ev_type == "goal") & (period < 5)
    home_goal_t = np.sort(ev_t[is_goal & ev_home])
    away_goal_t = np.sort(ev_t[is_goal & ~ev_home])
    lead = (np.searchsorted(home_goal_t, bounds[:-1], side="right")
            - np.searchsorted(away_goal_t, bounds[:-1], side="right")).astype(np.int64)
    fo = ev_type == "faceoff"
    fo_times = {int(t) for t in ev_t[fo]}
    zone = events_g.zone.to_numpy()
    fo_zone = {int(t): (z if h else _FLIP.get(z)) for t, z, h in
               zip(ev_t[fo], zone[fo], ev_home[fo]) if z in _FLIP}
    szone = [fo_zone.get(int(bounds[i])) for i in range(n)]
    ezone = [fo_zone.get(int(bounds[i + 1])) for i in range(n)]
    stype = ["faceoff" if int(bounds[i]) in fo_times else "fly" for i in range(n)]
    return lead, szone, ezone, stype


def classify_onice_match(home_n: int, away_n: int, sit_h, sit_a) -> str:
    """Compare reconstructed on-ice skater counts to the pbp situationCode counts on the shot."""
    if sit_h is None or sit_a is None:
        return "large"                       # unverifiable strength (rare) — count as a mismatch
    dh, da = abs(home_n - sit_h), abs(away_n - sit_a)
    if dh == 0 and da == 0:
        return "exact"
    return "within1" if (dh <= 1 and da <= 1) else "large"


def build_stints_for_game(shifts_g: pd.DataFrame, shots_g: pd.DataFrame, events_g: pd.DataFrame,
                          goalie_ids: set[int], home_team: str) -> list[dict]:
    """Ordered list of stint dicts for one game: on-ice personnel, model xGF, pbp shot-attempt
    volume (Corsi/Fenwick/SOG), score state and zone start. Orchestrates the pure builders above."""
    bounds = stint_bounds(shifts_g)
    n = len(bounds) - 1
    if n <= 0:
        return []
    home_sk, away_sk, home_g, away_g = personnel_per_stint(shifts_g, bounds, goalie_ids, home_team)
    hxg, axg = xgf_per_stint(bounds, shots_g)
    vol = volume_per_stint(bounds, events_g)
    lead, szone, ezone, stype = context_per_stint(bounds, events_g)
    out = []
    for i in range(n):
        hn, an = len(home_sk[i]), len(away_sk[i])
        out.append({
            "stint_idx": i, "start_g": int(bounds[i]), "end_g": int(bounds[i + 1]),
            "duration_s": int(bounds[i + 1] - bounds[i]),
            "home_skaters": home_sk[i], "away_skaters": away_sk[i],
            "home_goalie": home_g[i], "away_goalie": away_g[i],
            "home_n": hn, "away_n": an, "strength": f"{hn}v{an}",
            "home_xgf": round(float(hxg[i]), 4), "away_xgf": round(float(axg[i]), 4),
            "home_corsi": int(vol["home_corsi"][i]), "away_corsi": int(vol["away_corsi"][i]),
            "home_fen": int(vol["home_fen"][i]), "away_fen": int(vol["away_fen"][i]),
            "home_sog": int(vol["home_sog"][i]), "away_sog": int(vol["away_sog"][i]),
            "home_lead": int(lead[i]), "start_zone": szone[i], "end_zone": ezone[i],
            "start_type": stype[i],
        })
    return out


def _load(kind: str, season: int) -> pd.DataFrame | None:
    p = C.INTERIM / kind / f"{season}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def _shot_onice_rows(gid: int, sho_g: pd.DataFrame, stints: list[dict], bounds: np.ndarray) -> tuple[list[dict], int]:
    """Attach each shot to the stint it was taken in and emit a shots_onice record + the QC class.
    Returns (rows, n_after_last) where n_after_last counts shots past the last stint (dropped)."""
    times = sho_g.time_g.to_numpy()
    sidx = assign_shotlike(bounds, times)
    rows, after_last = [], 0
    for k, shot in enumerate(sho_g.itertuples(index=False)):
        j = int(sidx[k])
        if j < 0:
            if shot.time_g > bounds[-1]:
                after_last += 1
            continue
        st = stints[j]
        sit_h = int(shot.home_n) if pd.notna(shot.home_n) else None
        sit_a = int(shot.away_n) if pd.notna(shot.away_n) else None
        rows.append({
            "nhl_game_id": gid, "event_idx": int(shot.event_idx),
            "game_seconds": int(shot.time_g), "period": int(shot.period),
            "stint_idx": st["stint_idx"], "strength": st["strength"],
            "shooter_id": int(shot.shooter_id) if pd.notna(shot.shooter_id) else None,
            "shooter": shot.shooter if pd.notna(shot.shooter) else None,
            "is_home": int(shot.is_home), "xg": float(shot.xg) if pd.notna(shot.xg) else None,
            "event": shot.event, "goal": int(shot.goal),
            "shot_type": shot.shot_type if pd.notna(shot.shot_type) else None,
            "distance": float(shot.distance) if pd.notna(shot.distance) else None,
            "angle": float(shot.angle) if pd.notna(shot.angle) else None,
            "rebound": int(shot.rebound), "rush": int(shot.rush),
            "x": int(shot.x) if pd.notna(shot.x) else None,
            "y": int(shot.y) if pd.notna(shot.y) else None,
            "home_skaters": st["home_skaters"], "away_skaters": st["away_skaters"],
            "home_goalie": st["home_goalie"], "away_goalie": st["away_goalie"],
            "sit_home_n": sit_h, "sit_away_n": sit_a,
            "onice_match": classify_onice_match(st["home_n"], st["away_n"], sit_h, sit_a),
        })
    return rows, after_last


def process_season(season: int, limit: int | None = None) -> dict:
    shots = _load("shots", season)
    shifts = _load("shifts", season)
    roster = _load("roster", season)
    events = _load("events", season)
    if shots is None or shifts is None or roster is None or events is None:
        print(f"[stints] {season}: missing interim inputs, skipping")
        return {}

    # attach the model xG to each shot (left join — empty-net/penalty shots stay with xg=NaN)
    xgp = C.PROCESSED / "xg" / f"{season}.parquet"
    xg = pd.read_parquet(xgp, columns=["nhl_game_id", "event_idx", "xg"]) if xgp.exists() else None
    if xg is None:
        print(f"[stints] {season}: missing processed/xg — run `make xg` first")
        return {}
    shots = shots.merge(xg, on=["nhl_game_id", "event_idx"], how="left")

    goalie_ids = set(roster.loc[roster.position == "G", "player_id"])
    home_of = shots.groupby("nhl_game_id").home_team.first().to_dict()
    away_of = events.loc[~events.is_home].groupby("nhl_game_id").team.first().to_dict()
    games = sorted(set(shifts.nhl_game_id) & set(home_of))
    if limit:
        games = games[:limit]

    # group once (avoids re-scanning the season frames per game)
    shifts_by = dict(tuple(shifts.groupby("nhl_game_id")))
    shots_by = dict(tuple(shots.groupby("nhl_game_id")))
    events_by = dict(tuple(events.groupby("nhl_game_id")))
    empty_ev = events.iloc[0:0]

    all_stints, all_shot_onice = [], []
    matched = within1 = large = total = after_last = salvaged = skipped = 0
    for gid in games:
        sh_g, foreign = drop_foreign_shifts(shifts_by[gid], home_of[gid], away_of.get(gid))
        if foreign:
            salvaged += 1
            print(f"    !! {gid}: corrupt shiftchart — dropped foreign-team shifts {sorted(foreign)}, "
                  f"kept {home_of[gid]}/{away_of.get(gid)}")
        if home_of[gid] not in set(sh_g.team.unique()):
            skipped += 1
            print(f"    !! {gid}: home team {home_of[gid]} absent from shifts — skipping")
            continue
        sho_g = shots_by[gid].sort_values("time_g")
        ev_g = events_by.get(gid, empty_ev)
        stints = build_stints_for_game(sh_g, sho_g, ev_g, goalie_ids, home_of[gid])
        if not stints:
            continue
        for s in stints:
            s["nhl_game_id"] = gid
            all_stints.append(s)
        bounds = np.array([s["start_g"] for s in stints] + [stints[-1]["end_g"]], dtype=np.int64)
        rows, n_after = _shot_onice_rows(gid, sho_g, stints, bounds)
        after_last += n_after
        all_shot_onice.extend(rows)
        for r in rows:
            total += 1
            if r["onice_match"] == "exact":
                matched += 1
            elif r["onice_match"] == "within1":
                within1 += 1
            else:
                large += 1

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
    print(f"[stints] {season}: {len(games)} games, {len(stints_df):,} stints, {total:,} shots placed"
          f"{f' ({salvaged} corrupt shiftcharts cleaned)' if salvaged else ''}"
          f"{f' ({skipped} games skipped)' if skipped else ''}"
          f"{f' ({after_last} shots past last stint dropped)' if after_last else ''}")
    print(f"    on-ice match: exact={exact:.2%}  within±1={near:.2%}  "
          f"large-mismatch={large_rate:.3%}  illegal-stints={illegal} ({illegal_rate:.3%})")
    if total and (large_rate > (1 - ONICE_MATCH_FLOOR) or illegal_rate > 0.001):
        print(f"    !! WARNING: large-mismatch {large_rate:.3%} or illegal-stint {illegal_rate:.3%} "
              f"above tolerance -- investigate")
    return {"season": season, "games": len(games), "stints": len(stints_df), "shots": total,
            "exact": exact, "within1": near, "large": large, "illegal": illegal,
            "salvaged_games": salvaged, "skipped_games": skipped, "shots_after_last": after_last}


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
