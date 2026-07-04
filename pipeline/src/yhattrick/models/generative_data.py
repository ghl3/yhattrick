"""Model-input tables → per-fit design rows for the generative player model.

This module reads the clean FACT tables (`processed/stints`, `processed/shots_onice`) and DIMENSION
tables (`dimensions/players`, `dimensions/games`, `dimensions/arenas`) and turns them into the dense,
indexed arrays each stage fits on: the shared player index, the aging-curve / position inputs, the
arena recording-bias state index, and the rate / quality / conversion design-row dicts. It also runs
the conversion pre-calculation (the empirical-Bayes prior SDs for finishing and goalie save skill).

Nothing here evaluates the likelihood or touches the optimizer — the fit engine (`generative_model`)
calls these builders once per fit and hands the arrays to the objective from `generative_likelihood`.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .. import config as C
from .player_onice_model import roster_names
from .generative_features import RATE_CTX, base_ctx, load_stints as _load_stints
from .generative_likelihood import (
    _sigmoid,
    AGE_PEAK,
    AGE_SCALE,
    EPS,
    MAX_DEF,
    ALL_STRENGTHS,
    PRIOR_SD_FIN,
    PRIOR_SD_GSAVE,
    PRIOR_SD_FLOOR,
    MIN_SHOTS_FIN_EST,
    MIN_SHOTS_GSAVE_EST,
)

# ── data loaders (5v5) ──────────────────────────────────────────────────────────────────────────


def player_index(seasons):
    """Shared player index across ALL modeled strengths (EV+MA), so the per-strength rate fits align to
    one player list. Returns (players, idx)."""
    df = _load_stints(seasons, ALL_STRENGTHS)
    if df.empty:
        return [], {}
    players = sorted(set().union(*df.home_skaters, *df.away_skaters))
    return players, {p: i for i, p in enumerate(players)}


def _season_cols(seasons):
    """Map season -> season-indicator column (first season = reference, dropped). Empty for one season."""
    sl = sorted(set(seasons))
    return {s: i for i, s in enumerate(sl[1:])}, max(len(sl) - 1, 0)


# ── game venues (the arena recording-bias offsets) ──────────────────────────────────────────────


def _game_venues(seasons):
    """{nhl_game_id: canonical venue name} for the seasons, from the games + arenas DIMENSIONS
    (`make dims`). NHL scorekeeping varies by building (shot counts AND recorded locations), so the
    rate/quality stages carry per-venue offsets. Canonicalisation — sponsor-rename aliases, so a
    venue's bias states stay one random-walk chain, while real building moves stay split — lives in the
    arenas table."""
    gp = C.DIM / "games.parquet"
    ap = C.DIM / "arenas.parquet"
    if not gp.exists() or not ap.exists():
        raise FileNotFoundError(f"games/arenas dimensions missing ({C.DIM}); run `make dims` first")
    sset = set(int(s) for s in seasons)
    g = pd.read_parquet(gp, columns=["nhl_game_id", "season", "venue_id"])
    g = g[g.season.isin(sset)]
    a = pd.read_parquet(ap, columns=["venue_id", "canonical_name"])
    vid2name = dict(zip(a.venue_id.astype(int), a.canonical_name))
    return {int(gid): vid2name.get(int(vid)) for gid, vid in zip(g.nhl_game_id, g.venue_id)}


ARENA_MIN_GAMES = 20  # venues with fewer games in the fit window (outdoor/neutral sites)
#   get no offset — too few games to estimate one


def _arena_index(seasons, min_games=ARENA_MIN_GAMES):
    """(Venue, SEASON)-state index for the arena recording-bias offsets. Rink bias is a scorer-crew
    phenomenon — persistent across adjacent seasons, movable when crews change — so each major venue
    gets one state per season it hosts games, ridged to zero (ARENA_SD) and smoothed across seasons
    by a random-walk penalty (ARENA_RW_SD). Rare venues (< min_games in the window: outdoor/neutral
    sites) map to −1 (no offset). Returns (game→venue map, {(venue, season): col}, pair machinery
    dict or None)."""
    ven = _game_venues(seasons)
    if not ven:
        return {}, {}, None
    counts = pd.Series(list(ven.values())).value_counts()
    majors = {v for v, c in counts.items() if c >= min_games}
    sset = set(int(s) for s in seasons)
    pairs = sorted(
        {(v, int(str(g)[:4])) for g, v in ven.items() if v in majors and int(str(g)[:4]) in sset}
    )
    if not pairs:
        return ven, {}, None
    pmap = {p: i for i, p in enumerate(pairs)}
    pv = [p[0] for p in pairs]
    ps = np.array([p[1] for p in pairs], dtype=np.int64)
    same = np.array([pv[i + 1] == pv[i] for i in range(len(pv) - 1)], dtype=bool)
    e_prev = np.nonzero(same)[0].astype(np.int64)
    e_next = e_prev + 1
    mach = {
        "venue": pv,
        "season": ps,
        "e_prev": e_prev,
        "e_next": e_next,
        "e_gap": (ps[e_next] - ps[e_prev]).astype(np.float64),
    }
    return ven, pmap, mach


# ── age & position (the shared aging-curve inputs) ──────────────────────────────────────────────


def _birthdates(ids):
    """Birthdates for a list of player/goalie ids from the players DIMENSION (`make dims`), aligned to
    `ids`; NaT for any id absent from the dimension."""
    p = C.DIM / "players.parquet"
    if not p.exists():
        raise FileNotFoundError(f"players dimension missing ({p}); run `make dims` first")
    df = pd.read_parquet(p, columns=["player_id", "birthdate"])
    bd = dict(zip(df.player_id.astype(int), pd.to_datetime(df.birthdate, errors="coerce")))
    return pd.to_datetime(pd.Series([bd.get(int(x)) for x in ids]), errors="coerce")


def _season_age(born, season):
    """Float age at Jan 1 of season+1 (mid-season) for a datetime Series; NaN if unknown."""
    return (pd.Timestamp(year=int(season) + 1, month=1, day=1) - born).dt.days.to_numpy() / 365.25


def _age_position(players, seasons):
    """Per-player ages and positions for the aging curves. Returns
    {"age": {season: (P,) float, NaN if unknown}, "z": {season: (P,) float, 0 if unknown},
     "isD": (P,) float 0/1, "missing": int}. Age = years at Jan 1 of season+1 (mid-season);
    z = (age − AGE_PEAK)/AGE_SCALE. A missing birthdate maps to z = 0 — the player sits AT the
    curve's reference point (contributes no age signal) instead of biasing the curve. Position from
    the season rosters; D = 1, anything else (F, unknown) = 0."""
    P = len(players)
    born = _birthdates(players)
    missing = int(born.isna().sum())
    names = roster_names(list(seasons))
    isD = np.array([1.0 if names.get(int(p), {}).get("pos") == "D" else 0.0 for p in players])
    age, z = {}, {}
    for s in sorted(set(seasons)):
        a = _season_age(born, s)
        age[s] = a
        z[s] = np.where(np.isnan(a), 0.0, (a - AGE_PEAK) / AGE_SCALE)
    return {"age": age, "z": z, "isD": isD, "missing": missing}


def _age_cols(z, d):
    """Position-split age-basis columns for skaters: [F·z, F·z², D·z, D·z²] (last axis).
    `z`/`d` broadcast — scalars or arrays."""
    z, d = np.asarray(z, dtype=np.float64), np.asarray(d, dtype=np.float64)
    f = 1.0 - d
    return np.stack([f * z, f * z * z, d * z, d * z * z], axis=-1)


def _curve_val(w, z, d):
    """Evaluate a fitted position-split age curve (coeffs w = [Fz, Fz², Dz, Dz²]) at z for
    position d (0=F, 1=D). Broadcasts over arrays."""
    return (1.0 - d) * (w[0] * z + w[1] * z * z) + d * (w[2] * z + w[3] * z * z)


def _shooter_counts(seasons, strengths):
    """{(nhl_game_id, stint_idx): {shooter_id: fenwick_count}} for shots in the given strengths (the
    shooter-resolved response). A player's shots in a stint are his regardless of side."""
    out: dict[tuple, dict] = defaultdict(lambda: defaultdict(int))
    for s in seasons:
        p = C.PROCESSED / "shots_onice" / f"{s}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["nhl_game_id", "stint_idx", "shooter_id", "strength"])
        d = d[d.strength.isin(strengths) & d.shooter_id.notna()]
        g = d.groupby(["nhl_game_id", "stint_idx", "shooter_id"]).size()
        for (gid, sidx, pid), n in g.items():
            out[(int(gid), int(sidx))][int(pid)] = int(n)
    return out


AGE_CTX = [
    "shoot_D",
    "create_D",
    "def_D",
    "shoot_zF",
    "shoot_z2F",
    "shoot_zD",
    "shoot_z2D",
    "create_zF",
    "create_z2F",
    "create_zD",
    "create_z2D",
    "def_zF",
    "def_z2F",
    "def_zD",
    "def_z2D",
]


def rate_rows(seasons, strengths, dual, players, idx, agepos=None, states=False, arenas=True):
    """Shooter-resolved rate design for ONE strength bucket, on the SHARED player index. For each stint
    side that attacks (EV/dual: both sides; MA: only the more-skaters side) and each on-ice attacker j
    as focal shooter: one row with j's Fenwick count, j's index, the 4 teammate indices, the (≤5)
    defender indices padded to width 5 with a `def_mask`, offset log(t/3600), and context. Context =
    5 base cols + season indicators + (when `agepos` is given) the position-offset and age-basis
    columns for the shoot/create/def blocks (AGE_CTX — the shared F/D aging curves + D intercepts).
    `ctx_names` names every context column so downstream code extracts curve coefficients by name.
    Also returns per-player attacking/defending TOI and each player's last active season.

    `states=True` (the EV bucket) additionally emits the per-(player, season) UNIT machinery for the
    random-walk drift states: `unit_player`/`unit_season` (compact index over active pairs, sorted by
    player then season), `shooter_unit`/`team_unit`/`def_unit` gathers (int32; masked def slots → 0),
    the RW edge list (`e_prev`, `e_next` unit positions, `e_gap` season gaps), `first_mask` (which
    unit carries the level ridge), per-unit attacking TOI, and `unit_lut` ((P, nS) player×season →
    unit, −1 if inactive) for mapping goal rows in run()."""
    df = _load_stints(seasons, strengths)
    if df.empty:
        return None
    counts = _shooter_counts(seasons, strengths)
    P = len(players)
    scol, nseas = _season_cols(seasons)
    slist = sorted(set(seasons))
    AC = {s: _age_cols(agepos["z"][s], agepos["isD"]) for s in slist} if agepos else None
    isD = agepos["isD"] if agepos else None
    ven, acol_of, amach = _arena_index(seasons) if arenas else ({}, {}, None)

    shooter, team, dff, dmask, ctx, cnt, off_t, gid, seas_row, arena = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    toi_atk, toi_def = np.zeros(P), np.zeros(P)
    last_season = np.full(P, -1, dtype=np.int64)

    def emit_side(atk, dfd, atk_home, s, def_goalie):
        ac_i = acol_of.get((ven.get(int(s.nhl_game_id)), s.season), -1)  # −1 = rare/unknown venue
        c = counts.get((int(s.nhl_game_id), int(s.stint_idx)), {})
        ai = [idx[p] for p in atk]
        di = [idx[p] for p in dfd]
        nd = len(di)
        di_pad = di + [0] * (MAX_DEF - nd)  # pad to width 5 (masked slots point at 0)
        mrow = [1.0] * nd + [0.0] * (MAX_DEF - nd)
        base = base_ctx(atk_home, s)
        seas = [0.0] * nseas
        if s.season in scol:
            seas[scol[s.season]] = 1.0
        if AC is not None:
            ac = AC[s.season]
            a_ac = ac[ai]  # (5, 4) attacker age-basis rows
            a_sum = a_ac.sum(0)
            d_cols = list(ac[di].sum(0)) if nd else [0.0] * 4  # defender age-basis sum (all real)
            a_d = isD[ai]
            a_dsum = float(a_d.sum())
            d_dsum = float(isD[di].sum()) if nd else 0.0
        for t_i, j in enumerate(atk):
            row = base + seas
            if AC is not None:  # AGE_CTX: [D-offsets | shoot|create|def basis]
                row = (
                    row
                    + [float(a_d[t_i]), a_dsum - float(a_d[t_i]), d_dsum]
                    + list(a_ac[t_i])
                    + list(a_sum - a_ac[t_i])
                    + d_cols
                )
            shooter.append(idx[j])
            team.append([idx[p] for p in atk if p != j])
            dff.append(di_pad)
            dmask.append(mrow)
            ctx.append(row)
            cnt.append(float(c.get(int(j), 0)))
            off_t.append(s.duration_s)
            gid.append(def_goalie)
            seas_row.append(s.season)
            arena.append(ac_i)

    for s in df.itertuples():
        hn, an = len(s.home_skaters), len(s.away_skaters)
        dur = s.duration_s
        if dual:
            if hn != 5 or an != 5:
                continue
            emit_side(s.home_skaters, s.away_skaters, True, s, s.away_goalie)
            emit_side(s.away_skaters, s.home_skaters, False, s, s.home_goalie)
            for p in (*s.home_skaters, *s.away_skaters):
                toi_atk[idx[p]] += dur
                toi_def[idx[p]] += dur
                last_season[idx[p]] = max(last_season[idx[p]], s.season)
        else:
            if hn == an:
                continue  # no man-advantage
            atk_home = hn > an
            atk, dfd = (
                (s.home_skaters, s.away_skaters) if atk_home else (s.away_skaters, s.home_skaters)
            )
            emit_side(atk, dfd, atk_home, s, s.away_goalie if atk_home else s.home_goalie)
            for p in atk:
                toi_atk[idx[p]] += dur  # PP-attacker time
                last_season[idx[p]] = max(last_season[idx[p]], s.season)
            for p in dfd:
                toi_def[idx[p]] += dur  # PK-defender time
                last_season[idx[p]] = max(last_season[idx[p]], s.season)

    dur = np.asarray(off_t, dtype=np.float64)
    ctx_names = RATE_CTX + [f"season_{s}" for s in slist[1:]] + (AGE_CTX if AC is not None else [])
    out = {
        "players": players,
        "idx": idx,
        "dual": dual,
        "shooter_idx": np.asarray(shooter, dtype=np.int64),
        "team_idx": np.asarray(team, dtype=np.int64),
        "def_idx": np.asarray(dff, dtype=np.int64),
        "def_mask": np.asarray(dmask, dtype=np.float64),
        "Xctx": np.asarray(ctx, dtype=np.float64),
        "count": np.asarray(cnt, dtype=np.float64),
        "offset": np.log(np.clip(dur / 3600.0, 1e-9, None)),
        "dur": dur,
        "def_goalie": np.asarray(gid, dtype=object),
        "toi_atk": toi_atk,
        "toi_def": toi_def,
        "toi": toi_atk,
        "n_season_cols": nseas,
        "season_row": np.asarray(seas_row, dtype=np.int64),
        "seasons": slist,
        "ctx_names": ctx_names,
        "last_season": last_season,
    }
    if acol_of:
        out["arena_col"] = np.asarray(arena, dtype=np.int32)
        out["n_arenas"] = len(acol_of)
        out["arena_venue"] = amach["venue"]
        out["arena_season"] = amach["season"]
        out["arena_e_prev"] = amach["e_prev"]
        out["arena_e_next"] = amach["e_next"]
        out["arena_e_gap"] = amach["e_gap"]
    if states:
        out.update(_unit_machinery(out, P, slist))
    return out


def _unit_machinery(R, P, slist):
    """Build the per-(player, season) unit index for the RW drift states of one bucket: which pairs
    are active (appear in any row as shooter/teammate/defender), compact unit ids sorted by (player,
    season), unit-indexed row gathers, RW edges between a player's consecutive active seasons, the
    first-state mask, and per-unit attacking TOI (from the focal-shooter rows)."""
    sord = {s: i for i, s in enumerate(slist)}
    nS = len(slist)
    sh, tm, dfi = R["shooter_idx"], R["team_idx"], R["def_idx"]
    dmk = R["def_mask"]
    srow = np.array([sord[s] for s in R["season_row"]], dtype=np.int64)
    active = np.zeros((P, nS), dtype=bool)
    active[sh, srow] = True
    for j in range(tm.shape[1]):
        active[tm[:, j], srow] = True
    for j in range(dfi.shape[1]):
        m = dmk[:, j] > 0
        active[dfi[m, j], srow[m]] = True
    up, us = np.nonzero(active)  # sorted by player, then season
    lut = np.full((P, nS), -1, dtype=np.int64)
    lut[up, us] = np.arange(len(up))
    same = up[1:] == up[:-1]  # consecutive units of the same player
    e_prev = np.nonzero(same)[0].astype(np.int64)
    e_next = e_prev + 1
    seas_arr = np.array(slist, dtype=np.int64)
    first_mask = np.ones(len(up))
    first_mask[e_next] = 0.0  # only each player's first state is ridged
    toi_unit = np.zeros(len(up))
    np.add.at(toi_unit, lut[sh, srow], R["dur"])
    return {
        "unit_player": up.astype(np.int64),
        "unit_season": seas_arr[us],
        "unit_lut": lut,
        "n_units": int(len(up)),
        "shooter_unit": lut[sh, srow].astype(np.int32),
        "team_unit": lut[tm, srow[:, None]].astype(np.int32),
        "def_unit": np.where(dmk > 0, lut[dfi, srow[:, None]], 0).astype(np.int32),
        "e_prev": e_prev,
        "e_next": e_next,
        "e_gap": (seas_arr[us[e_next]] - seas_arr[us[e_prev]]).astype(np.float64),
        "first_mask": first_mask,
        "toi_unit": toi_unit,
    }


# ── quality data: each shot + (for goals) the observed primary creator ────────────────────────────


def quality_creator_rows(seasons, idx, strengths, agepos=None, arenas=True):
    """Per Fenwick shot in the given strengths (POOLED across EV+MA): shooter, 4 teammates, ≤5 defenders
    (padded to 5 with a `def_mask`), context [is_home, pp, season…, shooter_D, def_D], logit-xG, goal
    flag, a CREATOR LABEL, a strength label (0=EV, 1=MA), and the row's season (for mapping teammates
    to the rate stage's per-season units). MA keeps only shots taken by the more-skaters (PP) side —
    so teammates=4, defenders=4 — consistent with the rate stage (shorthanded shots are out of scope).
    Creator: for goals, which teammate (0–3) got the primary assist, 4=unassisted, or −1 when the
    credited assister is not an on-ice teammate (data glitch / goalie assist — latent, and excluded
    from the assist-credit anchor; F4). Non-goals −1 (latent).
    Creator labels come from `shots_onice.assist1_id`/`assist2_id` (event_idx == pbp goal sortOrder).
    """
    scol, nseas = _season_cols(seasons)
    isD = agepos["isD"] if agepos else None
    ven, acol_of, amach = _arena_index(seasons) if arenas else ({}, {}, None)
    shooter, team, dff, dmask, ctx, y, goal, creator, creator2, slab, seas_row, arena = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for s in seasons:
        p = C.PROCESSED / "shots_onice" / f"{s}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(
            p,
            columns=[
                "nhl_game_id",
                "strength",
                "is_home",
                "xg",
                "goal",
                "shooter_id",
                "assist1_id",
                "assist2_id",
                "home_skaters",
                "away_skaters",
            ],
        )
        d = d[d.strength.isin(strengths) & d.xg.notna() & d.shooter_id.notna()]
        for gid, sub in d.groupby("nhl_game_id"):
            ac_i = acol_of.get((ven.get(int(gid)), s), -1)  # −1 = rare/unknown venue
            for r in sub.itertuples():
                hs, as_ = list(r.home_skaters), list(r.away_skaters)
                atk, dfd = (hs, as_) if r.is_home == 1 else (as_, hs)
                is_ev = r.strength == "5v5"
                if len(atk) != 5:  # EV: 5v5; MA: shooter must be on the 5 (PP) side
                    continue
                if is_ev and len(dfd) != 5:
                    continue
                sid = int(r.shooter_id)
                if (
                    sid not in atk
                    or any(q not in idx for q in atk)
                    or any(q not in idx for q in dfd)
                ):
                    continue
                mates = [q for q in atk if q != sid]  # 4 teammates
                nd = len(dfd)
                di = [idx[q] for q in dfd] + [0] * (MAX_DEF - nd)
                mrow = [1.0] * nd + [0.0] * (MAX_DEF - nd)
                seas = [0.0] * nseas
                if s in scol:
                    seas[scol[s]] = 1.0
                shooter.append(idx[sid])
                team.append([idx[q] for q in mates])
                dff.append(di)
                dmask.append(mrow)
                row = [1.0 if r.is_home == 1 else 0.0, 0.0 if is_ev else 1.0] + seas
                if isD is not None:  # A2: position offsets (appended last so
                    row += [
                        float(isD[idx[sid]]),  #   the pp column stays at index 1)
                        float(sum(isD[idx[q]] for q in dfd)),
                    ]
                ctx.append(row)
                xg = float(np.clip(r.xg, EPS, 1 - EPS))
                y.append(np.log(xg / (1 - xg)))
                goal.append(int(r.goal))
                slab.append(0 if is_ev else 1)
                seas_row.append(s)
                arena.append(ac_i)
                if r.goal == 1:
                    ap = r.assist1_id
                    if pd.isna(ap):
                        creator.append(4)  # genuinely unassisted
                    elif int(ap) in mates:
                        creator.append(mates.index(int(ap)))
                    else:
                        creator.append(-1)  # assister not an on-ice teammate (data
                    ap2 = r.assist2_id  # glitch / goalie assist): latent, and
                    creator2.append(
                        mates.index(int(ap2))  # excluded from the assist-credit anchor
                        if pd.notna(ap2) and int(ap2) in mates
                        else -1
                    )
                else:
                    creator.append(-1)
                    creator2.append(-1)
    out = {
        "shooter_idx": np.asarray(shooter, dtype=np.int64),
        "team_idx": np.asarray(team, dtype=np.int64),
        "def_idx": np.asarray(dff, dtype=np.int64),
        "def_mask": np.asarray(dmask, dtype=np.float64),
        "Xctx": np.asarray(ctx, dtype=np.float64),
        "y": np.asarray(y, dtype=np.float64),
        "goal": np.asarray(goal, dtype=np.int64),
        "creator": np.asarray(creator, dtype=np.int64),
        "strength": np.asarray(slab, dtype=np.int64),
        "creator2": np.asarray(creator2, dtype=np.int64),
        "season": np.asarray(seas_row, dtype=np.int64),
        "ctx_names": ["home", "pp"]
        + [f"season_{s}" for s in sorted(set(seasons))[1:]]
        + (["shooter_D", "def_D"] if isD is not None else []),
    }
    if acol_of:
        out["arena_col"] = np.asarray(arena, dtype=np.int32)
        out["n_arenas"] = len(acol_of)
        out["arena_venue"] = amach["venue"]
        out["arena_season"] = amach["season"]
        out["arena_e_prev"] = amach["e_prev"]
        out["arena_e_next"] = amach["e_next"]
        out["arena_e_gap"] = amach["e_gap"]
    return out


# ── conversion data: each shot's shooter, facing goalie, observed xG, goal flag ────────────────────


def conversion_rows(seasons, idx, strengths, agepos=None):
    """Per Fenwick shot (POOLED across the given strengths) for the CONVERSION fit: shooter index,
    facing-goalie index, logit of the OBSERVED xG, goal flag, a strength label (0=EV, 1=MA), and the
    season. For MA, keeps only PP-side shots (shooter's team has the man-advantage), matching
    rate/quality. Rows whose shooter isn't in `idx`, or with a missing facing goalie, are dropped.
    Uses observed `xg` (fixed), never `qbar`, so the stage stays independent of Stages 1-2.
    `fin`/`gsave` are POOLED across strengths (per-strength slope/intercept in fit_conversion).
    With `agepos`, also builds the global context block `ctx` (named in `ctx_names`): per-season
    intercept offsets (F6 — league finishing drift, first season = reference), the shooter position
    offset + F/D age basis (finishing curve), and the goalie age basis (goalie aging curve)."""
    frames = []
    for s in seasons:
        p = C.PROCESSED / "shots_onice" / f"{s}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(
            p,
            columns=[
                "strength",
                "is_home",
                "xg",
                "goal",
                "shooter_id",
                "home_goalie",
                "away_goalie",
            ],
        )
        d = d[d.strength.isin(strengths) & d.xg.notna() & d.shooter_id.notna()].copy()
        # skater counts from the "HvA" strength string; shooter's own count vs the defenders'
        hn = d.strength.str.slice(0, 1).astype(int).to_numpy()
        an = d.strength.str.slice(2, 3).astype(int).to_numpy()
        is_home = d.is_home.to_numpy() == 1
        shooter_n = np.where(is_home, hn, an)
        def_n = np.where(is_home, an, hn)
        d["slab"] = np.where(shooter_n == def_n, 0, 1)  # 0 = EV (equal), 1 = MA (man-advantage)
        d = d[shooter_n >= def_n]  # EV, or PP-side shots only (drop shorthanded)
        d["goalie_id"] = np.where(
            d.is_home.to_numpy() == 1, d.away_goalie.to_numpy(), d.home_goalie.to_numpy()
        )
        d = d[d.goalie_id.notna()]
        d["sidx"] = d.shooter_id.astype(int).map(idx)
        d = d[d.sidx.notna()]
        d["season"] = s
        frames.append(d[["sidx", "goalie_id", "xg", "goal", "slab", "season"]])
    if not frames or sum(len(f) for f in frames) == 0:
        return None
    D = pd.concat(frames, ignore_index=True)
    goalies = sorted(int(g) for g in D.goalie_id.astype(int).unique())
    gmap = {g: i for i, g in enumerate(goalies)}
    xg = np.clip(D.xg.to_numpy(float), EPS, 1 - EPS)
    sidx = D.sidx.astype(int).to_numpy(np.int64)
    gidx = D.goalie_id.astype(int).map(gmap).to_numpy(np.int64)
    srow = D.season.to_numpy(np.int64)
    out = {
        "shooter_idx": sidx,
        "goalie_idx": gidx,
        "logit_xg": np.log(xg / (1 - xg)),
        "y": D.goal.to_numpy(np.float64),
        "strength": D.slab.to_numpy(np.int64),
        "season": srow,
        "goalies": goalies,
    }
    if agepos is not None:
        slist = sorted(set(seasons))
        cols = [(srow == s).astype(np.float64) for s in slist[1:]]  # per-season offsets (F6)
        names = [f"season_{s}" for s in slist[1:]]
        zsh = np.zeros(len(D))
        dsh = agepos["isD"][sidx]
        gborn = _birthdates(goalies)
        zg = np.zeros(len(D))
        for s in slist:
            m = srow == s
            zsh[m] = agepos["z"][s][sidx[m]]
            ag = _season_age(gborn, s)
            zg[m] = np.where(np.isnan(ag), 0.0, (ag - AGE_PEAK) / AGE_SCALE)[gidx[m]]
        cols += [dsh] + list(_age_cols(zsh, dsh).T) + [zg, zg * zg]
        names += ["shooter_D", "fin_zF", "fin_z2F", "fin_zD", "fin_z2D", "g_z", "g_z2"]
        out["ctx"] = np.column_stack(cols) if cols else np.zeros((len(D), 0))
        out["ctx_names"] = names
    return out


# ── conversion PRE-CALCULATION: empirical-Bayes prior SDs (recomputed each fit, then held fixed) ─────


def _eb_prior_sd(count, exp_goals, made, var, min_shots, vbar, fallback):
    """Empirical-Bayes prior SD for one conversion offset block (finishing OR goalie) — the ridge
    analogue of shooting_model._estimate_k. For each entity aggregate its shots: N, expected goals
    Σxg, actual goals, and summed Bernoulli variance Σxg(1−xg). The per-shot residual rate
    r = (goals − Σxg)/N has, ACROSS high-volume entities, a volume-weighted spread equal to
    (true talent variance) + (mean sampling variance); subtract the latter (method of moments) to
    isolate the talent variance on the goals/shot scale. Map that to the LOGIT scale the fit uses via
    the local link derivative (a small logit offset δ shifts a shot's goal prob by ≈ p(1−p)·δ, so
    r ≈ vbar·offset ⇒ sd_logit = sd_prob / vbar). Falls back to `fallback` when <2 entities clear
    `min_shots`. This is computed OUTSIDE the fit and is a FIXED hyperparameter within it."""
    m = count >= min_shots
    if int(m.sum()) < 2:
        return fallback
    N, E, M, V = count[m].astype(float), exp_goals[m], made[m], var[m]
    r = (M - E) / N  # per-shot residual rate (goals/shot above xG)
    W = N / N.sum()
    rbar = float(np.sum(W * r))
    wvar = float(np.sum(W * (r - rbar) ** 2))  # observed spread of r
    msamp = float(np.sum(W * (V / N**2)))  # mean sampling variance of r
    tau2_prob = max(wvar - msamp, 0.0)  # talent variance (goals/shot), floored at 0
    sd_logit = np.sqrt(tau2_prob) / max(vbar, 1e-9)  # map goals/shot → logit scale
    return float(max(sd_logit, PRIOR_SD_FLOOR))


def estimate_conversion_prior_sds(Cr, P):
    """Pre-calculation stage for the conversion fit: estimate the finishing and goalie prior SDs from
    THIS fit's data (empirical Bayes), so the ridge shrinkage is data-calibrated rather than hand-set.
    Recomputed each fit and then held FIXED during the fit. Returns (sd_fin, sd_gsave) on the logit
    scale. Talent SD is estimated only from high-volume entities (MIN_SHOTS_*), then applied to all.
    """
    G = len(Cr["goalies"])
    xg = _sigmoid(Cr["logit_xg"])  # recover observed xg from its logit
    v = xg * (1.0 - xg)
    vbar = float(v.mean()) if len(v) else 1e-9  # mean per-shot Bernoulli variance p(1−p)

    def blocks(idx, n):
        return (
            np.bincount(idx, minlength=n).astype(float),  # N
            np.bincount(idx, weights=xg, minlength=n),  # Σxg
            np.bincount(idx, weights=Cr["y"], minlength=n),  # goals
            np.bincount(idx, weights=v, minlength=n),
        )  # ΣV

    sc, se_, sm, sv = blocks(Cr["shooter_idx"], P)
    gc, ge, gm, gv = blocks(Cr["goalie_idx"], G)
    sd_fin = _eb_prior_sd(sc, se_, sm, sv, MIN_SHOTS_FIN_EST, vbar, PRIOR_SD_FIN)
    sd_gsave = _eb_prior_sd(gc, ge, gm, gv, MIN_SHOTS_GSAVE_EST, vbar, PRIOR_SD_GSAVE)
    return sd_fin, sd_gsave
