"""Shared row-building primitives for the generative model and its WAR consumers.

ONE implementation of the four things every consumer needs: load the stint facts, orient a stint to
the attacking team, build the rate-context vector, and gate which stint sides become design rows. The
fit (`generative_model.rate_rows`), the card WAR accounting (`generative_cards.war_rows`), and the WAR
audit (`generative_war_audit.rows_with_teams`) all build on these, so the orientation math — the
attacking-team zone flip and lead sign — is defined in one tested place.

Low-level: imports only config + numpy/pandas, never the model modules (they import from here).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C

# The 5 rate-context columns, oriented to the ATTACKING team.
RATE_CTX = ["home", "ozone", "dzone", "trail", "lead"]


def load_stints(seasons, strengths) -> pd.DataFrame:
    """Regular-season non-overload stints in the given strength states (e.g. ('5v5',) or
    ('5v4','4v5')). Keeps short <10s stints (the Poisson offset handles them). Adds a `season` column
    for the season fixed-effect."""
    frames = []
    for s in seasons:
        p = C.PROCESSED / "stints" / f"{s}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if "overload" not in df.columns:
            df["overload"] = False
        reg = (df.nhl_game_id // 10000) % 100 == 2
        sub = df[reg & df.strength.isin(strengths) & (~df.overload) & (df.duration_s >= 1)].copy()
        sub["season"] = s
        frames.append(sub)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def zone(atk_home, start_type, start_zone):
    """Attacking-team zone start: O/D flips for the away team. Returns (ozone, dzone) indicators."""
    if start_type != "faceoff" or start_zone not in ("O", "D"):
        return 0.0, 0.0
    az = start_zone if atk_home else ("O" if start_zone == "D" else "D")
    return (1.0, 0.0) if az == "O" else (0.0, 1.0)


def base_ctx(atk_home, st):
    """The 5 RATE_CTX columns for one stint side, oriented to the attacking team:
    [home, ozone, dzone, trail, lead]. `st` is a stint row (namedtuple / attr access)."""
    oz, dz = zone(atk_home, st.start_type, st.start_zone)
    lead = st.home_lead if atk_home else -st.home_lead
    return [1.0 if atk_home else 0.0, oz, dz, 1.0 if lead < 0 else 0.0, 1.0 if lead > 0 else 0.0]


def side_rows(seasons, strengths, dual, idx, gteam=None):
    """Per stint-side WAR design (one row per attacking 5-man side), on the shared player index:
    `atk` (n,5), `def` (n,5) padded + `dmask`, `goalie` id, `dur`, `ctx` = RATE_CTX. `dual` (EV) emits
    BOTH 5v5 sides; MA emits only the more-skaters (PP) side (5v4/5v3). Rows whose players are all in
    `idx` only. If `gteam` ({game_id: (home_abbrev, away_abbrev)}) is given, also attach the attacking
    / defending team label per row (`ateam`/`dteam`) and skip games absent from `gteam` — this is the
    audit's team-resolved variant. Returns a dict of arrays, or None if no rows."""
    df = load_stints(seasons, strengths)
    if df.empty:
        return None
    want_teams = gteam is not None
    atk, dfd, dmask, gid, dur, ctx, ateam, dteam = [], [], [], [], [], [], [], []

    def emit(A, B, home, st, goalie, pair):
        if any(q not in idx for q in (*A, *B)):
            return
        nb = len(B)
        dfd.append([idx[q] for q in B] + [0] * (5 - nb))
        dmask.append([1.0] * nb + [0.0] * (5 - nb))
        atk.append([idx[q] for q in A])
        ctx.append(base_ctx(home, st))
        gid.append(goalie)
        dur.append(st.duration_s)
        if want_teams:
            ht, at_ = pair
            ateam.append(ht if home else at_)
            dteam.append(at_ if home else ht)

    for st in df.itertuples():
        pair = gteam.get(int(st.nhl_game_id)) if want_teams else None
        if want_teams and pair is None:
            continue
        hn, an = len(st.home_skaters), len(st.away_skaters)
        if dual:
            if hn == 5 and an == 5:
                emit(st.home_skaters, st.away_skaters, True, st, st.away_goalie, pair)
                emit(st.away_skaters, st.home_skaters, False, st, st.home_goalie, pair)
        elif hn != an and max(hn, an) == 5:
            home = hn > an
            A, B = (
                (st.home_skaters, st.away_skaters) if home else (st.away_skaters, st.home_skaters)
            )
            emit(A, B, home, st, st.away_goalie if home else st.home_goalie, pair)

    if not atk:
        return None
    out = {
        "atk": np.asarray(atk, dtype=np.int64),
        "def": np.asarray(dfd, dtype=np.int64),
        "dmask": np.asarray(dmask),
        "goalie": np.asarray(gid, dtype=object),
        "dur": np.asarray(dur, dtype=np.float64),
        "ctx": np.asarray(ctx, dtype=np.float64),
    }
    if want_teams:
        out["ateam"] = np.asarray(ateam)
        out["dteam"] = np.asarray(dteam)
    return out
