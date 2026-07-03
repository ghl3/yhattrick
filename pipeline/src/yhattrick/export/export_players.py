"""Export player data to site JSON: the isolated-impact ratings plus the descriptive box score
(from our own NHL data), per-season trends, top linemates, and team(s).

Writes:
  data/games/players.json        index row per player (sortable table)
  data/games/player/<id>.json    full detail (headline impact, per-season stats+impact, linemates)
then syncs to web/public/data.

Usage:  uv run python -m yhattrick.export_players
"""
from __future__ import annotations

import json
import math
import shutil
from collections import defaultdict

import numpy as np
import pandas as pd

from .. import config as C
from ..models import player_onice_model as model
from ..models import shooting_model
from . import player_heatmap
from ..data.aggregates import ONICE_COLS

FULL_MIN_GAMES = 200     # treat a season as "full" (skip tiny dev slices)
MIN_EV_TOI = 100         # minutes of 5v5 ice time to appear in the table
SKATER_POS = {"C", "L", "R", "D"}
N_LINEMATES = 8

# --- isolated-impact (modeled RAPM) metrics: per-60 deltas, adjusted for linemates/competition ---
# metric -> higher_is_better (defence metrics are better when more negative)
METRICS = {"ev_off": True, "ev_def": False, "pp_off": True, "pk_def": False}
# percentile pool: only players with enough role ice time are "eligible" to be ranked against
ELIGIBILITY = {
    "ev_off": ("ev_off_toi", MIN_EV_TOI), "ev_def": ("ev_def_toi", MIN_EV_TOI),
    "pp_off": ("pp_off_toi", 40), "pk_def": ("pk_def_toi", 40),
}

# --- on-ice (raw, descriptive) metrics: the team's rate while the player is on the ice, NOT
# isolated. Their own first-class variables (xG and Corsi families + shares), each with a
# within-position percentile.  name -> (higher_is_better, eligibility_toi_col, threshold) ---
ONICE = {
    "ev_xgf60":   (True,  "ev_off_toi", MIN_EV_TOI),   # 5v5 on-ice expected goals for / 60
    "ev_xga60":   (False, "ev_def_toi", MIN_EV_TOI),   # 5v5 on-ice expected goals against / 60
    "ev_xgshare": (True,  "ev_off_toi", MIN_EV_TOI),   # xGF / (xGF+xGA)
    "ev_cf60":    (True,  "ev_off_toi", MIN_EV_TOI),   # 5v5 on-ice Corsi (shot attempts) for / 60
    "ev_ca60":    (False, "ev_def_toi", MIN_EV_TOI),   # 5v5 on-ice Corsi against / 60
    "ev_cfshare": (True,  "ev_off_toi", MIN_EV_TOI),   # CF / (CF+CA)  (classic Corsi %)
    "pp_xgf60":   (True,  "pp_off_toi", 40),           # power-play on-ice xGF / 60
    "pk_xga60":   (False, "pk_def_toi", 40),           # penalty-kill on-ice xGA / 60
}

# --- individual (on-puck) metrics: the player's OWN shooting & production, all situations. Rates
# are per-60 of all-situations TOI; finishing comes from finishing.py.  name -> (higher_better,
# eligibility_col, threshold) ---
SHOTS_MIN = 150          # min unblocked shots to rank shot-based metrics
INDIV_TOI_MIN = 200      # min all-situations minutes to rank production rates
FO_MIN = 200             # min faceoffs taken to rank faceoff win %
ZS_MIN = 200             # min 5v5 O/D-zone faceoff starts to rank zone-start %
INDIVIDUAL = {
    "shots60":     (True,  "shots", SHOTS_MIN),        # unblocked shots / 60 (shot generation)
    "xg_per_shot": (True,  "shots", SHOTS_MIN),        # avg xG per shot (shot quality / danger)
    "ixg60":       (True,  "shots", SHOTS_MIN),        # individual xG / 60 (= shots60 x xg_per_shot)
    "fin_per100":  (True,  "shots", SHOTS_MIN),        # finishing: goals above expected / 100 shots
    "g60":         (True,  "toi_all", INDIV_TOI_MIN),  # goals / 60
    "a60":         (True,  "toi_all", INDIV_TOI_MIN),  # assists / 60
    "a1_60":       (True,  "toi_all", INDIV_TOI_MIN),  # primary assists / 60 (more repeatable than total)
    "pen_drawn60": (True,  "toi_all", INDIV_TOI_MIN),  # penalties drawn / 60
    "pen_taken60": (False, "toi_all", INDIV_TOI_MIN),  # penalties taken / 60 (lower is better)
    "fo_win":      (True,  "c_fo", FO_MIN),            # faceoff win % (fraction; ranked above FO_MIN)
    "ozs":         (True,  "n_zs", ZS_MIN),            # offensive-zone-start % (deployment context, not a rating)
}
# --- player value: GOALS ATTRIBUTED. Every figure is goals we credit to the player; across the
# league the shares reconcile to actual goals (xG is calibrated, finishing = goals − xG). Built from
# the on-ice model: created/allowed = baseline ÷ on-ice skaters + RAPM coef (absolute attributed
# share, ≥0), plus the shooting model's finishing (goals − xG) and net penalties. Net = created +
# finishing − allowed + penalties; at 5v5 the baseline cancels in the difference (same slice on
# offense and defense). Two layers: deployment-free per-60 rates, and actual goals (rates × real
# role-TOI, summed across situations, then per game). See docs/metrics.md.
# name -> (higher_is_better, eligibility_toi_col, threshold) ---
VALUE = {
    "gnet_pg":         (True,  "ev_off_toi", MIN_EV_TOI),  # NET GOALS ADDED PER GAME — top-line metric
    "scoring60":       (True,  "ev_off_toi", MIN_EV_TOI),  # 5v5 own-shot production/60 (ixG + finishing)
    "playmaking60":    (True,  "ev_off_toi", MIN_EV_TOI),  # 5v5 creation for teammates/60 (on-ice − own ixG)
    "allow60":         (False, "ev_def_toi", MIN_EV_TOI),  # 5v5 goals allowed/60 (share, lower is better)
    "pp_scoring60":    (True,  "pp_off_toi", 40),          # power-play own-shot production/60
    "pp_playmaking60": (True,  "pp_off_toi", 40),          # power-play creation for teammates/60
    "pk_allow60":      (False, "pk_def_toi", 40),          # penalty-kill goals allowed/60 (lower is better)
    "pen_net60":       (True,  "toi_all", INDIV_TOI_MIN),  # net penalty goals/60 ((drawn−taken)×value)
    "g_net":           (True,  "ev_off_toi", MIN_EV_TOI),  # actual net goals (window total, all situations)
}
# value fields carried into the index row (scalars; percentiles added for the VALUE keys above)
VALUE_COLS = ["gnet_pg", "gcreate_pg", "gallow_pg", "scoring60", "playmaking60", "allow60",
              "pp_scoring60", "pp_playmaking60", "pk_allow60", "pen_net60",
              "g_created", "g_allowed", "g_fin", "g_pen", "g_net"]

# box-score columns carried into the per-season detail
BOX_COLS = ["gp", "toi_min", "g", "a1", "a2", "points", "sog", "icf", "blocks",
            "hits", "takeaways", "giveaways", "fo_won", "fo_lost", "pen_taken", "pen_drawn"]

# goals-attributed value metrics carried into the per-season detail (mirrors the headline cards so the
# by-season table is consistent with them); computed per season by season_value().
SEASON_VALUE_COLS = ["gnet_pg", "scoring60", "playmaking60", "allow60",
                     "pp_scoring60", "pp_playmaking60", "pk_allow60", "pen_net60"]


def _dump(obj) -> str:
    def san(o):
        if isinstance(o, dict):
            return {k: san(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [san(v) for v in o]
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, (np.floating, float)):
            f = float(o)
            return None if (math.isnan(f) or math.isinf(f)) else f
        if isinstance(o, np.bool_):
            return bool(o)
        return o
    return json.dumps(san(obj), separators=(",", ":"))


def full_seasons() -> list[int]:
    out = []
    for s in model.available_seasons():
        n = pd.read_parquet(C.PROCESSED / "stints" / f"{s}.parquet", columns=["nhl_game_id"]).nhl_game_id.nunique()
        if n >= FULL_MIN_GAMES:
            out.append(s)
    return out


def pooled_impact(seasons, names) -> pd.DataFrame:
    """One row per player across the window, with within-position percentiles (headline card)."""
    ev = model.fit_cached(seasons, model.SPECS["ev"], names)
    pp = model.fit_cached(seasons, model.SPECS["pp_pk"], names)
    df = ev.merge(pp.drop(columns=["name", "pos"]), on="player_id", how="left")
    df = df[df.pos.isin(SKATER_POS)].copy()
    df["group"] = np.where(df.pos == "D", "D", "F")
    # percentile within position group, ranked only against players eligible for that metric
    for col, higher in METRICS.items():
        toi_col, thr = ELIGIBILITY[col]
        elig = df[df[toi_col] >= thr]
        r = elig.groupby("group")[col].rank(pct=True)  # index-aligned; non-eligible -> NaN
        df[f"{col}_pct"] = ((r if higher else 1 - r) * 100).round(0)
    return df


def season_impact(season, names) -> pd.DataFrame:
    """Per-season ev_off/ev_def/pp_off/pk_def for one season (no percentiles)."""
    ev = model.fit_cached([season], model.SPECS["ev"], names)
    pp = model.fit_cached([season], model.SPECS["pp_pk"], names)
    if ev.empty:
        return pd.DataFrame()
    keep_ev = ["player_id", "ev_off", "ev_def"]
    keep_pp = ["player_id", "pp_off", "pk_def"]
    return ev[keep_ev].merge(pp[keep_pp], on="player_id", how="left")


def onice_table(allbox: pd.DataFrame) -> pd.DataFrame:
    """Career on-ice rates per player (sums pooled across seasons → per-60 rates and shares).
    Returns one row per player_id with the ONICE metric columns; NaN where no qualifying ice
    time. These are descriptive (raw, un-isolated) variables in their own right."""
    if not len(allbox) or not set(ONICE_COLS) <= set(allbox.columns):
        return pd.DataFrame(columns=["player_id", *ONICE])
    s = allbox.groupby("player_id")[ONICE_COLS].sum()

    def per60(num, den):
        return np.where(den > 0, num * 3600.0 / den.replace(0, np.nan), np.nan)

    def share(f, a):
        tot = f + a
        return np.where(tot > 0, f / tot.replace(0, np.nan), np.nan)

    out = pd.DataFrame({"player_id": s.index.astype(int)})
    out["ev_xgf60"] = per60(s.ev_xgf_on, s.ev_onice_s)
    out["ev_xga60"] = per60(s.ev_xga_on, s.ev_onice_s)
    out["ev_xgshare"] = share(s.ev_xgf_on, s.ev_xga_on)
    out["ev_cf60"] = per60(s.ev_cf_on, s.ev_onice_s)
    out["ev_ca60"] = per60(s.ev_ca_on, s.ev_onice_s)
    out["ev_cfshare"] = share(s.ev_cf_on, s.ev_ca_on)
    out["pp_xgf60"] = per60(s.pp_xgf_on, s.pp_onice_s)
    out["pk_xga60"] = per60(s.pk_xga_on, s.pk_onice_s)
    # offensive-zone-start share: O / (O + D) faceoff starts at 5v5. Classified as an individual
    # (deployment) metric in INDIVIDUAL; n_zs is its eligibility volume. Counts come from the same
    # on-ice stint sums, so they pool across seasons additively like the rest of ONICE_COLS.
    out["n_zs"] = (s.ev_oz_starts + s.ev_dz_starts).values
    out["ozs"] = share(s.ev_oz_starts, s.ev_dz_starts)
    return out.reset_index(drop=True)


def add_onice_percentiles(df: pd.DataFrame) -> None:
    """Add `<metric>_pct` (within position group, eligible pool only) for every ONICE metric."""
    for col, (higher, toi_col, thr) in ONICE.items():
        elig = df[df[toi_col] >= thr]
        r = elig.groupby("group")[col].rank(pct=True)
        df[f"{col}_pct"] = ((r if higher else 1 - r) * 100).round(0)


def career_totals(allbox: pd.DataFrame) -> pd.DataFrame:
    """Per-player career sums needed for the individual rates (all-situations TOI + production)."""
    cols = ["player_id", "toi_all", "c_g", "c_a", "c_a1", "c_pd", "c_pt", "fo_won", "c_fo"]
    if not len(allbox):
        return pd.DataFrame(columns=cols)
    s = allbox.groupby("player_id").agg(
        toi_s=("toi_s", "sum"), c_g=("g", "sum"), a1=("a1", "sum"), a2=("a2", "sum"),
        c_pd=("pen_drawn", "sum"), c_pt=("pen_taken", "sum"),
        fo_won=("fo_won", "sum"), fo_lost=("fo_lost", "sum")).reset_index()
    s["player_id"] = s.player_id.astype(int)
    s["toi_all"] = (s.toi_s / 60.0)                    # all-situations minutes
    s["c_a"] = s.a1 + s.a2
    s["c_a1"] = s.a1                                    # primary assists
    s["c_fo"] = s.fo_won + s.fo_lost                    # faceoffs taken (win + loss) = volume gate
    return s[cols]


def individual_table(fin: pd.DataFrame, career: pd.DataFrame) -> pd.DataFrame:
    """Per-player individual rates: shot volume/quality, ixG/60, finishing, scoring + penalties /60.
    `fin` from finishing.py (shots, ixg, fin_per100, ...); `career` from career_totals."""
    df = career.merge(fin[["player_id", "shots", "ixg", "fin_per100", "fin_per100_se", "fin_goals"]],
                      on="player_id", how="left")
    sec = df.toi_all * 60.0
    df["shots"] = df.shots.fillna(0.0)
    df["ixg"] = df.ixg.fillna(0.0)
    per60 = lambda n: np.where(sec > 0, n * 3600.0 / sec.replace(0, np.nan), np.nan)
    df["shots60"] = per60(df.shots)
    df["ixg60"] = per60(df.ixg)
    df["xg_per_shot"] = np.where(df.shots > 0, df.ixg / df.shots.replace(0, np.nan), np.nan)
    df["g60"] = per60(df.c_g)
    df["a60"] = per60(df.c_a)
    df["a1_60"] = per60(df.c_a1)
    df["pen_drawn60"] = per60(df.c_pd)
    df["pen_taken60"] = per60(df.c_pt)
    df["fo_win"] = np.where(df.c_fo > 0, df.fo_won / df.c_fo.replace(0, np.nan), np.nan)
    return df


def add_individual_percentiles(df: pd.DataFrame) -> None:
    """Add `<metric>_pct` (within position group, eligible pool only) for every INDIVIDUAL metric."""
    for col, (higher, elig_col, thr) in INDIVIDUAL.items():
        elig = df[df[elig_col] >= thr]
        r = elig.groupby("group")[col].rank(pct=True)
        df[f"{col}_pct"] = ((r if higher else 1 - r) * 100).round(0)


def shots_by_strength(seasons) -> pd.DataFrame:
    """Per-shooter own-shot Fenwick counts split into 5v5 and power-play (shooter on a 5v4) buckets.

    Finishing is one pooled per-shot rate (alpha) per player; to credit 5v5 and PP finishing to the
    right situational rate we apportion it by where the player's shots were taken. The modeled shot set
    (non-null xg) already excludes empty-net, so PP here matches pp_off's 5v4/4v5 universe."""
    frames = []
    for s in seasons:
        p = C.PROCESSED / "shots_onice" / f"{s}.parquet"
        if p.exists():
            df = pd.read_parquet(p, columns=["shooter_id", "is_home", "strength", "xg"])
            frames.append(df[df.shooter_id.notna() & df.xg.notna()])
    if not frames:
        return pd.DataFrame(columns=["player_id", "shots5", "shotspp"])
    df = pd.concat(frames, ignore_index=True)
    parts = df.strength.str.split("v", n=1, expand=True)
    hn, an = pd.to_numeric(parts[0], errors="coerce"), pd.to_numeric(parts[1], errors="coerce")
    shooter_n = np.where(df.is_home == 1, hn, an)
    opp_n = np.where(df.is_home == 1, an, hn)
    is5 = (df.strength.values == "5v5").astype(float)
    ispp = ((shooter_n == 5) & (opp_n == 4)).astype(float)
    # shot counts and own expected goals (ixG), split 5v5 / power play — ixG isolates the player's
    # own-shot offense so Playmaking = on-ice offense − own ixG (see value_table / docs/metrics.md)
    out = pd.DataFrame({"player_id": df.shooter_id.astype(int).values,
                        "shots5": is5, "shotspp": ispp,
                        "ixg5": is5 * df.xg.values, "ixgpp": ispp * df.xg.values})
    return out.groupby("player_id", as_index=False)[["shots5", "shotspp", "ixg5", "ixgpp"]].sum()


def penalty_value(seasons, allbox: pd.DataFrame) -> float:
    """Net goal value of one drawn minor penalty, derived from our own data: league goals scored on
    the power play (shooter on a 5v4) minus goals allowed shorthanded (shooter on a 4v5), per drawn
    penalty. Used to convert a skater's net penalties (drawn − taken) into goals on the same scale."""
    pp_gf = sh_gf = 0
    for s in seasons:
        p = C.PROCESSED / "shots_onice" / f"{s}.parquet"
        if not p.exists():
            continue
        g = pd.read_parquet(p, columns=["is_home", "strength", "goal"]).query("goal == 1")
        parts = g.strength.str.split("v", n=1, expand=True)
        hn, an = pd.to_numeric(parts[0], errors="coerce"), pd.to_numeric(parts[1], errors="coerce")
        shooter_n = np.where(g.is_home == 1, hn, an)
        opp_n = np.where(g.is_home == 1, an, hn)
        pp_gf += int(((shooter_n == 5) & (opp_n == 4)).sum())
        sh_gf += int(((shooter_n == 4) & (opp_n == 5)).sum())
    n_pen = float(allbox.pen_drawn.sum()) if len(allbox) and "pen_drawn" in allbox.columns else 0.0
    v = (pp_gf - sh_gf) / n_pen if n_pen > 0 else 0.0
    print(f"[value] penalty value V = {v:.4f} goals/penalty  (PP GF {pp_gf} − SH GF {sh_gf}) / {n_pen:.0f} drawn")
    return v


def value_table(pooled: pd.DataFrame, shots_strength: pd.DataFrame, pen_v: float) -> pd.DataFrame:
    """Goals attributed: absolute attributed shares (per 60) + actual-goal totals.

    Each player is credited his share of the expected goals created and allowed while on the ice —
    his on-ice xG split among the on-ice skaters (baseline ÷ skaters + his RAPM coefficient) — so the
    shares are ≥0 and reconcile to actual goals leaguewide (xG is calibrated). Finishing (goals − xG
    on his shots) and net penalties complete the ledger. Net = created + finishing − allowed +
    penalties; at 5v5 the baseline cancels in the difference (same slice on offense and defense), so
    the 5v5 net is his pure marginal differential. Actual goals scale each per-60 rate by real
    role-TOI and sum across situations, then per game (g_net/gp). All approximate per-stint (ridge
    shrinkage, tiny μ) but calibrated in aggregate; a roster's g_net does NOT sum to team
    differential. `pooled` must carry the model baselines (ev_off_base, pp_off_base) and `gp`.
    See docs/metrics.md."""
    cols = ["player_id", "ev_off", "ev_def", "ev_off_toi", "pp_off", "pk_def", "pp_off_toi",
            "pk_def_toi", "ev_off_base", "pp_off_base", "fin_per100", "fin_goals", "gp",
            "toi_all", "pen_drawn60", "pen_taken60", "ev_xgf60", "pp_xgf60"]
    df = pooled[cols].merge(shots_strength, on="player_id", how="left")
    for c in cols[1:] + ["shots5", "shotspp", "ixg5", "ixgpp"]:
        df[c] = df[c].fillna(0.0)
    safe = lambda num, den: np.where(den > 0, num / np.where(den == 0, np.nan, den), 0.0)
    alpha = df.fin_per100 / 100.0                 # finishing goals per shot
    blocks5 = df.ev_off_toi / 60.0                # 5v5 ice time in 60-min blocks (ev_off_toi==ev_def_toi)
    ppblocks = df.pp_off_toi / 60.0
    pkblocks = df.pk_def_toi / 60.0
    fin5_60 = safe(alpha * df.shots5, blocks5)     # 5v5 finishing goals / 60
    finpp_60 = safe(alpha * df.shotspp, ppblocks)  # power-play finishing goals / 60
    ixg5_60 = safe(df.ixg5, blocks5)               # his own 5v5 expected goals / 60
    ixgpp_60 = safe(df.ixgpp, ppblocks)            # his own power-play expected goals / 60
    pen_blocks = df.toi_all / 60.0                 # all-situations ice time in 60-min blocks (pen rates' base)
    pen_add = df.pen_drawn60 * pen_blocks * pen_v   # goals from penalties drawn (= drawn count × V)
    pen_sub = df.pen_taken60 * pen_blocks * pen_v   # goal cost of penalties taken
    # absolute attributed shares: the league baseline split among that side's on-ice skaters (5 at
    # even strength / on the PP, 4 on the PK) + the player's RAPM coefficient. ≥0 in practice.
    create60 = df.ev_off_base / 5.0 + df.ev_off    # 5v5 xG he creates (his share); incl. his own shots
    allow60 = df.ev_off_base / 5.0 + df.ev_def     # 5v5 opponent xG he allows (his share); lower better
    pp_create60 = df.pp_off_base / 5.0 + df.pp_off  # power-play xG created (5 PP skaters share)
    pk_allow60 = df.pp_off_base / 4.0 + df.pk_def   # penalty-kill xG allowed (4 PK skaters share)
    # Offense split into scoring (his own shots) and playmaking (creation for teammates) by
    # PROPORTION: φ = his own ixG ÷ team on-ice xGF = the fraction of on-ice chances that were his
    # own shots. Scoring = φ·offense + finishing; Playmaking = (1−φ)·offense. Both ≥0, and they sum
    # to offense + finishing, so this is a positive partition with no double-count and the net is
    # unchanged. φ is a partition of his ISOLATED offense (not a marginal estimate). See docs/metrics.md.
    phi5 = np.clip(safe(ixg5_60, df.ev_xgf60), 0.0, 1.0)
    phipp = np.clip(safe(ixgpp_60, df.pp_xgf60), 0.0, 1.0)
    out = pd.DataFrame({"player_id": df.player_id})
    out["scoring60"] = phi5 * create60 + fin5_60
    out["playmaking60"] = (1.0 - phi5) * create60
    out["allow60"] = allow60
    out["pp_scoring60"] = phipp * pp_create60 + finpp_60
    out["pp_playmaking60"] = (1.0 - phipp) * pp_create60
    out["pk_allow60"] = pk_allow60
    out["pen_net60"] = (df.pen_drawn60 - df.pen_taken60) * pen_v   # net penalty goals / 60 (all situations)
    # actual-goal totals (deployment-weighted, summed across situations) — the attribution ledger
    out["g_created"] = create60 * blocks5 + pp_create60 * ppblocks
    out["g_allowed"] = allow60 * blocks5 + pk_allow60 * pkblocks
    out["g_fin"] = df.fin_goals                        # finishing goals (all situations)
    out["g_pen"] = pen_add - pen_sub                   # net penalty goals
    out["g_net"] = out.g_created + out.g_fin - out.g_allowed + out.g_pen
    out["gcreate_pg"] = safe(out.g_created, df.gp)
    out["gallow_pg"] = safe(out.g_allowed, df.gp)
    out["gnet_pg"] = safe(out.g_net, df.gp)            # top-line: net goals attributed per game
    return out


def add_value_percentiles(df: pd.DataFrame) -> None:
    """Add `<metric>_pct` (within position group, eligible pool only) for every VALUE metric."""
    for col, (higher, toi_col, thr) in VALUE.items():
        elig = df[df[toi_col] >= thr]
        r = elig.groupby("group")[col].rank(pct=True)
        df[f"{col}_pct"] = ((r if higher else 1 - r) * 100).round(0)


def season_value(season: int, names: dict, pen_v: float, fin_s: pd.DataFrame, box_s: pd.DataFrame) -> pd.DataFrame:
    """Per-season goals-attributed value (gnet_pg + per-60 rates), mirroring the pooled `value_table`
    so the by-season table shows the SAME metrics as the headline cards. Single-season model fits are
    noisier than the pooled window, so read these as a trend, not a precise decomposition."""
    ev = model.fit_cached([season], model.SPECS["ev"], names)
    pp = model.fit_cached([season], model.SPECS["pp_pk"], names)
    if ev.empty or box_s is None or not len(box_s):
        return pd.DataFrame(columns=["player_id"])
    ps = ev.merge(pp.drop(columns=["name", "pos"], errors="ignore"), on="player_id", how="left")
    ot = onice_table(box_s)[["player_id", "ev_xgf60", "pp_xgf60"]]   # on-ice xGF for the φ scoring/playmaking split
    car = career_totals(box_s)[["player_id", "toi_all", "c_pd", "c_pt"]]
    gp_s = box_s.groupby("player_id", as_index=False).gp.sum()
    ps = ps.merge(ot, on="player_id", how="left").merge(car, on="player_id", how="left").merge(gp_s, on="player_id", how="left")
    sec = ps.toi_all.fillna(0.0) * 60.0
    rate = lambda n: np.where(sec > 0, n.fillna(0.0) * 3600.0 / sec.replace(0, np.nan), 0.0)
    ps["pen_drawn60"], ps["pen_taken60"] = rate(ps.c_pd), rate(ps.c_pt)
    if fin_s is not None and len(fin_s):
        fs = fin_s.copy()
        fs["fin_goals"] = (fs.fin_per100.fillna(0.0) / 100.0) * fs.shots.fillna(0.0)   # season_finishing omits fin_goals
        ps = ps.merge(fs[["player_id", "fin_per100", "fin_goals"]], on="player_id", how="left")
    else:
        ps["fin_per100"], ps["fin_goals"] = 0.0, 0.0
    vt = value_table(ps, shots_by_strength([season]), pen_v)
    return vt[["player_id", *SEASON_VALUE_COLS]]


def _loc(v):
    """NHL landing fields like birthCity are {'default': 'Toronto'}; others are plain strings."""
    return v.get("default") if isinstance(v, dict) else v


def player_bios(ids: set[int]) -> dict[int, dict]:
    """Bio + headshot per player from the cached NHL player-landing json (raw/nhl/players)."""
    out: dict[int, dict] = {}
    for pid in ids:
        p = C.RAW_PLAYERS / f"{pid}.json"
        if not p.exists():
            continue
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        dd = d.get("draftDetails") or {}
        out[int(pid)] = {
            "headshot": d.get("headshot"),
            "height_in": d.get("heightInInches"),
            "weight_lb": d.get("weightInPounds"),
            "shoots": d.get("shootsCatches"),
            "birth_date": d.get("birthDate"),
            "birth_city": _loc(d.get("birthCity")),
            "birth_state": _loc(d.get("birthStateProvince")),
            "birth_country": d.get("birthCountry"),
            "number": d.get("sweaterNumber"),
            "draft_year": dd.get("year"),
            "draft_overall": dd.get("overallPick"),
        }
    return out


def game_logs(seasons) -> dict[int, list]:
    """player_id -> their per-game box lines (most-recent-first) from processed/gamelog."""
    frames = []
    for s in seasons:
        p = C.PROCESSED / "gamelog" / f"{s}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
    if not frames:
        return {}
    df = pd.concat(frames, ignore_index=True).sort_values(["date", "game_id"], ascending=[False, False])
    out: dict[int, list] = defaultdict(list)
    for r in df.itertuples():
        out[int(r.player_id)].append({
            "game_id": int(r.game_id), "season": int(r.season),
            "date": None if pd.isna(r.date) else r.date,
            "team": r.team, "opp": r.opp, "home": bool(r.home),
            "gf": None if pd.isna(r.gf) else int(r.gf), "ga": None if pd.isna(r.ga) else int(r.ga),
            "result": None if (isinstance(r.result, float) and pd.isna(r.result)) else r.result,
            "toi_s": int(r.toi_s), "g": int(r.g), "a": int(r.a), "p": int(r.p),
            "sog": int(r.sog), "pen": int(r.pen),
        })
    return out


def linemates(seasons, names, top=N_LINEMATES) -> dict[int, list]:
    """Top 5v5 linemates per player by shared on-ice time (from stints)."""
    stints = model.load_stints(seasons, model.SPECS["ev"].strengths)
    pair: dict[tuple, float] = defaultdict(float)
    for s in stints.itertuples():
        for side in (s.home_skaters, s.away_skaters):
            a = list(side)
            for i in range(len(a)):
                for j in range(i + 1, len(a)):
                    x, y = (a[i], a[j]) if a[i] < a[j] else (a[j], a[i])
                    pair[(int(x), int(y))] += s.duration_s
    mates: dict[int, list] = defaultdict(list)
    for (x, y), dur in pair.items():
        mates[x].append((y, dur))
        mates[y].append((x, dur))
    out = {}
    for p, lst in mates.items():
        lst.sort(key=lambda t: -t[1])
        out[p] = [{"id": q, "name": names.get(q, {}).get("name", f"#{q}"),
                   "toi_min": round(d / 60.0, 1)} for q, d in lst[:top]]
    return out


def main() -> None:
    seasons = full_seasons()
    if not seasons:
        raise SystemExit("no full processed seasons available — run `make stints` first")
    print(f"[export] seasons {seasons}")
    names = model.roster_names(seasons)

    pooled = pooled_impact(seasons, names)
    # regular-season box only (playoffs are kept separate in the parquet via game_type)
    box = {}
    for s in seasons:
        p = C.INTERIM / "box" / f"{s}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            box[s] = df[df.game_type == "regular"].copy() if "game_type" in df.columns else df
    # career box totals across seasons (for the index + header)
    allbox = pd.concat(box.values(), ignore_index=True) if box else pd.DataFrame()
    # on-ice (raw, descriptive) metrics as first-class variables, with within-position percentiles
    pooled = pooled.merge(onice_table(allbox), on="player_id", how="left")
    add_onice_percentiles(pooled)

    # individual (on-puck) metrics: shooting, finishing, scoring, penalties — all situations.
    # finishing now comes from the joint shooter×goalie shooting_model (goalie-adjusted).
    fin_lams = shooting_model.pooled_lambdas(seasons, names)
    fin_df, _ = shooting_model.fit_cached(seasons, names)
    pooled = pooled.merge(individual_table(fin_df, career_totals(allbox)),
                          on="player_id", how="left")
    add_individual_percentiles(pooled)
    fin_season = {s: shooting_model.season_finishing(s, names, fin_lams) for s in seasons}

    # player value: net goals added — per-game top line + deployment-free per-60 rates + totals
    gp_df = (allbox.groupby("player_id", as_index=False).gp.sum() if len(allbox)
             else pd.DataFrame(columns=["player_id", "gp"]))
    pooled = pooled.merge(gp_df, on="player_id", how="left")
    pooled["gp"] = pooled.gp.fillna(0)
    pen_v = penalty_value(seasons, allbox)
    pooled = pooled.merge(value_table(pooled, shots_by_strength(seasons), pen_v), on="player_id", how="left")
    add_value_percentiles(pooled)

    simp = {s: season_impact(s, names) for s in seasons}
    # per-season goals-attributed value (gnet_pg + rates) so the by-season table matches the cards
    sval = {s: season_value(s, names, pen_v, fin_season.get(s), box.get(s)) for s in seasons}
    mates = linemates(seasons, names)
    glog = game_logs(seasons)
    heat = player_heatmap.build(seasons)

    # table players = pooled skaters with enough EV ice time
    table = pooled[pooled.ev_off_toi >= MIN_EV_TOI].copy()
    keep_ids = set(table.player_id)
    bios = player_bios({int(i) for i in keep_ids})

    # generative player-card block (Cards v2): merged verbatim when the artifact exists
    # (make generative-cards). Pure JSON merge — the JAX side stays in the experimental group.
    gen_players, gen_meta = {}, None
    gcp = C.MODELS / "gen_cards.json"
    if gcp.exists():
        gj = json.loads(gcp.read_text())
        gen_players = gj.get("players", {})
        gm = gj.get("meta", {})
        gen_meta = {k: gm.get(k) for k in ("kappa", "goals_per_win", "latest_season",
                                           "age_curves", "rw_sd", "seasons", "replacement",
                                           "replacement_values", "repl_band_pct",
                                           "fin_avg_p100", "lg_def_xga60")}
        print(f"[export] gen_cards: {len(gen_players)} players (fit: {gm.get('fit', '?')})")
    else:
        print("[export] no gen_cards.json — details ship without the gen block "
              "(run `make generative-cards`)")

    C.SITE_JSON.mkdir(parents=True, exist_ok=True)
    pdir = C.SITE_JSON / "player"
    pdir.mkdir(exist_ok=True)

    def _rnd(v, n):
        return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), n)

    def onval(r, c):   # round an on-ice value off an itertuples row (None if missing)
        return _rnd(getattr(r, c), 4 if c.endswith("share") else 2)

    def indval(r, c):  # round an individual value (fractions/small ones need more precision)
        return _rnd(getattr(r, c), 4 if c in ("xg_per_shot", "fo_win", "ozs") else 2)

    def valval(r, c):  # round a value metric: actual-goal totals to 1 dp, rates/per-game to 3 dp
        return _rnd(getattr(r, c), 1 if c.startswith("g_") else 3)  # g_created/g_allowed/g_fin/g_pen/g_net

    index = []
    for r in table.itertuples():
        pid = r.player_id
        career = allbox[allbox.player_id == pid] if len(allbox) else pd.DataFrame()
        gp = int(career.gp.sum()) if len(career) else 0
        goals = int(career.g.sum()) if len(career) else 0
        assists = int((career.a1 + career.a2).sum()) if len(career) else 0
        pts = int(career.points.sum()) if len(career) else 0
        team_list = sorted({t for ts in career.teams for t in ts}) if len(career) else []
        # most-recent team = primary team of the latest season the player appears in
        current_team = str(career.sort_values("season").iloc[-1].team) if len(career) else (team_list[0] if team_list else "")
        # per-team games + production (for team-roster views) from this player's game log
        by_team: dict[str, dict] = {}
        for grow in glog.get(int(pid), []):
            acc = by_team.setdefault(grow["team"], {"gp": 0, "g": 0, "a": 0, "p": 0, "toi_s": 0})
            acc["gp"] += 1
            acc["g"] += grow["g"]; acc["a"] += grow["a"]; acc["p"] += grow["p"]; acc["toi_s"] += grow["toi_s"]
        gnp = gen_players.get(str(pid))
        # generative-card columns for the index tables: the four headline metrics + the five skill
        # attributes (prefixed gen_ where a production-model column of the same name exists)
        gen_cols = {}
        if gnp:
            attrs = gnp["attrs"]
            for key, col in (("war", "war"), ("ga60", "ga60"), ("pp_ga60", "pp_ga60"),
                             ("pk_ga60", "pk_ga60"), ("scoring", "gen_scoring"),
                             ("shooting", "gen_shooting"), ("finishing", "gen_finishing"),
                             ("playmaking", "gen_playmaking"), ("defense", "gen_defense")):
                a = attrs.get(key)
                if a is not None:
                    gen_cols[col] = a["v"]
                    gen_cols[f"{col}_pct"] = a["pct"]
        index.append({
            "id": pid, "name": r.name, "pos": r.pos, "group": r.group,
            "team": current_team, "teams": team_list, "by_team": by_team,
            "ev_toi": r.ev_off_toi, "gp": gp, "g": goals, "a": assists, "points": pts,
            **gen_cols,
            **{c: getattr(r, c) for c in METRICS},
            **{f"{c}_pct": getattr(r, f"{c}_pct") for c in METRICS},
            **{c: onval(r, c) for c in ONICE},
            **{f"{c}_pct": getattr(r, f"{c}_pct") for c in ONICE},
            **{c: indval(r, c) for c in INDIVIDUAL},
            **{f"{c}_pct": getattr(r, f"{c}_pct") for c in INDIVIDUAL},
            **{c: valval(r, c) for c in VALUE_COLS},
            **{f"{c}_pct": getattr(r, f"{c}_pct") for c in VALUE},
        })

        # --- detail ---
        per_season, teams_seen = [], []
        for s in seasons:
            b = box.get(s)
            brow = b[b.player_id == pid] if b is not None else pd.DataFrame()
            if not len(brow):
                continue
            brow = brow.iloc[0]
            for t in brow.teams:
                if t not in teams_seen:
                    teams_seen.append(t)
            rec = {"season": s, "team": brow.team, **{c: int(brow[c]) if c not in ("toi_min",) else float(brow[c]) for c in BOX_COLS}}
            si = simp.get(s)
            if si is not None and len(si):
                ir = si[si.player_id == pid]
                if len(ir):
                    rec.update({m: (None if pd.isna(ir.iloc[0][m]) else round(float(ir.iloc[0][m]), 3)) for m in METRICS})
            sv = sval.get(s)
            if sv is not None and len(sv):
                vr = sv[sv.player_id == pid]
                if len(vr):
                    rec.update({c: (None if pd.isna(vr.iloc[0][c]) else round(float(vr.iloc[0][c]), 3)) for c in SEASON_VALUE_COLS})
            fs = fin_season.get(s)
            if fs is not None and len(fs):
                fr = fs[fs.player_id == pid]
                sec = float(brow.toi_min) * 60.0
                if len(fr) and sec > 0:
                    fr = fr.iloc[0]
                    rec["shots60"] = round(float(fr.shots) * 3600.0 / sec, 2)
                    rec["xg_per_shot"] = round(float(fr.ixg) / fr.shots, 4) if fr.shots > 0 else None
                    rec["fin_per100"] = None if pd.isna(fr.fin_per100) else round(float(fr.fin_per100), 2)
            per_season.append(rec)

        detail = {
            "id": pid, "name": r.name, "pos": r.pos, "group": r.group,
            "current_team": current_team, "teams": teams_seen, "seasons": seasons,
            "gp": gp, "g": goals, "a": assists, "points": pts,
            "impact": {m: {"v": getattr(r, m), "se": getattr(r, f"{m}_se"),
                           "toi": getattr(r, f"{m}_toi"), "pct": getattr(r, f"{m}_pct")} for m in METRICS},
            "onice": {c: {"v": onval(r, c), "pct": getattr(r, f"{c}_pct")} for c in ONICE},
            "individual": {c: {"v": indval(r, c), "pct": getattr(r, f"{c}_pct"),
                               **({"se": _rnd(getattr(r, "fin_per100_se"), 2)} if c == "fin_per100" else {})}
                           for c in INDIVIDUAL},
            "shooting": {"shots": int(_rnd(getattr(r, "shots"), 0) or 0),
                         "ixg": _rnd(getattr(r, "ixg"), 1),
                         "fin_goals": _rnd(getattr(r, "fin_goals"), 1)},
            "value": {
                "rates": {
                    "scoring60":       {"v": valval(r, "scoring60"), "pct": getattr(r, "scoring60_pct")},
                    "playmaking60":    {"v": valval(r, "playmaking60"), "pct": getattr(r, "playmaking60_pct")},
                    "allow60":         {"v": valval(r, "allow60"), "pct": getattr(r, "allow60_pct")},
                    "pp_scoring60":    {"v": valval(r, "pp_scoring60"), "pct": getattr(r, "pp_scoring60_pct")},
                    "pp_playmaking60": {"v": valval(r, "pp_playmaking60"), "pct": getattr(r, "pp_playmaking60_pct")},
                    "pk_allow60":      {"v": valval(r, "pk_allow60"), "pct": getattr(r, "pk_allow60_pct")},
                    "pen_net60":       {"v": valval(r, "pen_net60"), "pct": getattr(r, "pen_net60_pct")},
                },
                "actual": {
                    "create_pg": valval(r, "gcreate_pg"), "allow_pg": valval(r, "gallow_pg"),
                    "net_pg": valval(r, "gnet_pg"), "net_pg_pct": getattr(r, "gnet_pg_pct"),
                    "created": valval(r, "g_created"), "allowed": valval(r, "g_allowed"),
                    "fin": valval(r, "g_fin"), "pen": valval(r, "g_pen"), "net": valval(r, "g_net"),
                },
            },
            "per_season": per_season,
            "linemates": mates.get(pid, [])[:N_LINEMATES],
            "games": glog.get(pid, []),
            "heat": heat.get(pid),
            "bio": bios.get(pid),
            "gen": gnp,
            "gen_meta": gen_meta if gnp else None,
        }
        (pdir / f"{pid}.json").write_text(_dump(detail))

    index.sort(key=lambda x: -(x["gnet_pg"] if x["gnet_pg"] is not None else -1e9))
    (C.SITE_JSON / "players.json").write_text(_dump(index))
    print(f"[export] {len(index)} players -> players.json + player/<id>.json")

    dst = C.WEB_DATA
    (dst / "player").mkdir(parents=True, exist_ok=True)
    shutil.copy2(C.SITE_JSON / "players.json", dst / "players.json")
    n = 0
    for f in pdir.glob("*.json"):
        shutil.copy2(f, dst / "player" / f.name)
        n += 1
    print(f"[export] synced players.json + {n} player files -> {dst}")


if __name__ == "__main__":
    main()
