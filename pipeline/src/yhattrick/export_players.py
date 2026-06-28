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

from . import config as C
from . import player_onice_model as model
from . import finishing
from . import player_heatmap
from .aggregates import ONICE_COLS

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
INDIVIDUAL = {
    "shots60":     (True,  "shots", SHOTS_MIN),        # unblocked shots / 60 (shot generation)
    "xg_per_shot": (True,  "shots", SHOTS_MIN),        # avg xG per shot (shot quality / danger)
    "ixg60":       (True,  "shots", SHOTS_MIN),        # individual xG / 60 (= shots60 x xg_per_shot)
    "fin_per100":  (True,  "shots", SHOTS_MIN),        # finishing: goals above expected / 100 shots
    "g60":         (True,  "toi_all", INDIV_TOI_MIN),  # goals / 60
    "a60":         (True,  "toi_all", INDIV_TOI_MIN),  # assists / 60
    "pen_drawn60": (True,  "toi_all", INDIV_TOI_MIN),  # penalties drawn / 60
    "pen_taken60": (False, "toi_all", INDIV_TOI_MIN),  # penalties taken / 60 (lower is better)
}
# box-score columns carried into the per-season detail
BOX_COLS = ["gp", "toi_min", "g", "a1", "a2", "points", "sog", "icf", "blocks",
            "hits", "takeaways", "giveaways", "fo_won", "fo_lost", "pen_taken", "pen_drawn"]


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
    return out.reset_index(drop=True)


def add_onice_percentiles(df: pd.DataFrame) -> None:
    """Add `<metric>_pct` (within position group, eligible pool only) for every ONICE metric."""
    for col, (higher, toi_col, thr) in ONICE.items():
        elig = df[df[toi_col] >= thr]
        r = elig.groupby("group")[col].rank(pct=True)
        df[f"{col}_pct"] = ((r if higher else 1 - r) * 100).round(0)


def career_totals(allbox: pd.DataFrame) -> pd.DataFrame:
    """Per-player career sums needed for the individual rates (all-situations TOI + production)."""
    if not len(allbox):
        return pd.DataFrame(columns=["player_id", "toi_all", "c_g", "c_a", "c_pd", "c_pt"])
    s = allbox.groupby("player_id").agg(
        toi_s=("toi_s", "sum"), c_g=("g", "sum"), a1=("a1", "sum"), a2=("a2", "sum"),
        c_pd=("pen_drawn", "sum"), c_pt=("pen_taken", "sum")).reset_index()
    s["player_id"] = s.player_id.astype(int)
    s["toi_all"] = (s.toi_s / 60.0)                    # all-situations minutes
    s["c_a"] = s.a1 + s.a2
    return s[["player_id", "toi_all", "c_g", "c_a", "c_pd", "c_pt"]]


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
    df["pen_drawn60"] = per60(df.c_pd)
    df["pen_taken60"] = per60(df.c_pt)
    return df


def add_individual_percentiles(df: pd.DataFrame) -> None:
    """Add `<metric>_pct` (within position group, eligible pool only) for every INDIVIDUAL metric."""
    for col, (higher, elig_col, thr) in INDIVIDUAL.items():
        elig = df[df[elig_col] >= thr]
        r = elig.groupby("group")[col].rank(pct=True)
        df[f"{col}_pct"] = ((r if higher else 1 - r) * 100).round(0)


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

    # individual (on-puck) metrics: shooting, finishing, scoring, penalties — all situations
    fin_k = finishing.pooled_k(seasons, names)
    pooled = pooled.merge(individual_table(finishing.fit_cached(seasons, names), career_totals(allbox)),
                          on="player_id", how="left")
    add_individual_percentiles(pooled)
    fin_season = {s: finishing.season_finishing(s, names, fin_k) for s in seasons}

    simp = {s: season_impact(s, names) for s in seasons}
    mates = linemates(seasons, names)
    glog = game_logs(seasons)
    heat = player_heatmap.build(seasons)

    # table players = pooled skaters with enough EV ice time
    table = pooled[pooled.ev_off_toi >= MIN_EV_TOI].copy()
    keep_ids = set(table.player_id)
    bios = player_bios({int(i) for i in keep_ids})

    C.SITE_JSON.mkdir(parents=True, exist_ok=True)
    pdir = C.SITE_JSON / "player"
    pdir.mkdir(exist_ok=True)

    def _rnd(v, n):
        return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), n)

    def onval(r, c):   # round an on-ice value off an itertuples row (None if missing)
        return _rnd(getattr(r, c), 4 if c.endswith("share") else 2)

    def indval(r, c):  # round an individual value (xg_per_shot is small -> more precision)
        return _rnd(getattr(r, c), 4 if c == "xg_per_shot" else 2)

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
        index.append({
            "id": pid, "name": r.name, "pos": r.pos, "group": r.group,
            "team": current_team, "teams": team_list, "by_team": by_team,
            "ev_toi": r.ev_off_toi, "gp": gp, "g": goals, "a": assists, "points": pts,
            **{c: getattr(r, c) for c in METRICS},
            **{f"{c}_pct": getattr(r, f"{c}_pct") for c in METRICS},
            **{c: onval(r, c) for c in ONICE},
            **{f"{c}_pct": getattr(r, f"{c}_pct") for c in ONICE},
            **{c: indval(r, c) for c in INDIVIDUAL},
            **{f"{c}_pct": getattr(r, f"{c}_pct") for c in INDIVIDUAL},
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
            "per_season": per_season,
            "linemates": mates.get(pid, [])[:N_LINEMATES],
            "games": glog.get(pid, []),
            "heat": heat.get(pid),
            "bio": bios.get(pid),
        }
        (pdir / f"{pid}.json").write_text(_dump(detail))

    index.sort(key=lambda x: -x["ev_off"])
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
