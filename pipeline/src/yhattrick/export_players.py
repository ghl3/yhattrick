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

FULL_MIN_GAMES = 200     # treat a season as "full" (skip tiny dev slices)
MIN_EV_TOI = 100         # minutes of 5v5 ice time to appear in the table
SKATER_POS = {"C", "L", "R", "D"}
N_LINEMATES = 8

# impact metric -> higher_is_better (defence metrics are better when more negative)
METRICS = {"ev_off": True, "ev_def": False, "pp_off": True, "pk_def": False}
# percentile pool: only players with enough role ice time are "eligible" to be ranked against
ELIGIBILITY = {
    "ev_off": ("ev_off_toi", MIN_EV_TOI), "ev_def": ("ev_def_toi", MIN_EV_TOI),
    "pp_off": ("pp_off_toi", 40), "pk_def": ("pk_def_toi", 40),
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
    simp = {s: season_impact(s, names) for s in seasons}
    mates = linemates(seasons, names)

    # table players = pooled skaters with enough EV ice time
    table = pooled[pooled.ev_off_toi >= MIN_EV_TOI].copy()
    keep_ids = set(table.player_id)

    C.SITE_JSON.mkdir(parents=True, exist_ok=True)
    pdir = C.SITE_JSON / "player"
    pdir.mkdir(exist_ok=True)

    # career box totals across seasons (for the index + header)
    allbox = pd.concat(box.values(), ignore_index=True) if box else pd.DataFrame()

    index = []
    for r in table.itertuples():
        pid = r.player_id
        career = allbox[allbox.player_id == pid] if len(allbox) else pd.DataFrame()
        gp = int(career.gp.sum()) if len(career) else 0
        goals = int(career.g.sum()) if len(career) else 0
        assists = int((career.a1 + career.a2).sum()) if len(career) else 0
        pts = int(career.points.sum()) if len(career) else 0
        index.append({
            "id": pid, "name": r.name, "pos": r.pos, "group": r.group,
            "ev_toi": r.ev_off_toi, "gp": gp, "g": goals, "a": assists, "points": pts,
            **{c: getattr(r, c) for c in METRICS},
            **{f"{c}_pct": getattr(r, f"{c}_pct") for c in METRICS},
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
            per_season.append(rec)

        detail = {
            "id": pid, "name": r.name, "pos": r.pos, "group": r.group,
            "teams": teams_seen, "seasons": seasons,
            "gp": gp, "g": goals, "a": assists, "points": pts,
            "impact": {m: {"v": getattr(r, m), "se": getattr(r, f"{m}_se"),
                           "toi": getattr(r, f"{m}_toi"), "pct": getattr(r, f"{m}_pct")} for m in METRICS},
            "per_season": per_season,
            "linemates": mates.get(pid, [])[:N_LINEMATES],
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
