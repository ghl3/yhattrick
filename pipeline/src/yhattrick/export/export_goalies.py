"""Export goalie data to site JSON: the shrunken save-talent rating (GSAx) plus the descriptive
box, danger / situation / shot-type splits, per-season trend, shots-against heatmap and game log.

Goalies live at the SAME url + player id as skaters (web reads /data/player/<id>.json), so this
writes goalie detail files alongside the skater ones (export_players skips goalies) with a
`"kind":"goalie"` discriminator the page branches on. The index is its own file.

Writes:
  data/games/goalies.json        index row per goalie (sortable table, used by /players G filter + team page)
  data/games/player/<id>.json    full goalie detail (kind=goalie)
then syncs to web/public/data.

Usage:  uv run python -m yhattrick.export_goalies
"""
from __future__ import annotations

import shutil
from collections import defaultdict

import numpy as np
import pandas as pd

from .. import config as C
from ..models import goalie
from . import goalie_heatmap
from ..models.player_onice_model import roster_names
from .export_players import _dump, player_bios

MIN_SHOTS_FACED = 1000   # shots faced to be ranked (percentile pool) — a meaningful workload

# headline goalie metrics -> higher_is_better (GAA lower is better). Percentiles rank within the
# single goalie pool (no position split). gsax_per100 is the shrunk talent rate from goalie.py.
GOALIE_METRICS = {
    "gsax_per100": True, "gsax60": True, "sv_pct": True,
    "gaa": False, "hd_sv_pct": True, "qs_pct": True,
}
DANGER = ("ld", "md", "hd")
SITUATIONS = ("ev", "pk", "pp")

# columns summed when pooling per-season boxes into a career line. xGA/GSAx are over the Fenwick set
# (calibrated); saves/Sv%/shots-against display over shots on goal. Each split carries Fenwick
# SA/xGA/GA + the SOG count, so split GSAx reconciles to the headline and Sv% stays conventional.
_SUM_COLS = (["gp", "starts", "shutouts", "qs", "rbs", "toi_s", "sa", "sog_against", "xga_sog",
             "saves", "ga", "xga", "gsax"]
            + [f"{b}_{m}" for b in DANGER for m in ("sa", "xga", "ga", "sog")]
            + [f"{s}_{m}" for s in SITUATIONS for m in ("sa", "xga", "ga", "sog")])


def _seasons() -> list[int]:
    return [s for s in C.SEASONS if (C.PROCESSED / "goalie_box" / f"{s}.parquet").exists()]


def _load_boxes(seasons) -> dict[int, pd.DataFrame]:
    out = {}
    for s in seasons:
        p = C.PROCESSED / "goalie_box" / f"{s}.parquet"
        if p.exists():
            out[s] = pd.read_parquet(p)
    return out


def _career(boxes: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Pool the per-season boxes into one career row per goalie (additive sums) + team summary."""
    allbox = pd.concat(boxes.values(), ignore_index=True)
    g = allbox.groupby("player_id")[_SUM_COLS].sum().reset_index()
    g = g.merge(allbox.groupby("player_id").name.first().reset_index(), on="player_id", how="left")
    # current team = primary team of the latest season the goalie appears in; teams = union
    latest = allbox.sort_values("season").groupby("player_id").team.last()
    teams = (allbox.groupby("player_id").teams.apply(lambda ls: sorted({t for x in ls for t in x})))
    g = g.merge(pd.DataFrame({"team": latest, "teams": teams}).reset_index(), on="player_id", how="left")
    return g


def _rates(g: pd.DataFrame) -> pd.DataFrame:
    """Derive the rate metrics (Sv%, GAA, GSAx/60, HD Sv%, QS%) from the career sums."""
    safe = lambda n, d: np.where(d > 0, n / np.where(d == 0, np.nan, d), np.nan)
    g["sv_pct"] = safe(g.saves, g.sog_against)
    g["gaa"] = safe(g.ga * 3600.0, g.toi_s)
    g["gsax60"] = safe(g.gsax * 3600.0, g.toi_s)               # GSAx (Fenwick) per 60 min
    g["sa60"] = safe(g.sog_against * 3600.0, g.toi_s)          # shots on goal faced / 60
    g["hd_sv_pct"] = safe(g.hd_sog - g.hd_ga, g.hd_sog)        # high-danger save % (shots on goal)
    g["qs_pct"] = safe(g.qs, g.starts)
    return g


def _shottypes(seasons) -> dict[int, list]:
    """player_id -> [{shot_type, sa, sv_pct, gsax}] pooled across seasons, most-faced first."""
    frames = [pd.read_parquet(C.PROCESSED / "goalie_shottype" / f"{s}.parquet")
              for s in seasons if (C.PROCESSED / "goalie_shottype" / f"{s}.parquet").exists()]
    if not frames:
        return {}
    df = pd.concat(frames, ignore_index=True).groupby(["player_id", "shot_type"]).agg(
        sa=("sa", "sum"), xga=("xga", "sum"), ga=("ga", "sum"), sog=("sog", "sum")).reset_index()
    out: dict[int, list] = defaultdict(list)
    for r in df.sort_values("sog", ascending=False).itertuples():
        sog = int(r.sog)
        out[int(r.player_id)].append({
            "shot_type": r.shot_type, "sa": sog,                       # shots on goal of this type
            "sv_pct": round((sog - r.ga) / sog, 4) if sog else None,
            "gsax": round(float(r.xga - r.ga), 2),                     # Fenwick GSAx for this type
        })
    return out


def _gamelogs(seasons) -> dict[int, list]:
    frames = [pd.read_parquet(C.PROCESSED / "goalie_gamelog" / f"{s}.parquet")
              for s in seasons if (C.PROCESSED / "goalie_gamelog" / f"{s}.parquet").exists()]
    if not frames:
        return {}
    df = pd.concat(frames, ignore_index=True).sort_values(["date", "game_id"], ascending=[False, False])
    out: dict[int, list] = defaultdict(list)
    for r in df.itertuples():
        out[int(r.player_id)].append({
            "game_id": int(r.game_id), "season": int(r.season),
            "date": None if pd.isna(r.date) else r.date, "team": r.team, "opp": r.opp,
            "home": bool(r.home),
            "decision": None if (isinstance(r.decision, float) and pd.isna(r.decision)) else r.decision,
            "toi_s": int(r.toi_s), "sog_against": int(r.sog_against), "saves": int(r.saves),
            "ga": int(r.ga), "xga": round(float(r.xga), 2), "gsax": round(float(r.gsax), 2),
            "started": bool(r.started), "shutout": bool(r.shutout),
            "qs": bool(r.qs), "rbs": bool(r.rbs),
        })
    return out


def _f(v, n):
    return None if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else round(float(v), n)


def main() -> None:
    seasons = _seasons()
    if not seasons:
        raise SystemExit("no goalie_box seasons — run `make goalie-box` first")
    print(f"[export-goalies] seasons {seasons}")
    names = roster_names(seasons)

    boxes = _load_boxes(seasons)
    g = _rates(_career(boxes))

    # shrunken save-talent rating (GSAx/100, with CI) from goalie.py
    talent = goalie.fit_cached(seasons, names)[
        ["player_id", "gsax_per100", "gsax_per100_se", "gsax_saved"]]
    g = g.merge(talent, on="player_id", how="left")

    # percentiles within the single goalie pool (eligible = enough shots faced)
    elig = g[g.sa >= MIN_SHOTS_FACED]
    for col, higher in GOALIE_METRICS.items():
        r = elig[col].rank(pct=True)
        g[f"{col}_pct"] = ((r if higher else 1 - r) * 100).round(0)

    st = _shottypes(seasons)
    glog = _gamelogs(seasons)
    heat = goalie_heatmap.build(seasons)
    # per-season shrunk GSAx/100 (pooled k) for the trend
    k = goalie.pooled_k(seasons, names)
    season_talent = {s: goalie.season_goalie(s, names, k) for s in seasons}
    bios = player_bios({int(i) for i in g.player_id})

    C.SITE_JSON.mkdir(parents=True, exist_ok=True)
    pdir = C.SITE_JSON / "player"
    pdir.mkdir(exist_ok=True)

    index = []
    for r in g.itertuples():
        pid = int(r.player_id)
        metric = {m: {"v": _f(getattr(r, m), 4 if m in ("sv_pct", "hd_sv_pct") else 2),
                      "pct": _f(getattr(r, f"{m}_pct"), 0),
                      **({"se": _f(r.gsax_per100_se, 2)} if m == "gsax_per100" else {})}
                  for m in GOALIE_METRICS}
        index.append({
            "id": pid, "name": r.name, "team": r.team, "teams": list(r.teams),
            "gp": int(r.gp), "starts": int(r.starts), "sa": int(r.sog_against), "saves": int(r.saves),
            "ga": int(r.ga), "shutouts": int(r.shutouts),
            "sv_pct": _f(r.sv_pct, 4), "gaa": _f(r.gaa, 2),
            "xga": _f(r.xga, 1), "gsax": _f(r.xga - r.ga, 1),
            "gsax_per100": _f(r.gsax_per100, 2), "gsax_per100_se": _f(r.gsax_per100_se, 2),
            "gsax60": _f(r.gsax60, 2), "hd_sv_pct": _f(r.hd_sv_pct, 4),
            "qs": int(r.qs), "rbs": int(r.rbs), "qs_pct": _f(r.qs_pct, 3),
            **{f"{m}_pct": _f(getattr(r, f"{m}_pct"), 0) for m in GOALIE_METRICS},
        })

        # danger + situation splits — shots/Sv% over shots on goal, GSAx over the Fenwick set
        def _split(prefix):
            sog, ga, xga = getattr(r, f"{prefix}_sog"), getattr(r, f"{prefix}_ga"), getattr(r, f"{prefix}_xga")
            return {"sa": int(sog), "sv_pct": round((sog - ga) / sog, 4) if sog else None,
                    "gsax": round(float(xga - ga), 2)}
        danger = [{"bucket": b, **_split(b)} for b in DANGER]
        situation = [{"sit": s, **_split(s)} for s in SITUATIONS]

        # per-season trend
        per_season = []
        for s in seasons:
            b = boxes.get(s)
            brow = b[b.player_id == pid] if b is not None else pd.DataFrame()
            if not len(brow):
                continue
            br = brow.iloc[0]
            sog, ga = int(br.sog_against), int(br.ga)
            toi_s = float(br.toi_s)
            rec = {
                "season": s, "team": br.team, "gp": int(br.gp), "starts": int(br.starts),
                "toi_min": round(toi_s / 60.0, 1), "sa": sog, "sog_against": sog,
                "saves": int(br.saves), "ga": ga, "shutouts": int(br.shutouts),
                "sv_pct": round((sog - ga) / sog, 4) if sog else None,
                "gaa": round(ga * 3600.0 / toi_s, 2) if toi_s else None,
                "xga": round(float(br.xga), 1), "gsax": round(float(br.xga - ga), 1),
                "hd_sv_pct": round((br.hd_sog - br.hd_ga) / br.hd_sog, 4) if br.hd_sog else None,
                "qs": int(br.qs), "rbs": int(br.rbs), "gsax_per100": None,
            }
            stt = season_talent.get(s)
            if stt is not None and len(stt):
                tr = stt[stt.player_id == pid]
                if len(tr):
                    rec["gsax_per100"] = _f(tr.iloc[0].gsax_per100, 2)
            per_season.append(rec)

        detail = {
            "kind": "goalie", "id": pid, "name": r.name, "current_team": r.team,
            "teams": list(r.teams), "seasons": seasons,
            "gp": int(r.gp), "starts": int(r.starts), "sa": int(r.sa), "saves": int(r.saves),
            "ga": int(r.ga), "shutouts": int(r.shutouts),
            "sv_pct": _f(r.sv_pct, 4), "gaa": _f(r.gaa, 2), "gsax": _f(r.xga - r.ga, 1),
            "metric": metric, "danger": danger, "situation": situation,
            "shot_types": st.get(pid, []), "per_season": per_season,
            "games": glog.get(pid, []), "heat": heat.get(pid), "bio": bios.get(pid),
        }
        (pdir / f"{pid}.json").write_text(_dump(detail))

    index.sort(key=lambda x: (x["gsax_per100"] is None, -(x["gsax_per100"] or 0)))
    (C.SITE_JSON / "goalies.json").write_text(_dump(index))
    print(f"[export-goalies] {len(index)} goalies -> goalies.json + player/<id>.json")

    dst = C.WEB_DATA
    (dst / "player").mkdir(parents=True, exist_ok=True)
    shutil.copy2(C.SITE_JSON / "goalies.json", dst / "goalies.json")
    n = 0
    for r in g.itertuples():
        f = pdir / f"{int(r.player_id)}.json"
        if f.exists():
            shutil.copy2(f, dst / "player" / f.name)
            n += 1
    print(f"[export-goalies] synced goalies.json + {n} goalie files -> {dst}")


if __name__ == "__main__":
    main()
