"""The validation checks, grouped by pipeline stage. Each `check_*` reads the artifacts that stage
produced and yields `Check`s classified EXACT (algebraic identity / partition / join — a failure is a
bug) or APPROX (calibration / shrinkage / luck — holds at scale, warn-only). See `docs/modeling.md`
and `docs/metrics.md` for the invariants; nothing here refits a model."""

from __future__ import annotations

import json
from typing import Iterator

import pandas as pd

from .. import config as C
from ..models.player_onice_model import MIN_STINT_S
from .core import KNOWN_GAP_SEASONS, Check, approx, cell_len, exact, json_clean, label, pq, skip


# ===============================================================================================
# STAGE: interim  (parse-level self-consistency)
# ===============================================================================================
def check_interim(seasons: list[int]) -> Iterator[Check]:
    st = "interim"
    for s in seasons:
        sh = pq(
            C.INTERIM / "shifts" / f"{s}.parquet",
            columns=["nhl_game_id", "player_id", "start_g", "end_g", "duration_s"],
        )
        if sh is None:
            yield skip(f"shifts {s}", st, "EXACT", "no interim/shifts")
            continue
        bad_dur = int((sh.duration_s != (sh.end_g - sh.start_g)).sum())
        yield exact(f"shift duration = end−start ({s})", st, bad_dur == 0, bad_dur, 0)
        # no player's merged shift intervals overlap within a game (the _merge_player_intervals contract)
        o = sh.sort_values(["nhl_game_id", "player_id", "start_g"])
        nxt = o.groupby(["nhl_game_id", "player_id"]).start_g.shift(-1)
        overlaps = int(((nxt.notna()) & (o.end_g.values > nxt.values)).sum())
        yield exact(f"no overlapping shifts/player ({s})", st, overlaps == 0, overlaps, 0)

        shots = pq(C.INTERIM / "shots" / f"{s}.parquet", columns=["nhl_game_id", "event_idx"])
        if shots is not None:
            dup = int(shots.duplicated(["nhl_game_id", "event_idx"]).sum())
            yield exact(f"unique (game,event_idx) shot keys ({s})", st, dup == 0, dup, 0)


# ===============================================================================================
# STAGE: stints  (the intermediate join — partition + reconstruction invariants)
# ===============================================================================================
def check_stints(seasons: list[int]) -> Iterator[Check]:
    st = "stints"
    for s in seasons:
        df = pq(C.PROCESSED / "stints" / f"{s}.parquet")
        if df is None:
            yield skip(f"stints {s}", st, "EXACT", "no processed/stints")
            continue
        df = df.sort_values(["nhl_game_id", "stint_idx"]).reset_index(drop=True)

        # E1 partition: duration = end−start; consecutive stints touch (end == next start); idx is 0..n−1
        bad_dur = int((df.duration_s != (df.end_g - df.start_g)).sum())
        nxt_start = df.groupby("nhl_game_id").start_g.shift(-1)
        gaps = int(((nxt_start.notna()) & (df.end_g.values != nxt_start.values)).sum())
        idx_ok = bool((df.groupby("nhl_game_id").cumcount().values == df.stint_idx.values).all())
        yield exact(f"stint duration = end−start ({s})", st, bad_dur == 0, bad_dur, 0)
        yield exact(
            f"stints partition the game, no gaps ({s})",
            st,
            gaps == 0,
            gaps,
            0,
            "end_g[i] == start_g[i+1] within each game",
        )
        yield exact(f"stint_idx contiguous 0..n−1 ({s})", st, idx_ok, None, None)

        # E2 personnel ↔ strength
        hn_ok = int((df.home_n != df.home_skaters.map(cell_len)).sum())
        an_ok = int((df.away_n != df.away_skaters.map(cell_len)).sum())
        strength_bad = int(
            (df.strength != (df.home_n.astype(str) + "v" + df.away_n.astype(str))).sum()
        )
        yield exact(f"home_n == len(home_skaters) ({s})", st, hn_ok == 0, hn_ok, 0)
        yield exact(f"away_n == len(away_skaters) ({s})", st, an_ok == 0, an_ok, 0)
        yield exact(f"strength == 'home_n v away_n' ({s})", st, strength_bad == 0, strength_bad, 0)

        # A4 overload rate
        if "overload" in df.columns and len(df):
            rate = float(df.overload.mean())
            yield approx(
                f"overload-stint rate ({s})",
                st,
                rate,
                None,
                0.001,
                f"{int(df.overload.sum())} illegal (>6-skater) stints",
            )

        # E3 xGF partition: Σ stint xGF == Σ modeled-shot xg, per game (no double-count, no leak)
        so = pq(
            C.PROCESSED / "shots_onice" / f"{s}.parquet",
            columns=["nhl_game_id", "stint_idx", "xg", "goal"],
        )
        if so is not None:
            stint_xgf = df.assign(t=df.home_xgf + df.away_xgf).groupby("nhl_game_id").t.sum()
            shot_xg = so[so.xg.notna()].groupby("nhl_game_id").xg.sum()
            j = pd.concat([stint_xgf.rename("stint"), shot_xg.rename("shot")], axis=1).fillna(0.0)
            max_diff = float((j.stint - j.shot).abs().max()) if len(j) else 0.0
            yield exact(
                f"Σ stint xGF == Σ shot xg ({s})",
                st,
                max_diff < 0.02,
                round(max_diff, 4),
                0.0,
                "per-game; tolerance = 4-dp stint rounding",
            )

            # E5 every shot's stint_idx is valid (in range for its game), goal is binary
            nst = df.groupby("nhl_game_id").size()
            so2 = so.join(nst.rename("nst"), on="nhl_game_id")
            oob = int(((so2.stint_idx < 0) | (so2.stint_idx >= so2.nst)).sum())
            yield exact(f"shot stint_idx in range ({s})", st, oob == 0, oob, 0)
            yield exact(f"goal ∈ {{0,1}} ({s})", st, bool(so.goal.isin([0, 1]).all()), None, None)

            yield from _onice_quality(s, st)


def _onice_quality(season: int, stage: str) -> Iterator[Check]:
    so = pq(
        C.PROCESSED / "shots_onice" / f"{season}.parquet",
        columns=["onice_match", "event", "goal", "strength", "sit_home_n", "sit_away_n"],
    )
    if so is None or not len(so):
        return
    if "onice_match" in so.columns:
        good = float(so.onice_match.isin(["exact", "within1"]).mean())
        note = "known-gap season" if season in KNOWN_GAP_SEASONS else ""
        yield approx(f"on-ice match (exact+within1) ({season})", stage, good, 0.97, None, note)
    # A3 goal reconstructed-strength vs pbp situationCode
    g = so[(so.event == "goal") & so.sit_home_n.notna() & so.sit_away_n.notna()].copy()
    if len(g):
        parts = g.strength.str.split("v", expand=True)
        stint_even = pd.to_numeric(parts[0]) == pd.to_numeric(parts[1])
        sit_even = g.sit_home_n == g.sit_away_n
        mism = float((stint_even.values != sit_even.values).mean())
        yield approx(
            f"goal strength vs situationCode mismatch ({season})",
            stage,
            mism,
            None,
            0.03,
            "boundary rule files PP goals on the right stint",
        )


# ===============================================================================================
# STAGE: xg  (calibration of the chance layer)
# ===============================================================================================
def check_xg(seasons: list[int]) -> Iterator[Check]:
    st = "xg"
    pooled_xg = pooled_g = 0.0
    for s in seasons:
        so = pq(C.PROCESSED / "shots_onice" / f"{s}.parquet", columns=["xg", "goal"])
        if so is None:
            continue
        m = so[so.xg.notna()]
        xg, goals = float(m.xg.sum()), float(m.goal.sum())
        pooled_xg += xg
        pooled_g += goals
        rel = abs(xg - goals) / goals if goals else 0.0
        note = "known-gap season" if s in KNOWN_GAP_SEASONS else ""
        # per-season drift from a pooled isotonic fit is expected; the pooled total is the tight one
        yield approx(
            f"Σxg ≈ Σgoals ({s})", st, rel, None, 0.04, f"Σxg={xg:.0f} Σg={goals:.0f} {note}"
        )
    if pooled_g:
        rel = abs(pooled_xg - pooled_g) / pooled_g
        yield approx(
            "Σxg ≈ Σgoals (pooled, isotonic-calibrated)",
            st,
            rel,
            None,
            0.005,
            f"Σxg={pooled_xg:.0f} Σg={pooled_g:.0f}",
        )


# ===============================================================================================
# STAGE: models  (the additive identity + per-player conversion totals)
# ===============================================================================================
def check_models(seasons: list[int]) -> Iterator[Check]:
    st = "models"
    lab = label(seasons)

    # E8 per-player: fin_goals == α·shots ; gsax_saved == −γ·sa
    # NB: fin_per100/gsax_per100 are stored rounded, so these reconcile only to that precision
    # (the exact pre-rounding identity is covered by test_shooting_model / test_goal_accounting).
    fin = pq(C.MODELS / f"shooting_finishing_{lab}.parquet")
    if fin is not None and {"fin_per100", "fin_goals", "shots"} <= set(fin.columns):
        resid = float((fin.fin_goals - fin.fin_per100 / 100.0 * fin.shots).abs().max())
        yield exact(
            "fin_goals == (fin_per100/100)·shots",
            st,
            resid < 0.05,
            round(resid, 5),
            0.0,
            "to stored fin_per100 precision",
        )
        mx = (
            float(fin.loc[fin.shots >= 150, "fin_per100"].abs().max())
            if (fin.shots >= 150).any()
            else 0.0
        )
        yield approx("|finishing/100| within sane range (≥150 shots)", st, mx, None, 6.0)
    else:
        yield skip("finishing parquet", st, "EXACT", f"no shooting_finishing_{lab}.parquet")

    gl = pq(C.MODELS / f"shooting_goalie_{lab}.parquet")
    if gl is not None and {"gsax_per100", "gsax_saved", "sa"} <= set(gl.columns):
        resid = float((gl.gsax_saved - gl.gsax_per100 / 100.0 * gl.sa).abs().max())
        yield exact(
            "gsax_saved == (gsax_per100/100)·sa",
            st,
            resid < 0.1,
            round(resid, 5),
            0.0,
            "to stored gsax_per100 precision",
        )
        mx = float(gl.loc[gl.sa >= 200, "gsax_per100"].abs().max()) if (gl.sa >= 200).any() else 0.0
        yield approx("|GSAx/100| within sane range (≥200 SA)", st, mx, None, 2.0)

    # E7 league additive identity + A5 μ + A6 luck gap, read off the shipped goal-accounting meta
    meta_p = C.LOGS_MODEL / f"goal_accounting_{lab}.meta.json"
    if meta_p.exists():
        L = json.loads(meta_p.read_text())
        resid = abs(float(L.get("league_identity_resid", 1.0)))
        yield exact(
            "league identity: goals = Σxg+Σμ+Σfin+Σgoalie",
            st,
            resid < 0.5,
            round(resid, 4),
            0.0,
            f"goals={L.get('goals')} reconstructed={L.get('reconstructed')}",
        )
        n = max(int(L.get("n_shots", 1)), 1)
        mu = float(L.get("mu_total", 0.0)) / n
        yield approx(
            "μ (intercept/shot) is tiny", st, abs(mu), None, 0.002, f"μ={mu:.5f} goals/shot"
        )
        yield approx(
            "team-season GF luck gap (mean %)",
            st,
            float(L.get("team_gf_gap_pct_mean", 0.0)),
            None,
            12.0,
            "single-season finishing variance vs multi-year effects",
        )
    else:
        yield skip("goal-accounting meta", st, "EXACT", f"no goal_accounting_{lab}.meta.json")

    # A7 RAPM creation/suppression reconcile to ΣxGF: fold each player's baseline share back in
    # (baseline/5 + ev_off) and weight by his real role-TOI; summed leaguewide these should recover
    # the 5v5 xG that was actually created. APPROX by nature — ridge shrinkage + the dropped defence/
    # context terms leak, so the folded shares recover *most* (not all) of ΣxGF. This runs over the
    # FULL coefficient set (every skater), unlike the TOI-thresholded export-index ratio above.
    yield from _ev_creation_reconciliation(seasons, lab, st)

    # E9 goal-accounting team table: gf_recon == its components (internal consistency, no refit)
    team = pq(C.MODELS / f"goal_accounting_{lab}.parquet")
    if team is not None and {"gf_recon", "xgf", "mu_for", "fin_for", "opp_goalie"} <= set(
        team.columns
    ):
        # components stored at 2 dp -> reconcile to ~0.03 (5 rounded terms); exact pre-rounding
        d = float(
            (team.gf_recon - (team.xgf + team.mu_for + team.fin_for + team.opp_goalie)).abs().max()
        )
        yield exact(
            "team gf_recon == xgf+μ+fin+oppGoalie",
            st,
            d < 0.03,
            round(d, 4),
            0.0,
            "to stored 2-dp precision",
        )
        if {"ga_recon", "xga", "mu_against", "opp_fin", "own_goalie"} <= set(team.columns):
            d2 = float(
                (team.ga_recon - (team.xga + team.mu_against + team.opp_fin + team.own_goalie))
                .abs()
                .max()
            )
            yield exact(
                "team ga_recon == xga+μ+oppFin+ownGoalie",
                st,
                d2 < 0.03,
                round(d2, 4),
                0.0,
                "to stored 2-dp precision",
            )


def _league_5v5_xgf(seasons: list[int]) -> float:
    """Total 5v5 xGF over the RAPM model's exact stint universe (regular season, non-overload,
    duration ≥ MIN_STINT_S) — i.e. the regression target the on-ice coefficients were fit to."""
    tot = 0.0
    for s in seasons:
        d = pq(
            C.PROCESSED / "stints" / f"{s}.parquet",
            columns=["nhl_game_id", "strength", "overload", "duration_s", "home_xgf", "away_xgf"],
        )
        if d is None:
            continue
        reg = (d.nhl_game_id // 10000) % 100 == 2
        d = d[reg & (d.strength == "5v5") & (~d.overload) & (d.duration_s >= MIN_STINT_S)]
        tot += float((d.home_xgf + d.away_xgf).sum())
    return tot


def _ev_creation_reconciliation(seasons: list[int], lab: str, stage: str) -> Iterator[Check]:
    ev = pq(C.MODELS / f"ev_{lab}.parquet")
    need = {"ev_off", "ev_def", "ev_off_base", "ev_off_toi", "ev_def_toi"}
    if ev is None or not need <= set(ev.columns):
        yield skip("RAPM creation reconciliation", stage, "APPROX", f"no ev_{lab}.parquet")
        return
    target = _league_5v5_xgf(seasons)
    if target <= 0:
        yield skip("RAPM creation reconciliation", stage, "APPROX", "no 5v5 stints")
        return
    # fold the baseline share back in (baseline/5 + coef) and weight by role-TOI (minutes -> /60 = hrs)
    created = float(((ev.ev_off_base / 5.0 + ev.ev_off) * (ev.ev_off_toi / 60.0)).sum())
    allowed = float(((ev.ev_off_base / 5.0 + ev.ev_def) * (ev.ev_def_toi / 60.0)).sum())
    yield approx(
        "Σ created shares ≈ league 5v5 ΣxGF",
        stage,
        created / target,
        0.80,
        1.15,
        f"created={created:.0f} vs ΣxGF={target:.0f} ({100 * (created / target - 1):+.1f}%)",
    )
    yield approx(
        "Σ allowed shares ≈ league 5v5 ΣxGF",
        stage,
        allowed / target,
        0.75,
        1.15,
        f"allowed={allowed:.0f} vs ΣxGF={target:.0f} ({100 * (allowed / target - 1):+.1f}%); "
        "defence + context absorb more, so this reconciles looser than creation",
    )


# ===============================================================================================
# STAGE: gamelog  (per-game rollup == season box)
# ===============================================================================================
def check_gamelog(seasons: list[int]) -> Iterator[Check]:
    st = "gamelog"
    for s in seasons:
        gl = pq(
            C.PROCESSED / "gamelog" / f"{s}.parquet",
            columns=["player_id", "game_id", "g", "sog", "toi_s"],
        )
        box = pq(C.INTERIM / "box" / f"{s}.parquet")
        if gl is None or box is None:
            yield skip(f"gamelog rollup ({s})", st, "EXACT", "missing gamelog/box")
            continue
        bx = box[box.game_type == "regular"] if "game_type" in box.columns else box
        glr = gl[gl.game_id.map(C.is_regular_season)]
        a = glr.groupby("player_id")[["g", "sog", "toi_s"]].sum()
        b = bx.set_index("player_id")[["g", "sog", "toi_s"]]
        j = a.join(b, how="inner", lsuffix="_gl", rsuffix="_bx")
        for col in ("g", "sog", "toi_s"):
            bad = int((j[f"{col}_gl"] != j[f"{col}_bx"]).sum())
            yield exact(
                f"Σ gamelog {col} == box {col} ({s})",
                st,
                bad == 0,
                bad,
                0,
                f"{len(j)} players reconciled",
            )


# ===============================================================================================
# STAGE: goalies  (descriptive identities + split closure)
# ===============================================================================================
def check_goalies(seasons: list[int]) -> Iterator[Check]:
    st = "goalies"
    for s in seasons:
        gb = pq(C.PROCESSED / "goalie_box" / f"{s}.parquet")
        if gb is None or not len(gb):
            yield skip(f"goalie_box ({s})", st, "EXACT", "no goalie_box")
            continue
        d1 = float((gb.saves - (gb.sog_against - gb.ga)).abs().max())
        d2 = float((gb.gsax - (gb.xga - gb.ga)).abs().max())
        yield exact(f"saves == SOG_against − GA ({s})", st, d1 < 1e-6, round(d1, 6), 0.0)
        yield exact(f"GSAx == xGA − GA ({s})", st, d2 < 1e-6, round(d2, 6), 0.0)
        # split closure: danger buckets' SA sum to overall SA (the splits reconcile to the headline)
        dcols = [f"{b}_sa" for b in ("ld", "md", "hd") if f"{b}_sa" in gb.columns]
        if dcols:
            d3 = float((gb[dcols].sum(axis=1) - gb.sa).abs().max())
            yield exact(f"danger SA splits sum to total ({s})", st, d3 < 1e-6, round(d3, 6), 0.0)


# ===============================================================================================
# STAGE: export  (player value ledger + index/detail/file consistency + JSON validity)
# ===============================================================================================
def check_export(seasons: list[int]) -> Iterator[Check]:
    st = "export"
    idx_p = C.SITE_JSON / "players.json"
    if not idx_p.exists():
        yield skip("players.json", st, "EXACT", "no data/games/players.json")
        return
    raw = idx_p.read_text()
    yield exact("players.json has no NaN/Infinity", st, json_clean(raw), None, None)
    index = json.loads(raw)

    # E10 ledger: g_net == g_created + g_fin − g_allowed + g_pen. Each term is rounded to 1 dp, so 5
    # rounded values reconcile to ~0.25; the exact identity is in test_export_players.value_table.
    worst = 0.0
    over = 0
    for r in index:
        if r.get("g_net") is None:
            continue
        recon = (
            r.get("g_created", 0) + r.get("g_fin", 0) - r.get("g_allowed", 0) + r.get("g_pen", 0)
        )
        diff = abs(r["g_net"] - recon)
        worst = max(worst, diff)
        over += diff > 0.3
    yield exact(
        "g_net == created + fin − allowed + pen",
        st,
        over == 0,
        round(worst, 3),
        0.0,
        f"{len(index)} players; worst |Δ|={worst:.3f} (1-dp rounding)",
    )

    # E11 (JSON-level): percentiles ∈ [0,100]; attributed shares are ≥0 "in practice" (a handful of
    # extreme players can go slightly negative — see docs/metrics.md — so that one is APPROX).
    pcts = [r[k] for r in index for k in r if k.endswith("_pct") and r[k] is not None]
    pct_ok = all(0 <= p <= 100 for p in pcts)
    yield exact(
        "all percentiles ∈ [0,100]", st, pct_ok, None, None, f"{len(pcts)} percentile values"
    )
    shares = [
        r[k]
        for r in index
        for k in ("scoring60", "playmaking60", "allow60")
        if r.get(k) is not None
    ]
    neg_frac = sum(v < -1e-6 for v in shares) / len(shares) if shares else 0.0
    yield approx(
        "attributed shares ≥ 0 (in practice)",
        st,
        neg_frac,
        None,
        0.05,
        "extreme players may go slightly negative",
    )

    # E12 index ↔ detail files: every indexed player must have a detail file (stale extras tolerated)
    pdir = C.SITE_JSON / "player"
    if pdir.exists():
        files = {int(p.stem) for p in pdir.glob("*.json")}
        ids = {r["id"] for r in index}
        yield exact("every index player has a detail file", st, ids <= files, len(ids - files), 0)
        stale = len(files - ids)
        yield approx(
            "no stale player detail files",
            st,
            stale,
            None,
            0,
            f"{stale} files on disk not in the index (rebuild `player/` to clear)",
        )

    # A7 export-level reconciliation: index g_fin is a (thresholded) subset of league finishing goals
    g_fin_sum = sum(r.get("g_fin", 0) or 0 for r in index)
    fin = pq(C.MODELS / f"shooting_finishing_{label(seasons)}.parquet", columns=["fin_goals"])
    if fin is not None and fin.fin_goals.sum():
        yield approx(
            "Σ player g_fin vs Σ finishing goals (ratio)",
            st,
            g_fin_sum / float(fin.fin_goals.sum()),
            0.5,
            1.05,
            "index drops sub-threshold players, so ≤ 1",
        )

    for name in ("games.json", "goalies.json"):
        p = C.SITE_JSON / name
        if p.exists():
            yield exact(f"{name} has no NaN/Infinity", st, json_clean(p.read_text()), None, None)


# ===============================================================================================
# STAGE: games  (per-game JSON timeline self-consistency — sampled)
# ===============================================================================================
def check_games(seasons: list[int], sample: int) -> Iterator[Check]:
    st = "games"
    gdir = C.SITE_JSON / "game"
    idx_p = C.SITE_JSON / "games.json"
    if not gdir.exists() or not idx_p.exists():
        yield skip("per-game JSON", st, "EXACT", "no data/games/game/*.json")
        return
    index = json.loads(idx_p.read_text())
    gids = [g["game_id"] for g in index if (not seasons or g.get("season") in seasons)]
    # evenly sample across the index (deterministic) rather than checking thousands of files
    checked = gids if len(gids) <= sample else gids[:: max(1, len(gids) // sample)][:sample]
    worst_xgf = 0.0
    nan_leak = missing = 0
    for gid in checked:
        p = gdir / f"{gid}.json"
        if not p.exists():
            missing += 1
            continue
        raw = p.read_text()
        if not json_clean(raw):
            nan_leak += 1
        game = json.loads(raw)
        st_sum_h = round(sum(x["home_xgf"] for x in game["stints"]), 3)
        st_sum_a = round(sum(x["away_xgf"] for x in game["stints"]), 3)
        worst_xgf = max(
            worst_xgf,
            abs(st_sum_h - game["totals"]["home_xgf"]),
            abs(st_sum_a - game["totals"]["away_xgf"]),
        )
    note = f"sampled {len(checked)}/{len(gids)} games"
    yield exact("Σ stint xGF == game totals", st, worst_xgf < 1e-6, round(worst_xgf, 4), 0.0, note)
    yield exact("per-game JSON has no NaN/Infinity", st, nan_leak == 0, nan_leak, 0, note)
    if missing:
        yield exact("sampled games have a JSON file", st, False, missing, 0, f"{missing} missing")


# --- aggregate runner --------------------------------------------------------------------------
def run_all(seasons: list[int], sample: int = 60) -> list[Check]:
    checks: list[Check] = []
    checks += list(check_interim(seasons))
    checks += list(check_stints(seasons))
    checks += list(check_xg(seasons))
    checks += list(check_models(seasons))
    checks += list(check_gamelog(seasons))
    checks += list(check_goalies(seasons))
    checks += list(check_export(seasons))
    checks += list(check_games(seasons, sample))
    return checks
