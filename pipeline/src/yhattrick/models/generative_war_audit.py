"""WAR audit — the acceptance harness for every WAR change (first run: the 2026-07 audit).

Recomputes the latest season's WAR accounting with team labels and checks what the shipped
artifact can't check itself:

  A. league calibration — Σ model E[GF] vs actual goals in the fitted shot set (NO correction
     factor is applied anywhere; a material residual is a modeling finding, not a knob);
  B. per-team: model E[GF−GA] under actual lineups + the JOINT counterfactual (replace the whole
     team with replacements) vs actual goal differential and standings points;
  C. league totals: Σ per-player marginal GAR (the shipped WAR) vs Σ joint team GAR — the
     summed-marginals synergy factor (multiplicative model ⇒ marginals don't add to the joint);
  D. per-player sanity: |WAR| bounded by TOI, bottom tail named, WAR-rate ↔ GA/60 consistency.

Run after `make generative-cards`:
  uv run --group experimental python -m yhattrick.models.generative_war_audit [--season 2025]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd

from .. import config as C
from . import generative_cards as GC
from .generative_model import _sigmoid, EV_STRENGTHS, MA_STRENGTHS, AGE_SCALE
from .generative_features import side_rows

E_TOL = 0.02                # league ΣE vs actual: |residual| beyond this is a model finding
SCALE_TOL = 0.15            # card units vs reality: model F-mean scoring vs actual F goals/60


def rows_with_teams(season, key, strengths, dual, idx, gteam):
    """war_rows for ONE (season, bucket), plus the attacking/defending team label per side-row. Thin
    wrapper over the shared `side_rows` builder with team resolution (`gteam`)."""
    return side_rows([season], strengths, dual, idx, gteam=gteam)


def run_audit(season=None, fit_path=None):
    fit = GC.load_fit(fit_path)
    seasons = [int(s) for s in fit["seasons"]]
    S = int(season or max(seasons))
    t = GC.player_table(fit)
    n = len(t["id"])
    idx = {int(p): i for i, p in enumerate(t["id"])}
    kap = GC.kappa(fit)

    # values → GA/60 → replacement archetypes, exactly as build() derives them
    sc, pm, dfv = GC.values_at(fit, t, "ev", t["ev_shoot"], t["ev_create"], t["ev_def"],
                               t["fin_eff"], n_def=5)
    pp_sc, pp_pm, _ = GC.values_at(fit, t, "ma", t["pp_shoot"], t["pp_create"], t["pp_def"],
                                   t["fin_eff"], n_def=4)
    _, _, pk_df = GC.values_at(fit, t, "ma", t["pp_shoot"], t["pp_create"], t["pp_def"],
                               t["fin_eff"], n_def=4)
    vals = {"sc": sc, "pm": pm, "df": dfv, "pp_sc": pp_sc, "pp_pm": pp_pm, "pk_df": pk_df}
    base = GC.baselines(t, vals, kap)
    ga60 = GC.ga60_of(t, base, sc, pm, dfv, kap)
    bppsc = np.where(t["isD"] > 0, base["D"]["pp_sc"], base["F"]["pp_sc"])
    bpppm = np.where(t["isD"] > 0, base["D"]["pp_pm"], base["F"]["pp_pm"])
    bpkdf = np.where(t["isD"] > 0, base["D"]["pk_df"], base["F"]["pk_df"])
    pp_ga60 = (pp_sc - bppsc) + kap * (pp_pm - bpppm)
    pk_ga60 = kap * (pk_df - bpkdf)
    ev_el = t["toi_ev"] >= GC.EV_GATE
    pp_el = t["toi_pp"] >= GC.MA_GATE
    pk_el = t["toi_pk"] >= GC.MA_GATE
    ga_pct = GC.pct_within(ga60, ev_el, t["isD"])
    pp_pct = GC.pct_within(pp_ga60, pp_el, t["isD"])
    pk_pct = GC.pct_within(pk_ga60, pk_el, t["isD"])
    repl = {}
    for blk, arr, toi, elig, pct in (
            ("sh", t["ev_shoot"], t["toi_ev"], ev_el, ga_pct),
            ("cr", t["ev_create"], t["toi_ev"], ev_el, ga_pct),
            ("df", t["ev_def"], t["toi_ev"], ev_el, ga_pct),
            ("qs", t["qshoot_eff"], t["toi_ev"], ev_el, ga_pct),
            ("qd", t["qdef_eff"], t["toi_ev"], ev_el, ga_pct),
            ("fin", t["fin_eff"], t["toi_ev"], ev_el, ga_pct),
            ("pp_sh", t["pp_shoot"], t["toi_pp"], pp_el, pp_pct),
            ("pp_cr", t["pp_create"], t["toi_pp"], pp_el, pp_pct),
            ("pp_qs", t["qshoot_eff"], t["toi_pp"], pp_el, pp_pct),
            ("pp_fin", t["fin_eff"], t["toi_pp"], pp_el, pp_pct),
            ("pk_df", t["pp_def"], t["toi_pk"], pk_el, pk_pct),
            ("pk_qd", t["qdef_eff"], t["toi_pk"], pk_el, pk_pct)):
        repl[blk], _m = GC.band_mean(arr, toi, elig, pct, t["isD"])
    print("── replacement GA/60 bands (position mean of band members; the 'freely available' zero)")
    for lbl, metric, elig, pct, toi in (("ev", ga60, ev_el, ga_pct, t["toi_ev"]),
                                        ("pp", pp_ga60, pp_el, pp_pct, t["toi_pp"]),
                                        ("pk", pk_ga60, pk_el, pk_pct, t["toi_pk"])):
        _v, m = GC.band_mean(metric, toi, elig, pct, t["isD"])
        print(f"   {lbl}: {m}")

    games = json.load(open(C.WEB_DATA / "games.json"))
    gteam, gd = {}, defaultdict(lambda: {"gf": 0, "ga": 0, "pts": 0})
    for g in games:
        gteam[int(g["game_id"])] = (g["home"], g["away"])
        if int(g["season"]) != S or g.get("home_score") is None:
            continue
        hs, as_ = g["home_score"], g["away_score"]
        for tm, us, them in ((g["home"], hs, as_), (g["away"], as_, hs)):
            d = gd[tm]; d["gf"] += us; d["ga"] += them
            d["pts"] += 2 if us > them else (1 if g.get("ot") else 0)

    gsave_map = {int(g["id"]): g["gsave"] for g in fit["goalies"]}
    war_p = np.zeros(n)
    toi25 = np.zeros(n)
    joint = defaultdict(float)
    net_model = defaultdict(float)
    E_tot = {}
    f_hours = 0.0                                            # F slot-hours (value-scale check)
    for key, strengths, dual in (("ev", EV_STRENGTHS, True), ("ma", MA_STRENGTHS, False)):
        r = rows_with_teams(S, key, strengths, dual, idx, gteam)
        if r is None:
            continue
        rctx = fit["strengths"][key]["rate_ctx"]
        beta = np.array([rctx.get(c, 0.0) for c in ("home", "ozone", "dzone", "trail", "lead")])
        cx = np.exp(fit["strengths"][key]["rate_intercept"] + r["ctx"] @ beta
                    + rctx.get(f"season_{S}", 0.0))
        r["gsave"] = np.array([gsave_map.get(int(g), 0.0) if g == g and g is not None else 0.0
                               for g in r["goalie"]])
        mq = fit["mu_qual"]["ev" if key == "ev" else "ma"]
        zs = t["z_last"] - (t["last_season"] - S) / AGE_SCALE
        fin_s = GC.fin_eff_at(fit, t, np.where(np.isnan(t["age"]), 0.0, zs))
        kq = _sigmoid(mq + t["qdef_eff"]) / max(_sigmoid(mq), GC.EPS)
        psi0 = fit["strengths"][key]["psi0"]
        if key == "ev":
            sh = np.array([tr.get(str(S), {}).get("shoot", 0.0) for tr in t["trend"]])
            cr = np.array([tr.get(str(S), {}).get("create", 0.0) for tr in t["trend"]])
            df_ = np.array([tr.get(str(S), {}).get("def", 0.0) for tr in t["trend"]])
            rp = {"sh": repl["sh"], "cr": repl["cr"], "df": repl["df"]}
            r_qs, r_fin, r_qd = repl["qs"], repl["fin"], repl["qd"]
        else:
            sh, cr, df_ = t["pp_shoot"], t["pp_create"], t["pp_def"]
            rp = {"sh": repl["pp_sh"], "cr": repl["pp_cr"], "df": repl["pk_df"]}
            r_qs, r_fin, r_qd = repl["pp_qs"], repl["pp_fin"], repl["pk_qd"]
        gcl = GC.gbar_classes(fit, key, t["qshoot_eff"], fin_s, season=S)
        rgcl = GC.gbar_classes(fit, key, r_qs, r_fin, season=S)
        rp["g0"], rp["gF"], rp["gD"] = rgcl
        rp["kq"] = _sigmoid(mq + r_qd) / max(_sigmoid(mq), GC.EPS)

        ga, gdd, E = GC.war_bucket(r, n, sh, cr, df_, gcl, kq, r["gsave"], cx, rp,
                                   psi0, t["isD"])
        war_p += ga + gdd
        E_tot[key] = float(E.sum())
        if key == "ev":
            np.add.at(toi25, r["atk"], r["dur"][:, None] + np.zeros((1, 5)))
            f_hours = float(np.sum(r["dur"] * (t["isD"] < 0.5)[r["atk"]].sum(1)) / 3600.0)
        for tm, e in zip(r["ateam"], E):
            net_model[tm] += e
        for tm, e in zip(r["dteam"], E):
            net_model[tm] -= e
        # joint: whole-team replacement (attack side via bucket_E at repl params; defense closed-form)
        E_atk_repl = GC.bucket_E(r, rp["sh"], rp["cr"], df_, rgcl, kq, cx, psi0, t["isD"])
        dfB = (df_[r["def"]] * r["dmask"]).sum(1)
        lKB = (np.log(np.clip(kq[r["def"]], GC.EPS, None)) * r["dmask"]).sum(1)
        dfB_r = (rp["df"][r["def"]] * r["dmask"]).sum(1)
        lKB_r = (np.log(np.clip(rp["kq"][r["def"]], GC.EPS, None)) * r["dmask"]).sum(1)
        E_def_repl = E * np.exp(dfB_r - dfB + lKB_r - lKB)
        for tm, v in zip(r["ateam"], E - E_atk_repl):
            joint[tm] += v
        for tm, v in zip(r["dteam"], E_def_repl - E):
            joint[tm] += v
        print(f"  [{key}] rows {len(E):,}  ΣE[GF] {E.sum():.0f}")

    # value-scale check: card units must be real-world per-60 rates (the Hyman-case finding —
    # the reference environment ran ≈2× low before value_env)
    shots = pd.read_parquet(C.PROCESSED / "shots_onice" / f"{S}.parquet")
    ev5 = shots[shots.strength == "5v5"]
    isF_of = dict(zip(t["id"].tolist(), (t["isD"] < 0.5).tolist()))
    gF = int(sum(1 for sid, gl in zip(ev5.shooter_id, ev5.goal) if gl == 1 and isF_of.get(int(sid), True)))
    actual_f60 = gF / max(f_hours, 1e-9)                     # f_hours captured in the EV pass
    gate = (t["toi_ev"] >= GC.EV_GATE) & (t["isD"] < 0.5)
    model_f60 = float(np.average(sc[gate], weights=t["toi_ev"][gate]))
    ratio = model_f60 / max(actual_f60, 1e-9)
    print(f"\n── A0. card value scale ({S}): model F-mean scoring {model_f60:.3f} vs actual "
          f"F goals/60 {actual_f60:.3f}  (×{ratio:.2f}; tolerance ±{SCALE_TOL:.0%})")
    if abs(ratio - 1.0) > SCALE_TOL:
        print("   [FINDING] card units are off real-world scale — check value_env")

    print(f"\n── A. league calibration ({S}) — no correction factors anywhere")
    tot_E = sum(E_tot.values())
    rs = fit["conv"].get("recon_season", {})
    actual = (rs.get(str(S)) or rs.get(S) or [np.nan, np.nan])[1]
    resid = tot_E / actual - 1.0 if actual == actual else np.nan
    print(f"   model ΣE[GF] {tot_E:.0f} vs actual (fitted shot set) {actual:.0f}  "
          f"residual {resid:+.1%} (tolerance ±{E_TOL:.0%})")
    if abs(resid) > E_TOL:
        print("   [FINDING] residual exceeds tolerance — a model gap, not a knob to tune")

    print(f"\n── B/C. per-team ({S})")
    teams = sorted(joint, key=lambda x: -joint[x])
    jt = np.array([joint[tm] for tm in teams])
    nm = np.array([net_model[tm] for tm in teams])
    gdv = np.array([gd[tm]["gf"] - gd[tm]["ga"] for tm in teams])
    pts = np.array([gd[tm]["pts"] for tm in teams])
    for tm in teams[:5] + ["…"] + teams[-3:]:
        if tm == "…":
            print("   …")
            continue
        print(f"   {tm:4s} joint {joint[tm]:7.1f}  E[GF-GA] {net_model[tm]:7.1f}  "
              f"gdiff {gd[tm]['gf'] - gd[tm]['ga']:5d}  pts {gd[tm]['pts']:4d}")
    cor = lambda x, y: float(np.corrcoef(x, y)[0, 1])
    marg_total = float(war_p.sum())
    print(f"   league: Σjoint {jt.sum():.0f} goals ({jt.sum() / GC.GOALS_PER_WIN:.0f} WAR-equiv)  "
          f"Σmarginal {marg_total:.0f} ({marg_total / GC.GOALS_PER_WIN:.0f} WAR)  "
          f"synergy ×{marg_total / max(jt.sum(), 1e-9):.2f}")
    print(f"   corr(E[GF-GA], gdiff) {cor(nm, gdv):.3f}   corr(joint, gdiff) {cor(jt, gdv):.3f}   "
          f"corr(joint, pts) {cor(jt, pts):.3f}")

    print(f"\n── D. per-player ({S} marginal WAR = GAR/{GC.GOALS_PER_WIN:.0f})")
    war = war_p / GC.GOALS_PER_WIN
    toim = toi25 / 60.0
    low = toim < 200
    print(f"   <200 EV min: {int(low.sum())} players, max |WAR| {np.abs(war[low]).max():.2f}")
    full = toim >= 700
    for g, gm in (("F", t["isD"] < 0.5), ("D", t["isD"] > 0.5)):
        m = gm & full
        print(f"   full-time {g} (n={int(m.sum())}): mean {war[m].mean():+.2f}  "
              f"median {np.median(war[m]):+.2f}  p90 {np.percentile(war[m], 90):+.2f}")
    name = {}
    try:
        rowsj = json.load(open(C.WEB_DATA / "players.json"))
        name = {int(x["id"]): x["name"] for x in rowsj}
    except Exception:
        pass
    order = np.argsort(war)
    print("   bottom 6 / top 6:")
    for i in list(order[:6]) + list(order[-6:][::-1]):
        print(f"     {war[i]:6.2f}  {name.get(int(t['id'][i]), int(t['id'][i]))!s:24s} "
              f"{'D' if t['isD'][i] > 0 else 'F'}  {toim[i]:5.0f} EV min")
    el = full & ~np.isnan(ga60)
    rate = war[el] * GC.GOALS_PER_WIN / np.maximum(toim[el] / 60.0, 1e-9)
    print(f"   corr(WAR-per-hour, GA/60) full-time: {cor(rate, ga60[el]):.3f}")
    return {"E_resid": resid, "joint": float(jt.sum()), "marginal": marg_total,
            "corr_joint_gdiff": cor(jt, gdv)}


def main(argv=None):
    p = argparse.ArgumentParser(description="WAR audit — acceptance harness for WAR changes")
    p.add_argument("--season", type=int, default=None, help="audit season (default: latest fitted)")
    p.add_argument("--fit", type=str, default=None, help="fit JSON path (default: latest pooled)")
    args = p.parse_args(argv)
    run_audit(season=args.season, fit_path=args.fit)


if __name__ == "__main__":
    main()
