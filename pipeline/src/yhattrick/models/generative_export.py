"""Post-processing + CLI for the generative player model: turn a fit into per-60 goal values,
run the posterior-predictive check, print the leaderboards, and write the model JSON.

`run` fits the model (via generative_model.fit_all), collapses it to effective per-player parameters,
prices those into scoring / playmaking / defense values, projects to next season, and serialises
everything to `data/models/generative_model_<seasons>.json` — the artifact generative_cards consumes.
`make generative-model` reaches this through the thin CLI shim in generative_model.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .. import config as C
from .player_onice_model import roster_names
from .generative_likelihood import (
    _sigmoid,
    marginal_goal_prob,
    creator_mix,
    N_TM,
    N_DEF_EV,
    N_DEF_MA,
    EPS,
    SNIFF_MIN_TOI,
    SNIFF_MIN_TOI_MA,
    RW_SD_SHOOT,
    RW_SD_CREATE,
    RW_SD_DEF,
    ARENA_SD,
    ARENA_RW_SD,
    AGE_PEAK,
    AGE_SCALE,
)
from .generative_data import _curve_val
from .generative_model import (
    fit_all,
    effective_params,
    unit_effective,
    value_environment,
    _coef_map,
    _block_curve,
)


def player_values(rates, qual, conv, players, isD=None):
    """Per-strength deployment-free per-60 goal values. `rates` = {"ev": rate_fit, "ma": rate_fit}.
    For each strength (rate loadings are per-strength; quality/finishing loadings are POOLED, only the
    intercept splits):
       scoring(j)    = exp(mu_rate+shoot) · ḡ_j                                                   own shots, converted
       playmaking(p) = N_TM · exp(mu_rate) · (exp(create)−1) · sigmoid(mu_qual+qcreate)           teammate xG added
       defense(d)    = N_DEF · [exp(mu_rate)·sigmoid(mu_qual) − exp(mu_rate+def)·sigmoid(mu_qual+qdef)]  suppression
    ḡ_j is the model's own goals-per-shot: the conversion curve marginalized over BOTH the fitted
    shot-quality distribution (Beta, concentration beta_s — see marginal_goal_prob) AND the creator
    classes (unassisted / F-created / D-created; qcreate is parameterized with unassisted as the
    reference and created shots carry the negative position bumps, so skipping this marginalization
    prices every shot as unassisted ≈ +13% goals league-wide). Falls back to the point shortcut when
    beta_s / the qcreate pair / isD are unavailable (synthetic fixtures).
    EV → ev_scoring/playmaking/defense (N_DEF=5). MA → pp_scoring/pp_playmaking + pk_defense (N_DEF=4;
    the MA `def` loadings ARE the penalty-killers). All per 60; defense >0 = suppresses (good)."""
    out = {}
    qshoot, qcreate, qdef, fin = qual["qshoot"], qual["qcreate"], qual["qdef"], conv["fin"]
    s_conc = qual.get("beta_s")  # Stage-2 Beta concentration
    qc_pos = qual.get("qcreate_pos")  # the [F, D] creator-class pair
    if qc_pos is None and np.asarray(qcreate).shape == (2,):
        qc_pos = np.asarray(qcreate)
    for key, rate in rates.items():
        ml, mq = rate["intercept"], qual["mu_qual"][key]
        n_def = N_DEF_EV if key == "ev" else N_DEF_MA
        a, b = conv["a"][key], conv["b"][key]
        cr, shoot, defn = rate["create"], rate["shoot"], rate["def"]
        shots = np.exp(ml + shoot)
        q_own = np.clip(_sigmoid(mq + qshoot), EPS, 1 - EPS)
        if s_conc and qc_pos is not None and isD is not None:
            w0, wf, wd = creator_mix(rate["psi0"], isD, key)
            p_own = p_own0 = 0.0
            for w, qc in ((w0, 0.0), (wf, float(qc_pos[0])), (wd, float(qc_pos[1]))):
                qb = np.clip(_sigmoid(mq + qshoot + qc), EPS, 1 - EPS)
                p_own = p_own + w * marginal_goal_prob(qb, float(s_conc), a, b, fin)
                p_own0 = p_own0 + w * marginal_goal_prob(qb, float(s_conc), a, b)
        elif s_conc:  # Beta marginal, no creator classes
            p_own = marginal_goal_prob(q_own, float(s_conc), a, b, fin)
            p_own0 = marginal_goal_prob(q_own, float(s_conc), a, b)
        else:  # point shortcut (synthetic fixtures)
            lq = np.log(q_own / (1 - q_own))
            p_own = _sigmoid(a * lq + b + fin)
            p_own0 = _sigmoid(a * lq + b)
        base = np.exp(ml) * _sigmoid(mq)
        env = float(rate.get("value_env") or 1.0)  # average-environment factor (see
        out[key] = {  # value_environment; 1.0 = reference)
            "scoring": env * shots * p_own,  # own shots, converted to goals
            "finishing": env * shots * (p_own - p_own0),  # goals above xG-implied conversion
            "own_xg": env * shots * _sigmoid(mq + qshoot),
            "own_shots": env * shots,
            "playmaking": env * N_TM * np.exp(ml) * (np.exp(cr) - 1.0) * _sigmoid(mq + qcreate),
            "defense": env * n_def * (base - np.exp(ml + defn) * _sigmoid(mq + qdef)),
            "creator_share": np.exp(cr) / (np.exp(rate["psi0"]) + np.exp(cr) + (N_TM - 1)),
        }
    return out


def ppc(R, rate, qual, conv, key, seed=0, agepos=None):
    """Forward-simulate every shooter-stint row of one strength bucket (`key` ∈ {'ev','ma'}) and compare
    to actuals: shot count (rate), shot quality (pooled quality at the strength intercept), total goals
    (conversion at the strength's a/b). Rate gathers use the per-season UNIT states when present; the
    quality context is mapped from the rate row — home, season indicators, position offsets (F5); the
    conversion side includes the season offsets and finishing curve (the goalie AGE curve is omitted —
    the per-goalie gsave still applies). Defenders masked (≤5)."""
    rng = np.random.default_rng(seed)
    dmask = R.get("def_mask")
    if dmask is None:
        dmask = np.ones_like(R["def_idx"], dtype=np.float64)
    sh_u = R.get("shooter_unit", R["shooter_idx"])
    tm_u = R.get("team_unit", R["team_idx"])
    df_u = R.get("def_unit", R["def_idx"])
    eta_r = (
        rate["intercept"]
        + rate["shoot"][sh_u]
        + rate["create"][tm_u].sum(1)
        + (rate["def"][df_u] * dmask).sum(1)
        + R["Xctx"] @ rate["beta"]
    )
    acol = R.get("arena_col")
    if acol is not None and len(rate.get("arena_vec", [])):
        eta_r = eta_r + np.where(acol >= 0, rate["arena_vec"][np.clip(acol, 0, None)], 0.0)
    mu = np.exp(eta_r + R["offset"])
    if rate.get("count_model") == "nb" and rate.get("r"):
        r = rate["r"]
        N = rng.negative_binomial(r, r / (r + mu))
    else:
        N = rng.poisson(mu)

    # quality marginalized over the creator, at this strength's quality intercept (mu_qual[key]);
    # context mapped from the rate row: home + season indicators + position offsets (F5)
    qb = np.asarray(qual["beta"])
    nseas = R.get("n_season_cols", 0)
    qctx = qb[0] * R["Xctx"][:, 0]
    if nseas and len(qb) >= 2 + nseas:
        qctx = qctx + R["Xctx"][:, 5 : 5 + nseas] @ qb[2 : 2 + nseas]
    qcm = _coef_map(qual.get("ctx_names"), qb)
    isD = agepos["isD"] if agepos is not None else None
    if isD is not None:
        qctx = (
            qctx
            + qcm.get("shooter_D", 0.0) * isD[R["shooter_idx"]]
            + qcm.get("def_D", 0.0) * (isD[R["def_idx"]] * dmask).sum(1)
        )
    if acol is not None and len(qual.get("arena_vec", [])):
        qctx = qctx + np.where(acol >= 0, qual["arena_vec"][np.clip(acol, 0, None)], 0.0)
    base = (
        qual["mu_qual"][key]
        + qual["qshoot"][R["shooter_idx"]]
        + (qual["qdef"][R["def_idx"]] * dmask).sum(1)
        + qctx
    )
    qc2 = np.asarray(qual["qcreate"])
    if qc2.shape == (2,):  # A1: position-level creator bump
        qbump = qc2[
            (
                isD[R["team_idx"]] if isD is not None else np.zeros_like(R["team_idx"], dtype=float)
            ).astype(np.int64)
        ]
    else:  # legacy per-player array (synthetic tests)
        qbump = qc2[R["team_idx"]]
    sig5 = np.concatenate([_sigmoid(base)[:, None], _sigmoid(base[:, None] + qbump)], 1)
    lg = np.concatenate([np.full((len(base), 1), rate["psi0"]), rate["create"][tm_u]], 1)
    pi = np.exp(lg - lg.max(1, keepdims=True))
    pi /= pi.sum(1, keepdims=True)
    qpred_marg = (pi * sig5).sum(1)
    gsave_map = {g: conv["gsave"][i] for i, g in enumerate(conv["goalies"])}
    fin_row = conv["fin"][R["shooter_idx"]]  # per shooter-stint row
    soff = np.zeros(len(base))
    ccm = _coef_map(conv.get("ctx_names"), conv.get("beta", []))
    if agepos is not None and "season_row" in R and ccm:
        zsh = np.zeros(len(base))
        for s in sorted(set(int(x) for x in np.unique(R["season_row"]))):
            m = R["season_row"] == s
            zsh[m] = agepos["z"][s][R["shooter_idx"][m]]
            soff[m] = ccm.get(f"season_{s}", 0.0)
        fin_row = fin_row + _block_curve(
            ccm, "fin", zsh, isD[R["shooter_idx"]], off_name="shooter_D"
        )
    gsv_row = np.array(
        [gsave_map.get(int(g), 0.0) if pd.notna(g) else 0.0 for g in R["def_goalie"]]
    )

    rep = np.repeat(np.arange(len(N)), N)
    p_q = qpred_marg[rep]
    s = qual["beta_s"]
    q = rng.beta(s * p_q, s * (1 - p_q))  # this chance's drawn xG
    qc = np.clip(q, EPS, 1 - EPS)
    eta = (
        conv["a"][key] * np.log(qc / (1 - qc))
        + conv["b"][key]
        + soff[rep]
        + fin_row[rep]
        + gsv_row[rep]
    )
    p_goal = _sigmoid(eta)  # logit conversion; bounded
    goals_sim = int(rng.binomial(1, p_goal).sum())
    return {
        "shots_actual": int(R["count"].sum()),
        "shots_sim": int(N.sum()),
        "shots_expected": float(mu.sum()),
        "perrow_mean_actual": float(R["count"].mean()),
        "perrow_mean_sim": float(N.mean()),
        "perrow_var_actual": float(R["count"].var()),
        "perrow_var_sim": float(N.var()),
        "zero_frac_actual": float((R["count"] == 0).mean()),
        "zero_frac_sim": float((N == 0).mean()),
        "mean_xg_sim": float(q.mean()),
        "goals_sim": goals_sim,
        "n_sim_shots": int(rep.size),
    }


def _board(names, players, val, toi, label, se=None, higher=True, n=12, min_toi=SNIFF_MIN_TOI):
    elig = toi >= min_toi
    order = np.argsort(val * (-1 if higher else 1))
    order = [i for i in order if elig[i]][:n]
    print(f"\n{label}")
    for i in order:
        pid = players[i]
        nm = names.get(int(pid), {}).get("name", f"#{pid}")
        if se is not None:
            z = val[i] / se[i] if se[i] > 0 else 0.0
            flag = "" if abs(z) >= 2 else "  ⚠ low-conf"
            print(
                f"   {val[i]:+.3f} ±{1.96 * se[i]:.3f} (z={z:+.1f}){flag}  {nm:24s} ({toi[i] / 60:.0f} min)"
            )
        else:
            print(f"   {val[i]:+.3f}  {nm:24s} ({toi[i] / 60:.0f} min)")


def _arena_json(fit):
    """{venue: {season: coef}} from a fit's arena states (None when absent)."""
    vec = fit.get("arena_vec")
    if vec is None or not len(vec):
        return None
    out = {}
    for v, s, x in zip(fit["arena_venue"], fit["arena_season"], vec):
        out.setdefault(v, {})[int(s)] = round(float(x), 4)
    return out


def _curve_json(cm, blk, off_name=None):
    """JSON form of one fitted aging curve: coefficients, the D intercept offset, and the curve
    sampled over ages 18–40 per position (site-ready)."""
    return {
        "coef": {c: cm.get(f"{blk}_{c}", 0.0) for c in ("zF", "z2F", "zD", "z2D")},
        "d_offset": cm.get(off_name or f"{blk}_D", 0.0),
        "curve": {
            pos: {
                a: round(
                    float(
                        _curve_val(
                            [
                                cm.get(f"{blk}_zF", 0.0),
                                cm.get(f"{blk}_z2F", 0.0),
                                cm.get(f"{blk}_zD", 0.0),
                                cm.get(f"{blk}_z2D", 0.0),
                            ],
                            (a - AGE_PEAK) / AGE_SCALE,
                            d,
                        )
                    ),
                    4,
                )
                for a in range(18, 41, 2)
            }
            for pos, d in (("F", 0.0), ("D", 1.0))
        },
    }


def run(
    seasons,
    count_model="poisson",
    spg_scale=1.0,
    warm=True,
    reexport=False,
    ma_anchor_scale=1.0,
    ma_create_prior_sd=None,
    ma_def_prior_sd=None,
    ev_anchor_scale=1.0,
    create_prior_center=None,
):
    names = roster_names(seasons)
    M = fit_all(
        seasons,
        count_model=count_model,
        spg_scale=spg_scale,
        warm=warm,
        reexport=reexport,
        save_ckpt=not reexport,
        ma_anchor_scale=ma_anchor_scale,
        ma_create_prior_sd=ma_create_prior_sd,
        ma_def_prior_sd=ma_def_prior_sd,
        ev_anchor_scale=ev_anchor_scale,
        create_prior_center=create_prior_center,
    )
    players, agepos, Q = M["players"], M["agepos"], M["Q"]
    rates, spg, qual, conv, last_season = (
        M["rates"],
        M["spg"],
        M["qual"],
        M["conv"],
        M["last_season"],
    )

    # average-environment factor per bucket (card units = real-world per-60 rates)
    for key in rates:
        rates[key]["value_env"] = value_environment(rates[key], season=int(max(seasons)))
    print("  value environment: " + "  ".join(f"{k} ×{rates[k]['value_env']:.2f}" for k in rates))

    # effective per-player params (last state + position offset + curve) → values
    eff_rates, eff_qual, eff_conv = effective_params(
        rates, qual, conv, players, agepos, last_season
    )
    vals = player_values(eff_rates, eff_qual, eff_conv, players, isD=agepos["isD"])

    # pooled (shared) skill leaderboards — once. RAW player residuals, not effective params: the
    # position offsets / age curves are calibration terms (A2), and adding them back would turn a
    # talent board into a position board (e.g. the conversion shooter_D offset compensates the a>1
    # slope on low-xG point shots — every D would top "finishing").
    ev_toi = rates["ev"]["R"]["toi"]
    _board(
        names,
        players,
        qual["qshoot"],
        ev_toi,
        "TOP qshoot (own-shot danger above position baseline):",
    )
    _board(
        names,
        players,
        conv["fin"],
        ev_toi,
        "TOP FINISHING (log-odds above position/age baseline):",
        conv["se_fin"],
    )
    fz = (np.abs(conv["fin"]) > 2 * conv["se_fin"]) & (ev_toi >= SNIFF_MIN_TOI)
    print(
        f"  conversion identification: fin |z|>2 in {int(fz.sum())}/{int((ev_toi >= SNIFF_MIN_TOI).sum())} "
        f"eligible (weak-signal — expected)"
    )

    # per-strength rate/value leaderboards + posterior-predictive check
    ppcs = {}
    for key in rates:
        R = rates[key]["R"]
        off, dfn = R["toi_atk"], R["toi_def"]
        gate = SNIFF_MIN_TOI if key == "ev" else SNIFF_MIN_TOI_MA
        lbl = "EV" if key == "ev" else "PP"
        print(f"\n──[{lbl}] rate/value leaderboards ──")
        _board(
            names,
            players,
            rates[key]["create_last"],
            off,
            f"[{lbl}] TOP create (playmaking volume, last state):",
            rates[key]["se_create_last"],
            min_toi=gate,
        )
        _board(
            names,
            players,
            vals[key]["scoring"],
            off,
            f"[{lbl}] TOP SCORING (goals/60, own shots):",
            min_toi=gate,
        )
        _board(
            names,
            players,
            vals[key]["playmaking"],
            off,
            f"[{lbl}] TOP PLAYMAKING (xG/60):",
            min_toi=gate,
        )
        _board(
            names,
            players,
            vals[key]["defense"],
            dfn,
            f"[{'EV' if key == 'ev' else 'PK'}] BEST DEFENSE (xG/60 suppressed):",
            min_toi=gate,
        )
        pp = ppc(R, rates[key], qual, conv, key, agepos=agepos)
        slab = 0 if key == "ev" else 1
        pp["goals_actual"] = int(((Q["goal"] == 1) & (Q["strength"] == slab)).sum())
        ppcs[key] = pp
        print(
            f"  [{lbl}] PPC — shots a/s {pp['shots_actual']:,}/{pp['shots_sim']:,} "
            f"(model-exp {pp['shots_expected']:.0f})  goals a/s {pp['goals_actual']:,}/{pp['goals_sim']:,}"
        )

    # projection: ages advance to the target season, states stay at their last value (RW mean)
    proj = None
    if len(set(seasons)) >= 2:
        target = int(max(seasons)) + 1
        pr, pq, pc = effective_params(
            rates, qual, conv, players, agepos, last_season, target=target
        )
        pvals = player_values(pr, pq, pc, players, isD=agepos["isD"])
        proj = {"season": target, "vals": pvals}
        print(
            f"\n──[projection → {target}] (last RW state + aging curve; reference environment) ──"
        )
        _board(
            names,
            players,
            pvals["ev"]["scoring"],
            ev_toi,
            f"PROJECTED {target} EV SCORING (goals/60):",
        )
        _board(
            names,
            players,
            pvals["ev"]["playmaking"],
            ev_toi,
            f"PROJECTED {target} EV PLAYMAKING (xG/60):",
        )

    _save(
        seasons,
        players,
        rates,
        qual,
        conv,
        vals,
        ppcs,
        spg,
        agepos,
        last_season,
        (eff_rates, eff_qual, eff_conv),
        proj,
    )


def _save(seasons, players, rates, qual, conv, vals, ppcs, spg, agepos, last_season, eff, proj):
    label = "+".join(map(str, seasons))
    ev, ma = rates.get("ev"), rates.get("ma")
    eff_rates, eff_qual, eff_conv = eff
    qc = np.asarray(qual["qcreate"])
    curves = {}
    for key, rate in rates.items():
        cm = _coef_map(rate.get("ctx_names"), rate["beta"])
        if cm:
            for blk in ("shoot", "create", "def"):
                curves[f"{key}_{blk}"] = _curve_json(cm, blk)
    ccm = _coef_map(conv.get("ctx_names"), conv.get("beta", []))
    if ccm:
        curves["fin"] = _curve_json(ccm, "fin", off_name="shooter_D")
        curves["gsave_age"] = {"z": ccm.get("g_z", 0.0), "z2": ccm.get("g_z2", 0.0)}
    out = {
        "model": "generative_model_ev_ma",
        "seasons": list(seasons),
        "count_model": ev.get("count_model"),
        "nb_r": ev.get("r"),
        "shots_per_goal": spg,
        "value_env": {k: r.get("value_env") for k, r in rates.items()},
        "quality_intercept": float(qual["intercept"]),
        "mu_qual": qual["mu_qual"],
        "beta_s": qual["beta_s"],
        "qcreate": {
            "F": float(qc[0]),
            "D": float(qc[1]),
            "se_F": float(qual["se_qcreate"][0]),
            "se_D": float(qual["se_qcreate"][1]),
        },
        "quality_ctx": {n: float(v) for n, v in zip(qual.get("ctx_names") or [], qual["beta"])},
        "age_curves": curves,
        "rw_sd": {"shoot": RW_SD_SHOOT, "create": RW_SD_CREATE, "def": RW_SD_DEF},
        "arena_effects": {
            "ev": _arena_json(ev) if ev else None,
            "ma": _arena_json(ma) if ma else None,
            "quality": _arena_json(qual),
            "prior": {"sd": ARENA_SD, "rw_sd": ARENA_RW_SD},
        },
        "missing_birthdates": agepos["missing"],
        "conv": {
            "a": conv["a"],
            "b": conv["b"],
            "prior_sd_fin": conv["prior_sd_fin"],
            "prior_sd_gsave": conv["prior_sd_gsave"],
            "sum_p": conv["sum_p"],
            "sum_y": conv["sum_y"],
            "recon_season": conv.get("recon_season"),
            "ctx": {n: float(v) for n, v in zip(conv.get("ctx_names", []), conv.get("beta", []))},
        },
        "strengths": {
            k: {
                "rate_intercept": float(rates[k]["intercept"]),
                "psi0": float(rates[k]["psi0"]),
                "a2_q": rates[k].get("a2_q"),
                "n_a2": rates[k].get("n_a2"),
                "rate_ctx": {
                    n: float(v) for n, v in zip(rates[k].get("ctx_names") or [], rates[k]["beta"])
                },
                "ppc": ppcs.get(k),
            }
            for k in rates
        },
        "players": [],
        "goalies": [
            {
                "id": int(conv["goalies"][j]),
                "gsave": float(conv["gsave"][j]),
                "gsave_se": float(conv["se_gsave"][j]),
            }
            for j in range(len(conv["goalies"]))
        ],
    }
    trend = {}
    if ev is not None and ev.get("unit_season") is not None:
        ue = unit_effective(ev, agepos)
        for u, (p, s) in enumerate(zip(ev["unit_player"], ev["unit_season"])):
            trend.setdefault(int(p), {})[int(s)] = {
                "shoot": round(float(ue["shoot"][u]), 4),
                "create": round(float(ue["create"][u]), 4),
                "def": round(float(ue["def"][u]), 4),
            }
    for i in range(len(players)):
        ls = int(last_season[i])
        age = agepos["age"].get(ls, np.full(len(players), np.nan))[i] if ls > 0 else np.nan
        rec = {
            "id": int(players[i]),
            "pos": "D" if agepos["isD"][i] else "F",
            "age": round(float(age), 1) if np.isfinite(age) else None,
            "last_season": ls if ls > 0 else None,
            "toi_ev": float(ev["R"]["toi_atk"][i]) if ev else 0.0,
            "toi_pp": float(ma["R"]["toi_atk"][i]) if ma else 0.0,
            "toi_pk": float(ma["R"]["toi_def"][i]) if ma else 0.0,
            "qshoot": float(qual["qshoot"][i]),
            "qdef": float(qual["qdef"][i]),
            "n_create": int(qual["n_create"][i]),
            "fin": float(conv["fin"][i]),
            "fin_se": float(conv["se_fin"][i]),
            "ev_defense": float(vals["ev"]["defense"][i]) if "ev" in vals else 0.0,
            "pk_defense": float(vals["ma"]["defense"][i]) if "ma" in vals else 0.0,
        }
        for key, pre in [("ev", "ev"), ("ma", "pp")]:
            if key in rates:
                rec[f"{pre}_shoot"] = float(eff_rates[key]["shoot"][i])
                rec[f"{pre}_create"] = float(eff_rates[key]["create"][i])
                rec[f"{pre}_create_se"] = float(rates[key]["se_create_last"][i])
                rec[f"{pre}_def"] = float(eff_rates[key]["def"][i])
                rec[f"{pre}_scoring"] = float(vals[key]["scoring"][i])
                rec[f"{pre}_playmaking"] = float(vals[key]["playmaking"][i])
                rec[f"{pre}_finishing"] = float(vals[key]["finishing"][i])
        if i in trend:
            rec["trend"] = trend[i]
        out["players"].append(rec)
    if proj:
        pv = proj["vals"]
        pl = []
        for i in range(len(players)):
            rec = {
                "id": int(players[i]),
                "ev_scoring": float(pv["ev"]["scoring"][i]),
                "ev_playmaking": float(pv["ev"]["playmaking"][i]),
                "ev_defense": float(pv["ev"]["defense"][i]),
            }
            if "ma" in pv:
                rec.update(
                    pp_scoring=float(pv["ma"]["scoring"][i]),
                    pp_playmaking=float(pv["ma"]["playmaking"][i]),
                    pk_defense=float(pv["ma"]["defense"][i]),
                )
            pl.append(rec)
        out["projection"] = {
            "season": proj["season"],
            "players": pl,
            "note": "last RW state + aging curve at target-season age; "
            "reference environment; uncertainty widens by rw_sd per block",
        }
    C.MODELS.mkdir(parents=True, exist_ok=True)
    (C.MODELS / f"generative_model_{label}.json").write_text(json.dumps(out))
    print(f"\n  -> data/models/generative_model_{label}.json")


def main(argv=None):
    p = argparse.ArgumentParser(description="Experimental shooter-resolved generative model — POC")
    p.add_argument(
        "--season", type=int, default=None, help="one season (default: latest available)"
    )
    p.add_argument("--pool", action="store_true", help="pool all available seasons")
    p.add_argument(
        "--count",
        choices=["poisson", "nb"],
        default="poisson",
        help="count layer: poisson (Var=μ) or nb (negative binomial, Var=μ+μ²/r)",
    )
    p.add_argument(
        "--spg-scale",
        type=float,
        default=1.0,
        help="multiply the assist-credit weight (A3 sensitivity checks: 0.5 / 2.0)",
    )
    p.add_argument(
        "--ma-anchor-scale",
        type=float,
        default=0.25,
        help="scale the PP/PK bucket's assist-anchor weight (default: the value the "
        "2026-07 held-out calibration sweep selected — docs §7/§9)",
    )
    p.add_argument(
        "--ev-anchor-scale",
        type=float,
        default=0.25,
        help="scale the EV bucket's assist-anchor weight (default: the value the held-out "
        "sweep selected — docs §7/§9). At full weight the anchor overfits assist-ROLE into "
        "create (held-out teammate-shot corr barely beat naive counting); 0.25 predicts "
        "next-season teammate shots markedly better and relaxes the Kapanen-class drag (§5e)",
    )
    p.add_argument(
        "--ma-create-prior",
        type=float,
        default=0.04,
        help="create prior SD for the PP/PK bucket (default: sweep-selected)",
    )
    p.add_argument(
        "--ma-def-prior",
        type=float,
        default=0.10,
        help="def prior SD for the PP/PK bucket (default: sweep-selected)",
    )
    p.add_argument(
        "--create-prior-center",
        choices=["zero", "position-mean"],
        default="position-mean",
        help="EV create ridge target: 'position-mean' (default; held-out selected — shrinks a "
        "weakly-identified forward toward the forward baseline, not to 0) or 'zero' (legacy). §5e/§7",
    )
    p.add_argument(
        "--cold",
        action="store_true",
        help="ignore the θ̂ checkpoint (default: warm-start each stage from the last fit)",
    )
    p.add_argument(
        "--reexport",
        action="store_true",
        help="skip optimization: reuse the checkpoint's θ̂/SEs (exact same fit signature "
        "required) and just regenerate reports + JSON — for export-side changes",
    )
    args = p.parse_args(argv)
    sd = C.PROCESSED / "shots_onice"
    avail = sorted(int(f.stem) for f in sd.glob("*.parquet")) if sd.exists() else []
    if not avail:
        raise SystemExit("no processed shots — run `make stints` first")
    seasons = avail if args.pool else ([args.season] if args.season else [avail[-1]])
    run(
        seasons,
        count_model=args.count,
        spg_scale=args.spg_scale,
        warm=not args.cold,
        reexport=args.reexport,
        ma_anchor_scale=args.ma_anchor_scale,
        ma_create_prior_sd=args.ma_create_prior,
        ma_def_prior_sd=args.ma_def_prior,
        ev_anchor_scale=args.ev_anchor_scale,
        create_prior_center=(
            None if args.create_prior_center == "zero" else args.create_prior_center
        ),
    )
