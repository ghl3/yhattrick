"""Export the interactive model-explainer payload to site JSON.

The /models page reproduces the generative model's closed-form value equations client-side
(TypeScript port of `values_at`/`gbar_classes`/`creator_mix`/`marginal_goal_prob`). This exporter
ships everything that page needs in one small file:

  - the fitted global constants (intercepts, conversion map, creator mix, kappa, ...)
  - the 40-node Gauss-Legendre grid the Beta marginal uses (identical nodes -> identical numbers)
  - replacement archetypes, position baselines, and per-position skill distributions (slider scales)
  - curated player templates: real players' effective parameters + their Python-computed values,
    which the client asserts against at load time (the port must reproduce them)

Writes data/games/gen_model.json, then syncs to web/public/data.

Usage:  uv run --group experimental python -m yhattrick.export.export_gen_explainer
"""

from __future__ import annotations

import json
import shutil

import numpy as np

from .. import config as C
from ..models.generative_cards import (
    EV_GATE,
    kappa,
    player_table,
    values_at,
)
from ..models.generative_likelihood import _GL_W, _GL_X

QUANTS = [1, 5, 10, 25, 50, 75, 90, 95, 99]

# effective-parameter keys shipped per template / distribution: rate (sh/cr/df), quality
# (qs/qd/cq), conversion (fin) — everything values_at needs besides the fit constants
PARAM_KEYS = {
    "sh": "ev_shoot",
    "cr": "ev_create",
    "df": "ev_def",
    "qs": "qshoot_eff",
    "qd": "qdef_eff",
    "cq": "create_qual",
    "fin": "fin_eff",
}


def _load(path):
    with open(path) as f:
        return json.load(f)


def _names() -> dict[int, str]:
    idx = _load(C.SITE_JSON / "players.json")
    return {int(r["id"]): r["name"] for r in idx}


def _template(t, i, cards, names, label):
    pid = int(t["id"][i])
    card = cards.get(str(pid), {})
    attrs = card.get("attrs", {})
    n_seasons = max(len(t["trend"][i]), 1)
    out = {
        "id": pid,
        "name": names.get(pid, str(pid)),
        "pos": "D" if t["isD"][i] > 0 else "F",
        "age": None if np.isnan(t["age"][i]) else int(t["age"][i]),
        "params": {k: round(float(t[col][i]), 4) for k, col in PARAM_KEYS.items()},
        "toi_ev_min": round(float(t["toi_ev"][i]) / 60.0 / n_seasons, 1),
        # the Python-computed values the client-side port must reproduce (and displays)
        "expected": {
            "sc": round(float(t["ev_scoring"][i]), 4),
            "pm": round(float(t["ev_playmaking"][i]), 4),
            "df": round(float(t["ev_defense"][i]), 4),
            "ga60": (attrs.get("ga60") or {}).get("v"),
            "war": (attrs.get("war") or {}).get("v"),
        },
    }
    if label:
        out["label"] = label
    return out


def _pick(mask, score, taken):
    """Index of the highest `score` under `mask`, skipping already-taken players."""
    order = np.argsort(-np.where(mask, score, -np.inf))
    for i in order:
        if score[i] == score[i] and mask[i] and i not in taken:
            return int(i)
    return None


def build() -> dict:
    cards_doc = _load(C.MODELS / "gen_cards.json")
    meta = cards_doc["meta"]
    fit = _load(meta["fit"])
    cards = cards_doc["players"]
    names = _names()

    t = player_table(fit)
    elig = t["toi_ev"] >= EV_GATE
    # template curation wants recognizable CURRENT regulars: active in the latest season with
    # >=500 EV min in the window
    reg = (t["toi_ev"] >= 30000.0) & (t["last_season"] == fit["seasons"][-1])
    isF, isD = t["isD"] < 0.5, t["isD"] > 0.5

    # card reads per player (for template selection)
    def attr(key):
        return np.array(
            [
                ((cards.get(str(int(pid)), {}).get("attrs", {}).get(key) or {}).get("v"))
                if cards.get(str(int(pid))) is not None
                else None
                for pid in t["id"]
            ],
            dtype=object,
        )

    def num(key):
        a = attr(key)
        return np.array([float(v) if v is not None else np.nan for v in a])

    ga60, war = num("ga60"), num("war")
    pm_v, fin_v, prev_v = num("playmaking"), num("finishing"), num("prevented60")

    # curated archetypes, chosen by criteria so a refit re-picks sensibly
    taken: set[int] = set()
    picks: list[tuple[str, int | None]] = []
    for label, mask, score in (
        ("Elite dual-threat forward", reg & isF, war),
        ("Elite playmaker", reg & isF, pm_v),
        ("Elite finisher", reg & isF, fin_v),
        ("Defensive forward", reg & isF, prev_v),
        ("Offensive defenseman", reg & isD, ga60),
        ("Shutdown defenseman", reg & isD, prev_v),
        ("League-average forward", reg & isF, -np.abs(ga60 - np.nanmedian(ga60[elig & isF]))),
        ("Replacement-level forward", reg & isF, -np.abs(ga60)),
    ):
        i = _pick(mask, score, taken)
        if i is not None:
            taken.add(i)
            picks.append((label, i))

    templates = [_template(t, i, cards, names, label) for label, i in picks]

    # every gate-clearing player, so the simulator can start from ANY player's fitted skills
    # (the curated templates above are just quick picks into this list)
    players = [
        _template(t, int(i), cards, names, "")
        for i in np.flatnonzero(elig)
        if int(t["id"][i]) in names
    ]
    players.sort(key=lambda p: p["name"])

    # per-position quantiles of each effective parameter over EV-eligible players (slider scales
    # + percentile readouts)
    dist = {}
    for g, gm in (("F", isF & elig), ("D", isD & elig)):
        dist[g] = {
            k: [round(float(v), 4) for v in np.percentile(t[col][gm], QUANTS)]
            for k, col in PARAM_KEYS.items()
        }

    ev, ma = fit["strengths"]["ev"], fit["strengths"]["ma"]
    return {
        "fit_seasons": fit["seasons"],
        "n_players": len(fit["players"]),
        "constants": {
            "mu_rate": {"ev": ev["rate_intercept"], "ma": ma["rate_intercept"]},
            "psi0": {"ev": ev["psi0"], "ma": ma["psi0"]},
            "mu_qual": fit["mu_qual"],
            "beta_s": fit["beta_s"],
            "qcreate": {"F": fit["qcreate"]["F"], "D": fit["qcreate"]["D"]},
            "conv_a": fit["conv"]["a"],
            "conv_b": fit["conv"]["b"],
            "value_env": fit["value_env"],
            "kappa": round(kappa(fit), 4),
            "goals_per_win": meta["goals_per_win"],
            "shots_per_goal": fit["shots_per_goal"],
        },
        "gl40": {"x": list(_GL_X), "w": list(_GL_W)},
        "quantiles": QUANTS,
        "dist": dist,
        "replacement": meta["replacement"],
        "replacement_values": meta["replacement_values"],
        "baselines": meta["baselines"],
        "repl_band_pct": meta["repl_band_pct"],
        "fin_avg_p100": meta["fin_avg_p100"],
        "templates": templates,
        "players": players,
    }


def main() -> None:
    out = build()

    # sanity: values_at on every shipped player's (rounded) params must reproduce his shipped
    # values — the same tolerance the client-side port asserts at load
    fit = _load(_load(C.MODELS / "gen_cards.json")["meta"]["fit"])
    rows = out["players"]
    tt = {
        "isD": np.array([1.0 if r["pos"] == "D" else 0.0 for r in rows]),
        "qshoot_eff": np.array([r["params"]["qs"] for r in rows]),
        "qdef_eff": np.array([r["params"]["qd"] for r in rows]),
        "create_qual": np.array([r["params"]["cq"] for r in rows]),
    }
    sc, pm, df = values_at(
        fit, tt, "ev",
        np.array([r["params"]["sh"] for r in rows]),
        np.array([r["params"]["cr"] for r in rows]),
        np.array([r["params"]["df"] for r in rows]),
        np.array([r["params"]["fin"] for r in rows]),
        5.0,
    )
    for i, r in enumerate(rows):
        for got, want, k in ((sc[i], r["expected"]["sc"], "sc"), (pm[i], r["expected"]["pm"], "pm"), (df[i], r["expected"]["df"], "df")):
            if abs(got - want) > 0.02:
                raise AssertionError(f"{r['name']} {k}: recomputed {got:.4f} != shipped {want:.4f}")

    dst = C.SITE_JSON / "gen_model.json"
    dst.write_text(json.dumps(out))
    web = C.WEB_DATA / "gen_model.json"
    web.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(dst, web)
    kb = dst.stat().st_size / 1024
    print(f"wrote {dst} ({kb:.0f} KB) and synced to {web}")
    for tpl in out["templates"]:
        print(f"  template: {tpl['label']:<28} {tpl['name']} ({tpl['pos']}) war={tpl['expected']['war']}")


if __name__ == "__main__":
    main()
