r"""Held-out-season predictive harness for the generative model (roadmap #1).

Fits the full model on seasons ≤ --train-through, projects every player to the following season
(the RW-mean state + aging curve at next-season age), then scores the projection against what
ACTUALLY happened that season — on the season's real lineups — versus simpler reads of the same
training fit. This converts "the trajectories look right" into "they predict better", and is the
tool for tuning RW_SD_* (rerun with different constants, compare).

CANDIDATE per-player EV parameter sets (all from the ONE training fit; identical treatment of
intercept/context, so differences isolate the player-skill read):
  league-avg    every player at the prior mean — the "no player skill" floor (it keeps the full
                context incl. position/age composition, which needs no skill knowledge)
  pooled-mean   TOI-weighted mean of the player's per-season effective states — the static-pooled
                read (what you'd estimate with no drift and no projection)
  last-state    the player's LAST effective state (state + curve at his last age) — drift, no aging
  projection    last RW state + curve at TARGET-season age — the model's actual forecast

The skill candidates carry each player's aging curve + position offset inside their EFFECTIVE
parameters, so the holdout context applies only the base columns (home/zone/score) + the league
environment nowcast (final training season's FE) — applying the AGE/position columns too would
double-count terms with large nonzero league means (a ~2× Σ-shots inflation; the per-candidate
block-mean diagnostic line exists to catch exactly that class of bug).

SCORING (EV rate stage — own-shot generation on the held-out season's real stint rows):
  1. Row-level Poisson deviance of observed focal-shooter Fenwick counts under each candidate's
     rates (per-1000-row scale; lower is better). Real lineups ⇒ deployment identical across
     candidates. Arena states are omitted for all candidates (the target season's venue states
     don't exist in training — common treatment).
  2. Player-level own-shots/60: predicted (candidate rates summed over the player's actual rows)
     vs observed, TOI-weighted correlation + MAE over eligible players (≥ MIN_TOI_EVAL in the
     target season AND seen in training). Also scored: the naive baseline (the player's raw
     final-training-season shots/60 — the "just use last year's stats" bar).

Players unseen in training (rookies) sit at the prior mean for every candidate; the eligibility
gate keeps the player-level table honest.

The training side (candidate vectors, naive aggregates, context coefficients) is cached to
data/models/holdout_fit_<train>.npz so scoring iterations don't repay the ~hours-long fit:
  uv run --group experimental python -m yhattrick.models.generative_holdout            # 2021-24 → 2025
  uv run --group experimental python -m yhattrick.models.generative_holdout --rescore  # reuse cache
Output: printed tables + data/models/holdout_<target>.json.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .. import config as C
from . import generative_model as G

MIN_TOI_EVAL = 12000.0      # ≥ 200 EV minutes in the target season to enter the player-level table
CAND_ORDER = ("league-avg", "pooled-mean", "last-state", "projection")
BLOCKS = ("shoot", "create", "def")


def poisson_deviance(N, mu):
    """Total Poisson deviance 2Σ[μ − N + N·log(N/μ)] (the 0·log0 term is 0)."""
    return float(2.0 * np.sum(mu - N + np.where(N > 0, N * np.log(np.maximum(N, 1e-12) / mu), 0.0)))


def wpearson(x, y, w):
    """Weighted Pearson correlation."""
    w = w / w.sum()
    mx, my = np.sum(w * x), np.sum(w * y)
    cov = np.sum(w * (x - mx) * (y - my))
    return float(cov / np.sqrt(np.sum(w * (x - mx) ** 2) * np.sum(w * (y - my) ** 2)))


def wmae(x, y, w):
    """Weighted mean absolute error."""
    return float(np.sum(w * np.abs(x - y)) / w.sum())


def candidates(M, target):
    """Per-player (training-index) EV shoot/create/def parameter sets — see module docstring."""
    rates, qual, conv = M["rates"], M["qual"], M["conv"]
    players, agepos, ls = M["players"], M["agepos"], M["last_season"]
    P = len(players)
    re_last, _, _ = G.effective_params(rates, qual, conv, players, agepos, ls)
    re_proj, _, _ = G.effective_params(rates, qual, conv, players, agepos, ls, target=target)
    ev = rates["ev"]
    ue = G.unit_effective(ev, agepos)
    tw, up = ev["R"]["toi_unit"], ev["unit_player"]
    pooled = {}
    for blk in BLOCKS:
        ws = np.bincount(up, weights=tw, minlength=P)
        vs = np.bincount(up, weights=ue[blk] * tw, minlength=P)
        pooled[blk] = np.where(ws > 0, vs / np.maximum(ws, 1e-9), 0.0)
    return {
        "league-avg": {b: np.zeros(P) for b in BLOCKS},
        "pooled-mean": pooled,
        "last-state": {b: re_last["ev"][b] for b in BLOCKS},
        "projection": {b: re_proj["ev"][b] for b in BLOCKS},
    }


def _fit_train_side(train, train_through, target, count_model, spg_scale):
    """Run the training fit and reduce it to the slim scoring inputs; cached as an npz so scoring
    can iterate without re-fitting."""
    M = G.fit_all(train, count_model=count_model, spg_scale=spg_scale)
    evt = M["rates"]["ev"]
    Rt = evt["R"]
    P = len(M["players"])
    mlast = Rt["season_row"] == train_through                # naive baseline: final-season raw rates
    ts = {"players": np.array(M["players"], dtype=np.int64),
          "intercept": np.float64(evt["intercept"]),
          "ctx_names": np.array(evt["ctx_names"]), "beta": np.asarray(evt["beta"], dtype=np.float64),
          "s_last": np.bincount(Rt["shooter_idx"][mlast], weights=Rt["count"][mlast], minlength=P),
          "t_last": np.bincount(Rt["shooter_idx"][mlast], weights=Rt["dur"][mlast], minlength=P),
          "train": np.array(train, dtype=np.int64)}
    for name, c in candidates(M, target).items():
        for blk in BLOCKS:
            ts[f"cand_{name}_{blk}"] = c[blk]
    path = C.MODELS / f"holdout_fit_{train_through}.npz"
    C.MODELS.mkdir(parents=True, exist_ok=True)
    np.savez(path, **ts)
    print(f"[holdout] training side cached -> {path.name}")
    return ts


def evaluate(train_through, target=None, count_model="nb", spg_scale=1.0, rescore=False):
    sd = C.PROCESSED / "shots_onice"
    avail = sorted(int(f.stem) for f in sd.glob("*.parquet")) if sd.exists() else []
    train = [s for s in avail if s <= train_through]
    target = target or train_through + 1
    if target not in avail:
        raise SystemExit(f"target season {target} not in processed data {avail}")
    cache = C.MODELS / f"holdout_fit_{train_through}.npz"
    if rescore and cache.exists():
        print(f"[holdout] rescoring from {cache.name}")
        z = np.load(cache)
        ts = {k: z[k] for k in z.files}
    else:
        print(f"[holdout] train {train} → target {target} — fitting …")
        ts = _fit_train_side(train, train_through, target, count_model, spg_scale)
    players_t = ts["players"]
    idx_t = {int(p): i for i, p in enumerate(players_t)}

    # held-out EV rows on the target season's own player index (rookies included, at the prior mean)
    players_h, idx_h = G.player_index([target])
    agepos_h = G._age_position(players_h, [target])
    Rh = G.rate_rows([target], G.EV_STRENGTHS, True, players_h, idx_h, agepos_h,
                     states=False, arenas=False)
    Ph = len(players_h)
    seen = np.array([p in idx_t for p in players_h])
    print(f"[holdout] target rows {len(Rh['count']):,}  players {Ph} ({int(seen.sum())} seen in training)")

    # context by NAME: base columns for the skill candidates (their effective params already carry
    # curve + position offsets — see docstring); full columns for the no-skill league-avg floor
    cmt = G._coef_map([str(n) for n in ts["ctx_names"]], ts["beta"])
    nowcast = cmt.get(f"season_{train_through}", 0.0)
    ctx_base = np.full(len(Rh["count"]), nowcast)
    ctx_full = ctx_base.copy()
    base_names = set(G.RATE_CTX)
    for j, nm in enumerate(Rh["ctx_names"]):
        if nm in cmt:
            if nm in base_names:
                ctx_base = ctx_base + cmt[nm] * Rh["Xctx"][:, j]
            ctx_full = ctx_full + cmt[nm] * Rh["Xctx"][:, j]

    def to_hold(vec_t):
        out = np.zeros(Ph)
        for i, p in enumerate(players_h):
            j = idx_t.get(int(p))
            if j is not None:
                out[i] = vec_t[j]
        return out

    N = Rh["count"]
    toi_h = Rh["toi_atk"]
    obs_shots = np.bincount(Rh["shooter_idx"], weights=N, minlength=Ph)
    obs_rate = np.where(toi_h > 0, obs_shots / np.maximum(toi_h, 1.0) * 3600.0, 0.0)
    elig = (toi_h >= MIN_TOI_EVAL) & seen
    w = toi_h
    mu0 = float(ts["intercept"])

    out = {"train": [int(s) for s in ts["train"]], "target": int(target), "count_model": count_model,
           "n_rows": int(len(N)), "n_players": Ph, "n_seen": int(seen.sum()),
           "n_eligible": int(elig.sum()),
           "rw_sd": {"shoot": G.RW_SD_SHOOT, "create": G.RW_SD_CREATE, "def": G.RW_SD_DEF},
           "candidates": {}}
    print(f"\n[holdout] eligible for the player table: {int(elig.sum())} "
          f"(≥{MIN_TOI_EVAL / 60:.0f} EV min in {target} + trained)")
    print(f"{'candidate':14s} {'row-dev/1k':>11s} {'Σμ/ΣN':>8s} {'corr':>7s} {'MAE/60':>7s}")
    for name in CAND_ORDER:
        sh = to_hold(ts[f"cand_{name}_shoot"])
        cr = to_hold(ts[f"cand_{name}_create"])
        df = to_hold(ts[f"cand_{name}_def"])
        ctx = ctx_full if name == "league-avg" else ctx_base
        eta = (mu0 + sh[Rh["shooter_idx"]] + cr[Rh["team_idx"]].sum(1)
               + (df[Rh["def_idx"]] * Rh["def_mask"]).sum(1) + ctx)
        mu = np.exp(eta + Rh["offset"])
        dev = poisson_deviance(N, mu)
        pred_shots = np.bincount(Rh["shooter_idx"], weights=mu, minlength=Ph)
        pred_rate = np.where(toi_h > 0, pred_shots / np.maximum(toi_h, 1.0) * 3600.0, 0.0)
        corr = wpearson(pred_rate[elig], obs_rate[elig], w[elig])
        mae = wmae(pred_rate[elig], obs_rate[elig], w[elig])
        out["candidates"][name] = {"row_deviance_per_1k": dev / len(N) * 1000.0,
                                   "sum_mu": float(mu.sum()), "sum_N": float(N.sum()),
                                   "rate_corr": corr, "rate_mae60": mae}
        wm = w[seen] / max(w[seen].sum(), 1e-9)              # block-mean diagnostic (level-bug canary)
        print(f"{name:14s} {dev / len(N) * 1000.0:11.3f} {mu.sum() / N.sum():8.3f} "
              f"{corr:7.3f} {mae:7.3f}   blocks sh {np.sum(wm * sh[seen]):+.2f} "
              f"cr {np.sum(wm * cr[seen]):+.2f} df {np.sum(wm * df[seen]):+.2f}")

    # naive baseline: the player's raw final-training-season shots/60 predicts his target rate
    naive_t = np.where(ts["t_last"] > 0, ts["s_last"] / np.maximum(ts["t_last"], 1.0) * 3600.0, 0.0)
    naive_h = to_hold(naive_t)
    en = elig & (to_hold(ts["t_last"]) >= MIN_TOI_EVAL)
    corr = wpearson(naive_h[en], obs_rate[en], w[en])
    mae = wmae(naive_h[en], obs_rate[en], w[en])
    out["naive_last_season"] = {"rate_corr": corr, "rate_mae60": mae, "n": int(en.sum())}
    print(f"{'naive-' + str(train_through):14s} {'—':>11s} {'—':>8s} {corr:7.3f} {mae:7.3f}"
          f"   (n={int(en.sum())}, needs {train_through} TOI too)")
    lm = float(np.sum(w[elig] * obs_rate[elig]) / w[elig].sum())
    out["league_mean_mae60"] = wmae(np.full(int(elig.sum()), lm), obs_rate[elig], w[elig])
    print(f"{'league-mean':14s} {'—':>11s} {'—':>8s} {'0.000':>7s} {out['league_mean_mae60']:7.3f}")

    path = C.MODELS / f"holdout_{target}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"\n  -> {path}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Held-out-season predictive harness (generative model)")
    p.add_argument("--train-through", type=int, default=2024, help="last training season")
    p.add_argument("--target", type=int, default=None, help="held-out season (default: next)")
    p.add_argument("--count", choices=["poisson", "nb"], default="nb")
    p.add_argument("--spg-scale", type=float, default=1.0)
    p.add_argument("--rescore", action="store_true",
                   help="reuse the cached training side (holdout_fit_<train>.npz) — skip the fit")
    args = p.parse_args(argv)
    evaluate(args.train_through, args.target, count_model=args.count,
             spg_scale=args.spg_scale, rescore=args.rescore)


if __name__ == "__main__":
    main()
