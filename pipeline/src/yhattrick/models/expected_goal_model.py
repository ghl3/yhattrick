"""Expected-goals (xG) model: the probability an unblocked shot becomes a goal.

This xG is the calibrated chance-quality baseline of the additive theory of goals (see
docs/modeling.md): STAGE 1 (player_onice_model) distributes it across on-ice skaters, and STAGE 2
(shooting_model) splits the residual goal − xG between shooter and goalie. Strength (man-advantage)
is an xG feature, so the conversion layer can pool all situations.

For every unblocked shot (shot-on-goal / missed-shot / goal) the model estimates goal probability
from the shot's geometry and pre-shot context, all computed from the NHL play-by-play event stream
(`interim/events`).

Features:
  geometry      distance, angle, oriented x/y (attacking net at +89 via homeTeamDefendingSide)
  shot          shot_type, zone
  pre-shot      last_event_type, time/dist/speed since last event, rebound, rush
  context       angle_change (goalie lateral sweep), royal-road cross-slot pass, possession
                continuity (same-team last event), since-faceoff, skater counts (3v3/4v4/OT openness)
  state         strength differential, score differential, period, game time, home/away
  role          shooter is a defenseman (point shots) — a role covariate, NOT finishing skill
Shooter identity is excluded on purpose: xG is the chance quality; finishing skill lives in
finishing.py.

Model: XGBoost (binary:logistic) with GroupKFold-by-game out-of-fold predictions, then an isotonic
recalibration on the OOF probabilities — this flattens per-bin calibration bias and makes the league
total xG match goals. Validation (AUC / log-loss / Brier / reliability) is computed on the OOF
predictions.

Outputs:
  data/processed/xg/<season>.parquet   per-shot predictions
  data/models/xg_booster.json          fitted booster
  data/models/xg_isotonic.json         calibration mapping
  logs/model/xg_<label>.meta.json      metrics, reliability, importances
  web/public/data/xg_model.json        payload for the model-exploration page

Usage:
  uv run python -m yhattrick.models.expected_goal_model                 # all available seasons, pooled
  uv run python -m yhattrick.models.expected_goal_model --season 2024
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from .. import config as C
from ..data import shot_geom as G
from ..data.shot_geom import GOAL_X

ROYAL_ROAD_S = 4.0       # window for a cross-slot (royal-road) pass to count
N_SPLITS = 5             # GroupKFold folds (grouped by game)

# XGBoost hyperparameters (modest trees; the page can't retrain, so no live tuning)
XGB_PARAMS = dict(
    max_depth=6, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8,
    min_child_weight=5, gamma=0.0, reg_lambda=1.0, objective="binary:logistic",
    eval_metric="logloss", tree_method="hist", n_estimators=4000,
)
EARLY_STOP = 50

CAT_FEATURES = ["shot_type", "zone", "last_event_type"]
# Every feature describes the PLAY itself (geometry + pre-shot context + game state) — never who is
# on the ice. Player, team, goalie quality and deployment are deliberately excluded so the xG stays a
# pure chance-quality measure; player value is isolated downstream (RAPM / finishing) and would be
# double-counted if it leaked in here.
NUM_FEATURES = [
    "distance", "abs_angle", "x_adj", "abs_y", "time_since_last", "time_since_shot",
    "dist_from_last", "speed_from_last", "angle_change", "last_dist_to_net", "since_faceoff",
    "attempts_15s", "strength_diff", "shooter_skaters", "def_skaters", "score_diff", "period", "time_g",
    # boolean flags stored as int (off_wing may be NaN when handedness is unknown)
    "is_home", "rebound", "rush", "same_team_last", "royal_road", "last_behind_net", "off_wing",
]
FEATURES = NUM_FEATURES + CAT_FEATURES

# heatmap grid for the exploration page (offensive zone, attacking net on the right at +89)
HM_X = list(range(26, 90, 2))      # blue line-ish to the goal line
HM_Y = list(range(-42, 43, 2))
HM_SHOT_TYPES = ["wrist", "snap", "slap", "tip-in", "backhand", "deflected"]
HM_STRENGTHS = {"PP": (5, 4), "EV": (5, 5), "PK": (4, 5)}   # shooter vs defending skaters


def available_seasons() -> list[int]:
    d = C.INTERIM / "events"
    return sorted(int(p.stem) for p in d.glob("*.parquet")) if d.exists() else []


def handedness_map() -> dict[int, str]:
    """player_id -> 'L'/'R' from the cached NHL player-landing json (raw/nhl/players)."""
    hand: dict[int, str] = {}
    for p in C.RAW_PLAYERS.glob("*.json"):
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        sc = d.get("shootsCatches")
        if sc in ("L", "R"):
            hand[int(p.stem)] = sc
    return hand


# --- feature engineering -----------------------------------------------------
def build_features(ev: pd.DataFrame, hand: dict[int, str]) -> pd.DataFrame:
    """From the full event stream build one feature row per modeled (goalie-present) shot."""
    ev = ev.sort_values(["nhl_game_id", "time_g", "event_idx"]).reset_index(drop=True)
    g = ev.groupby("nhl_game_id", sort=False)

    # previous event (within the same game) for pre-shot context
    prev_x, prev_y = g.x.shift(1), g.y.shift(1)
    prev_t = g.time_g.shift(1)
    prev_type = g.type.shift(1)
    prev_zone = g.zone.shift(1)
    prev_home = g.is_home.shift(1)

    # time since the most recent faceoff in this game
    fo_time = ev.time_g.where(ev.type == "faceoff")
    last_fo = fo_time.groupby(ev.nhl_game_id).ffill()

    # time since the most recent shot ATTEMPT (rebounds/flurries beyond the immediate prior event)
    corsi = ev.type.isin(G.FENWICK + ["blocked-shot"])
    shot_t = ev.time_g.where(corsi).groupby(ev.nhl_game_id).ffill()
    prev_shot_t = shot_t.groupby(ev.nhl_game_id).shift(1)   # previous attempt, excluding this one

    # sustained pressure: attacking-team shot attempts in the prior 15s (cycle / scramble)
    attempts_15s = pd.Series(0.0, index=ev.index)
    cev = ev.loc[corsi, ["nhl_game_id", "is_home", "time_g"]]
    for _, grp in cev.groupby(["nhl_game_id", "is_home"], sort=False):
        t = grp.time_g.to_numpy()
        lo = np.searchsorted(t, t - 15, side="left")
        attempts_15s.loc[grp.index] = np.arange(len(t)) - lo   # earlier same-team attempts in window

    # running score (strictly before each event) per team
    home_goal = ((ev.type == "goal") & ev.is_home).astype(int)
    away_goal = ((ev.type == "goal") & ~ev.is_home).astype(int)
    home_score = home_goal.groupby(ev.nhl_game_id).cumsum() - home_goal
    away_score = away_goal.groupby(ev.nhl_game_id).cumsum() - away_goal

    sit = G.decode_situation(ev.situation_code)

    df = ev.assign(
        prev_x=prev_x, prev_y=prev_y, prev_type=prev_type, prev_zone=prev_zone,
        prev_home=prev_home, since_faceoff=(ev.time_g - last_fo),
        time_since_last=(ev.time_g - prev_t), time_since_shot=(ev.time_g - prev_shot_t),
        attempts_15s=attempts_15s, home_score=home_score, away_score=away_score, **sit,
    )

    # keep only modeled shots: unblocked, regular season, real strength, goalie in net
    df = df[df.type.isin(G.FENWICK)].copy()
    df = df[df.nhl_game_id.map(C.is_regular_season)]
    df = df[df.period < 5]                                    # drop shootout (regular-season SO = P5)
    df = df[df.x.notna() & df.y.notna() & df.primary_player_id.notna()]
    df = df[df[["away_goalie", "away_skaters", "home_skaters", "home_goalie"]].notna().all(axis=1)]

    is_home = df.is_home.to_numpy(bool)
    # orient to the attacking net at +89, then derive geometry (shared with clean.build_shots)
    sign = G.attack_sign(is_home, df.home_defending_side.to_numpy())
    x_adj, y_adj = G.oriented(df.x, df.y, sign)
    px_adj, py_adj = G.oriented(df.prev_x, df.prev_y, sign)

    dx = GOAL_X - x_adj
    df["x_adj"] = x_adj
    df["abs_y"] = np.abs(y_adj)
    df["distance"], df["abs_angle"] = G.distance_angle(x_adj, y_adj)
    signed_angle = np.degrees(np.arctan2(y_adj, dx))
    prev_signed = np.degrees(np.arctan2(py_adj, GOAL_X - px_adj))
    df["angle_change"] = np.abs(signed_angle - prev_signed)
    df["last_dist_to_net"] = np.sqrt((GOAL_X - px_adj) ** 2 + py_adj ** 2)
    df["last_behind_net"] = (px_adj > GOAL_X).astype(int)   # shot off a cycle / wrap from behind

    tsl = df.time_since_last.to_numpy(float)
    dist_last = np.sqrt((x_adj - px_adj) ** 2 + (y_adj - py_adj) ** 2)
    df["dist_from_last"] = dist_last
    df["speed_from_last"] = np.divide(dist_last, tsl, out=np.zeros_like(dist_last), where=tsl > 0)

    df["rebound"] = G.rebound_flag(df.prev_type, tsl)
    df["rush"] = G.rush_flag(df.prev_zone, tsl)
    df["same_team_last"] = (df.prev_home.to_numpy() == is_home).astype(int)
    df["royal_road"] = ((np.sign(py_adj) != np.sign(y_adj)) & (np.abs(py_adj) > 3)
                        & (np.abs(y_adj) > 3) & (tsl <= ROYAL_ROAD_S)).astype(int)

    shooter_sk = np.where(is_home, df.home_skaters, df.away_skaters)
    def_sk = np.where(is_home, df.away_skaters, df.home_skaters)
    def_goalie = np.where(is_home, df.away_goalie, df.home_goalie)
    df["shooter_skaters"] = shooter_sk
    df["def_skaters"] = def_sk
    df["strength_diff"] = shooter_sk - def_sk
    df["score_diff"] = np.where(is_home, df.home_score - df.away_score,
                                df.away_score - df.home_score)
    df["is_home"] = is_home.astype(int)
    # off-wing: shooting from the side opposite the stick hand (puck toward mid-ice / one-timer angle).
    # A handedness×shot-side partition, not a skill term; NaN where handedness is unknown.
    sc = df.primary_player_id.astype(int).map(hand)
    is_l, is_r = (sc == "L").to_numpy(), (sc == "R").to_numpy()
    right_side = y_adj < 0
    off_wing = np.where(is_l, right_side, ~right_side).astype(float)
    off_wing[~(is_l | is_r)] = np.nan
    df["off_wing"] = off_wing

    df["shot_type"] = df.shot_type.fillna("unknown").astype(str)
    df["last_event_type"] = df.prev_type.fillna("none").astype(str)
    df["zone"] = df.zone.fillna("O").astype(str)
    df["goal"] = (df.type == "goal").astype(int)
    df["shooter_id"] = df.primary_player_id.astype(int)

    # exclude empty-net (defending goalie pulled) and penalty-shot / malformed strengths
    df = df[(def_goalie == 1)]
    df = df[(df.shooter_skaters >= 3) & (df.def_skaters >= 3)]
    keep = ["nhl_game_id", "event_idx", "shooter_id", "goal", *FEATURES]   # time_g is in FEATURES
    return df[keep].reset_index(drop=True)


def load_shots(seasons: list[int], hand: dict[int, str]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        p = C.INTERIM / "events" / f"{s}.parquet"
        if p.exists():
            frames.append(build_features(pd.read_parquet(p), hand))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _as_model_frame(df: pd.DataFrame, categories: dict | None = None):
    """Cast feature columns to the dtypes XGBoost expects (categoricals as pandas category)."""
    X = df[FEATURES].copy()
    cats = {}
    for c in CAT_FEATURES:
        col = X[c].astype("category")
        if categories is not None:
            col = col.cat.set_categories(categories[c])
        X[c] = col
        cats[c] = list(X[c].cat.categories)
    return X, cats


# --- fit ---------------------------------------------------------------------
def fit(seasons: list[int]):
    """Returns (out, model, iso, meta, oof_cal) — oof_cal aligned row-for-row with out."""
    import xgboost as xgb

    df = load_shots(seasons, handedness_map())
    if df.empty:
        return df, None, None, {}, None
    X, categories = _as_model_frame(df)
    y = df.goal.to_numpy(int)
    groups = df.nhl_game_id.to_numpy()
    # start boosting at the true goal rate (helps the aggregate calibrate without an isotonic mean shift)
    common = dict(base_score=float(y.mean()), enable_categorical=True)

    # out-of-fold predictions via GroupKFold-by-game (no game leakage)
    oof = np.full(len(df), np.nan)
    best_iters = []
    gkf = GroupKFold(n_splits=N_SPLITS)
    for tr, va in gkf.split(X, y, groups):
        m = xgb.XGBClassifier(**XGB_PARAMS, **common, early_stopping_rounds=EARLY_STOP)
        m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], verbose=False)
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        best_iters.append(int(m.best_iteration) + 1)

    # isotonic recalibration on the out-of-fold probabilities
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(oof, y)
    oof_cal = iso.predict(oof)

    # final booster on all shots (rounds = mean of the per-fold best iteration)
    n_final = max(int(round(np.mean(best_iters))), 1)
    params = {**XGB_PARAMS, **common, "n_estimators": n_final}
    final = xgb.XGBClassifier(**params)
    final.fit(X, y, verbose=False)
    xg = iso.predict(final.predict_proba(X)[:, 1])

    out = df[["nhl_game_id", "event_idx", "time_g", "shooter_id", "goal",
              "distance", "abs_angle", "shot_type", "strength_diff", "rebound", "rush"]].copy()
    out["xg"] = np.round(xg, 5)

    meta = {
        "model": "xg", "seasons": sorted(int(s) for s in seasons), "n_shots": int(len(df)),
        "n_goals": int(y.sum()), "n_games": int(df.nhl_game_id.nunique()),
        "hyperparams": {**XGB_PARAMS, "n_estimators_final": n_final, "early_stopping": EARLY_STOP,
                        "fold_best_iters": best_iters},
        "metrics": _metrics(y, oof_cal),
        "metrics_uncalibrated": _metrics(y, oof),
        "reliability": _reliability(y, oof_cal),
        "importances": _importances(final),
        "categories": categories,
    }
    return out, final, iso, meta, oof_cal


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "auc": round(float(roc_auc_score(y, p)), 4),
        "logloss": round(float(log_loss(y, p)), 5),
        "brier": round(float(brier_score_loss(y, p)), 5),
        "total_xg": round(float(p.sum()), 1),
        "total_goals": int(y.sum()),
        "n": int(len(y)),
    }


RELI_BINS = [0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.7, 1.0]


def _reliability(y: np.ndarray, p: np.ndarray) -> list[dict]:
    idx = np.digitize(p, RELI_BINS[1:-1], right=False)
    rows = []
    for b in range(len(RELI_BINS) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({"p_lo": RELI_BINS[b], "p_hi": RELI_BINS[b + 1],
                     "pred": round(float(p[m].mean()), 4), "obs": round(float(y[m].mean()), 4),
                     "n": int(m.sum())})
    return rows


def _importances(model) -> list[dict]:
    booster = model.get_booster()
    score = booster.get_score(importance_type="gain")   # keyed by feature name
    total = sum(score.values()) or 1.0
    rows = [{"feature": f, "gain": round(v / total, 4)} for f, v in score.items()]
    return sorted(rows, key=lambda r: -r["gain"])


# --- exploration-page payload + heatmap --------------------------------------
def _heatmap(model, iso, categories: dict) -> dict:
    """Predicted xG over an offensive-zone grid for each (shot type x rebound x rush x strength)."""
    xs, ys = np.array(HM_X, float), np.array(HM_Y, float)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")        # shape (len(ys), len(xs))
    flat_x, flat_y = gx.ravel(), gy.ravel()
    dx = GOAL_X - flat_x
    base = {
        "distance": np.sqrt(dx ** 2 + flat_y ** 2),
        "abs_angle": np.degrees(np.arctan2(np.abs(flat_y), dx)),
        "x_adj": flat_x, "abs_y": np.abs(flat_y),
        "since_faceoff": 25.0, "score_diff": 0.0, "period": 1.0, "time_g": 600.0,
        "is_home": 1, "same_team_last": 1, "last_behind_net": 0, "off_wing": np.nan, "zone": "O",
    }
    combos = {}
    for st in HM_SHOT_TYPES:
        for reb in (0, 1):
            for rush in (0, 1):
                for sname, (ssk, dsk) in HM_STRENGTHS.items():
                    n = len(flat_x)
                    row = {**{k: np.full(n, v) if np.isscalar(v) else v for k, v in base.items()}}
                    row["shot_type"] = [st] * n
                    row["rebound"], row["rush"] = reb, rush
                    row["shooter_skaters"], row["def_skaters"] = ssk, dsk
                    row["strength_diff"] = ssk - dsk
                    # pre-shot context implied by the toggles
                    row["time_since_last"] = 2.5 if reb else (1.5 if rush else 18.0)
                    row["time_since_shot"] = 2.5 if reb else 30.0
                    row["attempts_15s"] = 1.0 if reb else 0.0
                    row["dist_from_last"] = 18.0 if reb else (60.0 if rush else 35.0)
                    row["speed_from_last"] = row["dist_from_last"] / row["time_since_last"]
                    row["angle_change"] = np.where(reb, 35.0, 0.0) + np.zeros(n)
                    row["last_dist_to_net"] = 12.0 if reb else 40.0
                    row["royal_road"] = 1 if reb else 0
                    row["last_event_type"] = (["shot-on-goal"] if reb else ["faceoff"]) * n
                    grid = pd.DataFrame(row)
                    X, _ = _as_model_frame(grid.rename_axis(None), categories)
                    p = iso.predict(model.predict_proba(X)[:, 1])
                    cells = np.round(p.reshape(len(ys), len(xs)), 4)
                    combos[f"{st}|{reb}|{rush}|{sname}"] = cells.tolist()
    return {"x": HM_X, "y": HM_Y, "combos": combos,
            "shot_types": HM_SHOT_TYPES, "strengths": list(HM_STRENGTHS)}


def export_web(meta: dict, model, iso, categories: dict) -> None:
    C.WEB_DATA.mkdir(parents=True, exist_ok=True)
    payload = {
        "seasons": meta["seasons"], "n_shots": meta["n_shots"], "n_goals": meta["n_goals"],
        "metrics": meta["metrics"], "reliability": meta["reliability"],
        "importances": meta["importances"][:15],
        "heatmap": _heatmap(model, iso, categories),
    }
    (C.WEB_DATA / "xg_model.json").write_text(json.dumps(payload))
    print(f"    -> web/public/data/xg_model.json")


# --- persistence -------------------------------------------------------------
def _write_meta(meta: dict) -> None:
    C.LOGS_MODEL.mkdir(parents=True, exist_ok=True)
    label = "+".join(map(str, meta["seasons"]))
    slim = {k: v for k, v in meta.items() if k != "categories"}
    (C.LOGS_MODEL / f"xg_{label}.meta.json").write_text(json.dumps(slim, indent=2))
    with (C.LOGS / "model_fits.jsonl").open("a") as f:
        f.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"),
                            "model": "xg", "seasons": meta["seasons"],
                            "metrics": meta["metrics"]}) + "\n")


def _save_artifacts(model, iso) -> None:
    C.MODELS.mkdir(parents=True, exist_ok=True)
    model.get_booster().save_model(str(C.MODELS / "xg_booster.json"))
    (C.MODELS / "xg_isotonic.json").write_text(json.dumps({
        "x": [round(float(v), 6) for v in iso.X_thresholds_],
        "y": [round(float(v), 6) for v in iso.y_thresholds_]}))


# --- reporting ---------------------------------------------------------------
def sniff(out: pd.DataFrame, meta: dict) -> None:
    m = meta["metrics"]
    print(f"    OOF: AUC={m['auc']}  logloss={m['logloss']}  Brier={m['brier']}  "
          f"Σxg={m['total_xg']} vs goals={m['total_goals']}")
    u = meta["metrics_uncalibrated"]
    print(f"    (pre-isotonic Σxg={u['total_xg']}, logloss={u['logloss']})")
    print("    top features (gain): " + ", ".join(
        f"{r['feature']} {r['gain']:.2f}" for r in meta["importances"][:8]))
    print("    reliability (pred -> obs):")
    for r in meta["reliability"]:
        print(f"      {r['p_lo']:.2f}-{r['p_hi']:.2f}: pred {r['pred']:.3f}  obs {r['obs']:.3f}  (n={r['n']:,})")
    hi = out.sort_values("xg", ascending=False).head(3)
    print(f"    highest-xG shots: " + ", ".join(
        f"{r.xg:.2f}@{r.distance:.0f}ft" for r in hi.itertuples()))


def run(seasons: list[int], pool: bool) -> None:
    groups = [seasons] if pool else [[s] for s in seasons]
    for grp in groups:
        label = "+".join(map(str, grp))
        out, model, iso, meta, _ = fit(grp)
        if out.empty:
            print(f"\n[xg {label}] no shots — skipping")
            continue
        print(f"\n=== xg — seasons {label} : {meta['n_shots']:,} shots, {meta['n_goals']:,} goals ===")
        # per-season prediction parquets (keyed by nhl_game_id + event_idx for downstream joins)
        (C.PROCESSED / "xg").mkdir(parents=True, exist_ok=True)
        out = out.assign(season=(out.nhl_game_id // 1_000_000).astype(int))
        for s, sp in out.groupby("season"):
            sp.drop(columns="season").to_parquet(C.PROCESSED / "xg" / f"{s}.parquet", index=False)
        _save_artifacts(model, iso)
        _write_meta(meta)
        export_web(meta, model, iso, meta["categories"])
        sniff(out, meta)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Fit the in-house xG model (raw NHL pbp only)")
    p.add_argument("--season", type=int, default=None, help="one season (default: all available)")
    p.add_argument("--pool", action="store_true", help="pool all available seasons into one fit")
    args = p.parse_args(argv)
    seasons = [args.season] if args.season else available_seasons()
    if not seasons:
        raise SystemExit("no interim events available — run `make clean-data` first")
    run(seasons, pool=args.pool or args.season is None)


if __name__ == "__main__":
    main()
