"""Per-player shot/goal location grids for the player-page heatmap.

For each player, bins their own unblocked shots over the offensive half (oriented so every shot
attacks the net at +89), accumulating three smoothed grids: shot count, goal count, and summed xG.
The UI derives all four views from these — Shots, Goals, Shot% (goals/shots) and average xG
(xG/shot) — so the data is mode-agnostic.

Returned (embedded into player detail by export_players) per player:
  {"x": [cell-center x...], "y": [cell-center y...],
   "s": [[shots]...], "g": [[goals]...], "xg": [[xg-sum]...]}   (each grid is [yi][xi])
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C

# offensive half: from just inside the blue line to the end boards; full ice width
XLO, XHI, YLO, YHI = 25.0, 100.0, -42.5, 42.5
BIN = 3.0  # ft per cell
MIN_SHOTS = 25  # players with fewer shots get no map
SMOOTH_SIGMA = 1.1  # gaussian smoothing, in cells

_XEDGES = np.arange(XLO, XHI + BIN, BIN)
_YEDGES = np.arange(YLO, YHI + BIN, BIN)
_XC = (_XEDGES[:-1] + _XEDGES[1:]) / 2
_YC = (_YEDGES[:-1] + _YEDGES[1:]) / 2
NX, NY = len(_XC), len(_YC)


def _kernel(sigma: float) -> np.ndarray:
    r = max(1, int(round(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    return k / k.sum()


def _smooth(a: np.ndarray, sigma: float = SMOOTH_SIGMA) -> np.ndarray:
    """Separable gaussian blur over a (NY, NX) grid (mode='same')."""
    k = _kernel(sigma)
    a = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, a)
    a = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, a)
    return a


def build(seasons) -> dict[int, dict]:
    frames = []
    for s in seasons:
        p = C.PROCESSED / "shots_onice" / f"{s}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p, columns=["shooter_id", "x", "y", "goal", "xg"]))
    if not frames:
        return {}
    df = pd.concat(frames, ignore_index=True).dropna(subset=["x", "y", "shooter_id"])

    # orient: fold both ends together so every shot attacks the net at +89 (matches how `distance`
    # was computed — sign(x) is the attacking direction for shot events).
    sign = np.where(df.x.values >= 0, 1.0, -1.0)
    xo = df.x.values * sign
    yo = df.y.values * sign
    ix = np.floor((xo - XLO) / BIN).astype(int)
    iy = np.floor((yo - YLO) / BIN).astype(int)
    keep = (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NY)

    d = pd.DataFrame(
        {
            "pid": df.shooter_id.values[keep].astype(np.int64),
            "cell": (iy[keep] * NX + ix[keep]),
            "goal": df.goal.values[keep].astype(float),
            "xg": np.nan_to_num(df.xg.values[keep].astype(float)),
        }
    )
    ncells = NX * NY
    out: dict[int, dict] = {}
    for pid, grp in d.groupby("pid"):
        cells = grp.cell.values
        if len(cells) < MIN_SHOTS:
            continue
        S = np.bincount(cells, minlength=ncells).astype(float)
        G = np.bincount(cells, weights=grp.goal.values, minlength=ncells)
        X = np.bincount(cells, weights=grp.xg.values, minlength=ncells)
        S, G, X = (_smooth(a.reshape(NY, NX)) for a in (S, G, X))
        out[int(pid)] = {
            "x": [round(float(v), 1) for v in _XC],
            "y": [round(float(v), 1) for v in _YC],
            "s": [[round(float(v), 3) for v in row] for row in S],
            "g": [[round(float(v), 3) for v in row] for row in G],
            "xg": [[round(float(v), 4) for v in row] for row in X],
        }
    return out
