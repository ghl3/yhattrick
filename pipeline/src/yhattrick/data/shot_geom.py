"""Shared shot geometry + pre-shot context, derived from NHL play-by-play.

Both the shot table (clean.py) and the xG features (xg.py) orient coordinates and flag
rebound/rush the same way; keeping that here stops the two from drifting apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GOAL_X = 89.0  # goal line distance from center (NHL coords)
FENWICK = [
    "shot-on-goal",
    "missed-shot",
    "goal",
]  # unblocked shot events (carry a shooter + can score)
REBOUND_S = 3.0  # a shot within this many seconds of a prior shot is a rebound
RUSH_S = 4.0  # a shot this soon after a neutral/defensive-zone event is a rush


def attack_sign(is_home, home_defending_side) -> np.ndarray:
    """+1 when the attacking net is already at +89 in raw coords, else -1.

    The home team defends `home_defending_side`, so it attacks the other end."""
    home_attack = np.where(np.asarray(home_defending_side) == "right", -1.0, 1.0)
    return np.where(np.asarray(is_home, dtype=bool), home_attack, -home_attack)


def oriented(x, y, sign) -> tuple[np.ndarray, np.ndarray]:
    """Raw (x, y) -> coordinates with the attacking net at +89 (x_adj, y_adj)."""
    return np.asarray(x, float) * sign, np.asarray(y, float) * sign


def distance_angle(x_adj, y_adj) -> tuple[np.ndarray, np.ndarray]:
    """Distance to the net and absolute shot angle (degrees) from oriented coordinates."""
    dx = GOAL_X - x_adj
    return np.sqrt(dx**2 + y_adj**2), np.degrees(np.arctan2(np.abs(y_adj), dx))


def rebound_flag(prev_type, time_since_last) -> np.ndarray:
    """1 if the previous event was a shot within REBOUND_S seconds."""
    prev_shot = np.isin(np.asarray(prev_type, dtype=object), FENWICK)
    return (prev_shot & (np.asarray(time_since_last, float) <= REBOUND_S)).astype(int)


def rush_flag(prev_zone, time_since_last) -> np.ndarray:
    """1 if the previous event was in the neutral/defensive zone within RUSH_S seconds."""
    nz = np.isin(np.asarray(prev_zone, dtype=object), ["N", "D"])
    return (nz & (np.asarray(time_since_last, float) <= RUSH_S)).astype(int)


def decode_situation(code: pd.Series) -> pd.DataFrame:
    """NHL situationCode 'AGAS HSHG' -> away goalie/skaters, home skaters/goalie (4 digits)."""
    s = code.astype("string").str.zfill(4)
    ok = s.str.len() == 4
    d = pd.DataFrame(index=code.index)
    for i, name in enumerate(["away_goalie", "away_skaters", "home_skaters", "home_goalie"]):
        d[name] = pd.to_numeric(s.str[i], errors="coerce").where(ok)
    return d
