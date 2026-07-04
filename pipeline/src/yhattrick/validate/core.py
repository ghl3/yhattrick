"""Shared types + helpers for the validation stage: the `Check` record, pass/warn/fail
constructors, small IO helpers, and the console reporter. Stage check functions live in
`checks.py`; the CLI is `__main__.py`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .. import config as C

# seasons whose at-scale checks are allowed to wobble (sparse/gappy source data); APPROX only.
KNOWN_GAP_SEASONS = {2025}

ICONS = {"pass": "✓", "warn": "⚠", "fail": "✗", "skip": "–"}


@dataclass
class Check:
    name: str
    stage: str
    kind: str  # 'EXACT' | 'APPROX'
    status: str  # 'pass' | 'warn' | 'fail' | 'skip'
    observed: object = None
    expected: object = None
    detail: str = ""


def _fmt(v):
    return round(v, 6) if isinstance(v, float) else v


def exact(name, stage, ok, observed=None, expected=None, detail="") -> Check:
    return Check(name, stage, "EXACT", "pass" if ok else "fail", observed, expected, detail)


def approx(name, stage, value, lo, hi, detail="") -> Check:
    """APPROX checks are warn-only (never fail the run); they flag a value outside its band."""
    ok = (lo is None or value >= lo) and (hi is None or value <= hi)
    band = f"[{'' if lo is None else _fmt(lo)}, {'' if hi is None else _fmt(hi)}]"
    return Check(name, stage, "APPROX", "pass" if ok else "warn", _fmt(value), band, detail)


def skip(name, stage, kind, detail) -> Check:
    return Check(name, stage, kind, "skip", detail=detail)


# --- small IO helpers --------------------------------------------------------------------------
def seasons_present(season: int | None) -> list[int]:
    d = C.PROCESSED / "stints"
    have = sorted(int(p.stem) for p in d.glob("*.parquet")) if d.exists() else []
    return [season] if season else have


def pq(path, **kw) -> pd.DataFrame | None:
    return pd.read_parquet(path, **kw) if path.exists() else None


def label(seasons: Iterable[int]) -> str:
    return "+".join(map(str, sorted(seasons)))


def cell_len(x) -> int:
    """len() of a stint personnel cell (list / ndarray / None)."""
    return 0 if x is None else len(x)


def json_clean(text: str) -> bool:
    """The exporters emit strict JSON (NaN/inf -> null). A bare NaN/Infinity token would be a leak."""
    return ("NaN" not in text) and ("Infinity" not in text)


# --- reporter ----------------------------------------------------------------------------------
def report(checks: list[Check]) -> dict:
    stages: list[str] = []
    for c in checks:
        if c.stage not in stages:
            stages.append(c.stage)
    print()
    for stage in stages:
        print(f"  ── {stage} " + "─" * max(2, 40 - len(stage)))
        for c in (x for x in checks if x.stage == stage):
            line = f"    {ICONS[c.status]} [{c.kind:6}] {c.name}"
            if c.status in ("fail", "warn") and c.observed is not None:
                line += f"  (got {c.observed}, want {c.expected})"
            elif c.status == "skip":
                line += f"  ({c.detail})"
            elif c.detail and c.status == "pass":
                line += f"  — {c.detail}"
            print(line)
    tally = {k: sum(c.status == k for c in checks) for k in ICONS}
    exact_fail = sum(c.status == "fail" and c.kind == "EXACT" for c in checks)
    print(
        f"\n  summary: {tally['pass']} pass · {tally['warn']} warn · {tally['fail']} fail · "
        f"{tally['skip']} skip   ({exact_fail} EXACT failures)"
    )
    return {"tally": tally, "exact_fail": exact_fail}
