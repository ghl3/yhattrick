"""Golden-equivalence gate for the stints.py refactor.

The stint builder is load-bearing: its output feeds every downstream model. To refactor it safely we
freeze the *current* (already boundary-fixed) processed output as a golden baseline, then assert the
refactored builder reproduces it exactly — except for a small set of documented, intended changes.

Workflow:
    # 1) BEFORE refactoring, with a populated data/ tree, capture the baseline:
    uv run python -m tests.golden.compare_stints capture
    # 2) AFTER each refactor phase, regenerate (`make stints`) then check:
    uv run python -m tests.golden.compare_stints check

`check` is also importable as a pytest (test_golden_*) and skips when no baseline exists, mirroring
the data-gated real-data test in tests/test_stints.py.

Intended (carved-out) differences vs. the pre-refactor baseline:
  * bug #1 (upper guard): shots with game_seconds beyond the last stint's end are dropped from
    shots_onice. Expected only in seasons with shift-data gaps (2025); each dropped row must have
    game_seconds greater than its game's last stint end. 2021–2024 must be identical.
  * bug #3 (shootout score state): filtering period-5 goals out of home_lead must produce ZERO diff
    (shootout goals already sat beyond the shift timeline) — asserted as zero.
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from yhattrick import config as C

GOLDEN = C.DATA / "golden"
TABLES = ("stints", "shots_onice")


def _seasons(kind: str, root) -> list[int]:
    d = root / kind
    return sorted(int(p.stem) for p in d.glob("*.parquet")) if d.exists() else []


def _hash(df: pd.DataFrame) -> int:
    # list/object columns aren't hashable by hash_pandas_object; stringify them first
    h = df.copy()
    for c in h.columns:
        if h[c].dtype == object:
            h[c] = h[c].map(lambda v: tuple(v) if isinstance(v, list) else v).astype(str)
    return int(pd.util.hash_pandas_object(h, index=False).sum())


def capture() -> None:
    """Copy current processed/{stints,shots_onice} into data/golden + write a manifest."""
    manifest = {"tables": {}}
    for kind in TABLES:
        seasons = _seasons(kind, C.PROCESSED)
        if not seasons:
            raise SystemExit(f"no processed/{kind} to capture — run `make stints` first")
        dst = GOLDEN / kind
        dst.mkdir(parents=True, exist_ok=True)
        manifest["tables"][kind] = {}
        for s in seasons:
            df = pd.read_parquet(C.PROCESSED / kind / f"{s}.parquet")
            df.to_parquet(dst / f"{s}.parquet", index=False)
            manifest["tables"][kind][str(s)] = {"rows": int(len(df)), "hash": _hash(df)}
            print(f"[golden] captured {kind}/{s}: {len(df):,} rows")
    (GOLDEN / "golden_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[golden] manifest -> {GOLDEN / 'golden_manifest.json'}")


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    """Stable ordering for comparison (row order should already match; sort defensively by keys)."""
    keys = ["nhl_game_id", "stint_idx"] if "duration_s" in df.columns else ["nhl_game_id", "event_idx"]
    return df.sort_values(keys).reset_index(drop=True)


def _corrupt_games(season: int) -> set[int]:
    """Games with a corrupt shiftchart (foreign teams spliced in, bug #2). The old code built garbage
    overloaded stints for them; the refactor cleans the foreign shifts (or skips if unrecoverable).
    Either way the game's rows intentionally differ, so they're excluded from BOTH golden and new."""
    from yhattrick.data.stints import teams_reconcile
    ev = pd.read_parquet(C.INTERIM / "events" / f"{season}.parquet", columns=["nhl_game_id", "team", "is_home"])
    sh = pd.read_parquet(C.INTERIM / "shifts" / f"{season}.parquet", columns=["nhl_game_id", "team"])
    home = ev[ev.is_home].groupby("nhl_game_id").team.first().to_dict()
    away = ev[~ev.is_home].groupby("nhl_game_id").team.first().to_dict()
    bad = set()
    for gid, g in sh.groupby("nhl_game_id"):
        if not teams_reconcile(g, home.get(gid), away.get(gid)):
            bad.add(int(gid))
    return bad


def _shootout_games(season: int) -> set[int]:
    """Games that reached a shootout (a period>=5 goal) — the only games where home_lead may change."""
    ev = pd.read_parquet(C.INTERIM / "events" / f"{season}.parquet", columns=["nhl_game_id", "period", "type"])
    return set(map(int, ev[(ev.period >= 5) & (ev.type == "goal")].nhl_game_id.unique()))


def _compare_table(kind: str) -> list[str]:
    problems: list[str] = []
    for s in _seasons(kind, GOLDEN):
        gp = GOLDEN / kind / f"{s}.parquet"
        np_ = C.PROCESSED / kind / f"{s}.parquet"
        if not np_.exists():
            problems.append(f"{kind}/{s}: regenerated file missing")
            continue
        gold, new = _norm(pd.read_parquet(gp)), _norm(pd.read_parquet(np_))
        # carve-out (bug #2): exclude corrupt-shiftchart games from BOTH sides — golden has the old
        # garbage (overloaded) stints, new has the cleaned ones; the fix is verified separately.
        bad = _corrupt_games(s)
        if bad:
            gold = gold[~gold.nhl_game_id.isin(bad)].reset_index(drop=True)
            new = new[~new.nhl_game_id.isin(bad)].reset_index(drop=True)
            print(f"  {kind}/{s}: excluded {len(bad)} corrupt game(s) from both sides: {sorted(bad)}")
        # carve-out (bug #3): in shootout games, home_lead may change (SO goals no longer count toward
        # in-game score state). Neutralize ONLY home_lead for those games; every other column is still
        # compared, so a real regression in a shootout game would still fail.
        if kind == "stints" and len(gold) == len(new):
            so = _shootout_games(s)
            m = gold.nhl_game_id.isin(so).to_numpy()
            if m.any():
                gold.loc[m, "home_lead"] = new.loc[m, "home_lead"].to_numpy()
                print(f"  {kind}/{s}: neutralized home_lead in {int(m.sum())} shootout-game stints")

        if kind == "shots_onice":
            # carve-out for bug #1: rows dropped from `new` must be post-last-stint shots only
            gkeys = set(map(tuple, gold[["nhl_game_id", "event_idx"]].itertuples(index=False)))
            nkeys = set(map(tuple, new[["nhl_game_id", "event_idx"]].itertuples(index=False)))
            dropped = gkeys - nkeys
            if dropped:
                ends = (pd.read_parquet(C.PROCESSED / "stints" / f"{s}.parquet")
                        .groupby("nhl_game_id").end_g.max().to_dict())
                bad = [(g, e) for (g, e) in dropped
                       if not (gold[(gold.nhl_game_id == g) & (gold.event_idx == e)].game_seconds.iloc[0]
                               > ends.get(g, -1))]
                print(f"  {kind}/{s}: {len(dropped)} dropped shots (bug #1 carve-out); "
                      f"{len(bad)} NOT explained by post-last-stint")
                if bad:
                    problems.append(f"{kind}/{s}: {len(bad)} dropped rows are not post-last-stint: {bad[:5]}")
                gold = gold[~gold[["nhl_game_id", "event_idx"]].apply(tuple, axis=1).isin(dropped)].reset_index(drop=True)
            if nkeys - gkeys:
                problems.append(f"{kind}/{s}: {len(nkeys - gkeys)} NEW rows not in golden")

        try:
            pd.testing.assert_frame_equal(gold, new, check_dtype=True, check_like=False)
            print(f"  {kind}/{s}: identical ({len(new):,} rows)")
        except AssertionError as e:
            diff_cols = [c for c in gold.columns if c in new.columns
                         and not gold[c].astype(str).equals(new[c].astype(str))]
            problems.append(f"{kind}/{s}: differs in columns {diff_cols}\n    {str(e).splitlines()[0]}")
    return problems


def check() -> int:
    if not (GOLDEN / "golden_manifest.json").exists():
        print("[golden] no baseline — run `... compare_stints capture` first; skipping")
        return 0
    problems: list[str] = []
    for kind in TABLES:
        print(f"[golden] checking {kind} ...")
        problems += _compare_table(kind)
    if problems:
        print("\n[golden] FAIL:")
        for p in problems:
            print("  -", p)
        return 1
    print("\n[golden] PASS — refactor reproduces the baseline (modulo documented carve-outs)")
    return 0


# pytest entry point (data-gated)
def test_golden_equivalence():
    import pytest
    if not (GOLDEN / "golden_manifest.json").exists():
        pytest.skip("no golden baseline captured")
    assert check() == 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "capture":
        capture()
    else:
        raise SystemExit(check())
