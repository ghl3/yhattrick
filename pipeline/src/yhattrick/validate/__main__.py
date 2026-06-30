"""CLI for the validation stage.

Report-only by design: prints a grouped pass/warn/fail table and exits 0. Pass `--strict` to exit
non-zero on any EXACT failure (for a future CI gate).

Usage:
  uv run python -m yhattrick.validate                  # all seasons present in processed/
  uv run python -m yhattrick.validate --season 2024
  uv run python -m yhattrick.validate --strict         # non-zero exit if an EXACT invariant fails
"""
from __future__ import annotations

import argparse

from .checks import run_all
from .core import label, report, seasons_present


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Validate inference-stage invariants across exported artifacts")
    p.add_argument("--season", type=int, default=None, help="one season (default: all in processed/)")
    p.add_argument("--strict", action="store_true", help="exit non-zero on any EXACT failure")
    p.add_argument("--games-sample", type=int, default=60, help="per-game JSON files to spot-check")
    args = p.parse_args(argv)
    seasons = seasons_present(args.season)
    if not seasons:
        raise SystemExit("no processed stints — run `make stints` first")
    print(f"=== validate — seasons {label(seasons)} ===")
    checks = run_all(seasons, args.games_sample)
    res = report(checks)
    if args.strict and res["exact_fail"]:
        raise SystemExit(f"{res['exact_fail']} EXACT invariant(s) violated")


if __name__ == "__main__":
    main()
