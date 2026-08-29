"""One-shot in-season updater: pull newly-finished games, rebuild the season's tables, refresh
the models, re-export the site JSON, validate, publish to R2, and deploy the site.

Stateless and idempotent: every decision is derived from the NHL schedule and what is on disk,
so an interrupted run is simply re-run. A file lock guards against overlapping runs (safe to
invoke from cron). When the schedule shows no newly-finished games and the interim tables are
current, the run stops after the fetch check ("up to date") without touching models or the site.

Daily xG uses the FROZEN booster from the last full fit (score-only), so per-game JSON for
already-exported games stays byte-identical — the R2 publish then uploads only genuinely new
games and the immutable edge cache stays honest. Refit deliberately (`--refit-xg`, or `make xg`)
rather than daily.

The generative model (JAX, experimental dep group) is out of scope here: run
`make generative-model` + `make generative-cards` deliberately; exports keep merging the latest
gen_cards.json either way.

Usage:
  uv run python -m yhattrick.update                  # current season, full update
  uv run python -m yhattrick.update --dry-run        # what would run (read-only schedule check)
  uv run python -m yhattrick.update --force          # full chain even with no new games
  uv run python -m yhattrick.update --fetch-only     # stop after the raw fetch
  uv run python -m yhattrick.update --refit-xg       # full xG refit instead of frozen scoring
  uv run python -m yhattrick.update --no-publish --no-deploy   # local-only rebuild
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from . import config as C
from .data import download

DEPLOY_TIMEOUT = 900  # seconds for `vercel --prod`


# --- environment -------------------------------------------------------------
def load_env_file(path: Path) -> int:
    """Export KEY=VALUE lines from a dotenv file (R2 credentials live in pipeline/.env).
    Existing environment variables win. Returns how many variables were set."""
    import os

    if not path.exists():
        return 0
    n = 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip("'\"")
        if k and k not in os.environ:
            os.environ[k] = v
            n += 1
    return n


# --- planning ----------------------------------------------------------------
def resolve_season(arg: int | None, today: dt.date | None = None) -> int:
    season = arg if arg is not None else C.current_season(today)
    if season not in C.SEASONS:
        raise SystemExit(
            f"season {season} is not in config.SEASONS {C.SEASONS} — add it there first "
            f"(see docs/updating.md for the season-start checklist)"
        )
    return season


def fetch_plan(season: int, schedule: list[dict]) -> dict:
    """What the fetch stage would do, from the schedule + local files only (no downloads).

    {"states": {state: count}, "missing": [ids], "stale": [ids], "pending": count} where
    `missing` are finished games with a pbp or shiftchart file absent from disk and `stale`
    are finished games whose cached pbp has no plays (fetched before the game was played)."""
    states: dict[str, int] = {}
    missing, stale = [], []
    for g in schedule:
        states[g["state"]] = states.get(g["state"], 0) + 1
        if g["state"] not in C.FINAL_GAME_STATES:
            continue
        p = C.RAW_PBP / f"{g['id']}.json"
        if not p.exists() or not (C.RAW_SHIFTS / f"{g['id']}.json").exists():
            missing.append(g["id"])
        elif download._pbp_file_stale(p):
            stale.append(g["id"])
    pending = sum(n for s, n in states.items() if s not in C.FINAL_GAME_STATES)
    return {"states": states, "missing": missing, "stale": stale, "pending": pending}


def interim_stale(season: int) -> bool:
    """True when any of the season's interim tables is missing or older than any raw input
    (pbp, shiftcharts, or HTML TOI reports) — i.e. a previous run fetched but died mid-clean,
    so the processing chain must run even though this run fetched nothing new. All four tables
    are checked because clean writes them in sequence (shifts, events, roster, shots) and a
    crash can leave the early ones fresh."""
    outs = [C.INTERIM / t / f"{season}.parquet" for t in ("shifts", "events", "roster", "shots")]
    raws = [p for d in (C.RAW_PBP, C.RAW_SHIFTS) for p in d.glob(f"{season}*.json")]
    # HTML TOI reports are named T{H,V}<game6>.HTM (no season prefix); scan them all — an
    # over-trigger just causes one extra clean of the season, which is idempotent.
    raws += list(C.RAW_HTMLSHIFTS.glob("*.HTM"))
    if not raws:
        return False
    if not all(p.exists() for p in outs):
        return True
    newest = max(p.stat().st_mtime for p in raws)
    return newest > min(p.stat().st_mtime for p in outs)


# --- stages ------------------------------------------------------------------
def build_stages(season: int, *, refit_xg: bool, publish: bool, deploy: bool) -> list[tuple]:
    """The processing chain, in dependency order — one (name, thunk) per stage. Mirrors
    `make all` scoped to one season; models refit on the pooled window; validate gates the
    outward-facing publish/deploy steps."""
    s = str(season)

    def _clean():
        from .data import clean

        clean.main(["--season", s])

    def _handedness():
        download.download_handedness()

    def _xg():
        from .models import expected_goal_model

        expected_goal_model.main(["--pool"] if refit_xg else ["--score-only", "--season", s])

    def _stints():
        from .data import stints

        stints.main(["--season", s])

    def _box():
        from .data import aggregates

        aggregates.main(["--season", s])

    def _model():
        from .models import player_onice_model

        player_onice_model.main(["--pool"])

    def _shooting():
        from .models import shooting_model

        shooting_model.main(["--pool"])

    def _goal_accounting():
        from .models import goal_accounting

        goal_accounting.main([])

    def _games():
        from .export import export_games

        export_games.main(["--season", s])

    def _gamelog():
        from .data import gamelog

        gamelog.main(["--season", s])

    def _players():
        from .export import export_players

        export_players.main()

    def _goalie_gamelog():
        from .data import goalie_gamelog

        goalie_gamelog.main(["--season", s])

    def _goalie_box():
        from .data import goalie_aggregates

        goalie_aggregates.main(["--season", s])

    def _goalies():
        from .export import export_goalies

        export_goalies.main()

    def _validate():
        from .validate.__main__ import main as validate_main

        validate_main(["--strict"])

    def _publish():
        from .export import publish as publish_mod

        load_env_file(C.PIPELINE_DIR / ".env")
        publish_mod.main([])

    stages = [
        ("clean", _clean),
        ("handedness", _handedness),
        ("xg-refit" if refit_xg else "xg-score", _xg),
        ("stints", _stints),
        ("box", _box),
        ("model", _model),
        ("shooting", _shooting),
        ("goal-accounting", _goal_accounting),
        ("games", _games),
        ("gamelog", _gamelog),
        ("players", _players),
        ("goalie-gamelog", _goalie_gamelog),
        ("goalie-box", _goalie_box),
        ("goalies", _goalies),
        ("validate", _validate),
    ]
    if publish:
        stages.append(("publish", _publish))
    if deploy:
        stages.append(("deploy", deploy_web))
    return stages


def deploy_web() -> None:
    """Production deploy of the site via the Vercel CLI (ships the refreshed public/data JSON;
    the heavy per-game files are excluded by .vercelignore and served from R2). The team scope
    comes from the linked project — without it the CLI answers "Not authorized" for a
    team-owned project even when logged in."""
    cmd = ["npx", "--yes", "vercel", "deploy", "--prod", "--yes"]
    link = C.WEB_DIR / ".vercel" / "project.json"
    if link.exists():
        org = json.loads(link.read_text()).get("orgId")
        if org:
            cmd += ["--scope", org]
    r = subprocess.run(cmd, cwd=C.WEB_DIR, capture_output=True, text=True, timeout=DEPLOY_TIMEOUT)
    tail = "\n".join((r.stdout + "\n" + r.stderr).strip().splitlines()[-6:])
    print(tail)
    if r.returncode != 0:
        raise RuntimeError(f"vercel deploy exited {r.returncode}")


def run_stages(stages: list[tuple]) -> list[dict]:
    """Run stages in order, timing each; the first failure stops the chain."""
    results = []
    for name, fn in stages:
        print(f"\n── {name} " + "─" * max(0, 60 - len(name)))
        t0 = time.time()
        try:
            fn()
        except SystemExit as e:  # stage CLIs signal fatal problems via SystemExit
            if e.code in (None, 0):
                results.append({"stage": name, "secs": round(time.time() - t0, 1), "ok": True})
                continue
            results.append({"stage": name, "secs": round(time.time() - t0, 1), "ok": False})
            _summarize(results)
            raise SystemExit(f"[update] FAILED at {name}: {e}")
        except Exception as e:
            results.append({"stage": name, "secs": round(time.time() - t0, 1), "ok": False})
            _summarize(results)
            raise SystemExit(f"[update] FAILED at {name}: {e}")
        results.append({"stage": name, "secs": round(time.time() - t0, 1), "ok": True})
    return results


def _summarize(results: list[dict]) -> None:
    if not results:
        return
    print("\n[update] stage summary:")
    for r in results:
        mark = "✓" if r["ok"] else "✗"
        print(f"    {mark} {r['stage']:<16} {r['secs']:>7.1f}s")
    print(f"    total {sum(r['secs'] for r in results):.1f}s")


def _append_run_log(record: dict) -> None:
    C.LOGS.mkdir(parents=True, exist_ok=True)
    with (C.LOGS / "update_runs.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")


# --- locking -----------------------------------------------------------------
@contextmanager
def update_lock():
    C.LOGS.mkdir(parents=True, exist_ok=True)
    f = (C.LOGS / "update.lock").open("w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        raise SystemExit("[update] another update is already running (logs/update.lock)")
    try:
        yield
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


# --- entry point -------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="One-shot in-season data/model/site update")
    p.add_argument(
        "--season", type=int, default=None, help="season start year (default: the current one)"
    )
    p.add_argument(
        "--dry-run", action="store_true", help="report the plan; only reads the NHL schedule"
    )
    p.add_argument("--force", action="store_true", help="run the full chain even with no new games")
    p.add_argument("--fetch-only", action="store_true", help="stop after the raw fetch")
    p.add_argument(
        "--refit-xg",
        action="store_true",
        help="full pooled xG refit instead of frozen-booster scoring (re-touches every game)",
    )
    p.add_argument("--no-publish", action="store_true", help="skip the R2 upload")
    p.add_argument("--no-deploy", action="store_true", help="skip the Vercel deploy")
    args = p.parse_args(argv)

    # progress must reach the teed log as it happens, not when the block buffer fills
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    season = resolve_season(args.season)
    started = dt.datetime.now().isoformat(timespec="seconds")
    print(f"[update] season {C.season_label(season)} · {started}")

    with update_lock():
        schedule = download.schedule_for_season(season)
        plan = fetch_plan(season, schedule)
        states = ", ".join(f"{s or '?'}:{n}" for s, n in sorted(plan["states"].items()))
        print(
            f"[update] schedule: {len(schedule)} games ({states}) · "
            f"{len(plan['missing'])} finished games to fetch · {len(plan['stale'])} to repair"
        )

        if args.dry_run:
            for label, ids in (("fetch", plan["missing"]), ("repair", plan["stale"])):
                if ids:
                    shown = ", ".join(map(str, ids[:10])) + (" …" if len(ids) > 10 else "")
                    print(f"    would {label}: {shown}")
            will_process = bool(plan["missing"] or plan["stale"]) or interim_stale(season)
            print(
                "[update] dry run — would "
                + (
                    "run the full processing chain"
                    if will_process or args.force
                    else "be up to date"
                )
            )
            return

        gsum = download.download_games(season, schedule=schedule)
        hsum = download.download_html_shifts(season)
        new_raw = gsum["fetched"] + gsum["repaired"] + hsum["fetched"]

        if args.fetch_only:
            print(f"[update] fetch-only: {new_raw} new raw files")
            return
        if new_raw == 0 and not interim_stale(season) and not args.force:
            print("[update] up to date — no newly-finished games, nothing to rebuild")
            _append_run_log({"ts": started, "season": season, "new_raw": 0, "up_to_date": True})
            return

        stages = build_stages(
            season,
            refit_xg=args.refit_xg,
            publish=not args.no_publish,
            deploy=not args.no_deploy,
        )
        print(f"[update] {new_raw} new raw files -> running {len(stages)} stages")
        results = run_stages(stages)
        _summarize(results)
        _append_run_log(
            {
                "ts": started,
                "season": season,
                "new_raw": new_raw,
                "pending": gsum.get("pending", 0),
                "stages": results,
                "ok": True,
            }
        )
        print(f"[update] done · {new_raw} new raw files · {gsum.get('pending', 0)} games pending")


if __name__ == "__main__":
    main()
