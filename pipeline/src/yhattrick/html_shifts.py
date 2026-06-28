"""Parse NHL HTML time-on-ice (TOI) reports into shift intervals.

Used as the shift source when the JSON shiftcharts feed is empty (late-2024-25 onward — the legacy
`api.nhle.com/stats/rest` shiftcharts stopped being populated). `TH` = home team, `TV` = away team.
Per player the report has a `playerHeading` cell ("<NUM> <LAST>, <FIRST>") followed by 5 cells per
shift — [Shift#, Period, "elapsed / remaining", "elapsed / remaining", "MM:SS"] (the trailing Event
cell carries an extra `rborder` class and is excluded by the class filter, so groups of 5 are clean).

Sweater number → playerId is resolved from the game's pbp `rosterSpots` (exact per game; handles
trades and number changes). Output matches the JSON shift path: per-game
`(player_id, name, team, team_id, period, start_g, end_g)`, which `clean.clean_shifts` merges,
numbers, and writes identically.
"""
from __future__ import annotations

import json
import re

from bs4 import BeautifulSoup

from . import config as C

# td classes that make up a shift block: the player header + the 5 data columns
_CELL_CLASSES = ["playerHeading + border", "lborder + bborder"]
_PERIOD = {"1": 1, "2": 2, "3": 3, "OT": 4}   # shootout / 2OT+ rows are dropped


def _mmss(s: str) -> int | None:
    m = re.match(r"^(\d+):(\d{2})$", s.strip())
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


def _game_sec(period: int, mmss: str) -> int | None:
    """Period-elapsed MM:SS -> seconds since opening faceoff (matches clean.game_sec)."""
    sec = _mmss(mmss)
    return None if sec is None else (period - 1) * C.PERIOD_SECONDS + sec


def _pbp_name(r: dict) -> str:
    return f"{r.get('firstName', {}).get('default', '')} {r.get('lastName', {}).get('default', '')}".strip()


def parse_report(html: str) -> list[tuple[int, str, int, int, int]]:
    """Return [(sweater_number, header_name, period, start_g, end_g), ...] from one TH/TV report."""
    soup = BeautifulSoup(html, "lxml")
    cells = soup.find_all("td", class_=_CELL_CLASSES)
    rows: list[tuple[int, str, int, int, int]] = []
    num: int | None = None
    name = ""
    buf: list[str] = []

    def flush() -> None:
        if num is None:
            buf.clear()
            return
        for k in range(0, len(buf) - 4, 5):              # [shift#, per, start, end, dur]
            _, per, start, end, dur = (b.strip() for b in buf[k:k + 5])
            period = _PERIOD.get(per)
            if period is None:                            # SO / 2OT+ — skip
                continue
            sg = _game_sec(period, start.split("/")[0])
            eg = _game_sec(period, end.split("/")[0])
            d = _mmss(dur.split("/")[0])
            if sg is None:
                continue
            if eg is None and d is not None:              # missing end time -> start + duration
                eg = sg + d
            if eg is None or eg <= sg:
                continue
            rows.append((num, name, period, sg, eg))
        buf.clear()

    for c in cells:
        text = c.get_text().replace("\xa0", " ").strip()
        if "playerHeading" in (c.get("class") or []) or "playerHeading" in " ".join(c.get("class") or []):
            flush()
            m = re.match(r"^(\d+)\s+(.*)$", text)         # "4 GOSTISBEHERE, SHAYNE"
            num, name = (int(m.group(1)), m.group(2)) if m else (None, "")
        elif num is not None:
            buf.append(text)
    flush()
    return rows


def shifts_for_game(gid: int) -> list[tuple[int, str, str, int, int, int, int]]:
    """(player_id, name, team, team_id, period, start_g, end_g) from the game's TH+TV reports.

    Sweater numbers are resolved against the game's pbp rosterSpots (TH→home, TV→away)."""
    pbp = json.loads((C.RAW_PBP / f"{gid}.json").read_text())
    out: list[tuple[int, str, str, int, int, int, int]] = []
    unresolved: list[tuple[str, int, str]] = []
    for hv, team in (("H", pbp["homeTeam"]), ("V", pbp["awayTeam"])):
        nmap = {r["sweaterNumber"]: (r["playerId"], _pbp_name(r))
                for r in pbp["rosterSpots"] if r["teamId"] == team["id"]}
        path = C.RAW_HTMLSHIFTS / f"T{hv}{C.game6(gid)}.HTM"
        if not path.exists():
            continue
        for num, hdr_name, period, sg, eg in parse_report(path.read_text(encoding="latin-1")):
            hit = nmap.get(num)
            if hit is None:
                unresolved.append((hv, num, hdr_name))
                continue
            pid, pname = hit
            out.append((pid, pname, team["abbrev"], team["id"], period, sg, eg))
    if unresolved:
        print(f"[htmlshifts] {gid}: {len(unresolved)} unresolved sweaters e.g. {unresolved[:4]}")
    return out
