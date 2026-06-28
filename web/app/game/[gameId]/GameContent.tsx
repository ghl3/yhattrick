"use client";
import { createContext, useContext, useDeferredValue, useMemo, useState, type ReactNode } from "react";
import Link from "next/link";
import useSWRImmutable from "swr/immutable";
import type { Game, PlayerAgg, Stint, TimelineEvent } from "@/lib/types";
import { mmss } from "@/lib/format";
import { gameDetailUrl } from "@/lib/data";
import EventCard from "@/components/EventCard";
import GameSkeleton, { TimelineSkeleton } from "./GameSkeleton";

// A finished game's timeline never changes, so use the immutable variant — SWR caches it for the
// session and dedups, so repeat visits are instant with no refetch. The standard data/error/loading
// API (not suspense) SSRs cleanly to the skeleton and hydrates without conflict — and replaces the
// useEffect/useState data-fetch plumbing with one declarative hook.
const fetchGame = (url: string): Promise<Game> =>
  fetch(url).then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))));

const PERIOD_NAMES: Record<number, string> = { 1: "1st", 2: "2nd", 3: "3rd", 4: "OT", 5: "SO" };

// name -> player id for the players in this game (skaters with a player page; used by events)
const LinkMap = createContext<Map<string, number>>(new Map());
// player id -> player aggregate, to resolve normalized stint on-ice ids to names/positions
const ByIdMap = createContext<Map<number, PlayerAgg>>(new Map());

// a player name that links to the player page when we have one (skaters only)
function PName({ name }: { name?: string | null }) {
  const map = useContext(LinkMap);
  if (!name) return null;
  const id = map.get(name);
  // prefetch={false}: a game has thousands of player-name links; letting Next prefetch the RSC for
  // each would flood the network and the dynamic /player route. Navigation on click is still fast.
  return id ? (
    <Link href={`/player/${id}`} prefetch={false} onClick={(e) => e.stopPropagation()}>
      {name}
    </Link>
  ) : (
    <span>{name}</span>
  );
}

// a player id that links to the player page (skaters only); resolves name via the game roster
function PNameById({ id }: { id: number }) {
  const byId = useContext(ByIdMap);
  const p = byId.get(id);
  const name = p?.name ?? `#${id}`;
  return p && p.pos !== "G" ? <PName name={name} /> : <span>{name}</span>;
}

function PlayerList({ ids }: { ids: number[] }) {
  return (
    <>
      {ids.map((id, i) => (
        <span key={id}>
          {i > 0 ? ", " : ""}
          <PNameById id={id} />
        </span>
      ))}
    </>
  );
}

function Collapsible({ title, children, open: initial = true }: { title: ReactNode; children: ReactNode; open?: boolean }) {
  const [open, setOpen] = useState(initial);
  return (
    <div className="panel">
      <div className={`collapse-head ${open ? "open" : ""}`} onClick={() => setOpen((o) => !o)}>
        <span className="chev">▶</span>
        {title}
      </div>
      {open && <div style={{ marginTop: 12 }}>{children}</div>}
    </div>
  );
}

function isSpecial(strength: string) {
  const [a, b] = strength.split("v").map(Number);
  return a !== b || a > 5;
}

function EventRow({ e }: { e: TimelineEvent }) {
  const [open, setOpen] = useState(false);
  const cls =
    e.type === "goal" ? "ev-goal" : e.type === "penalty" ? "ev-penalty" : e.type.includes("shot") ? "ev-shot" : "";
  const isShot = e.xg != null;
  const located = e.x != null && e.y != null;
  return (
    <li className={located ? "clickable" : ""}>
      <div className="ev-line" onClick={located ? () => setOpen((o) => !o) : undefined}>
        <span className="et">{e.clock.split(" ")[1]}</span>
        <span className={`ev-type ${cls}`}>{e.type.replace(/-/g, " ")}</span>
        <span className="muted">{e.team ?? ""}</span>
        <span><PName name={e.player} /></span>
        {e.detail && <span className="muted">({e.detail})</span>}
        {isShot && (
          <span className="ev-xg">
            {e.shot_type ? `${e.shot_type} ` : ""}
            {e.distance != null ? `${e.distance}ft · ` : ""}xG {e.xg!.toFixed(3)}
          </span>
        )}
        {located && <span className="ev-expand">{open ? "▾" : "▸"}</span>}
      </div>
      {open && <EventCard e={e} />}
    </li>
  );
}


// player ids present in `cur` but not `prev` (came on), and vice versa (went off)
function diff(prev: number[], cur: number[]) {
  const prevSet = new Set(prev);
  const curSet = new Set(cur);
  return {
    on: cur.filter((id) => !prevSet.has(id)),
    off: prev.filter((id) => !curSet.has(id)),
  };
}

function ChangeGroup({ team, on, off }: { team: string; on: number[]; off: number[] }) {
  if (!on.length && !off.length) return null;
  return (
    <span className="grp">
      {team}{" "}
      {on.map((id) => (
        <span key={"on" + id} className="on">
          +<PNameById id={id} />{" "}
        </span>
      ))}
      {off.map((id) => (
        <span key={"off" + id} className="off">
          −<PNameById id={id} />{" "}
        </span>
      ))}
    </span>
  );
}

function StintBlock({ s, prev, home, away }: { s: Stint; prev?: Stint; home: string; away: string }) {
  const [open, setOpen] = useState(s.events.length > 0);
  const byId = useContext(ByIdMap);
  const goalie = (id: number | null) => (id != null ? byId.get(id)?.name ?? `#${id}` : null);
  const hd = prev ? diff(prev.home_skaters, s.home_skaters) : null;
  const ad = prev ? diff(prev.away_skaters, s.away_skaters) : null;
  const hasChange = (hd && (hd.on.length || hd.off.length)) || (ad && (ad.on.length || ad.off.length));
  return (
    <div className={`stint ${s.overload ? "overload" : ""}`}>
      <div className="stint-head" onClick={() => setOpen((o) => !o)}>
        <span className="clk">
          {s.clock_start.split(" ")[1]}–{s.clock_end.split(" ")[1]}
        </span>
        <span className={`strength ${isSpecial(s.strength) ? "special" : ""}`}>{s.strength}</span>
        <span className="dur">{s.duration_s}s</span>
        {s.overload && <span className="pill bad">overload</span>}
        {s.events.length > 0 && <span className="muted">· {s.events.length} ev</span>}
        <span className="xgf">
          xGF <span className="h">{s.home_xgf.toFixed(2)}</span> / <span className="a">{s.away_xgf.toFixed(2)}</span>
        </span>
      </div>
      {prev && (
        <div className="changes">
          {hasChange ? (
            <>
              <ChangeGroup team={home} on={hd!.on} off={hd!.off} />
              <ChangeGroup team={away} on={ad!.on} off={ad!.off} />
            </>
          ) : (
            <span className="start">no skater change (goalie/clock)</span>
          )}
        </div>
      )}
      {open && (
        <div className="stint-body">
          <div className="lines">
            <div className="line home">
              <div className="lab">Home skaters {s.home_goalie != null ? "" : "· goalie pulled"}</div>
              <div className="names">
                <PlayerList ids={s.home_skaters} />
                {s.home_goalie != null && <span className="g"> · G {goalie(s.home_goalie)}</span>}
              </div>
            </div>
            <div className="line away">
              <div className="lab">Away skaters {s.away_goalie != null ? "" : "· goalie pulled"}</div>
              <div className="names">
                <PlayerList ids={s.away_skaters} />
                {s.away_goalie != null && <span className="g"> · G {goalie(s.away_goalie)}</span>}
              </div>
            </div>
          </div>
          {s.events.length > 0 && (
            <ul className="events">
              {s.events.map((e, i) => (
                <EventRow key={i} e={e} />
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}

function PeriodSection({ period, stints, home, away }: { period: number; stints: Stint[]; home: string; away: string }) {
  const [open, setOpen] = useState(true);
  const goals = stints.flatMap((s) => s.events).filter((e) => e.type === "goal");
  const hg = goals.filter((g) => g.team === home).length;
  const ag = goals.filter((g) => g.team === away).length;
  return (
    <div className="period-section">
      <div className="period-head" onClick={() => setOpen((o) => !o)}>
        <span>Period {PERIOD_NAMES[period] ?? period}</span>
        <span className="pscore">
          {away} {ag} – {hg} {home}
        </span>
        <span style={{ marginLeft: "auto", fontWeight: 400, opacity: 0.85 }}>
          {stints.length} stints
        </span>
      </div>
      {open && (
        <div className="timeline">
          {stints.map((s, i) => (
            <StintBlock key={s.idx} s={s} prev={stints[i - 1]} home={home} away={away} />
          ))}
        </div>
      )}
    </div>
  );
}

type PlayerSortKey = "name" | "pos" | "shifts" | "toi_s" | "g" | "a1" | "a2" | "pts" | "shots" | "xg";
const PLAYER_COLS: { key: PlayerSortKey; label: string; title?: string }[] = [
  { key: "name", label: "Player" },
  { key: "pos", label: "Pos" },
  { key: "shifts", label: "Shifts" },
  { key: "toi_s", label: "TOI" },
  { key: "g", label: "G", title: "Goals" },
  { key: "a1", label: "A1", title: "Primary assists" },
  { key: "a2", label: "A2", title: "Secondary assists" },
  { key: "pts", label: "P", title: "Points (G + A1 + A2)" },
  { key: "shots", label: "Sh", title: "Shot attempts" },
  { key: "xg", label: "xG", title: "Expected goals (our model)" },
];

function PlayerTable({ side, players }: { side: "home" | "away"; players: PlayerAgg[] }) {
  const [sort, setSort] = useState<{ key: PlayerSortKey; desc: boolean }>({ key: "toi_s", desc: true });
  const rows = useMemo(() => {
    const r = players.filter((p) => p.side === side);
    r.sort((a, b) => {
      const va = a[sort.key];
      const vb = b[sort.key];
      const cmp = typeof va === "string" ? va.localeCompare(vb as string) : (va as number) - (vb as number);
      return sort.desc ? -cmp : cmp;
    });
    return r;
  }, [players, side, sort]);

  const clickSort = (key: PlayerSortKey) =>
    setSort((s) => (s.key === key ? { key, desc: !s.desc } : { key, desc: key !== "name" && key !== "pos" }));

  return (
    <table className="players">
      <caption>{side === "home" ? "Home" : "Away"} — {rows[0]?.team}</caption>
      <thead>
        <tr>
          {PLAYER_COLS.map((c) => (
            <th key={c.key} className="sortable" title={c.title} onClick={() => clickSort(c.key)}>
              {c.label}
              {sort.key === c.key ? (sort.desc ? " ▼" : " ▲") : ""}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((p) => (
          <tr key={p.id}>
            <td>{p.pos === "G" ? p.name : <Link href={`/player/${p.id}`} prefetch={false}>{p.name}</Link>}</td>
            <td>{p.pos}</td>
            <td>{p.shifts}</td>
            <td>{mmss(p.toi_s)}</td>
            <td>{p.g}</td>
            <td>{p.a1}</td>
            <td>{p.a2}</td>
            <td>{p.pts}</td>
            <td>{p.shots}</td>
            <td>{p.xg.toFixed(2)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function GameContent({ gameId }: { gameId: string }) {
  const { data: game, error } = useSWRImmutable<Game>(gameDetailUrl(gameId), fetchGame);

  // The score/summary/box render from `game` and commit immediately; the heavy 200+ stint timeline
  // renders from the deferred copy. Seeding useDeferredValue with `undefined` means the first commit
  // paints the summary + a timeline skeleton, then React builds the large stint subtree at low
  // priority in a separate, non-blocking commit — so the real content's first paint never janks.
  const deferredGame = useDeferredValue(game, undefined);
  const timelinePending = deferredGame !== game;

  const byPeriod = useMemo(() => {
    if (!deferredGame) return [];
    const groups = new Map<number, Stint[]>();
    for (const s of deferredGame.stints) {
      const period = Math.floor(s.start / 1200) + 1;
      (groups.get(period) ?? groups.set(period, []).get(period)!).push(s);
    }
    return [...groups.entries()].sort((a, b) => a[0] - b[0]);
  }, [deferredGame]);

  // skater name -> id, so any name in the timeline can link to that player's page
  const nameToId = useMemo(
    () => new Map((game?.players ?? []).filter((p) => p.pos !== "G").map((p) => [p.name, p.id])),
    [game]
  );
  // id -> player, to resolve the normalized stint on-ice ids to names/positions
  const byId = useMemo(
    () => new Map((game?.players ?? []).map((p) => [p.id, p] as const)),
    [game]
  );

  if (error) return <div className="loading">Couldn’t load game {gameId} ({error.message}).</div>;
  if (!game) return <GameSkeleton />;
  const t = game.totals;

  return (
    <LinkMap.Provider value={nameToId}>
      <ByIdMap.Provider value={byId}>
      <Link className="backlink" href="/">
        ← all games
      </Link>

      <div className="panel">
        <div className="gv-header">
          <div className="scoreline">
            <span className="team">
              {game.away} <span className="score">{game.away_score}</span>
            </span>
            <span className="at">@</span>
            <span className="team">
              {game.home} <span className="score">{game.home_score}</span>
            </span>
          </div>
          <div className="gv-meta">
            {game.date} · game {game.game_id}
          </div>
        </div>
        <div className="statgrid">
          <div className="stat">
            <span className="v">{t.away_xgf.toFixed(2)} / {t.home_xgf.toFixed(2)}</span>
            <span className="k">xGF away / home</span>
          </div>
          <div className="stat">
            <span className="v">{t.away_shots} / {t.home_shots}</span>
            <span className="k">shots away / home</span>
          </div>
          <div className="stat">
            <span className="v">{game.stints.length}</span>
            <span className="k">stints</span>
          </div>
        </div>
      </div>

      <Collapsible title="Player aggregates (shifts · TOI · shots · xG)" open={false}>
        <div className="two-col">
          <PlayerTable side="away" players={game.players} />
          <PlayerTable side="home" players={game.players} />
        </div>
      </Collapsible>

      <div className="panel">
        <h2>Timeline — stints &amp; events</h2>
        {timelinePending || !deferredGame ? (
          <TimelineSkeleton />
        ) : (
          byPeriod.map(([period, stints]) => (
            <PeriodSection key={period} period={period} stints={stints} home={game.home} away={game.away} />
          ))
        )}
      </div>
      </ByIdMap.Provider>
    </LinkMap.Provider>
  );
}
