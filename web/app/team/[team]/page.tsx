"use client";
import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { type ColumnDef, type SortingState } from "@tanstack/react-table";
import type { GameIndexRow, GoalieRow, PlayerRow } from "@/lib/types";
import { pctColor, seasonLabel } from "@/lib/format";
import { teamFullName, teamLogo } from "@/lib/teams";
import { DataTable } from "@/components/DataTable";

const svFmt = (v: number | null) => (v == null ? "—" : v.toFixed(3).replace(/^0/, ""));

// goalies who appeared for this team (career totals — we don't split goalie stats by team)
const goalieColumns = (): ColumnDef<GoalieRow>[] => [
  { header: "Goalie", accessorKey: "name", cell: (c) => c.getValue<string>() },
  { header: "GP", accessorKey: "gp", id: "gp", cell: (c) => <span className="num">{c.getValue<number>()}</span>, meta: { title: "Career games played (all teams)" } },
  { header: "Sv%", accessorKey: "sv_pct", id: "sv_pct", cell: (c) => <span className="num">{svFmt(c.getValue<number | null>())}</span>, meta: { title: "Career save %" } },
  { header: "GAA", accessorKey: "gaa", id: "gaa", cell: (c) => <span className="num">{c.getValue<number | null>()?.toFixed(2) ?? "—"}</span>, meta: { title: "Career goals-against average" } },
  { header: "GSAx", accessorKey: "gsax", id: "gsax", cell: (c) => { const v = c.getValue<number | null>(); return <span className="num">{v == null ? "—" : `${v >= 0 ? "+" : ""}${v.toFixed(1)}`}</span>; }, meta: { title: "Career goals saved above expected" } },
];

// Roster stats are for THIS team only (from each player's game log), so a traded player shows just
// their games/production while on this team — not career totals (those live on the player page).
const rosterColumns = (team: string): ColumnDef<PlayerRow>[] => {
  const st = (p: PlayerRow) => p.by_team?.[team];
  const numCol = (header: string, id: string, get: (p: PlayerRow) => number, title: string): ColumnDef<PlayerRow> => ({
    header, id, accessorFn: get, meta: { title },
    cell: (c) => <span className="num">{c.getValue<number>()}</span>,
  });
  return [
    { header: "Player", accessorKey: "name", cell: (c) => c.getValue<string>() },
    { header: "Pos", accessorKey: "pos" },
    {
      header: "Status",
      id: "status",
      accessorFn: (p) => (p.team === team ? "Current" : "Former"),
      cell: (c) => {
        const v = c.getValue<string>();
        return <span className={v === "Current" ? "status-current" : "muted"}>{v}</span>;
      },
    },
    numCol("GP", "gp", (p) => st(p)?.gp ?? 0, `Games played for ${team}`),
    numCol("G", "g", (p) => st(p)?.g ?? 0, `Goals for ${team}`),
    numCol("A", "a", (p) => st(p)?.a ?? 0, `Assists for ${team}`),
    numCol("P", "points", (p) => st(p)?.p ?? 0, `Points for ${team}`),
    {
      header: "TOI", id: "toi", accessorFn: (p) => st(p)?.toi_s ?? 0, meta: { title: `Total time on ice for ${team} (minutes)` },
      cell: (c) => <span className="num">{Math.round(c.getValue<number>() / 60)}</span>,
    },
    // player-card headline metrics (overall, not split by team — the isolation already removes team context)
    {
      header: "WAR", id: "war", accessorFn: (p) => p.war ?? null,
      meta: { title: "Wins above replacement, latest season (all teams; 0 = replacement level)" },
      cell: (c) => {
        const p = c.row.original;
        return p.war == null ? <span className="metric-cell muted">—</span> : (
          <span className="metric-cell" style={{ background: pctColor(p.war_pct ?? null) }}>{p.war.toFixed(2)}</span>
        );
      },
    },
    {
      header: "GA/60", id: "ga60", accessorFn: (p) => p.ga60 ?? null,
      meta: { title: "Goals Added per 60 above a replacement-level player at his position (5v5)" },
      cell: (c) => {
        const p = c.row.original;
        return p.ga60 == null ? <span className="metric-cell muted">—</span> : (
          <span className="metric-cell" style={{ background: pctColor(p.ga60_pct ?? null) }}>
            {`${p.ga60 >= 0 ? "+" : ""}${p.ga60.toFixed(2)}`}
          </span>
        );
      },
    },
  ];
};

export default function Team() {
  const params = useParams<{ team: string }>();
  const router = useRouter();
  const team = (params.team || "").toUpperCase();
  // default: current players first, then most games played
  const [rosterSort, setRosterSort] = useState<SortingState>([{ id: "status", desc: false }, { id: "gp", desc: true }]);
  const rosterCols = useMemo(() => rosterColumns(team), [team]);
  const goalieCols = useMemo(() => goalieColumns(), []);
  const [goalieSort, setGoalieSort] = useState<SortingState>([{ id: "gsax", desc: true }]);
  const [players, setPlayers] = useState<PlayerRow[] | null>(null);
  const [goalies, setGoalies] = useState<GoalieRow[] | null>(null);
  const [games, setGames] = useState<GameIndexRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      fetch(`/data/players.json`).then((r) => (r.ok ? r.json() : Promise.reject(r.status))),
      fetch(`/data/games.json`).then((r) => (r.ok ? r.json() : Promise.reject(r.status))),
      fetch(`/data/goalies.json`).then((r) => (r.ok ? r.json() : Promise.reject(r.status))),
    ])
      .then(([p, g, go]) => { setPlayers(p); setGames(g); setGoalies(go); })
      .catch((e) => setError(String(e)));
  }, []);

  const roster = useMemo(
    () => (players ?? []).filter((p) => !!p.by_team?.[team]),
    [players, team]
  );
  const teamGoalies = useMemo(
    () => (goalies ?? []).filter((g) => g.teams.includes(team)),
    [goalies, team]
  );
  const sched = useMemo(
    () => (games ?? []).filter((g) => g.home === team || g.away === team)
      .sort((a, b) => (b.date ?? "").localeCompare(a.date ?? "")),
    [games, team]
  );
  // group the schedule by season (sched already date-sorted desc, so each season stays reverse-chron)
  const bySeason = useMemo(() => {
    const m = new Map<number, GameIndexRow[]>();
    for (const g of sched) (m.get(g.season) ?? m.set(g.season, []).get(g.season)!).push(g);
    return [...m.entries()].sort((a, b) => b[0] - a[0]);
  }, [sched]);

  // per-season summary from this team's games (record + goals/xG for & against), newest first
  type SeasonSummary = { gp: number; w: number; l: number; otl: number; gf: number; ga: number; xgf: number; xga: number; sf: number; sa: number };
  const summaries = useMemo(() => {
    const m = new Map<number, SeasonSummary>();
    for (const g of sched) {
      const home = g.home === team;
      const gf = (home ? g.home_score : g.away_score) ?? 0;
      const ga = (home ? g.away_score : g.home_score) ?? 0;
      const s = m.get(g.season) ?? { gp: 0, w: 0, l: 0, otl: 0, gf: 0, ga: 0, xgf: 0, xga: 0, sf: 0, sa: 0 };
      s.gp++;
      s.gf += gf; s.ga += ga;
      s.xgf += (home ? g.home_xgf : g.away_xgf) ?? 0;
      s.xga += (home ? g.away_xgf : g.home_xgf) ?? 0;
      s.sf += (home ? g.home_shots : g.away_shots) ?? 0;
      s.sa += (home ? g.away_shots : g.home_shots) ?? 0;
      if (gf > ga) s.w++; else if (g.ot) s.otl++; else s.l++;
      m.set(g.season, s);
    }
    return [...m.entries()].sort((a, b) => b[0] - a[0]);
  }, [sched, team]);

  if (error) return <div className="loading">Failed to load data ({error}).</div>;
  if (!players || !games) return <div className="loading">Loading {team}…</div>;

  return (
    <div>
      <Link className="backlink" href="/teams">← all teams</Link>

      <div className="panel">
        <div className="team-hero">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img className="team-logo" src={teamLogo(team)} alt={team} loading="lazy" />
          <div>
            <h2 className="player-name">{teamFullName(team)}</h2>
            <span className="player-meta">{team} · {roster.length} players · {sched.length} games (regular season, all covered years)</span>
          </div>
        </div>
      </div>

      <div className="panel">
        <h2>By season</h2>
        <table className="games ptable">
          <thead>
            <tr>
              <th>Season</th>
              <th>GP</th>
              <th>Record</th>
              <th title="Points (2 per win, 1 per overtime loss)">Pts</th>
              <th title="Goals for">GF</th>
              <th title="Goals against">GA</th>
              <th title="Goal differential">Diff</th>
              <th title="Expected goals for">xGF</th>
              <th title="Expected goals against">xGA</th>
              <th title="Share of on-ice expected goals">xG%</th>
            </tr>
          </thead>
          <tbody>
            {summaries.map(([season, s]) => {
              const diff = s.gf - s.ga;
              const xgPct = s.xgf + s.xga > 0 ? (s.xgf / (s.xgf + s.xga)) * 100 : null;
              return (
                <tr key={season}>
                  <td>{seasonLabel(season)}</td>
                  <td className="num">{s.gp}</td>
                  <td className="num">{s.w}-{s.l}-{s.otl}</td>
                  <td className="num">{2 * s.w + s.otl}</td>
                  <td className="num">{s.gf}</td>
                  <td className="num">{s.ga}</td>
                  <td className="num">{diff > 0 ? `+${diff}` : diff}</td>
                  <td className="num">{Math.round(s.xgf)}</td>
                  <td className="num">{Math.round(s.xga)}</td>
                  <td className="num">{xgPct != null ? `${xgPct.toFixed(1)}%` : "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <p className="muted card-note">Regular season. xG totals are summed from our model; xG% is xGF / (xGF + xGA).</p>
      </div>

      <div className="panel">
        <h2>Skaters</h2>
        <DataTable
          data={roster}
          columns={rosterCols}
          sorting={rosterSort}
          onSortingChange={setRosterSort}
          rowHref={(p) => `/player/${p.id}`}
          className="games ptable"
        />
        <p className="muted card-note">Every skater who appeared for {team} in our data. GP, G, A, P and TOI are for {team} only; WAR and GA/60 are the player&apos;s overall card metrics (color = percentile vs position). Click a skater for full career detail.</p>
      </div>

      {teamGoalies.length > 0 && (
        <div className="panel">
          <h2>Goalies</h2>
          <DataTable
            data={teamGoalies}
            columns={goalieCols}
            sorting={goalieSort}
            onSortingChange={setGoalieSort}
            rowHref={(g) => `/player/${g.id}`}
            className="games ptable"
          />
          <p className="muted card-note">Goalies who appeared for {team}. Stats are career totals across all teams — click a goalie for the full breakdown.</p>
        </div>
      )}

      <div className="panel">
        <h2>Games</h2>
        {bySeason.map(([season, gs]) => (
          <div key={season} className="season-group">
            <h3 className="season-head">{seasonLabel(season)} <span className="muted">· {gs.length} games</span></h3>
            <table className="games gtable">
              <thead>
                <tr><th>Date</th><th>Matchup</th><th>Result</th><th>Score</th></tr>
              </thead>
              <tbody>
                {gs.map((g) => {
                  const isHome = g.home === team;
                  const us = isHome ? g.home_score : g.away_score;
                  const them = isHome ? g.away_score : g.home_score;
                  const res = us == null || them == null ? "" : us > them ? "W" : us < them ? "L" : "T";
                  return (
                    <tr key={g.game_id} className="rowlink" onClick={() => router.push(`/game/${g.game_id}`)}>
                      <td>{g.date ?? "—"}</td>
                      <td>
                        {g.away === team ? <strong>{g.away}</strong> : g.away} @ {g.home === team ? <strong>{g.home}</strong> : g.home}
                      </td>
                      <td className={`result result-${res.toLowerCase()}`}>{res}</td>
                      <td className="num">{g.away_score}–{g.home_score}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ))}
      </div>
    </div>
  );
}
