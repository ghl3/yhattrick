"use client";
import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { type ColumnDef, type SortingState } from "@tanstack/react-table";
import type { GameIndexRow, GoalieRow, PlayerRow } from "@/lib/types";
import { pctColor, seasonLabel } from "@/lib/format";
import { teamFullName, teamLogo } from "@/lib/teams";
import { SUMMARY_HEADERS, summarizeTeamGames, summaryCells } from "@/lib/teamSeason";
import { DataTable } from "@/components/DataTable";

const svFmt = (v: number | null) => (v == null ? "—" : v.toFixed(3).replace(/^0/, ""));

const statusCol = <T,>(isCurrent: (row: T) => boolean): ColumnDef<T> => ({
  header: "Status",
  id: "status",
  accessorFn: (row) => (isCurrent(row) ? "Current" : "Former"),
  cell: (c) => {
    const v = c.getValue<string>();
    return <span className={v === "Current" ? "status-current" : "muted"}>{v}</span>;
  },
});

// Roster stats are for THIS team in THIS season only (from each player's game log), so a traded
// player shows just what he did here that year — career totals live on the player page.
const rosterColumns = (team: string, sKey: string, seasonName: string): ColumnDef<PlayerRow>[] => {
  const st = (p: PlayerRow) => p.by_team?.[team]?.[sKey];
  const numCol = (header: string, id: string, get: (p: PlayerRow) => number, title: string): ColumnDef<PlayerRow> => ({
    header, id, accessorFn: get, meta: { title },
    cell: (c) => <span className="num">{c.getValue<number>()}</span>,
  });
  return [
    { header: "Player", accessorKey: "name", cell: (c) => c.getValue<string>() },
    { header: "Pos", accessorKey: "pos" },
    statusCol<PlayerRow>((p) => p.team === team),
    numCol("GP", "gp", (p) => st(p)?.gp ?? 0, `Games played for ${team} in ${seasonName}`),
    numCol("G", "g", (p) => st(p)?.g ?? 0, `Goals for ${team} in ${seasonName}`),
    numCol("A", "a", (p) => st(p)?.a ?? 0, `Assists for ${team} in ${seasonName}`),
    numCol("P", "points", (p) => st(p)?.p ?? 0, `Points for ${team} in ${seasonName}`),
    {
      header: "TOI", id: "toi", accessorFn: (p) => st(p)?.toi_s ?? 0, meta: { title: `Total time on ice for ${team} in ${seasonName} (minutes)` },
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
      header: "NGA/60", id: "ga60", accessorFn: (p) => p.ga60 ?? null,
      meta: { title: "Net Goals Added per 60 above a replacement-level player at his position (5v5)" },
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

// Goalie lines are the SEASON's totals pooled across teams (goalie stats aren't split by team).
const goalieColumns = (team: string, sKey: string, seasonName: string): ColumnDef<GoalieRow>[] => {
  const st = (g: GoalieRow) => g.by_season?.[sKey];
  return [
    { header: "Goalie", accessorKey: "name", cell: (c) => c.getValue<string>() },
    statusCol<GoalieRow>((g) => g.team === team),
    {
      header: "GP", id: "gp", accessorFn: (g: GoalieRow) => st(g)?.gp ?? 0,
      cell: (c) => <span className="num">{c.getValue<number>()}</span>, meta: { title: `Games played in ${seasonName} (all teams)` },
    },
    {
      header: "Sv%", id: "sv_pct", accessorFn: (g: GoalieRow) => st(g)?.sv_pct ?? null,
      cell: (c) => <span className="num">{svFmt(c.getValue<number | null>())}</span>, meta: { title: `Save % in ${seasonName}` },
    },
    {
      header: "GAA", id: "gaa", accessorFn: (g: GoalieRow) => st(g)?.gaa ?? null,
      cell: (c) => <span className="num">{c.getValue<number | null>()?.toFixed(2) ?? "—"}</span>, meta: { title: `Goals-against average in ${seasonName}` },
    },
    {
      header: "GSAx", id: "gsax", accessorFn: (g: GoalieRow) => st(g)?.gsax ?? null,
      cell: (c) => { const v = c.getValue<number | null>(); return <span className="num">{v == null ? "—" : `${v >= 0 ? "+" : ""}${v.toFixed(1)}`}</span>; },
      meta: { title: `Goals saved above expected in ${seasonName}` },
    },
  ];
};

export default function TeamSeason() {
  const params = useParams<{ team: string; season: string }>();
  const router = useRouter();
  const team = (params.team || "").toUpperCase();
  const season = Number(params.season);
  const sKey = String(season);
  const seasonName = seasonLabel(season);

  const [rosterSort, setRosterSort] = useState<SortingState>([{ id: "status", desc: false }, { id: "gp", desc: true }]);
  const [goalieSort, setGoalieSort] = useState<SortingState>([{ id: "status", desc: false }, { id: "gp", desc: true }]);
  const rosterCols = useMemo(() => rosterColumns(team, sKey, seasonName), [team, sKey, seasonName]);
  const goalieCols = useMemo(() => goalieColumns(team, sKey, seasonName), [team, sKey, seasonName]);
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
    () => (players ?? []).filter((p) => !!p.by_team?.[team]?.[sKey]),
    [players, team, sKey]
  );
  const teamGoalies = useMemo(
    () => (goalies ?? []).filter((g) => !!g.by_season?.[sKey]?.teams?.includes(team)),
    [goalies, team, sKey]
  );
  const sched = useMemo(
    () => (games ?? []).filter((g) => g.season === season && (g.home === team || g.away === team))
      .sort((a, b) => (b.date ?? "").localeCompare(a.date ?? "")),
    [games, team, season]
  );
  const summary = useMemo(() => summarizeTeamGames(sched, team).get(season), [sched, team, season]);

  if (error) return <div className="loading">Failed to load data ({error}).</div>;
  if (!players || !games || !goalies) return <div className="loading">Loading {team} {seasonName}…</div>;

  return (
    <div>
      <Link className="backlink" href={`/team/${team}`}>← {team} seasons</Link>

      <div className="panel">
        <div className="team-hero">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img className="team-logo" src={teamLogo(team)} alt={team} loading="lazy" />
          <div>
            <h2 className="player-name">{teamFullName(team)} · {seasonName}</h2>
            <span className="player-meta">{team} · {roster.length} skaters · {teamGoalies.length} goalies · {sched.length} games (regular season)</span>
          </div>
        </div>
        {summary ? (
          <table className="games ptable">
            <thead>
              <tr>{SUMMARY_HEADERS.map((h) => <th key={h.label} title={h.title}>{h.label}</th>)}</tr>
            </thead>
            <tbody>
              <tr>{summaryCells(summary).map((v, i) => <td key={i} className="num">{v}</td>)}</tr>
            </tbody>
          </table>
        ) : (
          <p className="muted card-note">No games for {team} in {seasonName}.</p>
        )}
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
        <p className="muted card-note">Skaters who played for {team} in {seasonName}. Current means the skater&apos;s most recent game in our data was for {team}; Former covers everyone since traded, signed elsewhere, or out of the league. GP, G, A, P and TOI are for {team} in this season only; WAR and NGA/60 are the player&apos;s overall card metrics (color = percentile vs position). Click a skater for full career detail.</p>
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
          <p className="muted card-note">Goalies who appeared for {team} in {seasonName}. Season stats pool the goalie&apos;s whole season across teams. Current means the goalie&apos;s most recent game was for {team}. Click a goalie for the full breakdown.</p>
        </div>
      )}

      <div className="panel">
        <h2>Games</h2>
        <table className="games gtable">
          <thead>
            <tr><th>Date</th><th>Matchup</th><th>Result</th><th>Score</th></tr>
          </thead>
          <tbody>
            {sched.map((g) => {
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
    </div>
  );
}
