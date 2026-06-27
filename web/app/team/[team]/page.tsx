"use client";
import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import type { GameIndexRow, PlayerRow } from "@/lib/types";
import { seasonLabel } from "@/lib/format";

export default function Team() {
  const params = useParams<{ team: string }>();
  const team = (params.team || "").toUpperCase();
  const [players, setPlayers] = useState<PlayerRow[] | null>(null);
  const [games, setGames] = useState<GameIndexRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      fetch(`/data/players.json`).then((r) => (r.ok ? r.json() : Promise.reject(r.status))),
      fetch(`/data/games.json`).then((r) => (r.ok ? r.json() : Promise.reject(r.status))),
    ])
      .then(([p, g]) => { setPlayers(p); setGames(g); })
      .catch((e) => setError(String(e)));
  }, []);

  const roster = useMemo(
    () => (players ?? []).filter((p) => (p.teams ?? []).includes(team)).sort((a, b) => b.points - a.points),
    [players, team]
  );
  const sched = useMemo(
    () => (games ?? []).filter((g) => g.home === team || g.away === team)
      .sort((a, b) => (a.date ?? "").localeCompare(b.date ?? "")),
    [games, team]
  );
  // group the schedule by season (sched already date-sorted, so each season stays chronological)
  const bySeason = useMemo(() => {
    const m = new Map<number, GameIndexRow[]>();
    for (const g of sched) (m.get(g.season) ?? m.set(g.season, []).get(g.season)!).push(g);
    return [...m.entries()].sort((a, b) => a[0] - b[0]);
  }, [sched]);

  if (error) return <div className="loading">Failed to load data ({error}).</div>;
  if (!players || !games) return <div className="loading">Loading {team}…</div>;

  return (
    <div>
      <Link className="backlink" href="/teams">← all teams</Link>

      <div className="panel">
        <div className="player-head">
          <h2 className="player-name">{team}</h2>
          <span className="player-meta">{roster.length} players · {sched.length} games (regular season, all covered years)</span>
        </div>
      </div>

      <div className="panel">
        <h2>Players</h2>
        <table className="games ptable">
          <thead>
            <tr><th>Player</th><th>Pos</th><th>GP</th><th>G</th><th>A</th><th>P</th><th>EV TOI</th></tr>
          </thead>
          <tbody>
            {roster.map((p) => (
              <tr key={p.id}>
                <td><Link href={`/player/${p.id}`}>{p.name}</Link></td>
                <td>{p.pos}</td>
                <td className="num">{p.gp}</td>
                <td className="num">{p.g}</td>
                <td className="num">{p.a}</td>
                <td className="num">{p.points}</td>
                <td className="num">{Math.round(p.ev_toi)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="muted card-note">Players who appeared for {team} in any covered season (totals are career across all teams).</p>
      </div>

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
                    <tr key={g.game_id}>
                      <td><Link href={`/game/${g.game_id}`}>{g.date ?? "—"}</Link></td>
                      <td>
                        <Link href={`/game/${g.game_id}`}>
                          {g.away === team ? <strong>{g.away}</strong> : g.away} @ {g.home === team ? <strong>{g.home}</strong> : g.home}
                        </Link>
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
