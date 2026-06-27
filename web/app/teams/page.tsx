"use client";
import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import type { GameIndexRow, PlayerRow } from "@/lib/types";

type TeamRow = { team: string; players: number; games: number };

export default function Teams() {
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

  const teams = useMemo<TeamRow[]>(() => {
    if (!players || !games) return [];
    const pc = new Map<string, number>();
    for (const p of players) for (const t of p.teams ?? []) pc.set(t, (pc.get(t) ?? 0) + 1);
    const gc = new Map<string, number>();
    for (const g of games) for (const t of [g.home, g.away]) gc.set(t, (gc.get(t) ?? 0) + 1);
    const all = new Set<string>([...pc.keys(), ...gc.keys()]);
    return [...all].sort().map((team) => ({ team, players: pc.get(team) ?? 0, games: gc.get(team) ?? 0 }));
  }, [players, games]);

  if (error) return <div className="loading">Failed to load data ({error}). Run <code>make players</code>.</div>;
  if (!players || !games) return <div className="loading">Loading teams…</div>;

  return (
    <div className="panel">
      <div className="toolbar">
        <span className="muted">{teams.length} teams · click a team for its players and games</span>
      </div>
      <div className="team-grid">
        {teams.map((t) => (
          <Link key={t.team} href={`/team/${t.team}`} className="team-card">
            <span className="team-abbr">{t.team}</span>
            <span className="team-meta">{t.players} players · {t.games} games</span>
          </Link>
        ))}
      </div>
    </div>
  );
}
