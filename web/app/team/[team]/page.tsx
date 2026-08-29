"use client";
import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import type { GameIndexRow } from "@/lib/types";
import { seasonLabel } from "@/lib/format";
import { teamFullName, teamLogo } from "@/lib/teams";
import { SUMMARY_HEADERS, summarizeTeamGames, summaryCells } from "@/lib/teamSeason";

// Team hub: the season selector. Each season row links to /team/<team>/<season>, which holds
// that season's roster, goalies, and games.
export default function Team() {
  const params = useParams<{ team: string }>();
  const router = useRouter();
  const team = (params.team || "").toUpperCase();
  const [games, setGames] = useState<GameIndexRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`/data/games.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then(setGames)
      .catch((e) => setError(String(e)));
  }, []);

  const summaries = useMemo(
    () => (games ? [...summarizeTeamGames(games, team).entries()].sort((a, b) => b[0] - a[0]) : []),
    [games, team]
  );
  const totalGames = useMemo(() => summaries.reduce((n, [, s]) => n + s.gp, 0), [summaries]);

  if (error) return <div className="loading">Failed to load data ({error}).</div>;
  if (!games) return <div className="loading">Loading {team}…</div>;

  return (
    <div>
      <Link className="backlink" href="/teams">← all teams</Link>

      <div className="panel">
        <div className="team-hero">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img className="team-logo" src={teamLogo(team)} alt={team} loading="lazy" />
          <div>
            <h2 className="player-name">{teamFullName(team)}</h2>
            <span className="player-meta">{team} · {summaries.length} seasons · {totalGames} games (regular season)</span>
          </div>
        </div>
      </div>

      <div className="panel">
        <h2>Seasons</h2>
        <table className="games ptable">
          <thead>
            <tr>
              <th>Season</th>
              {SUMMARY_HEADERS.map((h) => <th key={h.label} title={h.title}>{h.label}</th>)}
            </tr>
          </thead>
          <tbody>
            {summaries.map(([season, s]) => (
              <tr key={season} className="rowlink" onClick={() => router.push(`/team/${team}/${season}`)}>
                <td><Link href={`/team/${team}/${season}`} onClick={(e) => e.stopPropagation()}>{seasonLabel(season)}</Link></td>
                {summaryCells(s).map((v, i) => <td key={i} className="num">{v}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
        <p className="muted card-note">Select a season for that year&apos;s roster, goalies, and games. xG totals are summed from our model; xG% is xGF / (xGF + xGA).</p>
      </div>
    </div>
  );
}
