import type { GameIndexRow } from "@/lib/types";

// One team's season line computed from the games index: record + goals/xG/shots for and against.
export type SeasonSummary = {
  gp: number; w: number; l: number; otl: number;
  gf: number; ga: number; xgf: number; xga: number; sf: number; sa: number;
};

export function summarizeTeamGames(games: GameIndexRow[], team: string): Map<number, SeasonSummary> {
  const m = new Map<number, SeasonSummary>();
  for (const g of games) {
    if (g.home !== team && g.away !== team) continue;
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
  return m;
}

// Header cells + one row of season metadata, shared by the team hub table and the season page.
export const SUMMARY_HEADERS: { label: string; title?: string }[] = [
  { label: "GP" },
  { label: "Record" },
  { label: "Pts", title: "Points (2 per win, 1 per overtime loss)" },
  { label: "GF", title: "Goals for" },
  { label: "GA", title: "Goals against" },
  { label: "Diff", title: "Goal differential" },
  { label: "xGF", title: "Expected goals for" },
  { label: "xGA", title: "Expected goals against" },
  { label: "xG%", title: "Share of on-ice expected goals" },
];

export function summaryCells(s: SeasonSummary): (string | number)[] {
  const diff = s.gf - s.ga;
  const xgPct = s.xgf + s.xga > 0 ? ((s.xgf / (s.xgf + s.xga)) * 100).toFixed(1) + "%" : "—";
  return [
    s.gp,
    `${s.w}-${s.l}-${s.otl}`,
    2 * s.w + s.otl,
    s.gf,
    s.ga,
    diff > 0 ? `+${diff}` : diff,
    Math.round(s.xgf),
    Math.round(s.xga),
    xgPct,
  ];
}
