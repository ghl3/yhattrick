// Shapes of the JSON the pipeline writes to web/public/data (see build_games.py).

export interface GameIndexRow {
  game_id: number;
  season: number;
  date: string | null;
  home: string;
  away: string;
  home_score: number | null;
  away_score: number | null;
  n_stints: number;
  n_shots: number;
  n_events: number;
  onice_exact: number | null; // fraction of shots whose on-ice counts matched MoneyPuck
  large_mismatch: number; // shots with a >1 on-ice count disagreement
  overload_stints: number; // stints with an illegal (>6) skater count
}

export interface PlayerRef {
  id: number;
  name: string;
  pos: string | null;
  number: number | null;
}

export interface PlayerAgg extends PlayerRef {
  team: string;
  side: "home" | "away";
  shifts: number;
  toi_s: number;
  g: number;
  a1: number;
  a2: number;
  pts: number;
  shots: number;
  xg: number;
}

export type OniceMatch = "exact" | "within1" | "large";

export interface TimelineEvent {
  t: number;
  clock: string;
  type: string;
  team: string | null;
  x: number | null;
  y: number | null;
  player?: string | null;
  detail?: string | null;
  zone?: string | null;
  xGoal?: number;
  onice_match?: OniceMatch;
  shot_type?: string | null;
  distance?: number | null;
  angle?: number | null;
  rebound?: boolean;
  rush?: boolean;
}

export interface Stint {
  idx: number;
  start: number;
  end: number;
  clock_start: string;
  clock_end: string;
  duration_s: number;
  strength: string; // e.g. "5v5", "5v4"
  overload: boolean;
  home_skaters: PlayerRef[];
  away_skaters: PlayerRef[];
  home_goalie: PlayerRef[];
  away_goalie: PlayerRef[];
  home_xgf: number;
  away_xgf: number;
  events: TimelineEvent[];
}

export interface GameTotals {
  home_xgf: number;
  away_xgf: number;
  home_shots: number;
  away_shots: number;
  home_score: number | null;
  away_score: number | null;
}

export interface Game {
  game_id: number;
  date: string | null;
  home: string;
  away: string;
  home_score: number | null;
  away_score: number | null;
  totals: GameTotals;
  players: PlayerAgg[];
  stints: Stint[];
}
