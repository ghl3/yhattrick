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

// --- model / player views (export_model.py) ---
export type MetricKey = "ev_off" | "ev_def" | "pp_off" | "pk_def";

export interface PlayerRow {
  id: number;
  name: string;
  pos: string;
  group: "F" | "D";
  ev_toi: number;
  ev_off: number;
  ev_off_pct: number;
  ev_def: number;
  ev_def_pct: number;
  pp_off: number;
  pp_off_pct: number;
  pk_def: number;
  pk_def_pct: number;
}

export interface PlayerMetric {
  v: number;
  se: number;
  toi: number;
  pct: number;
}

export interface SeasonRow {
  season: number;
  team: string;
  gp: number;
  toi_min: number;
  g: number;
  a1: number;
  a2: number;
  points: number;
  sog: number;
  icf: number;
  blocks: number;
  hits: number;
  takeaways: number;
  giveaways: number;
  fo_won: number;
  fo_lost: number;
  pen_taken: number;
  pen_drawn: number;
  so_g?: number;
  so_att?: number;
  ev_off?: number | null;
  ev_def?: number | null;
  pp_off?: number | null;
  pk_def?: number | null;
}

export interface Linemate {
  id: number;
  name: string;
  toi_min: number;
}

export interface PlayerDetail {
  id: number;
  name: string;
  pos: string;
  group: "F" | "D";
  teams: string[];
  seasons: number[];
  gp: number;
  g: number;
  a: number;
  points: number;
  impact: Record<MetricKey, PlayerMetric>;
  per_season: SeasonRow[];
  linemates: Linemate[];
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
