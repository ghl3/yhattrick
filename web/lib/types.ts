// Shapes of the JSON the pipeline writes to web/public/data (see export_games.py).

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
  onice_exact: number | null; // fraction of shots whose on-ice counts matched the pbp situationCode
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
  xg?: number;
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
  // on-ice personnel as player ids; resolve to name/pos via the game's players[] (normalized JSON)
  home_skaters: number[];
  away_skaters: number[];
  home_goalie: number | null;
  away_goalie: number | null;
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

// --- model / player views (export_players.py) ---
// modeled, isolated impact (RAPM, per-60 deltas)
export type MetricKey = "ev_off" | "ev_def" | "pp_off" | "pk_def";
// raw, descriptive on-ice rates (the team's rate while the player is on the ice, not isolated)
export type OniceKey =
  | "ev_xgf60" | "ev_xga60" | "ev_xgshare"
  | "ev_cf60" | "ev_ca60" | "ev_cfshare"
  | "pp_xgf60" | "pk_xga60";
// individual (on-puck) rates — the player's own shooting & production, all situations
export type IndividualKey =
  | "shots60" | "xg_per_shot" | "ixg60" | "fin_per100"
  | "g60" | "a60" | "pen_drawn60" | "pen_taken60";

export interface PlayerRow {
  id: number;
  name: string;
  pos: string;
  group: "F" | "D";
  teams: string[];
  ev_toi: number;
  gp: number;
  g: number;
  a: number;
  points: number;
  // isolated impact + percentiles
  ev_off: number;
  ev_off_pct: number;
  ev_def: number;
  ev_def_pct: number;
  pp_off: number;
  pp_off_pct: number;
  pk_def: number;
  pk_def_pct: number;
  // on-ice rates + percentiles (value keys from OniceKey, plus `${key}_pct`)
  ev_xgf60: number | null;
  ev_xga60: number | null;
  ev_xgshare: number | null;
  ev_cf60: number | null;
  ev_ca60: number | null;
  ev_cfshare: number | null;
  pp_xgf60: number | null;
  pk_xga60: number | null;
  ev_xgf60_pct: number | null;
  ev_xga60_pct: number | null;
  ev_xgshare_pct: number | null;
  ev_cf60_pct: number | null;
  ev_ca60_pct: number | null;
  ev_cfshare_pct: number | null;
  pp_xgf60_pct: number | null;
  pk_xga60_pct: number | null;
  // individual rates + percentiles
  shots60: number | null;
  xg_per_shot: number | null;
  ixg60: number | null;
  fin_per100: number | null;
  g60: number | null;
  a60: number | null;
  pen_drawn60: number | null;
  pen_taken60: number | null;
  shots60_pct: number | null;
  xg_per_shot_pct: number | null;
  ixg60_pct: number | null;
  fin_per100_pct: number | null;
  g60_pct: number | null;
  a60_pct: number | null;
  pen_drawn60_pct: number | null;
  pen_taken60_pct: number | null;
}

export interface PlayerMetric {
  v: number;
  se: number;
  toi: number;
  pct: number;
}

export interface OniceMetric {
  v: number | null;
  pct: number | null;
}

export interface IndividualMetric {
  v: number | null;
  pct: number | null;
  se?: number; // only finishing carries a CI
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
  shots60?: number | null;
  xg_per_shot?: number | null;
  fin_per100?: number | null;
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
  onice: Record<OniceKey, OniceMetric>;
  individual: Record<IndividualKey, IndividualMetric>;
  shooting: { shots: number; ixg: number | null; fin_goals: number | null };
  per_season: SeasonRow[];
  linemates: Linemate[];
}

// --- xG model exploration page (xg.py -> web/public/data/xg_model.json) ---
export interface XgMetrics {
  auc: number;
  logloss: number;
  brier: number;
  total_xg: number;
  total_goals: number;
  n: number;
}
export interface ReliabilityBin {
  p_lo: number;
  p_hi: number;
  pred: number;
  obs: number;
  n: number;
}
export interface XgModel {
  seasons: number[];
  n_shots: number;
  n_goals: number;
  metrics: XgMetrics; // out-of-fold
  reliability: ReliabilityBin[];
  importances: { feature: string; gain: number }[];
  heatmap: {
    x: number[]; // grid x (ft), attacking net at +89
    y: number[]; // grid y (ft)
    combos: Record<string, number[][]>; // key `${shot_type}|${rebound}|${rush}|${strength}` -> [y][x] xG
    shot_types: string[];
    strengths: string[];
  };
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
