// Shapes of the JSON the pipeline writes to web/public/data (see export_games.py).

export interface GameIndexRow {
  game_id: number;
  season: number;
  date: string | null;
  home: string;
  away: string;
  home_score: number | null;
  away_score: number | null;
  home_xgf: number;
  away_xgf: number;
  home_shots: number;
  away_shots: number;
  ot: boolean;
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
// player value: goals attributed, play-time-independent per-60 shares (by situation). Offense is
// split into scoring (own shots) and playmaking (creation for teammates).
export type ValueRateKey =
  | "scoring60" | "playmaking60" | "allow60"
  | "pp_scoring60" | "pp_playmaking60" | "pk_allow60" | "pen_net60";

export interface PlayerRow {
  id: number;
  name: string;
  pos: string;
  group: "F" | "D";
  team: string;
  teams: string[];
  // per-team games + production for team-roster views (key = team abbrev)
  by_team: Record<string, { gp: number; g: number; a: number; p: number; toi_s: number }>;
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
  // value: goals attributed — net per game (top line) + per-60 shares + window totals
  gnet_pg: number | null;
  gcreate_pg: number | null;
  gallow_pg: number | null;
  scoring60: number | null;
  playmaking60: number | null;
  allow60: number | null;
  pp_scoring60: number | null;
  pp_playmaking60: number | null;
  pk_allow60: number | null;
  pen_net60: number | null;
  g_created: number | null;
  g_allowed: number | null;
  g_fin: number | null;
  g_pen: number | null;
  g_net: number | null;
  gnet_pg_pct: number | null;
  scoring60_pct: number | null;
  playmaking60_pct: number | null;
  allow60_pct: number | null;
  pp_scoring60_pct: number | null;
  pp_playmaking60_pct: number | null;
  pk_allow60_pct: number | null;
  pen_net60_pct: number | null;
  g_net_pct: number | null;
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

export interface GameLogRow {
  game_id: number;
  season: number;
  date: string | null;
  team: string;
  opp: string;
  home: boolean;
  gf: number | null;
  ga: number | null;
  result: string | null; // "W" | "L" | "OTL" | "T"
  toi_s: number;
  g: number;
  a: number;
  p: number;
  sog: number;
  pen: number;
}

export interface PlayerDetail {
  id: number;
  name: string;
  pos: string;
  group: "F" | "D";
  current_team: string;
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
  value: PlayerValue;
  per_season: SeasonRow[];
  linemates: Linemate[];
  games: GameLogRow[];
  heat: PlayerHeat | null;
  bio: PlayerBio | null;
}

// player value (export_players.py): GOALS ATTRIBUTED. `rates` are deployment-free per-60 shares by
// situation (created/allowed = baseline ÷ on-ice skaters + RAPM coef; ≥0; allow lower is better);
// `actual` are season totals (rates × real role-TOI, summed across situations). Net = created +
// finishing − allowed + penalties. See docs/metrics.md.
export interface PlayerValue {
  rates: {
    scoring60: OniceMetric;
    playmaking60: OniceMetric;
    allow60: OniceMetric;
    pp_scoring60: OniceMetric;
    pp_playmaking60: OniceMetric;
    pk_allow60: OniceMetric;
    pen_net60: OniceMetric;
  };
  actual: {
    create_pg: number | null; allow_pg: number | null; net_pg: number | null; net_pg_pct: number | null;
    created: number | null; allowed: number | null; fin: number | null; pen: number | null; net: number | null;
  };
}

export interface PlayerBio {
  headshot: string | null;
  height_in: number | null;
  weight_lb: number | null;
  shoots: string | null;
  birth_date: string | null;
  birth_city: string | null;
  birth_state: string | null;
  birth_country: string | null;
  number: number | null;
  draft_year: number | null;
  draft_overall: number | null;
}

// per-player shot-location grids (player_heatmap.py). Each grid is [yi][xi] over cell centers x/y.
export interface PlayerHeat {
  x: number[];
  y: number[];
  s: number[][];  // smoothed shot count
  g: number[][];  // smoothed goal count
  xg: number[][]; // smoothed summed xG
}

// --- goalies (export_goalies.py) ---
// GSAx = expected goals against (xGA, over the modeled Fenwick set) − goals against; the headline
// gsax_per100 is the empirical-Bayes-shrunk save-talent rate. Shots-against / Sv% / GAA display over
// shots on goal; GSAx (and every split's GSAx) is over the calibrated Fenwick set so they reconcile.
export type GoalieMetricKey =
  | "gsax_per100" | "gsax60" | "sv_pct" | "gaa" | "hd_sv_pct" | "qs_pct";

export interface GoalieRow {
  id: number;
  name: string;
  team: string;
  teams: string[];
  gp: number;
  starts: number;
  sa: number;          // shots on goal faced
  saves: number;
  ga: number;
  shutouts: number;
  sv_pct: number | null;
  gaa: number | null;
  xga: number | null;  // expected goals against (Fenwick)
  gsax: number | null; // xGA − GA
  gsax_per100: number | null;
  gsax_per100_se: number | null;
  gsax60: number | null;
  hd_sv_pct: number | null;
  qs: number;
  rbs: number;
  qs_pct: number | null;
  // player-value analog (added=0; prevented=net=shrunk GSAx total; gprev60 = GSAx/60)
  g_added: number;
  g_prevented: number | null;
  g_net: number | null;
  gnet_pg: number | null;
  gprev60: number | null;
  // + `${GoalieMetricKey}_pct` percentile fields
  gsax_per100_pct: number | null;
  gsax60_pct: number | null;
  sv_pct_pct: number | null;
  gaa_pct: number | null;
  hd_sv_pct_pct: number | null;
  qs_pct_pct: number | null;
}

export interface GoalieSeasonRow {
  season: number;
  team: string;
  gp: number;
  starts: number;
  toi_min: number;
  sa: number;
  sog_against: number;
  saves: number;
  ga: number;
  shutouts: number;
  sv_pct: number | null;
  gaa: number | null;
  xga: number | null;
  gsax: number | null;
  hd_sv_pct: number | null;
  qs: number;
  rbs: number;
  gsax_per100: number | null;
}

export interface GoalieSplit {
  sa: number;          // shots on goal in this slice
  sv_pct: number | null;
  gsax: number | null; // Fenwick GSAx (reconciles to the total)
}
export type GoalieDangerSplit = GoalieSplit & { bucket: "ld" | "md" | "hd" };
export type GoalieSituationSplit = GoalieSplit & { sit: "ev" | "pk" | "pp" };
export type GoalieShotTypeSplit = GoalieSplit & { shot_type: string };

export interface GoalieGameLogRow {
  game_id: number;
  season: number;
  date: string | null;
  team: string;
  opp: string;
  home: boolean;
  decision: string | null; // W | L | OTL | null (relief)
  toi_s: number;
  sog_against: number;
  saves: number;
  ga: number;
  xga: number | null;
  gsax: number | null;
  started: boolean;
  shutout: boolean;
  qs: boolean;
  rbs: boolean;
}

export interface GoalieDetail {
  kind: "goalie";
  id: number;
  name: string;
  current_team: string;
  teams: string[];
  seasons: number[];
  gp: number;
  starts: number;
  sa: number;
  saves: number;
  ga: number;
  shutouts: number;
  sv_pct: number | null;
  gaa: number | null;
  gsax: number | null;
  metric: Record<GoalieMetricKey, { v: number | null; pct: number | null; se?: number }>;
  value: {
    rates: { gprev60: OniceMetric };
    actual: { added: number; prevented: number | null; net: number | null; net_pg: number | null };
  };
  danger: GoalieDangerSplit[];
  situation: GoalieSituationSplit[];
  shot_types: GoalieShotTypeSplit[];
  per_season: GoalieSeasonRow[];
  games: GoalieGameLogRow[];
  heat: PlayerHeat | null;
  bio: PlayerBio | null;
}

// the player detail JSON is one of these two (discriminated on `kind`; skaters omit it)
export type AnyPlayerDetail = PlayerDetail | GoalieDetail;
export function isGoalie(d: AnyPlayerDetail): d is GoalieDetail {
  return (d as GoalieDetail).kind === "goalie";
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
