// TypeScript port of the generative model's closed-form value equations
// (pipeline/src/yhattrick/models/generative_cards.py: values_at / gbar_classes / creator_mix,
// and generative_likelihood.py: marginal_goal_prob). Constants + quadrature nodes arrive in
// /data/gen_model.json (export_gen_explainer.py); each shipped player template carries its
// Python-computed values so the port can be asserted against them at load time.

export interface GenParams {
  sh: number;   // own-shot volume (log rate)
  cr: number;   // creation volume (log rate, per teammate)
  df: number;   // opponent shot suppression (log rate; <0 good)
  qs: number;   // own-shot danger (logit-xG, effective)
  qd: number;   // opponent danger suppression (logit-xG, effective)
  cq: number;   // on-ice creation quality (logit-xG lift on teammate shots)
  fin: number;  // finishing (logit-conversion, effective)
}

export interface GenTemplate {
  id: number; name: string; label?: string; pos: "F" | "D"; age: number | null;
  params: GenParams; toi_ev_min: number;
  expected: { sc: number; pm: number; df: number; ga60: number | null; war: number | null };
}

export interface GenModelData {
  fit_seasons: number[];
  n_players: number;
  constants: {
    mu_rate: { ev: number; ma: number };
    psi0: { ev: number; ma: number };
    mu_qual: { ev: number; ma: number };
    beta_s: number;
    qcreate: { F: number; D: number };
    conv_a: { ev: number; ma: number };
    conv_b: { ev: number; ma: number };
    value_env: { ev: number; ma: number };
    kappa: number;
    goals_per_win: number;
    shots_per_goal: { ev: number; ma: number };
  };
  gl40: { x: number[]; w: number[] };
  quantiles: number[];
  dist: Record<"F" | "D", Record<keyof GenParams, number[]>>;
  replacement: Record<string, Record<"F" | "D", Record<string, number>>>;
  replacement_values: Record<"F" | "D", { sc: number; pm: number; df: number; pp_sc: number; pp_pm: number; pk_df: number }>;
  baselines: Record<"F" | "D", { sc: number; pm: number; df: number }>;
  repl_band_pct: [number, number];
  fin_avg_p100: Record<"F" | "D", number>;
  templates: GenTemplate[];     // curated archetype quick-picks
  players: GenTemplate[];       // every gate-clearing player (simulator starting points)
}

const EPS = 1e-9;
export const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));
export const logit = (p: number) => Math.log(p / (1 - p));
const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));

// ── regularized incomplete beta + inverse (Lanczos lgamma + Lentz continued fraction) ──────────
const LANCZOS = [
  676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
  12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
];
function lgamma(x: number): number {
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  x -= 1;
  let a = 0.99999999999980993;
  for (let i = 0; i < LANCZOS.length; i++) a += LANCZOS[i] / (x + i + 1);
  const t = x + 7.5;
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

function betacf(a: number, b: number, x: number): number {
  const MAXIT = 200, FPMIN = 1e-300, TINY_EPS = 3e-14;
  const qab = a + b, qap = a + 1, qam = a - 1;
  let c = 1, d = 1 - (qab * x) / qap;
  if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= MAXIT; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d; h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < TINY_EPS) break;
  }
  return h;
}

/** I_x(a, b): P(X <= x) for X ~ Beta(a, b). */
export function betainc(a: number, b: number, x: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const lbeta = lgamma(a) + lgamma(b) - lgamma(a + b);
  const front = Math.exp(a * Math.log(x) + b * Math.log(1 - x) - lbeta);
  return x < (a + 1) / (a + b + 2)
    ? (front * betacf(a, b, x)) / a
    : 1 - (front * betacf(b, a, 1 - x)) / b;
}

/** Inverse of I_x(a, b): the Beta quantile. Newton with bisection safeguarding. */
export function betaincinv(a: number, b: number, p: number): number {
  if (p <= 0) return 0;
  if (p >= 1) return 1;
  let lo = 0, hi = 1, x = a / (a + b);
  const lbeta = lgamma(a) + lgamma(b) - lgamma(a + b);
  for (let it = 0; it < 100; it++) {
    const f = betainc(a, b, x) - p;
    if (f > 0) hi = x; else lo = x;
    const logPdf = (a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - lbeta;
    const step = f / Math.exp(logPdf);
    let nx = x - step;
    if (!(nx > lo && nx < hi)) nx = 0.5 * (lo + hi); // Newton left the bracket -> bisect
    if (Math.abs(nx - x) < 1e-12) return nx;
    x = nx;
  }
  return x;
}

// ── the model's own goals-per-shot (Beta-marginalized conversion) ───────────────────────────────
/** E[sigmoid(a·logit(x) + b + fin)] with x ~ Beta(s·qbar, s·(1−qbar)), by 40-node Gauss-Legendre
 * quadrature in CDF space — the same computation as the pipeline's marginal_goal_prob. */
export function marginalGoalProb(m: GenModelData, qbar: number, fin: number, key: "ev" | "ma" = "ev"): number {
  const s = m.constants.beta_s;
  const a = m.constants.conv_a[key], b = m.constants.conv_b[key];
  const q = clamp(qbar, EPS, 1 - EPS);
  const al = s * q, be = s * (1 - q);
  let total = 0;
  const { x: nodes, w: weights } = m.gl40;
  for (let k = 0; k < nodes.length; k++) {
    const x = clamp(betaincinv(al, be, nodes[k]), EPS, 1 - EPS);
    total += weights[k] * sigmoid(a * Math.log(x / (1 - x)) + b + fin);
  }
  return total;
}

/** (g0, gF, gD): goals-per-shot per creator class (unassisted / F-created / D-created). */
export function gbarClasses(m: GenModelData, qsEff: number, fin: number, key: "ev" | "ma" = "ev"): [number, number, number] {
  const mq = m.constants.mu_qual[key];
  const g = (qc: number) => marginalGoalProb(m, sigmoid(mq + qsEff + qc), fin, key);
  return [g(0), g(m.constants.qcreate.F), g(m.constants.qcreate.D)];
}

/** Creator-class weights (unassisted, F-created, D-created) in the reference environment. */
export function creatorMix(m: GenModelData, pos: "F" | "D", key: "ev" | "ma" = "ev"): [number, number, number] {
  const w0 = Math.exp(m.constants.psi0[key]) / (Math.exp(m.constants.psi0[key]) + 4);
  const nF = key === "ev" ? (pos === "D" ? 3 : 2) : pos === "D" ? 4 : 3;
  const wF = (1 - w0) * (nF / 4);
  return [w0, wF, 1 - w0 - wF];
}

export interface GenValues {
  shots60: number;      // own unblocked shots per 60
  xgPerShot: number;    // mean xG of his own shots (creator-mix weighted)
  gbar: number;         // the model's goals-per-shot on his shots
  sc: number;           // Scoring, goals/60
  pmVol: number;        // Playmaking volume half, xG/60
  pmQual: number;       // Playmaking quality half, xG/60
  pm: number;           // Playmaking total, xG/60
  df: number;           // Defense: opponent xG erased per 60
  defAllowed60: number; // his 5-defender share of opponent xG/60
}

/** The deployment-free per-60 values — a port of generative_cards.values_at (EV bucket). */
export function valuesAt(m: GenModelData, pos: "F" | "D", p: GenParams): GenValues {
  const c = m.constants;
  const mu = c.mu_rate.ev, mq = c.mu_qual.ev, env = c.value_env.ev;
  const qcPos = c.qcreate[pos];
  const shots60 = env * Math.exp(mu + p.sh);
  const [g0, gF, gD] = gbarClasses(m, p.qs, p.fin);
  const [w0, wF, wD] = creatorMix(m, pos);
  const gbar = w0 * g0 + wF * gF + wD * gD;
  const sc = shots60 * gbar;
  const xgPerShot =
    w0 * sigmoid(mq + p.qs) + wF * sigmoid(mq + p.qs + c.qcreate.F) + wD * sigmoid(mq + p.qs + c.qcreate.D);
  const pmVol = env * 4 * Math.exp(mu) * (Math.exp(p.cr) - 1) * sigmoid(mq + qcPos);
  const pmQual = env * 4 * Math.exp(mu) * Math.exp(p.cr) * (sigmoid(mq + qcPos + p.cq) - sigmoid(mq + qcPos));
  const base = Math.exp(mu) * sigmoid(mq);
  const allowed = Math.exp(mu + p.df) * sigmoid(mq + p.qd);
  return {
    shots60, xgPerShot, gbar, sc,
    pmVol, pmQual, pm: pmVol + pmQual,
    df: env * 5 * (base - allowed),
    defAllowed60: env * 5 * allowed,
  };
}

/** GA/60 vs the replacement archetype (the WAR zero), split created/prevented. */
export function ga60Of(m: GenModelData, pos: "F" | "D", v: GenValues) {
  const r = m.replacement_values[pos];
  const k = m.constants.kappa;
  const created = v.sc - r.sc + k * (v.pm - r.pm);
  const prevented = k * (v.df - r.df);
  return { created, prevented, ga60: created + prevented };
}

/** Equal-footing season WAR: GA/60 × EV minutes ÷ goals-per-win. (The site's real WAR replays
 * actual stints across EV+PP+PK; this is the 5v5, average-deployment approximation.) */
export function warOf(m: GenModelData, ga60: number, evMinutes: number): number {
  return (ga60 * (evMinutes / 60)) / m.constants.goals_per_win;
}

// ── deployment demo: observed on-ice production under a chosen environment ─────────────────────
// The value equations above run in the league-average environment (value_env). These rates layer
// relative-to-average lineup deltas on top, exactly where the model's rate/quality equations put
// them: teammate creation multiplies his shot rate; teammate creation quality lifts his shots' xG.
export interface LineupContext {
  tmCreate: number; // Σ create of his 4 teammates, relative to average (0 = average line)
  tmCq: number;     // Σ create_qual of his teammates, relative to average
  oppDef: number;   // Σ def of the 5 opponents, relative to average (<0 = stingier)
  oppQd: number;    // Σ qdef of the 5 opponents, relative to average
}

export function onIceRates(m: GenModelData, pos: "F" | "D", p: GenParams, ctx: LineupContext) {
  const c = m.constants;
  const shots60 = c.value_env.ev * Math.exp(c.mu_rate.ev + p.sh + ctx.tmCreate + ctx.oppDef);
  const [w0, wF, wD] = creatorMix(m, pos);
  const shift = ctx.tmCq + ctx.oppQd;
  const g = (qc: number) => marginalGoalProb(m, sigmoid(c.mu_qual.ev + p.qs + qc + shift), p.fin);
  const gbar = w0 * g(0) + wF * g(c.qcreate.F) + wD * g(c.qcreate.D);
  return { shots60, gbar, goals60: shots60 * gbar };
}

/** Percentile (0-100) of `value` against a shipped quantile grid, by linear interpolation. */
export function pctOf(quantiles: number[], grid: number[], value: number): number {
  if (value <= grid[0]) return quantiles[0];
  if (value >= grid[grid.length - 1]) return quantiles[quantiles.length - 1];
  for (let i = 1; i < grid.length; i++) {
    if (value <= grid[i]) {
      const t = (value - grid[i - 1]) / Math.max(grid[i] - grid[i - 1], 1e-12);
      return quantiles[i - 1] + t * (quantiles[i] - quantiles[i - 1]);
    }
  }
  return quantiles[quantiles.length - 1];
}

/** Assert the port reproduces the Python-computed values shipped with each template. */
export function verifyPort(m: GenModelData, tol = 0.02): string[] {
  const errs: string[] = [];
  for (const t of m.templates) {
    const v = valuesAt(m, t.pos, t.params);
    for (const [got, want, k] of [
      [v.sc, t.expected.sc, "sc"],
      [v.pm, t.expected.pm, "pm"],
      [v.df, t.expected.df, "df"],
    ] as const) {
      if (Math.abs(got - want) > tol) errs.push(`${t.name} ${k}: ts=${got.toFixed(4)} py=${want.toFixed(4)}`);
    }
  }
  return errs;
}
