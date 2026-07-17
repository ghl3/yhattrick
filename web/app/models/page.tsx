"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { XgModel } from "@/lib/types";
import {
  type GenModelData, type GenParams, type GenTemplate,
  ga60Of, onIceRates, pctOf, valuesAt, verifyPort, warOf,
} from "@/lib/genmath";

const f1 = (v: number) => v.toFixed(1);
const f2 = (v: number) => v.toFixed(2);
const f3 = (v: number) => v.toFixed(3);
const signed2 = (v: number) => `${v >= 0 ? "+" : ""}${v.toFixed(2)}`;

// ── shot builder: click the rink, price the shot ────────────────────────────────────────────────
const STRENGTH_LABELS: Record<string, string> = { PP: "Power play", EV: "Even strength", PK: "Penalty kill" };
const SHOT_LABELS: Record<string, string> = {
  wrist: "Wrist", snap: "Snap", slap: "Slap", "tip-in": "Tip-in", backhand: "Backhand", deflected: "Deflected",
};
const FEATURE_LABELS: Record<string, string> = {
  distance: "Distance to net",
  abs_y: "Lateral position (off-center)",
  abs_angle: "Shot angle",
  x_adj: "Depth in zone",
  time_since_last: "Time since last event",
  time_since_shot: "Time since last shot",
  attempts_15s: "Pressure (attempts in last 15s)",
  dist_from_last: "Distance from last event",
  speed_from_last: "Puck speed into shot",
  angle_change: "Angle change (goalie sweep)",
  last_dist_to_net: "Last event distance to net",
  last_behind_net: "Last event behind net",
  off_wing: "Off-wing (handedness × side)",
  since_faceoff: "Time since faceoff",
  strength_diff: "Strength (skater advantage)",
  shooter_skaters: "Shooters on ice",
  def_skaters: "Defenders on ice",
  score_diff: "Score margin",
  period: "Period",
  time_g: "Game time",
  is_home: "Home shooter",
  rebound: "Rebound",
  rush: "Rush",
  same_team_last: "Possession kept",
  royal_road: "Cross-slot pass (royal road)",
  shooter_is_d: "Shooter is a defenseman",
  shot_type: "Shot type",
  zone: "Zone",
  last_event_type: "Last event type",
};

// danger ramp: pale (low xG) -> amber -> red (high)
function heat(xg: number, cap = 0.45): string {
  const t = Math.min(1, Math.max(0, xg / cap));
  const lerp = (a: number, b: number, u: number) => Math.round(a + (b - a) * u);
  if (t < 0.5) {
    const u = t / 0.5;
    return `rgb(${lerp(238, 245, u)},${lerp(245, 180, u)},${lerp(251, 0, u)})`;
  }
  const u = (t - 0.5) / 0.5;
  return `rgb(${lerp(245, 200, u)},${lerp(180, 30, u)},${lerp(0, 30, u)})`;
}

const FT = 4.2;
const X0 = 24, X1 = 100;
const W = (X1 - X0) * FT, H = 85 * FT;
const px = (x: number) => (x - X0) * FT;
const py = (y: number) => (y + 42.5) * FT;

function ShotBuilder({ m }: { m: XgModel }) {
  const { x, y, combos, shot_types, strengths } = m.heatmap;
  const [shot, setShot] = useState("wrist");
  const [strength, setStrength] = useState("EV");
  const [rebound, setRebound] = useState(false);
  const [rush, setRush] = useState(false);
  const [pt, setPt] = useState({ x: 78, y: 4 }); // slot shot, in rink feet
  const svgRef = useRef<SVGSVGElement | null>(null);

  const grid = combos[`${shot}|${rebound ? 1 : 0}|${rush ? 1 : 0}|${strength}`] ?? [];
  const cw = (x.length > 1 ? x[1] - x[0] : 2) * FT;
  const ch = (y.length > 1 ? y[1] - y[0] : 2) * FT;

  const nearest = (arr: number[], v: number) =>
    arr.reduce((bi, a, i) => (Math.abs(a - v) < Math.abs(arr[bi] - v) ? i : bi), 0);
  const xi = nearest(x, pt.x), yi = nearest(y, pt.y);
  const xg = grid[yi]?.[xi] ?? null;
  const lgAvg = m.metrics.total_xg / m.metrics.n;

  const onClick = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = svgRef.current!.getBoundingClientRect();
    const fx = ((e.clientX - rect.left) / rect.width) * W / FT + X0;
    const fy = ((e.clientY - rect.top) / rect.height) * H / FT - 42.5;
    setPt({ x: Math.min(99, Math.max(26, fx)), y: Math.min(42, Math.max(-42, fy)) });
  };

  return (
    <div>
      <div className="xg-controls">
        <label>Shot type
          <select name="shot-type" value={shot} onChange={(e) => setShot(e.target.value)}>
            {shot_types.map((s) => <option key={s} value={s}>{SHOT_LABELS[s] ?? s}</option>)}
          </select>
        </label>
        <label>Strength
          <select name="shot-strength" value={strength} onChange={(e) => setStrength(e.target.value)}>
            {strengths.map((s) => <option key={s} value={s}>{STRENGTH_LABELS[s] ?? s}</option>)}
          </select>
        </label>
        <label className="chk"><input type="checkbox" checked={rebound} onChange={(e) => setRebound(e.target.checked)} /> Rebound</label>
        <label className="chk"><input type="checkbox" checked={rush} onChange={(e) => setRush(e.target.checked)} /> Rush</label>
      </div>

      <svg ref={svgRef} className="xg-rink clickable" viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: W }} onClick={onClick}>
        {grid.map((rowArr, gyi) =>
          rowArr.map((v, gxi) => (
            <rect key={`${gxi},${gyi}`} x={px(x[gxi]) - cw / 2} y={py(y[gyi]) - ch / 2}
              width={cw + 0.5} height={ch + 0.5} fill={heat(v)} />
          ))
        )}
        <rect x={0.5} y={0.5} width={W - 1} height={H - 1} rx={26 * FT} ry={26 * FT} fill="none" stroke="#cfe0f0" />
        <line x1={px(25)} y1={0} x2={px(25)} y2={H} stroke="#9fc1e6" strokeWidth={2} />
        <line x1={px(89)} y1={py(-39)} x2={px(89)} y2={py(39)} stroke="#d98c8c" strokeWidth={1.5} />
        <rect x={px(89)} y={py(-2)} width={6} height={py(2) - py(-2)} fill="none" stroke="#d98c8c" strokeWidth={1.5} />
        <circle cx={px(69)} cy={py(22)} r={15 * FT} fill="none" stroke="#d98c8c" strokeWidth={1} opacity={0.6} />
        <circle cx={px(69)} cy={py(-22)} r={15 * FT} fill="none" stroke="#d98c8c" strokeWidth={1} opacity={0.6} />
        {/* the placed shot */}
        <circle cx={px(pt.x)} cy={py(pt.y)} r={7} fill="#fff" stroke="#1b2a3a" strokeWidth={2.5} />
        <circle cx={px(pt.x)} cy={py(pt.y)} r={2.4} fill="#1b2a3a" />
      </svg>

      <div className="xg-legend">
        <span>0.0</span>
        <span className="bar" style={{ background: `linear-gradient(90deg, ${heat(0)}, ${heat(0.11)}, ${heat(0.22)}, ${heat(0.33)}, ${heat(0.45)})` }} />
        <span>0.45+ xG</span>
      </div>
      {xg != null && (
        <div className="shot-readout">
          <span className="big">{f3(xg)} xG</span>
          <span>about 1 goal per {Math.max(1, Math.round(1 / xg))} such shots</span>
          <span>{(xg / lgAvg).toFixed(1)}× the average unblocked shot ({f3(lgAvg)})</span>
        </div>
      )}
      <p className="section-sub" style={{ marginTop: 6 }}>
        Click anywhere in the zone to move the shot; attacking net at right. The map is the model
        itself, evaluated over a location grid for the chosen shot type and situation. Pre-shot
        movement matters as much as location: toggle rebound or rush and watch the same spot
        double in value.
      </p>
    </div>
  );
}

// ── the three-stage equations, term by term ─────────────────────────────────────────────────────
// Each term is a colored chip: blue = his skills, green = his teammates, red = the opposition,
// grey = league baselines. Hovering (or focusing) a term explains it in the hint bar.
type TermKind = "player" | "tm" | "opp" | "base" | "fn";
type Hint = { name: string; def: string } | null;

// hover/focus a term chip -> its plain-language meaning shows in the hint bar
function useEqTerms() {
  const [hint, setHint] = useState<Hint>(null);
  const T = (label: React.ReactNode, kind: TermKind, name: string, def: string) => (
    <span
      className={`term ${kind}`} tabIndex={0}
      onMouseEnter={() => setHint({ name, def })} onMouseLeave={() => setHint(null)}
      onFocus={() => setHint({ name, def })} onBlur={() => setHint(null)}
    >{label}</span>
  );
  return { hint, T };
}

function StageEquations() {
  const { hint, T } = useEqTerms();

  return (
    <div className="eqs">
      <div className="stage">
        <div className="stage-head">
          <span className="stage-num">1</span>
          <span className="stage-name">Rate</span>
          <span className="stage-q">how many shots?</span>
        </div>
        <div className="stage-eq">
          shots<sub>j</sub> <span className="op">∼</span>{" "}
          {T("Poisson", "fn", "Poisson", "shot counts scatter around the rate like independent arrivals do")}(
          {T("exp", "fn", "exp", "the effects add on a log scale, so each term multiplies the rate")}({" "}
          {T(<>μ<sub>rate</sub></>, "base", "μ_rate", "the league-baseline shot rate: an average lineup, average context")}
          <span className="op"> + </span>
          {T(<>shoot<sub>j</sub></>, "player", "shoot", "his own shot volume: how often he shoots, above or below baseline")}
          <span className="op"> + </span>
          {T(<>Σ create<sub>tm</sub></>, "tm", "Σ create", "his four teammates' creation, summed: playmakers raise how often he shoots")}
          <span className="op"> + </span>
          {T(<>Σ def<sub>opp</sub></>, "opp", "Σ def", "the five defenders' suppression, summed: good defense cuts everyone's shots")}
          {" "}) <span className="op">·</span>{" "}
          {T("minutes", "base", "minutes", "stint length: exposure, more ice means more shots")} )
        </div>
      </div>

      <div className="stage">
        <div className="stage-head">
          <span className="stage-num">2</span>
          <span className="stage-name">Quality</span>
          <span className="stage-q">how dangerous?</span>
        </div>
        <div className="stage-eq">
          xG<sub>i</sub> <span className="op">∼</span>{" "}
          {T("Beta", "fn", "Beta", "each shot's danger is a random draw around that mean, with a fitted spread: some looks beat their setup, some fizzle")}( <span className="op">mean =</span>{" "}
          {T("sigmoid", "fn", "sigmoid", "squashes an unbounded score into a 0 to 1 probability")}({" "}
          {T(<>μ<sub>qual</sub></>, "base", "μ_qual", "the league-baseline shot danger: the average unblocked look")}
          <span className="op"> + </span>
          {T(<>qshoot<sub>j</sub></>, "player", "qshoot", "the danger of his own shots: where and how he gets his looks")}
          <span className="op"> + </span>
          {T(<>Σ create_qual<sub>tm</sub></>, "tm", "Σ create_qual", "his teammates' creation quality: elite setup men make every shot more dangerous")}
          <span className="op"> + </span>
          {T(<>Σ qdef<sub>opp</sub></>, "opp", "Σ qdef", "the defenders' danger suppression: pushing shots to the outside")}
          {" "}) )
        </div>
      </div>

      <div className="stage">
        <div className="stage-head">
          <span className="stage-num">3</span>
          <span className="stage-name">Conversion</span>
          <span className="stage-q">does it go in?</span>
        </div>
        <div className="stage-eq">
          goal<sub>i</sub> <span className="op">∼</span>{" "}
          {T("Bernoulli", "fn", "Bernoulli", "a coin flip at the shot's final goal probability")}(
          {T("sigmoid", "fn", "sigmoid", "squashes an unbounded score into a 0 to 1 probability")}({" "}
          {T(<>a·logit(xG<sub>i</sub>) + b</>, "base", "a·logit(xG) + b", "the fitted map from the shot's xG to real goal odds; keeps league totals exact")}
          <span className="op"> + </span>
          {T(<>fin<sub>j</sub></>, "player", "fin", "his finishing: converts above or below what his shot quality predicts")}
          <span className="op"> + </span>
          {T(<>gsave<sub>goalie</sub></>, "opp", "gsave", "the goalie's saves above expected: a good goalie pulls this below zero")}
          {" "}) )
        </div>
      </div>

      <div className="eq-hint">
        {hint ? (
          <span><b>{hint.name}</b> · {hint.def}</span>
        ) : (
          <span className="eq-legend">
            <span><i className="sw player" />his skills</span>
            <span><i className="sw tm" />his teammates</span>
            <span><i className="sw opp" />the opposition</span>
            <span><i className="sw base" />league baseline</span>
            <span className="lg-note">hover any term</span>
          </span>
        )}
      </div>
    </div>
  );
}

// ── the value equations: GA/60, GAR, WAR, same visual language as the stages ───────────────────
function ValueEquations({ kappa, gpw }: { kappa: number; gpw: number }) {
  const { hint, T } = useEqTerms();
  const kappaChip = T("κ", "base", "κ", `${kappa}, the league's fitted goals-per-xG: prices chance quality into goals`);

  return (
    <div className="eqs">
      <div className="stage">
        <div className="stage-head">
          <span className="stage-name">GA/60</span>
          <span className="stage-q">how good is he, on equal footing?</span>
        </div>
        <div className="stage-eq">
          <div className="eq-line">
            GA/60 <span className="op">=</span> ({" "}
            {T("Scoring", "player", "Scoring", "his modeled goals per 60 from his own shots: volume, danger, finishing")}
            <span className="op"> − </span>
            {T(<>Scoring<sub>repl</sub></>, "base", "Scoring_repl", "what the replacement archetype at his position scores per 60")}
            {" "})
          </div>
          <div className="eq-line cont">
            <span className="op">+</span> {kappaChip} <span className="op">·</span> ({" "}
            {T("Playmaking", "player", "Playmaking", "the chance value he creates for teammates per 60, volume and quality")}
            <span className="op"> − </span>
            {T(<>Playmaking<sub>repl</sub></>, "base", "Playmaking_repl", "the replacement archetype's creation per 60")}
            {" "})
          </div>
          <div className="eq-line cont">
            <span className="op">+</span> {kappaChip} <span className="op">·</span> ({" "}
            {T("Defense", "player", "Defense", "the opponent chance value he erases per 60")}
            <span className="op"> − </span>
            {T(<>Defense<sub>repl</sub></>, "base", "Defense_repl", "the replacement archetype's suppression per 60")}
            {" "})
          </div>
        </div>
      </div>

      <div className="stage">
        <div className="stage-head">
          <span className="stage-name">GAR</span>
          <span className="stage-q">how many goals did his actual season add?</span>
        </div>
        <div className="stage-eq">
          <div className="eq-line">
            GAR <span className="op">=</span>{" "}
            {T("Σ his actual stints", "fn", "Σ stints", "summed over every shift he really played: 5v5, power play, penalty kill")}
            {" "}[{" "}
            {T(<>E(GF−GA <span className="op">|</span> him)</>, "player", "E(GF−GA | him)", "the stint's expected goal difference with him in his slot: real linemates, opponents, goalie")}
            <span className="op"> − </span>
            {T(<>E(GF−GA <span className="op">|</span> replacement)</>, "base", "E(GF−GA | replacement)", "the same stint re-priced with the replacement archetype in his slot")}
            {" "}]
          </div>
        </div>
      </div>

      <div className="stage">
        <div className="stage-head">
          <span className="stage-name">WAR</span>
          <span className="stage-q">and in wins</span>
        </div>
        <div className="stage-eq">
          <div className="eq-line">
            WAR <span className="op">=</span> GAR <span className="op">÷</span>{" "}
            {T(`${gpw} goals per win`, "base", "goals per win", "the standard exchange rate from goal differential to standings wins")}
          </div>
        </div>
      </div>

      <div className="eq-hint">
        {hint ? (
          <span><b>{hint.name}</b> · {hint.def}</span>
        ) : (
          <span className="eq-legend">
            <span><i className="sw player" />his modeled value</span>
            <span><i className="sw base" />the replacement baseline</span>
            <span className="lg-note">hover any term</span>
          </span>
        )}
      </div>
    </div>
  );
}

// ── xG diagnostics: what drives the prediction + how well it is calibrated ─────────────────────
function Importances({ m }: { m: XgModel }) {
  const maxGain = Math.max(...m.importances.map((i) => i.gain));
  return (
    <div className="xg-imp">
      {m.importances.map((i) => (
        <div key={i.feature} className="xg-imp-row">
          <span className="xg-imp-name">{FEATURE_LABELS[i.feature] ?? i.feature}</span>
          <span className="xg-imp-bar"><span style={{ width: `${(i.gain / maxGain) * 100}%` }} /></span>
          <span className="xg-imp-val">{(i.gain * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
}

function Calibration({ m }: { m: XgModel }) {
  const data = useMemo(() => {
    const max = Math.max(...m.reliability.map((b) => b.pred), 0.4);
    return [{ pred: 0, ideal: 0 }, ...m.reliability.map((b) => ({ pred: b.pred, ours: b.obs })), { pred: max, ideal: max }];
  }, [m]);
  const mt = m.metrics;
  return (
    <div>
      <div className="chart-wrap">
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={data} margin={{ top: 8, right: 16, bottom: 18, left: 0 }}>
            <CartesianGrid stroke="#e7f1fb" />
            <XAxis dataKey="pred" type="number" domain={[0, "dataMax"]} tickFormatter={(v) => v.toFixed(2)}
              tick={{ fontSize: 11, fill: "#6b7e90" }} label={{ value: "predicted xG", position: "insideBottom", offset: -8, fontSize: 11, fill: "#6b7e90" }} />
            <YAxis type="number" domain={[0, "dataMax"]} tickFormatter={(v) => v.toFixed(2)}
              tick={{ fontSize: 11, fill: "#6b7e90" }} width={44} label={{ value: "actual rate", angle: -90, position: "insideLeft", fontSize: 11, fill: "#6b7e90" }} />
            <Tooltip formatter={(v: unknown) => (typeof v === "number" ? v.toFixed(3) : String(v))} />
            <Line dataKey="ideal" name="perfect" stroke="#9aa7b4" strokeWidth={1} strokeDasharray="4 4" dot={false} connectNulls />
            <Line dataKey="ours" name="our xG" stroke="#2f6cb0" strokeWidth={2.5} dot={{ r: 3 }} connectNulls />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="table-scroll">
        <table className="games repl-table">
          <tbody>
            <tr><td>AUC (discrimination)</td><td className="num">{mt.auc.toFixed(4)}</td></tr>
            <tr><td>Log-loss</td><td className="num">{mt.logloss.toFixed(4)}</td></tr>
            <tr><td>Brier score</td><td className="num">{mt.brier.toFixed(4)}</td></tr>
            <tr><td>Total xG vs. {mt.total_goals.toLocaleString()} goals</td><td className="num">{Math.round(mt.total_xg).toLocaleString()}</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── the player lab ──────────────────────────────────────────────────────────────────────────────
type SliderDef = { key: keyof GenParams; label: string; sub: string; invert?: boolean };
const SLIDERS: { group: string; items: SliderDef[] }[] = [
  {
    group: "Shooting", items: [
      { key: "sh", label: "Shot volume", sub: "how often he shoots" },
      { key: "qs", label: "Shot danger", sub: "how dangerous his shot locations and looks are" },
      { key: "fin", label: "Finishing", sub: "converts above or below what his looks predict" },
    ],
  },
  {
    group: "Playmaking", items: [
      { key: "cr", label: "Creation volume", sub: "how much more his teammates shoot with him on" },
      { key: "cq", label: "Creation quality", sub: "how much more dangerous their shots become" },
    ],
  },
  {
    group: "Defense", items: [
      { key: "df", label: "Suppression volume", sub: "opponent shots he erases", invert: true },
      { key: "qd", label: "Suppression quality", sub: "opponent shot danger he erases", invert: true },
    ],
  },
];

function Bar({ v, max, color = "#2f6cb0" }: { v: number; max: number; color?: string }) {
  const w = Math.min(100, Math.max(0, (v / max) * 100));
  return <span className="hbar"><span style={{ width: `${w}%`, background: color }} /></span>;
}

// signed bar on a shared ±max scale, zero line in the middle
function SignedBar({ v, max, color }: { v: number; max: number; color: string }) {
  const half = Math.min(50, (Math.abs(v) / max) * 50);
  return (
    <span className="wbar">
      <span className="zero" />
      <span className="fill" style={{
        left: v >= 0 ? "50%" : `${50 - half}%`, width: `${half}%`, background: color,
      }} />
    </span>
  );
}

function PlayerLab({ g }: { g: GenModelData }) {
  const mcd = g.templates[0];
  const [sel, setSel] = useState<GenTemplate | null>(mcd);
  const [pos, setPos] = useState<"F" | "D">(mcd.pos);
  const [params, setParams] = useState<GenParams>({ ...mcd.params });
  const [minutes, setMinutes] = useState(Math.round(mcd.toi_ev_min));
  const [query, setQuery] = useState("");
  const [ctx, setCtx] = useState({ tmCreate: 0, tmCq: 0, oppDef: 0, oppQd: 0 });

  const matches = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (q.length < 2) return [];
    return g.players.filter((p) => p.name.toLowerCase().includes(q)).slice(0, 8);
  }, [g, query]);

  const pick = (t: GenTemplate) => {
    setSel(t); setPos(t.pos); setParams({ ...t.params });
    setMinutes(Math.round(t.toi_ev_min)); setQuery("");
    setCtx({ tmCreate: 0, tmCq: 0, oppDef: 0, oppQd: 0 });
  };
  const touched = sel != null && JSON.stringify(params) !== JSON.stringify(sel.params);

  const v = useMemo(() => valuesAt(g, pos, params), [g, pos, params]);
  const ga = ga60Of(g, pos, v);
  const war = warOf(g, ga.ga60, minutes);
  const onIce = useMemo(() => onIceRates(g, pos, params, ctx), [g, pos, params, ctx]);
  const base = useMemo(() => onIceRates(g, pos, params, { tmCreate: 0, tmCq: 0, oppDef: 0, oppQd: 0 }), [g, pos, params]);
  const repl = g.replacement_values[pos];
  const kap = g.constants.kappa;

  const range = (k: keyof GenParams) => {
    const grid = g.dist[pos][k];
    const lo = grid[0], hi = grid[grid.length - 1], pad = 0.3 * (hi - lo);
    return { min: lo - pad, max: hi + pad };
  };
  const pct = (k: keyof GenParams, invert?: boolean) => {
    const p = pctOf(g.quantiles, g.dist[pos][k], params[k]);
    return Math.round(invert ? 100 - p : p);
  };

  const wfMax = Math.max(1.2, Math.abs(v.sc - repl.sc), kap * Math.abs(v.pm - repl.pm), kap * Math.abs(v.df - repl.df), Math.abs(ga.ga60)) * 1.15;

  return (
    <div>
      {/* picker */}
      <div className="lab-picker">
        <div className="typeahead">
          <input name="player-search" value={query} placeholder="Search any player…" onChange={(e) => setQuery(e.target.value)} />
          {matches.length > 0 && (
            <ul className="ta-list">
              {matches.map((p) => (
                <li key={p.id} onClick={() => pick(p)}>{p.name} <span className="muted">{p.pos}{p.age != null ? ` · ${p.age}` : ""}</span></li>
              ))}
            </ul>
          )}
        </div>
        <div className="chips">
          {g.templates.map((t) => (
            <button key={t.id} className={`chip ${sel?.id === t.id ? "active" : ""}`} title={t.name} onClick={() => pick(t)}>
              {t.label}
            </button>
          ))}
        </div>
      </div>

      {sel && (
        <p className="lab-loaded">
          Skills loaded from <Link href={`/player/${sel.id}`}>{sel.name}</Link> ({sel.pos}
          {sel.age != null ? `, ${sel.age}` : ""}){sel.label ? ` · ${sel.label.toLowerCase()}` : ""}.
          Card GA/60 {sel.expected.ga60 != null ? f2(sel.expected.ga60) : "—"} · engine WAR{" "}
          {sel.expected.war != null ? f2(sel.expected.war) : "—"}.{touched ? " Sliders adjusted: this is now a hypothetical player." : ""}
        </p>
      )}

      <div className="lab-grid">
        {/* sliders */}
        <div>
          {SLIDERS.map((grp) => (
            <div key={grp.group} className="slider-group">
              <h3>{grp.group}</h3>
              {grp.items.map((s) => {
                const { min, max } = range(s.key);
                return (
                  <div key={s.key} className="slider-row" title={s.sub}>
                    <div className="slider-head">
                      <span>{s.label}</span>
                      <span className="muted">{params[s.key].toFixed(3)} · {pct(s.key, s.invert)}th pctile</span>
                    </div>
                    <input type="range" min={min} max={max} step={(max - min) / 200}
                      value={params[s.key]}
                      onChange={(e) => setParams({ ...params, [s.key]: Number(e.target.value) })} />
                    <div className="slider-sub">{s.sub}</div>
                  </div>
                );
              })}
            </div>
          ))}
          <div className="slider-group">
            <h3>Position</h3>
            <div className="seg" style={{ marginTop: 4 }}>
              {(["F", "D"] as const).map((p) => (
                <button key={p} className={pos === p ? "active" : ""} onClick={() => setPos(p)}>
                  {p === "F" ? "Forward" : "Defenseman"}
                </button>
              ))}
            </div>
            <div className="slider-sub" style={{ marginTop: 6 }}>
              Sets the replacement baseline, the creator mix on his shots, and the danger priced on
              his setups.
            </div>
          </div>
        </div>

        {/* outputs */}
        <div>
          <div className="out-block">
            <h3>His own shots, 5v5 per 60</h3>
            <div className="funnel">
              <div className="frow"><span className="fl">Shots</span><Bar v={v.shots60} max={16} /><span className="fv">{f1(v.shots60)}</span></div>
              <div className="frow"><span className="fl">xG per shot</span><Bar v={v.xgPerShot} max={0.14} /><span className="fv">{f3(v.xgPerShot)}</span></div>
              <div className="frow"><span className="fl">Goals (Scoring)</span><Bar v={v.sc} max={1.3} /><span className="fv">{f2(v.sc)}</span></div>
            </div>
            <div className="out-note">shots60 · goals-per-shot = {f1(v.shots60)} × {f3(v.gbar)} = {f2(v.sc)} goals/60; finishing is inside goals-per-shot</div>
          </div>

          <div className="out-block">
            <h3>Chances created for teammates</h3>
            <div className="funnel">
              <div className="frow"><span className="fl">Extra volume</span><Bar v={Math.max(0, v.pmVol)} max={2.4} color="#4a90d9" /><span className="fv">{f2(v.pmVol)}</span></div>
              <div className="frow"><span className="fl">Extra danger</span><Bar v={Math.max(0, v.pmQual)} max={2.4} color="#9db8d4" /><span className="fv">{f2(v.pmQual)}</span></div>
              <div className="frow"><span className="fl">Playmaking</span><Bar v={Math.max(0, v.pm)} max={2.4} /><span className="fv">{f2(v.pm)} xG/60</span></div>
            </div>
            <div className="out-note">volume prices the extra teammate shots he causes; quality prices the danger he adds to all of them</div>
          </div>

          <div className="out-block">
            <h3>Opponent chances erased</h3>
            <div className="funnel">
              <div className="frow"><span className="fl">They&apos;d get</span><Bar v={v.defAllowed60 + v.df} max={6} color="#b6c4d2" /><span className="fv">{f2(v.defAllowed60 + v.df)}</span></div>
              <div className="frow"><span className="fl">He allows</span><Bar v={v.defAllowed60} max={6} color="#d98c8c" /><span className="fv">{f2(v.defAllowed60)}</span></div>
              <div className="frow"><span className="fl">Defense</span><SignedBar v={v.df} max={1.4} color={v.df >= 0 ? "#3a9d6a" : "#cf4f4f"} /><span className="fv">{signed2(v.df)} xG/60</span></div>
            </div>
            <div className="out-note">his 5-defender share of opponent xG, vs a league-average defender share</div>
          </div>

          <div className="out-block accent">
            <h3>Net Goals Added /60, vs replacement</h3>
            <div className="funnel">
              <div className="frow"><span className="fl">Scoring</span><SignedBar v={v.sc - repl.sc} max={wfMax} color="#2f6cb0" /><span className="fv">{signed2(v.sc - repl.sc)}</span></div>
              <div className="frow"><span className="fl">Playmaking ×κ</span><SignedBar v={kap * (v.pm - repl.pm)} max={wfMax} color="#4a90d9" /><span className="fv">{signed2(kap * (v.pm - repl.pm))}</span></div>
              <div className="frow"><span className="fl">Defense ×κ</span><SignedBar v={kap * (v.df - repl.df)} max={wfMax} color="#3a9d6a" /><span className="fv">{signed2(kap * (v.df - repl.df))}</span></div>
              <div className="frow total"><span className="fl">GA/60</span><SignedBar v={ga.ga60} max={wfMax} color="#1b2a3a" /><span className="fv">{signed2(ga.ga60)}</span></div>
            </div>
            <div className="out-note">
              κ = {kap} prices xG into goals; replacement {pos} produces {f2(repl.sc)} scoring, {f2(repl.pm)} playmaking, {f2(repl.df)} defense
            </div>
          </div>

          <div className="out-block accent">
            <h3>A season of this player</h3>
            <div className="war-row">
              <div>
                <div className="slider-head"><span>5v5 minutes</span><span className="muted">{minutes} min</span></div>
                <input type="range" min={200} max={1600} step={10} value={minutes} onChange={(e) => setMinutes(Number(e.target.value))} />
              </div>
              <div className="war-num">
                <span className="big">{f2(war)}</span>
                <span className="k">wins above replacement</span>
              </div>
            </div>
            <div className="out-note">
              {signed2(ga.ga60)} GA/60 × {minutes} min ÷ {g.constants.goals_per_win}{" "}goals per win.
              Equal footing, 5v5 only; the site&apos;s WAR replays his actual shifts, linemates and
              opposition across 5v5 + special teams{sel && sel.expected.war != null ? ` (${sel.name}: ${f2(sel.expected.war)})` : ""}.
            </div>
          </div>

          {/* deployment */}
          <div className="out-block">
            <h3>Same skills, different linemates</h3>
            <div className="chips" style={{ marginBottom: 8 }}>
              <button className={`chip ${ctx.tmCreate === 0 && ctx.oppDef === 0 ? "active" : ""}`} onClick={() => setCtx({ tmCreate: 0, tmCq: 0, oppDef: 0, oppQd: 0 })}>Average line</button>
              <button className={`chip ${ctx.tmCreate > 0 ? "active" : ""}`} onClick={() => setCtx({ tmCreate: 0.5, tmCq: 0.12, oppDef: 0, oppQd: 0 })}>Elite top line</button>
              <button className={`chip ${ctx.tmCreate < 0 ? "active" : ""}`} onClick={() => setCtx({ tmCreate: -0.3, tmCq: -0.05, oppDef: 0, oppQd: 0 })}>Fourth line</button>
              <button className={`chip ${ctx.oppDef < 0 ? "active" : ""}`} onClick={() => setCtx({ ...ctx, oppDef: -0.3, oppQd: -0.04 })}>Shutdown opposition</button>
            </div>
            <div className="funnel">
              <div className="frow"><span className="fl">His shots/60 on ice</span><Bar v={onIce.shots60} max={20} /><span className="fv">{f1(onIce.shots60)} <span className="muted">({signed2(onIce.shots60 - base.shots60)})</span></span></div>
              <div className="frow"><span className="fl">His goals/60 on ice</span><Bar v={onIce.goals60} max={1.6} /><span className="fv">{f2(onIce.goals60)} <span className="muted">({signed2(onIce.goals60 - base.goals60)})</span></span></div>
            </div>
            <div className="out-note">
              Teammate creation multiplies his shot rate; teammate creation quality lifts every
              shot&apos;s danger. His production moves, his skill sliders don&apos;t: the cards are
              deployment-free, and WAR prices the deployment he actually got.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── replacement + wins reference table ─────────────────────────────────────────────────────────
function ReplacementTable({ g }: { g: GenModelData }) {
  const rows: { key: "sc" | "pm" | "df"; label: string; unit: string }[] = [
    { key: "sc", label: "Scoring", unit: "goals/60" },
    { key: "pm", label: "Playmaking", unit: "xG/60" },
    { key: "df", label: "Defense", unit: "xG erased/60" },
  ];
  return (
    <div className="table-scroll">
    <table className="games repl-table">
      <thead><tr><th>5v5 value</th><th>Replacement F</th><th>Average F</th><th>Replacement D</th><th>Average D</th></tr></thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.key}>
            <td>{r.label} <span className="muted">({r.unit})</span></td>
            <td className="num">{f2(g.replacement_values.F[r.key])}</td>
            <td className="num">{f2(g.baselines.F[r.key])}</td>
            <td className="num">{f2(g.replacement_values.D[r.key])}</td>
            <td className="num">{f2(g.baselines.D[r.key])}</td>
          </tr>
        ))}
      </tbody>
    </table>
    </div>
  );
}

// ── page ────────────────────────────────────────────────────────────────────────────────────────
export default function ModelsPage() {
  const [g, setG] = useState<GenModelData | null>(null);
  const [xg, setXg] = useState<XgModel | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [portErrs, setPortErrs] = useState<string[] | null>(null);

  useEffect(() => {
    fetch("/data/gen_model.json")
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then((data: GenModelData) => { setG(data); setPortErrs(verifyPort(data)); })
      .catch((e) => setError(String(e)));
    fetch("/data/xg_model.json")
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then(setXg)
      .catch((e) => setError(String(e)));
  }, []);

  if (error) return <div className="loading">Failed to load model data ({error}). Run <code>make players</code> and the explainer export.</div>;
  if (!g || !xg) return <div className="loading">Loading the models…</div>;

  const c = g.constants;
  const seasons = `${g.fit_seasons[0]}-${String(g.fit_seasons[0] + 1).slice(-2)} to ${g.fit_seasons[g.fit_seasons.length - 1]}-${String(g.fit_seasons[g.fit_seasons.length - 1] + 1).slice(-2)}`;

  return (
    <div className="mx">
      <div className="panel">
        <h2 style={{ marginTop: 0 }}>How ŷTrick works</h2>
        <p>
          Everything on this site comes from two models fit to raw NHL play-by-play: a shot model
          that prices every unblocked shot in expected goals (xG), and a generative player model
          that infers, from five seasons of shifts ({seasons}, {g.n_players.toLocaleString()} skaters,{" "}
          {xg.n_shots.toLocaleString()}{" "}shots), the skills that produce those shots. Player value is
          then a calculation, not a judgment call: evaluate the skills against a replacement-level
          player, price chances into goals, and goals into wins. This page walks the whole chain and
          lets you drive it: price a shot yourself, then load any player&apos;s fitted skills and
          bend them to see exactly how shots, goals, and WAR respond.
        </p>
      </div>

      <div className="panel">
        <h2>1 · Every shot gets a price</h2>
        <p>
          The foundation is a gradient-boosted model trained on {xg.n_shots.toLocaleString()}{" "}unblocked
          shots. For each one it predicts the chance it becomes a goal from where the shot came from
          (distance, angle, depth), what kind of shot it was, and what happened just before it: a
          cross-slot pass, a rebound, a rush, sustained pressure. The score is calibrated
          out-of-sample, so the league&apos;s {Math.round(xg.metrics.total_xg).toLocaleString()} total
          xG lands on its {xg.metrics.total_goals.toLocaleString()} actual goals. No shooter identity
          goes in: xG is the price of the <em>chance</em>, so that shooting talent can be measured
          against it later.
        </p>
        <ShotBuilder m={xg} />
        <h3 className="mx-h3">What drives the prediction</h3>
        <p className="section-sub">Relative importance (gain) of each input across the model&apos;s trees.</p>
        <Importances m={xg} />
      </div>

      <div className="panel">
        <h2>2 · A game is a stream of stints; skills produce the shots</h2>
        <p>
          Between substitutions, ten skaters and two goalies are on the ice: a <em>stint</em>. The
          generative model writes down how a stint produces shots and goals, as three chained
          stages:
        </p>
        <StageEquations />
        <p>
          Each verb in that process is a per-player skill: <strong>shoot</strong> (his own shot
          volume), <strong>create</strong> (how much more his teammates shoot with him on the ice),{" "}
          <strong>create_qual</strong> (how much more dangerous their shots become),{" "}
          <strong>qshoot</strong> (the danger of his own shots), <strong>def</strong> and{" "}
          <strong>qdef</strong> (the volume and danger he erases from opponents), and{" "}
          <strong>fin</strong>{" "}(goals above what his shot locations predict). Every skill is fit
          jointly across all stints, so it is isolated from linemates, competition, arena
          scorekeeping, and age: McDavid&apos;s wingers shoot more because he is on the ice, and the
          model knows exactly how much of that is him. Skills drift season to season on a random
          walk, which is what the trajectory chart on each player page shows, and small samples
          shrink toward the positional average rather than showing noise as talent.
        </p>
        <p>
          Because the model is generative, it runs in both directions: fit it to attribute real
          goals, or hand it a lineup and simulate. The lab below does the second, live.
        </p>
      </div>

      <div className="panel wide">
        <h2>3 · The player lab</h2>
        <p className="section-sub">
          Load any skater&apos;s fitted skills, then drag. Every number recomputes from the model&apos;s
          own equations, in your browser.
        </p>
        <PlayerLab g={g} />
        {portErrs != null && (
          <p className="port-note">
            {portErrs.length === 0
              ? `Checked on load: this page's math reproduces the pipeline's values for all ${g.players.length} shipped players.`
              : `Port check FAILED for ${portErrs.length} players: ${portErrs[0]}`}
          </p>
        )}
      </div>

      <div className="panel">
        <h2>4 · Replacement level, and turning goals into wins</h2>
        <p>
          Skill rates only become <em>value</em> against a baseline. The baseline here is the{" "}
          <strong>replacement-level player</strong>: the TOI-weighted average of everyone ranked in
          the {g.repl_band_pct[0]}th to {g.repl_band_pct[1]}th percentile at his position, the
          freely-available call-up a team can always find. Net Goals Added /60 evaluates the value
          equations for the player and for that archetype and takes the difference, pricing xG into
          goals at κ = {c.kappa}{" "}(the league&apos;s fitted goals-per-xG). WAR then swaps him out of
          every shift he actually played: expected goals with him, minus expected goals with the
          replacement in his slot, accumulated over the real season and divided by{" "}
          {c.goals_per_win}{" "}goals per win.
        </p>
        <ValueEquations kappa={c.kappa} gpw={c.goals_per_win} />
        <ReplacementTable g={g} />
        <p className="section-sub" style={{ marginTop: 10 }}>
          GA/60 answers &quot;how good is he, on equal footing?&quot; (deployment-free). WAR answers
          &quot;how much did he actually add?&quot; (deployment-in: real minutes, real linemates,
          real opposition, 5v5 + power play + penalty kill). A durable decent player can out-WAR a
          brilliant part-timer; that is the point, not a bug.
        </p>
      </div>

      <div className="panel">
        <h2>5 · What it gets right, and what it doesn&apos;t price</h2>
        <p>
          The shot model is scored out-of-fold: shots the model never saw during training. Its
          predicted rates track the real goal rates across the full danger range, and the totals
          reconcile:
        </p>
        <Calibration m={xg} />
        <p>
          The player model is validated on held-out seasons: fitted skills predict next season&apos;s
          shot rates at correlation ≈ 0.86, and its creation skill predicts next season&apos;s{" "}
          <em>teammate</em>{" "}shot rates better than raw counting stats, the test a lineup-aware
          decomposition has to win. The value engine&apos;s expected goals reconcile to the league&apos;s
          actual goals within about 1%, with no correction factors.
        </p>
        <p>
          Honest gaps: penalties drawn and taken are valued on the site but sit outside WAR;
          shorthanded goals, empty-netters, 5v3s and 3v3 overtime are unpriced (players who score
          them get a note on the WAR card); goalie value uses the simpler GSAx model. Replacement
          level runs deeper than public models, so WAR totals here are larger: compare WAR within
          this site, never across models.
        </p>
      </div>
    </div>
  );
}
