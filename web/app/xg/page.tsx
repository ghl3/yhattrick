"use client";
import { useEffect, useMemo, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { XgModel } from "@/lib/types";
import { seasonLabel } from "@/lib/format";

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

const STRENGTH_LABELS: Record<string, string> = { PP: "Power play", EV: "Even strength", PK: "Penalty kill" };
const SHOT_LABELS: Record<string, string> = {
  wrist: "Wrist", snap: "Snap", slap: "Slap", "tip-in": "Tip-in", backhand: "Backhand", deflected: "Deflected",
};

// danger ramp: pale (low xG) -> amber -> red (high xG)
function heat(xg: number, cap = 0.45): string {
  const t = Math.min(1, Math.max(0, xg / cap));
  const lerp = (a: number, b: number, u: number) => Math.round(a + (b - a) * u);
  if (t < 0.5) {
    const u = t / 0.5; // pale blue -> amber
    return `rgb(${lerp(238, 245, u)},${lerp(245, 180, u)},${lerp(251, 0, u)})`;
  }
  const u = (t - 0.5) / 0.5; // amber -> red
  return `rgb(${lerp(245, 200, u)},${lerp(180, 30, u)},${lerp(0, 30, u)})`;
}

// --- offensive-zone heatmap (attacking net on the right at x=89) ---
const FT = 4.2;
const X0 = 24, X1 = 100; // show from just behind the blue line to the end boards
const W = (X1 - X0) * FT;
const H = 85 * FT;
const px = (x: number) => (x - X0) * FT;
const py = (y: number) => (y + 42.5) * FT;
const RED = "#d98c8c", BLUE = "#9fc1e6";

function Heatmap({ m }: { m: XgModel }) {
  const { x, y, combos, shot_types, strengths } = m.heatmap;
  const [shot, setShot] = useState("wrist");
  const [strength, setStrength] = useState("EV");
  const [rebound, setRebound] = useState(false);
  const [rush, setRush] = useState(false);
  const grid = combos[`${shot}|${rebound ? 1 : 0}|${rush ? 1 : 0}|${strength}`] ?? [];
  const cw = (x.length > 1 ? x[1] - x[0] : 2) * FT;
  const ch = (y.length > 1 ? y[1] - y[0] : 2) * FT;

  return (
    <div>
      <div className="xg-controls">
        <label>Shot type
          <select value={shot} onChange={(e) => setShot(e.target.value)}>
            {shot_types.map((s) => <option key={s} value={s}>{SHOT_LABELS[s] ?? s}</option>)}
          </select>
        </label>
        <label>Strength
          <select value={strength} onChange={(e) => setStrength(e.target.value)}>
            {strengths.map((s) => <option key={s} value={s}>{STRENGTH_LABELS[s] ?? s}</option>)}
          </select>
        </label>
        <label className="chk"><input type="checkbox" checked={rebound} onChange={(e) => setRebound(e.target.checked)} /> Rebound</label>
        <label className="chk"><input type="checkbox" checked={rush} onChange={(e) => setRush(e.target.checked)} /> Rush</label>
      </div>

      <svg className="xg-rink" viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: W }}>
        {/* xG cells */}
        {grid.map((rowArr, yi) =>
          rowArr.map((v, xi) => (
            <rect key={`${xi},${yi}`} x={px(x[xi]) - cw / 2} y={py(y[yi]) - ch / 2}
              width={cw + 0.5} height={ch + 0.5} fill={heat(v)} />
          ))
        )}
        {/* rink markings */}
        <rect x={0.5} y={0.5} width={W - 1} height={H - 1} rx={26 * FT} ry={26 * FT} fill="none" stroke="#cfe0f0" />
        <line x1={px(25)} y1={0} x2={px(25)} y2={H} stroke={BLUE} strokeWidth={2} />
        <line x1={px(89)} y1={py(-39)} x2={px(89)} y2={py(39)} stroke={RED} strokeWidth={1.5} />
        <rect x={px(89)} y={py(-2)} width={6} height={py(2) - py(-2)} fill="none" stroke={RED} strokeWidth={1.5} />
        <path d={`M ${px(89)} ${py(-4)} Q ${px(89) - 6 * FT} ${py(0)} ${px(89)} ${py(4)} Z`} fill="none" stroke={BLUE} strokeWidth={1} />
        <circle cx={px(69)} cy={py(22)} r={15 * FT} fill="none" stroke={RED} strokeWidth={1} opacity={0.6} />
        <circle cx={px(69)} cy={py(-22)} r={15 * FT} fill="none" stroke={RED} strokeWidth={1} opacity={0.6} />
        <circle cx={px(69)} cy={py(22)} r={2} fill={RED} /><circle cx={px(69)} cy={py(-22)} r={2} fill={RED} />
      </svg>

      <div className="xg-legend">
        <span>0.0</span>
        <span className="bar" style={{ background: `linear-gradient(90deg, ${heat(0)}, ${heat(0.11)}, ${heat(0.22)}, ${heat(0.33)}, ${heat(0.45)})` }} />
        <span>0.45+ xG</span>
      </div>
    </div>
  );
}

export default function XgPage() {
  const [m, setM] = useState<XgModel | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`/data/xg_model.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then(setM)
      .catch((e) => setError(String(e)));
  }, []);

  const reliData = useMemo(() => {
    if (!m) return [];
    const max = Math.max(...m.reliability.map((b) => b.pred), 0.4);
    const pts = m.reliability.map((b) => ({ pred: b.pred, ours: b.obs }));
    return [{ pred: 0, ideal: 0 }, ...pts, { pred: max, ideal: max }];
  }, [m]);

  if (error) return <div className="loading">Failed to load the xG model ({error}). Run <code>make xg</code>.</div>;
  if (!m) return <div className="loading">Loading xG model…</div>;

  const mt = m.metrics;
  const maxGain = Math.max(...m.importances.map((i) => i.gain));

  return (
    <div>
      <div className="panel">
        <div className="player-head">
          <h2 className="player-name">Expected goals (xG) model</h2>
          <span className="player-meta">
            {seasonLabel(m.seasons[0])} – {seasonLabel(m.seasons[m.seasons.length - 1])} ·{" "}
            {m.n_shots.toLocaleString()} unblocked shots · {m.n_goals.toLocaleString()} goals
          </span>
        </div>
        <p className="section-sub">
          A gradient-boosted model that scores each unblocked shot by its chance of going in, from shot
          geometry and pre-shot context in the NHL play-by-play. Below: the danger map, how well the
          predictions match real goal rates, and what the model leans on.
        </p>
      </div>

      <div className="panel">
        <h2>Shot danger map</h2>
        <p className="section-sub">Predicted xG by location for the chosen shot type and situation. Attacking net at right.</p>
        <Heatmap m={m} />
      </div>

      <div className="xg-two">
        <div className="panel">
          <h2>Calibration</h2>
          <p className="section-sub">Predicted xG vs. the actual goal rate, in bins. On the dashed line = perfectly calibrated.</p>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={reliData} margin={{ top: 8, right: 16, bottom: 18, left: 0 }}>
                <CartesianGrid stroke="#e7f1fb" />
                <XAxis dataKey="pred" type="number" domain={[0, "dataMax"]} tickFormatter={(v) => v.toFixed(2)}
                  tick={{ fontSize: 11, fill: "#6b7e90" }} label={{ value: "predicted xG", position: "insideBottom", offset: -8, fontSize: 11, fill: "#6b7e90" }} />
                <YAxis type="number" domain={[0, "dataMax"]} tickFormatter={(v) => v.toFixed(2)}
                  tick={{ fontSize: 11, fill: "#6b7e90" }} width={44} label={{ value: "actual rate", angle: -90, position: "insideLeft", fontSize: 11, fill: "#6b7e90" }} />
                <Tooltip formatter={(v: unknown) => (typeof v === "number" ? v.toFixed(3) : String(v))} />
                <Line dataKey="ideal" name="ideal" stroke="#9aa7b4" strokeWidth={1} strokeDasharray="4 4" dot={false} connectNulls />
                <Line dataKey="ours" name="our xG" stroke="#2f6cb0" strokeWidth={2.5} dot={{ r: 3 }} connectNulls />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="xg-key">
            <span><i style={{ background: "#2f6cb0" }} /> our xG</span>
            <span><i style={{ background: "#9aa7b4" }} /> perfect</span>
          </div>
        </div>

        <div className="panel">
          <h2>Accuracy</h2>
          <p className="section-sub">Out-of-fold, on all {mt.n.toLocaleString()} modeled shots.</p>
          <table className="games xg-metrics">
            <tbody>
              <tr><td>AUC (discrimination)</td><td className="num">{mt.auc.toFixed(4)}</td></tr>
              <tr><td>Log-loss</td><td className="num">{mt.logloss.toFixed(4)}</td></tr>
              <tr><td>Brier score</td><td className="num">{mt.brier.toFixed(4)}</td></tr>
              <tr>
                <td>Total xG vs. {mt.total_goals.toLocaleString()} goals</td>
                <td className="num">{Math.round(mt.total_xg).toLocaleString()}</td>
              </tr>
            </tbody>
          </table>
          <p className="muted card-note">
            AUC measures ranking dangerous shots above safe ones; log-loss and Brier reward sharp,
            calibrated probabilities. Total xG lands on the real goal count.
          </p>
        </div>
      </div>

      <div className="panel">
        <h2>What drives the prediction</h2>
        <p className="section-sub">Relative importance (gain) of each input across the model&apos;s trees.</p>
        <div className="xg-imp">
          {m.importances.map((i) => (
            <div key={i.feature} className="xg-imp-row">
              <span className="xg-imp-name">{FEATURE_LABELS[i.feature] ?? i.feature}</span>
              <span className="xg-imp-bar"><span style={{ width: `${(i.gain / maxGain) * 100}%` }} /></span>
              <span className="xg-imp-val">{(i.gain * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
