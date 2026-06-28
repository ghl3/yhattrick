"use client";
import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import {
  CartesianGrid, Line, LineChart, PolarAngleAxis, PolarGrid, PolarRadiusAxis,
  Radar, RadarChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import type { GameLogRow, IndividualKey, MetricKey, OniceKey, PlayerDetail, PlayerHeat, SeasonRow } from "@/lib/types";
import { mmss, pctColor } from "@/lib/format";

const seasonLabel = (s: number) => `${s}-${String(s + 1).slice(-2)}`;
const heightStr = (inches?: number | null) => (inches ? `${Math.floor(inches / 12)}'${inches % 12}"` : null);
const ageFrom = (iso?: string | null): number | null => {
  if (!iso) return null;
  const b = new Date(iso), now = new Date();
  let a = now.getFullYear() - b.getFullYear();
  const m = now.getMonth() - b.getMonth();
  if (m < 0 || (m === 0 && now.getDate() < b.getDate())) a--;
  return a;
};

const IMPACT: { key: MetricKey; name: string; explain: string }[] = [
  { key: "ev_off", name: "Even-Strength Offense", explain: "Expected goals he adds per 60 at 5-on-5, vs. an average player." },
  { key: "ev_def", name: "Even-Strength Defense", explain: "Change in expected goals allowed per 60 at 5-on-5, vs. an average player — negative means fewer." },
  { key: "pp_off", name: "Power-Play Offense", explain: "Expected goals he adds per 60 on the power play, vs. an average player." },
  { key: "pk_def", name: "Penalty-Kill Defense", explain: "Change in expected goals allowed per 60 on the penalty kill, vs. an average player — negative means fewer." },
];

// on-ice (raw, descriptive) metrics — the team's rate while the player is on the ice
const num3 = (v: number) => v.toFixed(3);
const num2 = (v: number) => v.toFixed(2);
const num1 = (v: number) => v.toFixed(1);
// 5-on-5 unless the name says otherwise, so no "EV" prefix is needed
const ONICE: { key: OniceKey; name: string; explain: string; fmt: (v: number) => string }[] = [
  { key: "ev_xgf60", name: "Expected Goals For", explain: "Team's expected goals per 60 at 5-on-5 when he's on the ice.", fmt: num2 },
  { key: "ev_xga60", name: "Expected Goals Against", explain: "Team's expected goals allowed per 60 at 5-on-5 when he's on the ice.", fmt: num2 },
  { key: "ev_cf60", name: "Shot Attempts For", explain: "Team's shot attempts (Corsi) per 60 at 5-on-5 when he's on the ice.", fmt: num1 },
  { key: "ev_ca60", name: "Shot Attempts Against", explain: "Opponent shot attempts per 60 at 5-on-5 when he's on the ice.", fmt: num1 },
  { key: "pp_xgf60", name: "Power-Play Expected Goals For", explain: "Team's expected goals per 60 on the power play when he's on the ice.", fmt: num2 },
  { key: "pk_xga60", name: "Penalty-Kill Expected Goals Against", explain: "Team's expected goals allowed per 60 on the penalty kill when he's on the ice.", fmt: num2 },
];

// individual (on-puck) metrics — the player's own shooting & production, all situations
const INDIV: { key: IndividualKey; name: string; explain: string; fmt: (v: number) => string; signed?: boolean; ci?: boolean }[] = [
  { key: "shots60", name: "Shot Rate", explain: "His own unblocked shots per 60.", fmt: num1 },
  { key: "xg_per_shot", name: "Shot Quality", explain: "Average danger of his shots (xG per shot).", fmt: num3 },
  { key: "ixg60", name: "Expected Goal Rate", explain: "Expected goals from his own shots per 60.", fmt: num2 },
  { key: "fin_per100", name: "Finishing", explain: "Goals above expected per 100 of his shots.", fmt: num2, signed: true, ci: true },
  { key: "g60", name: "Goal Rate", explain: "His goals per 60 (all situations).", fmt: num2 },
  { key: "a60", name: "Assist Rate", explain: "His assists per 60 (all situations).", fmt: num2 },
  { key: "pen_drawn60", name: "Penalty Draw Rate", explain: "Penalties he drew per 60.", fmt: num2 },
  { key: "pen_taken60", name: "Penalty Take Rate", explain: "Penalties he took per 60.", fmt: num2 },
];

// per-season columns; `graph` ones are clickable to chart over time
type Col = { key: string; label: string; title: string; get: (r: SeasonRow) => number | null; fmt?: (v: number) => string; graph: boolean };
const COLS: Col[] = [
  { key: "gp", label: "GP", title: "Games played", get: (r) => r.gp, graph: true },
  { key: "toi_min", label: "TOI", title: "Time on ice (min)", get: (r) => r.toi_min, fmt: (v) => String(Math.round(v)), graph: true },
  { key: "g", label: "G", title: "Goals", get: (r) => r.g, graph: true },
  { key: "a", label: "A", title: "Assists", get: (r) => r.a1 + r.a2, graph: true },
  { key: "points", label: "P", title: "Points", get: (r) => r.points, graph: true },
  { key: "sog", label: "SOG", title: "Shots on goal", get: (r) => r.sog, graph: true },
  { key: "hits", label: "Hits", title: "Hits", get: (r) => r.hits, graph: true },
  { key: "takeaways", label: "Tk", title: "Takeaways", get: (r) => r.takeaways, graph: true },
  { key: "giveaways", label: "Gv", title: "Giveaways", get: (r) => r.giveaways, graph: true },
  { key: "pen_drawn", label: "PenD", title: "Penalties drawn", get: (r) => r.pen_drawn, graph: true },
  { key: "ev_off", label: "EV O", title: "EV offense impact (xGF/60)", get: (r) => r.ev_off ?? null, fmt: (v) => v.toFixed(2), graph: true },
  { key: "ev_def", label: "EV D", title: "EV defense impact (xGA/60)", get: (r) => r.ev_def ?? null, fmt: (v) => v.toFixed(2), graph: true },
  { key: "pp_off", label: "PP", title: "PP offense impact (xGF/60)", get: (r) => r.pp_off ?? null, fmt: (v) => v.toFixed(2), graph: true },
  { key: "pk_def", label: "PK", title: "PK defense impact (xGA/60)", get: (r) => r.pk_def ?? null, fmt: (v) => v.toFixed(2), graph: true },
  { key: "shots60", label: "Sh/60", title: "Unblocked shots per 60", get: (r) => r.shots60 ?? null, fmt: (v) => v.toFixed(1), graph: true },
  { key: "xg_per_shot", label: "xG/sh", title: "Average shot quality (xG per shot)", get: (r) => r.xg_per_shot ?? null, fmt: (v) => v.toFixed(3), graph: true },
  { key: "fin_per100", label: "Fin", title: "Finishing: goals above expected per 100 shots", get: (r) => r.fin_per100 ?? null, fmt: (v) => v.toFixed(2), graph: true },
];

// shared box: a colored bar with the metric NAME and its percentile ("54% among forwards"), then
// the raw value + error and a short description below. `value` is the formatted number node.
function BoxShell({ name, pctile, group, explain, value, footer }: {
  name: string; pctile: number | null; group: string; explain: string;
  value: React.ReactNode; footer?: React.ReactNode;
}) {
  const has = pctile != null;
  const grp = group === "D" ? "defensemen" : "forwards";
  return (
    <div className="metric-box">
      <div className="mb-bar" style={{ background: has ? pctColor(pctile) : "var(--accent-softer)" }}>
        <span className="mb-head">{name}</span>
        <span className="mb-rank">
          {has ? <><span className="mb-pctile">{Math.round(pctile!)}%</span><span className="mb-vs">among {grp}</span></>
               : <span className="mb-vs">no qualifying ice time</span>}
        </span>
      </div>
      <div className="mb-body">
        <div className="mb-val">{value}{footer}</div>
        <div className="mb-blurb">{explain}</div>
      </div>
    </div>
  );
}

// modeled, isolated impact: signed per-60 delta with a 95% CI
function MetricBox({ name, explain, v, se, toi, pctile, group }: {
  name: string; explain: string; v: number | null; se: number | null; toi: number | null; pctile: number | null; group: string;
}) {
  const has = v != null && pctile != null;
  return (
    <BoxShell name={name} pctile={has ? pctile : null} group={group} explain={explain}
      value={has ? <>{v! >= 0 ? "+" : ""}{v!.toFixed(3)} <span className="mb-ci">± {(1.96 * (se ?? 0)).toFixed(3)}</span></> : <span className="muted">—</span>}
      footer={toi ? <span className="mb-toi"> · {Math.round(toi)} min</span> : null} />
  );
}

// descriptive rate: plain value (optionally signed, optionally with a 95% CI for finishing)
function OniceBox({ name, explain, v, pctile, group, fmt, se, signed }: {
  name: string; explain: string; v: number | null; pctile: number | null; group: string;
  fmt: (v: number) => string; se?: number; signed?: boolean;
}) {
  const has = v != null && pctile != null;
  return (
    <BoxShell name={name} pctile={has ? pctile : null} group={group} explain={explain}
      value={v != null
        ? <>{signed && v >= 0 ? "+" : ""}{fmt(v)}{se != null && <span className="mb-ci"> ± {(1.96 * se).toFixed(2)}</span>}</>
        : <span className="muted">—</span>} />
  );
}

// --- shot map (offensive half, attacking net on the right at x=89); mirrors the xG page rink ---
const HFT = 4.2;
const HX0 = 24, HX1 = 100;
const HW = (HX1 - HX0) * HFT;
const HH = 85 * HFT;
const hpx = (x: number) => (x - HX0) * HFT;
const hpy = (y: number) => (y + 42.5) * HFT;
const HRED = "#d98c8c", HBLUE = "#9fc1e6";

// pale -> amber -> red ramp (t in 0..1)
function ramp(t: number): string {
  t = Math.min(1, Math.max(0, t));
  const lerp = (a: number, b: number, u: number) => Math.round(a + (b - a) * u);
  if (t < 0.5) {
    const u = t / 0.5;
    return `rgb(${lerp(238, 245, u)},${lerp(245, 180, u)},${lerp(251, 0, u)})`;
  }
  const u = (t - 0.5) / 0.5;
  return `rgb(${lerp(245, 200, u)},${lerp(180, 30, u)},${lerp(0, 30, u)})`;
}

type HeatMode = "shots" | "goals" | "shotpct" | "xg";
const HEAT_MODES: { key: HeatMode; label: string }[] = [
  { key: "shots", label: "Shots" },
  { key: "goals", label: "Goals" },
  { key: "shotpct", label: "Shot %" },
  { key: "xg", label: "Avg xG" },
];

function ShotMap({ heat }: { heat: PlayerHeat }) {
  const [mode, setMode] = useState<HeatMode>("shots");
  const { x, y, s, g, xg } = heat;
  const cw = (x.length > 1 ? x[1] - x[0] : 3) * HFT;
  const ch = (y.length > 1 ? y[1] - y[0] : 3) * HFT;

  const maxS = useMemo(() => Math.max(0, ...s.flat()), [s]);
  const maxG = useMemo(() => Math.max(0, ...g.flat()), [g]);
  const eps = 0.06 * maxS;                                  // hide ratios in barely-shot-from areas
  const RATIO_CAP = mode === "shotpct" ? 0.3 : 0.32;       // goals/shot and xG/shot both top out ~0.3

  // each cell -> a 0..1 color intensity, or null to leave blank (sparse cells in ratio modes)
  const intensity = (xi: number, yi: number): number | null => {
    const sv = s[yi][xi];
    if (mode === "shots") return maxS > 0 ? sv / maxS : 0;
    if (mode === "goals") return maxG > 0 ? g[yi][xi] / maxG : 0;
    if (sv < eps) return null;
    const r = (mode === "shotpct" ? g[yi][xi] : xg[yi][xi]) / sv;
    return r / RATIO_CAP;
  };

  const legend =
    mode === "shots" ? ["fewer", "more shots"] :
    mode === "goals" ? ["fewer", "more goals"] :
    mode === "shotpct" ? ["0%", "30%+ score"] : ["0", "0.30+ xG/shot"];

  return (
    <div>
      <div className="seg heat-seg">
        {HEAT_MODES.map((m) => (
          <button key={m.key} className={mode === m.key ? "active" : ""} onClick={() => setMode(m.key)}>
            {m.label}
          </button>
        ))}
      </div>
      <svg className="xg-rink" viewBox={`0 0 ${HW} ${HH}`} width="100%" style={{ maxWidth: HW }}>
        {y.map((_, yi) =>
          x.map((_, xi) => {
            const t = intensity(xi, yi);
            if (t == null || t <= 0.001) return null;
            return (
              <rect key={`${xi},${yi}`} x={hpx(x[xi]) - cw / 2} y={hpy(y[yi]) - ch / 2}
                width={cw + 0.5} height={ch + 0.5} fill={ramp(t)} />
            );
          })
        )}
        {/* rink markings */}
        <rect x={0.5} y={0.5} width={HW - 1} height={HH - 1} rx={26 * HFT} ry={26 * HFT} fill="none" stroke="#cfe0f0" />
        <line x1={hpx(25)} y1={0} x2={hpx(25)} y2={HH} stroke={HBLUE} strokeWidth={2} />
        <line x1={hpx(89)} y1={hpy(-39)} x2={hpx(89)} y2={hpy(39)} stroke={HRED} strokeWidth={1.5} />
        <rect x={hpx(89)} y={hpy(-2)} width={6} height={hpy(2) - hpy(-2)} fill="none" stroke={HRED} strokeWidth={1.5} />
        <path d={`M ${hpx(89)} ${hpy(-4)} Q ${hpx(89) - 6 * HFT} ${hpy(0)} ${hpx(89)} ${hpy(4)} Z`} fill="none" stroke={HBLUE} strokeWidth={1} />
        <circle cx={hpx(69)} cy={hpy(22)} r={15 * HFT} fill="none" stroke={HRED} strokeWidth={1} opacity={0.6} />
        <circle cx={hpx(69)} cy={hpy(-22)} r={15 * HFT} fill="none" stroke={HRED} strokeWidth={1} opacity={0.6} />
        <circle cx={hpx(69)} cy={hpy(22)} r={2} fill={HRED} /><circle cx={hpx(69)} cy={hpy(-22)} r={2} fill={HRED} />
      </svg>
      <div className="xg-legend">
        <span>{legend[0]}</span>
        <span className="bar" style={{ background: `linear-gradient(90deg, ${ramp(0)}, ${ramp(0.25)}, ${ramp(0.5)}, ${ramp(0.75)}, ${ramp(1)})` }} />
        <span>{legend[1]}</span>
      </div>
    </div>
  );
}

// percentile radar across the headline skills (0 = no qualifying ice time / bottom, 100 = best)
function ProfileRadar({ p }: { p: PlayerDetail }) {
  const data = [
    { axis: "EV Off", v: p.impact.ev_off.pct },
    { axis: "EV Def", v: p.impact.ev_def.pct },
    { axis: "PP Off", v: p.impact.pp_off.pct },
    { axis: "PK Def", v: p.impact.pk_def.pct },
    { axis: "Shot Vol", v: p.individual.shots60.pct },
    { axis: "Shot Qual", v: p.individual.xg_per_shot.pct },
    { axis: "Finishing", v: p.individual.fin_per100.pct },
    { axis: "Playmaking", v: p.individual.a60.pct },
  ].map((d) => ({ ...d, v: d.v ?? 0 }));
  return (
    <ResponsiveContainer width="100%" height={360}>
      <RadarChart data={data} outerRadius="70%">
        <PolarGrid stroke="#e7f1fb" />
        <PolarAngleAxis dataKey="axis" tick={{ fontSize: 11, fill: "#6b7e90" }} />
        <PolarRadiusAxis domain={[0, 100]} tickCount={5} angle={90} tick={{ fontSize: 9, fill: "#b6c4d2" }} />
        <Radar dataKey="v" stroke="#2f6cb0" fill="#4a90d9" fillOpacity={0.45} />
        <Tooltip formatter={(v: unknown) => `${Math.round(v as number)}th pct`} />
      </RadarChart>
    </ResponsiveContainer>
  );
}

function GameLog({ games }: { games: GameLogRow[] }) {
  const router = useRouter();
  if (!games?.length) return null;
  return (
    <div className="panel">
      <h2>Game log</h2>
      <p className="section-sub">Every game, most recent first — regular season and playoffs.</p>
      <div className="gamelog-wrap">
        <table className="players gamelog">
          <thead>
            <tr>
              <th>Date</th>
              <th>Team</th>
              <th>Opp</th>
              <th>Result</th>
              <th title="Time on ice">TOI</th>
              <th title="Goals">G</th>
              <th title="Assists">A</th>
              <th title="Points">P</th>
              <th title="Shots on goal">SOG</th>
              <th title="Penalties taken">PEN</th>
            </tr>
          </thead>
          <tbody>
            {games.map((g) => (
              <tr key={g.game_id} className="rowlink" onClick={() => router.push(`/game/${g.game_id}`)}>
                <td>{g.date ?? "—"}</td>
                <td><Link href={`/team/${g.team}`} prefetch={false} onClick={(e) => e.stopPropagation()}>{g.team}</Link></td>
                <td className="muted">{g.home ? "vs" : "@"} <Link href={`/team/${g.opp}`} prefetch={false} onClick={(e) => e.stopPropagation()}>{g.opp}</Link></td>
                <td className={`result result-${(g.result ?? "").toLowerCase()}`}>
                  {g.result}{g.gf != null && g.ga != null ? ` ${g.gf}–${g.ga}` : ""}
                </td>
                <td className="num">{mmss(g.toi_s)}</td>
                <td className="num">{g.g}</td>
                <td className="num">{g.a}</td>
                <td className="num">{g.p}</td>
                <td className="num">{g.sog}</td>
                <td className="num">{g.pen}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function Player() {
  const params = useParams<{ id: string }>();
  const id = params.id;
  const [p, setP] = useState<PlayerDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [stat, setStat] = useState("points");

  useEffect(() => {
    setP(null);
    fetch(`/data/player/${id}.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then(setP)
      .catch((e) => setError(String(e)));
  }, [id]);

  const col = COLS.find((c) => c.key === stat)!;
  const chartData = useMemo(
    () => (p ? p.per_season.map((r) => ({ season: seasonLabel(r.season), value: col.get(r) })) : []),
    [p, col]
  );

  if (error) return <div className="loading">Failed to load player {id} ({error}).</div>;
  if (!p) return <div className="loading">Loading player…</div>;

  const last = p.seasons[p.seasons.length - 1];

  return (
    <div>
      <Link className="backlink" href="/players">← all players</Link>

      <div className="panel">
        <div className="player-hero">
          {p.bio?.headshot ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img className="player-mug" src={p.bio.headshot} alt={p.name} loading="lazy" />
          ) : null}
          <div className="player-hero-main">
            <h2 className="player-name">{p.name}</h2>
            <span className="player-meta">
              {p.bio?.number != null ? `#${p.bio.number} · ` : ""}
              {p.pos} · {p.group === "D" ? "Defenseman" : "Forward"} · {p.teams.join(", ")}
            </span>
            <div className="bio-row">
              {p.bio?.shoots && <span><b>Shoots</b> {p.bio.shoots}</span>}
              {heightStr(p.bio?.height_in) && <span><b>Ht</b> {heightStr(p.bio?.height_in)}</span>}
              {p.bio?.weight_lb && <span><b>Wt</b> {p.bio.weight_lb} lb</span>}
              {ageFrom(p.bio?.birth_date) != null && <span><b>Age</b> {ageFrom(p.bio?.birth_date)}</span>}
              {p.bio?.birth_city && (
                <span><b>Born</b> {p.bio.birth_city}{p.bio.birth_state ? `, ${p.bio.birth_state}` : ""}{p.bio.birth_country ? `, ${p.bio.birth_country}` : ""}</span>
              )}
              {p.bio?.draft_overall ? <span><b>Draft</b> {p.bio.draft_year} · #{p.bio.draft_overall}</span> : null}
            </div>
          </div>
        </div>
        <div className="statgrid">
          <div className="stat"><span className="v">{p.gp}</span><span className="k">games</span></div>
          <div className="stat"><span className="v">{p.g}</span><span className="k">goals</span></div>
          <div className="stat"><span className="v">{p.a}</span><span className="k">assists</span></div>
          <div className="stat"><span className="v">{p.points}</span><span className="k">points</span></div>
          <div className="stat"><span className="v">{seasonLabel(p.seasons[0])} – {seasonLabel(last)}</span><span className="k">regular seasons</span></div>
        </div>
      </div>

      <div className="panel">
        <h2>Isolated impact</h2>
        <p className="section-sub">His effect on the game, adjusted for the teammates and competition he played with. ± is a 95% range.</p>
        <div className="metric-grid">
          {IMPACT.map((m) => {
            const d = p.impact[m.key];
            return <MetricBox key={m.key} name={m.name} explain={m.explain} v={d.v} se={d.se} toi={d.toi} pctile={d.pct} group={p.group} />;
          })}
        </div>
      </div>

      <div className="panel">
        <h2>Individual Rates</h2>
        <p className="section-sub">His own play with the puck — all situations, per 60 minutes unless noted.</p>
        <div className="metric-grid">
          {INDIV.map((m) => {
            const d = p.individual[m.key];
            return <OniceBox key={m.key} name={m.name} explain={m.explain} v={d.v} pctile={d.pct} group={p.group} fmt={m.fmt} se={m.ci ? d.se : undefined} signed={m.signed} />;
          })}
        </div>
      </div>

      <div className="panel">
        <h2>On-Ice Team Rates</h2>
        <p className="section-sub">His team&apos;s rates while he was on the ice, not adjusted for teammates — 5-on-5 unless a box says power play or penalty kill.</p>
        <div className="metric-grid">
          {ONICE.map((m) => {
            const d = p.onice[m.key];
            return <OniceBox key={m.key} name={m.name} explain={m.explain} v={d.v} pctile={d.pct} group={p.group} fmt={m.fmt} />;
          })}
        </div>
      </div>

      <div className="panel">
        <h2>By season — click a stat to chart it</h2>
        <div className="chart-wrap">
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={chartData} margin={{ top: 8, right: 16, bottom: 4, left: -8 }}>
              <CartesianGrid stroke="#e7f1fb" />
              <XAxis dataKey="season" tick={{ fontSize: 12, fill: "#6b7e90" }} />
              <YAxis tick={{ fontSize: 12, fill: "#6b7e90" }} width={48} />
              <Tooltip formatter={(v: unknown) => (typeof v === "number" && col.fmt ? col.fmt(v) : String(v))} />
              <Line type="monotone" dataKey="value" name={col.label} stroke="#2f6cb0" strokeWidth={2} dot={{ r: 3 }} connectNulls />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <table className="players season-table">
          <thead>
            <tr>
              <th>Season</th>
              <th>Team</th>
              {COLS.map((c) => (
                <th key={c.key} className={`sortable ${stat === c.key ? "graphed" : ""}`} title={c.title} onClick={() => setStat(c.key)}>
                  {c.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {[...p.per_season].reverse().map((r) => (
              <tr key={r.season}>
                <td>{seasonLabel(r.season)}</td>
                <td>{r.team}</td>
                {COLS.map((c) => {
                  const v = c.get(r);
                  return (
                    <td key={c.key} className={stat === c.key ? "graphed" : ""}>
                      {v == null ? "—" : c.fmt ? c.fmt(v) : v}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="panel">
        <h2>Top 5v5 linemates</h2>
        <div className="linemates">
          {p.linemates.map((lm) => (
            <Link key={lm.id} href={`/player/${lm.id}`} className="linemate">
              <span className="lm-name">{lm.name}</span>
              <span className="lm-toi">{Math.round(lm.toi_min)} min 5v5</span>
            </Link>
          ))}
        </div>
      </div>

      <div className="panel">
        <h2>Shot map &amp; profile</h2>
        <div className="viz-row">
          {p.heat && (
            <div className="viz-cell">
              <p className="section-sub">Where he shoots from, both ends folded to one — toggle shot volume, goals, shooting %, and average shot quality (xG). Attacking net at right.</p>
              <ShotMap heat={p.heat} />
            </div>
          )}
          <div className="viz-cell">
            <p className="section-sub">Percentile rank across key skills, within position ({p.group === "D" ? "defensemen" : "forwards"}). Further out is better.</p>
            <ProfileRadar p={p} />
          </div>
        </div>
      </div>

      <GameLog games={p.games ?? []} />
    </div>
  );
}
