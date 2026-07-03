"use client";
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import {
  CartesianGrid, ComposedChart, Line, LineChart, PolarAngleAxis, PolarGrid, PolarRadiusAxis,
  Radar, RadarChart, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis,
} from "recharts";
import type {
  AnyPlayerDetail, GameLogRow, GoalieDetail, GoalieMetricKey, GoalieSeasonRow,
  IndividualKey, OniceKey, PlayerBio, PlayerDetail, PlayerHeat, SeasonRow,
} from "@/lib/types";
import { isGoalie } from "@/lib/types";
import { mmss, pctColor } from "@/lib/format";
import { teamFullName, teamLogo } from "@/lib/teams";

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

const num3 = (v: number) => v.toFixed(3);
const num2 = (v: number) => v.toFixed(2);
const num1 = (v: number) => v.toFixed(1);
const signed = (v: number, fmt: (n: number) => string) => `${v >= 0 ? "+" : ""}${fmt(v)}`;
const svFmt = (v: number) => v.toFixed(3).replace(/^0/, ""); // .915
const pctFmt = (v: number) => (v * 100).toFixed(1); // fraction 0–1 -> "54.2"

// 5-on-5 unless the name says otherwise, so no "EV" prefix is needed
const ONICE: { key: OniceKey; name: string; explain: string; fmt: (v: number) => string; unit: string }[] = [
  { key: "ev_xgf60", name: "Expected Goals For", explain: "His team's expected goals per 60 at 5-on-5 while he's on the ice. A team rate, not isolated to him.", fmt: num2, unit: "xG/60" },
  { key: "ev_xga60", name: "Expected Goals Against", explain: "His team's expected goals allowed per 60 at 5-on-5 while he's on the ice. A team rate, not isolated; lower is better.", fmt: num2, unit: "xG/60" },
  { key: "ev_xgshare", name: "Expected Goals Share", explain: "His team's share of the expected goals (chance quality) at 5-on-5 while he's on the ice: xGF ÷ (xGF + xGA). Above 50% means out-chancing opponents.", fmt: pctFmt, unit: "%" },
  { key: "ev_cf60", name: "Shot Attempts For", explain: "His team's shot attempts (Corsi) per 60 at 5-on-5 while he's on the ice. A team rate.", fmt: num1, unit: "attempts/60" },
  { key: "ev_ca60", name: "Shot Attempts Against", explain: "Opponent shot attempts per 60 at 5-on-5 while he's on the ice. A team rate; lower is better.", fmt: num1, unit: "attempts/60" },
  { key: "ev_cfshare", name: "Shot Attempt Share", explain: "His team's share of all shot attempts at 5-on-5 while he's on the ice: Corsi for ÷ (for + against). Above 50% means controlling play.", fmt: pctFmt, unit: "%" },
  { key: "pp_xgf60", name: "Power-Play Expected Goals For", explain: "His team's expected goals per 60 on the power play while he's on the ice. A team rate.", fmt: num2, unit: "xG/60" },
  { key: "pk_xga60", name: "Penalty-Kill Expected Goals Against", explain: "His team's expected goals allowed per 60 on the penalty kill while he's on the ice. A team rate; lower is better.", fmt: num2, unit: "xG/60" },
];

const INDIV: { key: IndividualKey; name: string; explain: string; fmt: (v: number) => string; unit: string; signed?: boolean; ci?: boolean }[] = [
  { key: "shots60", name: "Shot Rate", explain: "His own unblocked shots per 60.", fmt: num1, unit: "shots/60" },
  { key: "xg_per_shot", name: "Shot Quality", explain: "Average danger of his shots (expected goals per shot).", fmt: num3, unit: "xG/shot" },
  { key: "ixg60", name: "Expected Goal Rate", explain: "Expected goals from his own shots per 60.", fmt: num2, unit: "xG/60" },
  { key: "fin_per100", name: "Finishing", explain: "Goals he scores above expected on his own shots, per 100 shots.", fmt: num2, unit: "goals/100 shots", signed: true, ci: true },
  { key: "g60", name: "Goal Rate", explain: "His goals per 60 (all situations).", fmt: num2, unit: "goals/60" },
  { key: "a60", name: "Assist Rate", explain: "His assists per 60 (all situations).", fmt: num2, unit: "assists/60" },
  { key: "a1_60", name: "Primary Assist Rate", explain: "First assists per 60, the pass that directly set up the goal.", fmt: num2, unit: "assists/60" },
  { key: "pen_drawn60", name: "Penalty Draw Rate", explain: "Penalties he drew per 60.", fmt: num2, unit: "drawn/60" },
  { key: "pen_taken60", name: "Penalty Take Rate", explain: "Penalties he took per 60.", fmt: num2, unit: "taken/60" },
  { key: "fo_win", name: "Faceoff Win %", explain: "Share of his faceoffs won.", fmt: pctFmt, unit: "%" },
  { key: "ozs", name: "Off. Zone Start %", explain: "Share of 5-on-5 shifts starting with an offensive-zone faceoff. Deployment context, not a skill rating.", fmt: pctFmt, unit: "%" },
];

// Player-stats section = metrics counted straight from raw events (the xG/finishing ones are modeled
// and live in the Modeled-impact section instead). Two rows of four.
const STAT_KEYS: IndividualKey[] = ["shots60", "g60", "a60", "a1_60", "pen_drawn60", "pen_taken60", "fo_win", "ozs"];

// per-season columns; `graph` ones are clickable to chart over time
type Col = { key: string; label: string; title: string; get: (r: SeasonRow) => number | null; fmt?: (v: number) => string; graph: boolean };
const COLS: Col[] = [
  { key: "gp", label: "GP", title: "Games played", get: (r) => r.gp, graph: true },
  { key: "toi_min", label: "TOI", title: "Time on ice (min)", get: (r) => r.toi_min, fmt: (v) => String(Math.round(v)), graph: true },
  { key: "g", label: "G", title: "Goals", get: (r) => r.g, graph: true },
  { key: "a", label: "A", title: "Assists", get: (r) => r.a1 + r.a2, graph: true },
  { key: "points", label: "P", title: "Points", get: (r) => r.points, graph: true },
  { key: "sog", label: "SOG", title: "Shots on goal", get: (r) => r.sog, graph: true },
  // goals-attributed value — the same metrics as the headline cards (per season)
  { key: "gnet_pg", label: "Net G/GP", title: "Net goals added per game (all situations)", get: (r) => r.gnet_pg ?? null, fmt: (v) => v.toFixed(2), graph: true },
  { key: "scoring60", label: "Scoring", title: "Goals from his own shots per 60 (5-on-5)", get: (r) => r.scoring60 ?? null, fmt: (v) => v.toFixed(2), graph: true },
  { key: "playmaking60", label: "Playmkg", title: "Expected goals he creates for teammates per 60 (5-on-5)", get: (r) => r.playmaking60 ?? null, fmt: (v) => v.toFixed(2), graph: true },
  { key: "allow60", label: "Defense", title: "His share of expected goals allowed per 60 (5-on-5; lower is better)", get: (r) => r.allow60 ?? null, fmt: (v) => v.toFixed(2), graph: true },
  { key: "pp_value", label: "PP", title: "Power-play offense per 60 (scoring + playmaking)", get: (r) => (r.pp_scoring60 == null && r.pp_playmaking60 == null) ? null : (r.pp_scoring60 ?? 0) + (r.pp_playmaking60 ?? 0), fmt: (v) => v.toFixed(2), graph: true },
  { key: "pk_allow60", label: "PK", title: "Penalty-kill expected goals allowed per 60 (lower is better)", get: (r) => r.pk_allow60 ?? null, fmt: (v) => v.toFixed(2), graph: true },
  { key: "pen_net60", label: "Pen", title: "Net goals from penalties drawn minus taken per 60", get: (r) => r.pen_net60 ?? null, fmt: (v) => v.toFixed(2), graph: true },
  { key: "shots60", label: "Sh/60", title: "Unblocked shots per 60", get: (r) => r.shots60 ?? null, fmt: (v) => v.toFixed(1), graph: true },
  { key: "xg_per_shot", label: "xG/sh", title: "Average shot quality (xG per shot)", get: (r) => r.xg_per_shot ?? null, fmt: (v) => v.toFixed(3), graph: true },
  { key: "fin_per100", label: "Fin", title: "Finishing: goals above expected per 100 shots", get: (r) => r.fin_per100 ?? null, fmt: (v) => v.toFixed(2), graph: true },
];

// shared box: a colored bar with the metric NAME and its percentile, then the value + a
// description. When `detail` is given, clicking the card flips it over and enlarges it to a
// front-and-center detail view (FLIP animation: the panel starts transformed onto the card's
// rect with the card face showing, then rotates 180° while travelling to screen center).
function BoxShell({ name, pctile, groupLabel, explain, value, footer, rankNote, detail }: {
  name: string; pctile: number | null; groupLabel: string; explain: string;
  value: React.ReactNode; footer?: React.ReactNode; rankNote?: string; detail?: React.ReactNode;
}) {
  const boxRef = useRef<HTMLDivElement | null>(null);
  const innerRef = useRef<HTMLDivElement | null>(null);
  // closed -> measure (overlay mounted hidden, compute card->center transform) -> open -> closing
  const [phase, setPhase] = useState<"closed" | "measure" | "open" | "closing">("closed");
  const [fromT, setFromT] = useState<string | null>(null);

  useLayoutEffect(() => {
    if (phase === "measure" && boxRef.current && innerRef.current) {
      const a = boxRef.current.getBoundingClientRect();
      const b = innerRef.current.getBoundingClientRect();
      setFromT(`translate(${a.left - b.left}px, ${a.top - b.top}px) ` +
               `scale(${a.width / b.width}, ${a.height / b.height})`);
      requestAnimationFrame(() => requestAnimationFrame(() => setPhase("open")));
    }
  }, [phase]);
  useEffect(() => {
    if (phase === "closed") return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setPhase("closing"); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [phase]);

  // clicking opens the detail — unless the user is selecting text on the card
  const openDetail = () => {
    const sel = window.getSelection();
    if (sel && sel.type === "Range" && sel.toString().length > 0) return;
    setPhase("measure");
  };

  const has = pctile != null;
  const barBg = has ? pctColor(pctile) : "var(--accent-softer)";
  const bar = (withClose: boolean) => (
    <div className="mb-bar" style={{ background: barBg }}>
      <span className="mb-head">{name}</span>
      <span className="mb-rank">
        {has ? <><span className="mb-pctile">{Math.round(pctile!)}%</span><span className="mb-vs">among {groupLabel}</span></>
             : <span className="mb-vs">{rankNote ?? "not enough volume"}</span>}
        {withClose && <button className="flip-close" aria-label="Close" onClick={() => setPhase("closing")}>×</button>}
      </span>
    </div>
  );

  return (
    <>
      <div ref={boxRef} className={`metric-box${detail ? " has-detail" : ""}`}
        style={phase !== "closed" ? { visibility: "hidden" } : undefined}
        role={detail ? "button" : undefined} tabIndex={detail ? 0 : undefined}
        title={detail ? "Click for the full definition" : undefined}
        onClick={detail ? openDetail : undefined}
        onKeyDown={detail ? (e) => { if (e.key === "Enter" || e.key === " ") setPhase("measure"); } : undefined}>
        {bar(false)}
        <div className="mb-body">
          <div className="mb-val">{value}{footer}</div>
          <div className="mb-blurb">{explain}</div>
        </div>
      </div>
      {phase !== "closed" && detail && (
        <div className={`flip-overlay${phase === "open" ? " on" : ""}`} onClick={() => setPhase("closing")}>
          <div ref={innerRef} className="flip-inner" onClick={(e) => e.stopPropagation()}
            style={{
              transform: phase === "open" ? "rotateY(180deg)" : fromT ?? undefined,
              visibility: fromT ? "visible" : "hidden",
            }}
            onTransitionEnd={(e) => {
              if (phase === "closing" && e.propertyName === "transform") { setPhase("closed"); setFromT(null); }
            }}>
            <div className="flip-front">
              {bar(false)}
              <div className="mb-body">
                <div className="mb-val">{value}{footer}</div>
                <div className="mb-blurb">{explain}</div>
              </div>
            </div>
            <div className="flip-back">
              {bar(true)}
              <div className="flip-back-body">
                <div className="mb-val">{value}{footer}</div>
                <div className="modal-body">{detail}</div>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

// format a card value: normalize "−0.00" to "0.00"; prepend + on non-negative when signed
const fmtVal = (v: number, fmt: (n: number) => string, sgn?: boolean) => {
  let s = fmt(v);
  if (/^-0(\.0+)?$/.test(s)) s = s.slice(1);
  return sgn && !s.startsWith("-") ? `+${s}` : s;
};

// descriptive / attributed-share box: a value with a unit, optionally signed, optionally with a 95% CI
function OniceBox({ name, explain, v, pctile, group, fmt, se, signed: sgn, unit, detail }: {
  name: string; explain: string; v: number | null; pctile: number | null; group: string;
  fmt: (v: number) => string; se?: number; signed?: boolean; unit?: string; detail?: React.ReactNode;
}) {
  return (
    <BoxShell name={name} pctile={v != null ? pctile : null} groupLabel={group === "D" ? "defensemen" : "forwards"} explain={explain} detail={detail}
      value={v != null
        ? <>{fmtVal(v, fmt, sgn)}{se != null && <span className="mb-ci"> ± {(1.96 * se).toFixed(2)}</span>}{unit && <span className="mb-unit">{unit}</span>}</>
        : <span className="muted">—</span>} />
  );
}

// player-value box: a goals number (per-60 share, per-game, or season total) + unit, colored by
// percentile. `signed` (default true) prepends + on differential metrics (net, penalties).
function ValueBox({ name, explain, v, pctile, group, fmt, footer, unit, signed: sgn = true, rankNote, detail }: {
  name: string; explain: string; v: number | null; pctile: number | null; group: string;
  fmt: (v: number) => string; footer?: React.ReactNode; unit?: string; signed?: boolean; rankNote?: string;
  detail?: React.ReactNode;
}) {
  return (
    <BoxShell name={name} pctile={v != null ? pctile : null} groupLabel={group === "D" ? "defensemen" : "forwards"} explain={explain} rankNote={rankNote} detail={detail}
      value={v != null ? <>{fmtVal(v, fmt, sgn)}{unit && <span className="mb-unit">{unit}</span>}</> : <span className="muted">—</span>} footer={footer} />
  );
}

// little equation table for the drill-downs: labeled rows, a sum rule, a result line
function Eq({ rows }: { rows: { label: React.ReactNode; val: string; kind?: "sum" | "res" }[] }) {
  return (
    <table className="eq">
      <tbody>
        {rows.map((r, i) => (
          <tr key={i} className={r.kind ? `eq-${r.kind}` : ""}>
            <td>{r.label}</td><td className="num">{r.val}</td>
          </tr>
        ))}
      </tbody>
    </table>
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

function ShotMap({ heat, variant = "skater" }: { heat: PlayerHeat; variant?: "skater" | "goalie" }) {
  const [mode, setMode] = useState<HeatMode>("shots");
  const { x, y, s, g, xg } = heat;
  const cw = (x.length > 1 ? x[1] - x[0] : 3) * HFT;
  const ch = (y.length > 1 ? y[1] - y[0] : 3) * HFT;

  const goalie = variant === "goalie";
  const modes: { key: HeatMode; label: string }[] = [
    { key: "shots", label: goalie ? "Shots Against" : "Shots" },
    { key: "goals", label: goalie ? "Goals Against" : "Goals" },
    { key: "shotpct", label: goalie ? "Goal %" : "Shot %" },
    { key: "xg", label: goalie ? "Avg xGA" : "Avg xG" },
  ];

  const maxS = useMemo(() => Math.max(0, ...s.flat()), [s]);
  const maxG = useMemo(() => Math.max(0, ...g.flat()), [g]);
  const eps = 0.06 * maxS;                                  // hide ratios in barely-shot-from areas
  const RATIO_CAP = mode === "shotpct" ? 0.3 : 0.32;       // goals/shot and xG/shot both top out ~0.3

  const intensity = (xi: number, yi: number): number | null => {
    const sv = s[yi][xi];
    if (mode === "shots") return maxS > 0 ? sv / maxS : 0;
    if (mode === "goals") return maxG > 0 ? g[yi][xi] / maxG : 0;
    if (sv < eps) return null;
    const r = (mode === "shotpct" ? g[yi][xi] : xg[yi][xi]) / sv;
    return r / RATIO_CAP;
  };

  const legend =
    mode === "shots" ? ["fewer", goalie ? "more shots faced" : "more shots"] :
    mode === "goals" ? ["fewer", goalie ? "more goals allowed" : "more goals"] :
    mode === "shotpct" ? ["0%", goalie ? "30%+ beat him" : "30%+ score"] : ["0", "0.30+ xG/shot"];

  return (
    <div>
      <div className="seg heat-seg">
        {modes.map((m) => (
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

// wrap long axis labels onto multiple lines (one word per line), vertically centered on the anchor
function RadarTick({ x, y, textAnchor, payload }: { x: number; y: number; textAnchor: "start" | "middle" | "end"; payload: { value: string } }) {
  const words = String(payload.value).split(" ");
  return (
    <text x={x} y={y} textAnchor={textAnchor} fontSize={10} fill="#6b7e90">
      {words.map((w, i) => (
        <tspan key={i} x={x} dy={i === 0 ? -(words.length - 1) * 5 : 10}>{w}</tspan>
      ))}
    </text>
  );
}

// percentile radar across the generative card attributes (0 = bottom / no qualifying ice time).
function ProfileRadar({ p }: { p: PlayerDetail }) {
  const g = p.gen!;
  const data = [
    { axis: "Scoring", v: g.attrs.scoring.pct },
    { axis: "Shooting", v: g.attrs.shooting.pct },
    { axis: "Finishing", v: g.attrs.finishing.pct },
    { axis: "Playmaking", v: g.attrs.playmaking.pct },
    { axis: "Power-Play Impact", v: g.attrs.pp_ga60.pct },
    { axis: "Penalty-Kill Defense", v: g.attrs.pk_ga60.pct },
    { axis: "Defense", v: g.attrs.defense.pct },
  ].map((d) => ({ ...d, v: d.v ?? 0 }));
  return (
    <ResponsiveContainer width="100%" height={360}>
      <RadarChart data={data} outerRadius="68%">
        <PolarGrid stroke="#e7f1fb" />
        <PolarAngleAxis dataKey="axis" tick={<RadarTick x={0} y={0} textAnchor="middle" payload={{ value: "" }} />} />
        <PolarRadiusAxis domain={[0, 100]} tickCount={5} angle={90} tick={{ fontSize: 9, fill: "#b6c4d2" }} />
        <Radar dataKey="v" stroke="#2f6cb0" fill="#4a90d9" fillOpacity={0.45} />
        <Tooltip formatter={(v: unknown) => `${Math.round(v as number)}th pct`} />
      </RadarChart>
    </ResponsiveContainer>
  );
}

// --- shared bio hero (identical header for skaters and goalies) ---
function BioHero({ name, metaLine, currentTeam, teams, bio }: {
  name: string; metaLine: React.ReactNode; currentTeam: string; teams: string[]; bio: PlayerBio | null;
}) {
  const former = teams.filter((t) => t !== currentTeam);
  return (
    <div className="player-hero">
      {bio?.headshot ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img className="player-mug" src={bio.headshot} alt={name} loading="lazy" />
      ) : null}
      <div className="player-hero-main">
        <h2 className="player-name">{name}</h2>
        <span className="player-meta">{metaLine}</span>
        <div className="player-teams">
          {currentTeam && (
            <Link href={`/team/${currentTeam}`} className="cur-team">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={teamLogo(currentTeam)} alt={currentTeam} loading="lazy" />
              {teamFullName(currentTeam)}
            </Link>
          )}
          {former.length > 0 ? (
            <span className="muted former-teams">
              Formerly{" "}
              {former.map((t, i) => (
                <span key={t}>{i > 0 ? ", " : ""}<Link href={`/team/${t}`}>{t}</Link></span>
              ))}
            </span>
          ) : null}
        </div>
        <div className="bio-row">
          {bio?.shoots && <span><b>Catches</b> {bio.shoots}</span>}
          {heightStr(bio?.height_in) && <span><b>Ht</b> {heightStr(bio?.height_in)}</span>}
          {bio?.weight_lb && <span><b>Wt</b> {bio.weight_lb} lb</span>}
          {ageFrom(bio?.birth_date) != null && <span><b>Age</b> {ageFrom(bio?.birth_date)}</span>}
          {bio?.birth_city && (
            <span><b>Born</b> {bio.birth_city}{bio.birth_state ? `, ${bio.birth_state}` : ""}{bio.birth_country ? `, ${bio.birth_country}` : ""}</span>
          )}
          {bio?.draft_overall ? <span><b>Draft</b> {bio.draft_year} · #{bio.draft_overall}</span> : null}
        </div>
      </div>
    </div>
  );
}

function GameLog({ games }: { games: GameLogRow[] }) {
  const router = useRouter();
  if (!games?.length) return null;
  return (
    <div className="panel">
      <h2>Game log</h2>
      <p className="section-sub">Every game, most recent first, regular season and playoffs.</p>
      <div className="gamelog-wrap">
        <table className="players gamelog">
          <thead>
            <tr>
              <th>Date</th><th>Team</th><th>Opp</th><th>Result</th>
              <th title="Time on ice">TOI</th><th title="Goals">G</th><th title="Assists">A</th>
              <th title="Points">P</th><th title="Shots on goal">SOG</th><th title="Penalties taken">PEN</th>
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

// ── generative Player Card (Cards v2): current skill inferred by the shot-generation model ──────
const sg2 = (v: number | null | undefined) => (v == null ? "—" : signed(v, num2));
const sg1 = (v: number | null | undefined) => (v == null ? "—" : signed(v, num1));

function GenCards({ p }: { p: PlayerDetail }) {
  const g = p.gen!;
  const A = g.attrs;
  const meta = p.gen_meta;
  const rv = meta?.replacement_values?.[g.pos];
  const kap = meta?.kappa ?? 1;
  const gpw = meta?.goals_per_win ?? 6;
  const posN = p.group === "D" ? "defenseman" : "forward";
  const seasonLbl = seasonLabel(g.last_season);

  // drill-down bodies: the full definition, with this player's numbers plugged in
  const replNote = (
    <p className="modal-note">
      Replacement level = the 8–12th-percentile regular at his position, measured separately for
      5-on-5, PP, and PK: the player a team can call up or claim for free. All value cards use it
      as their zero.
    </p>
  );
  const wg = g.war.goals;
  const wgSum = wg ? (wg.ev_atk ?? 0) + (wg.ev_def ?? 0) + (wg.pp ?? 0) + (wg.pk ?? 0) : null;
  const warDetail = (
    <>
      <p>
        For every shift he played in {seasonLbl}, the model computes his team&apos;s expected goals
        for and against with him on the ice, then again with a replacement-level {posN} in his slot:
        same linemates, opponents, zone starts, and goalies. Those differences, summed and
        converted to wins:
      </p>
      {wg && (
        <Eq rows={[
          { label: "even-strength offense", val: `${sg1(wg.ev_atk)} goals` },
          { label: "even-strength defense", val: sg1(wg.ev_def) },
          { label: "power play", val: sg1(wg.pp) },
          { label: "penalty kill", val: sg1(wg.pk) },
          { label: "goals above replacement", val: sg1(wgSum), kind: "sum" },
          { label: "÷ goals per win", val: gpw.toFixed(1) },
          { label: "WAR", val: sg2(A.war.v), kind: "res" },
        ]} />
      )}
      {g.unpriced_goals?.latest ? (
        <p>
          {`WAR doesn't price ${g.unpriced_goals.latest} of his goals this season (${
            Object.entries(g.unpriced_goals.detail ?? {}).map(([k, n]) => `${n}× ${k}`).join(", ")
          }): shorthanded, empty-net, and extra-attacker situations.`}
        </p>
      ) : null}
      <p className="modal-note">
        Percentile is among this season&apos;s regulars (200+ even-strength minutes). Over the full{" "}
        {meta?.seasons?.length ?? 5}-season window: {g.war.total != null ? num2(g.war.total) : "—"} WAR
        (EV {sg2(g.war.ev)}, PP {sg2(g.war.pp)}, PK {sg2(g.war.pk)}).
      </p>
      {replNote}
    </>
  );
  const c = g.components;
  const rawDef = c?.def_erased60 ?? null;   // xG erased before the position-average re-centering
  const evRows = (which: "all" | "created" | "prevented", resLabel: string, res: number | null) => {
    if (!rv || A.scoring.v == null || A.playmaking.v == null || rawDef == null) return null;
    const rows = [];
    if (which !== "prevented") {
      rows.push({ label: `scoring: his ${num2(A.scoring.v)} − replacement ${num2(rv.sc)}`,
                  val: sg2(A.scoring.v - rv.sc) });
      rows.push({ label: `playmaking: (${num2(A.playmaking.v)} − ${num2(rv.pm)}) × κ`,
                  val: sg2(kap * (A.playmaking.v - rv.pm)) });
    }
    if (which !== "created") {
      rows.push({ label: `defense: (his ${num2(rawDef)} erased − replacement ${num2(rv.df)}) × κ`,
                  val: sg2(kap * (rawDef - rv.df)) });
    }
    rows.push({ label: resLabel, val: sg2(res), kind: "res" as const });
    return rows;
  };
  const kapNote = (
    <p className="modal-note">
      Playmaking and defense are valued in expected goals; κ = {kap.toFixed(2)} converts them to
      goals at the league rate. Rates are skill-now at 5-on-5 in an average environment; WAR is
      the season-total counterpart over his actual usage.
    </p>
  );
  const addedRows = evRows("all", "Goals Added /60", A.ga60.v);
  const addedDetail = addedRows && (
    <>
      <p>His modeled 5-on-5 production per 60 minus a replacement {posN}&apos;s, component by component:</p>
      <Eq rows={addedRows} />
      {kapNote}{replNote}
    </>
  );
  const createdRows = evRows("created", "Goals Created /60", A.created60?.v ?? null);
  const createdDetail = createdRows && (
    <>
      <p>The offense half of Goals Added: his own scoring plus the chances he creates, each minus
        the replacement level:</p>
      <Eq rows={createdRows} />
      {kapNote}{replNote}
    </>
  );
  const preventedRows = evRows("prevented", "Goals Prevented /60", A.prevented60?.v ?? null);
  const preventedDetail = preventedRows && (
    <>
      <p>The defense half of Goals Added: opponent chance value he erases (shots suppressed plus
        danger suppressed), minus the replacement level:</p>
      <Eq rows={preventedRows} />
      {kapNote}{replNote}
    </>
  );
  const ppDetail = c && rv && c.pp_scoring != null && c.pp_playmaking != null ? (
    <>
      <p>His modeled power-play production per 60 minus a replacement PP regular&apos;s. The
        replacement here is the 8–12th-percentile player among those who actually play PP minutes,
        not the league-average PP player:</p>
      <Eq rows={[
        { label: `scoring: his ${num2(c.pp_scoring)} − replacement ${num2(rv.pp_sc)}`,
          val: sg2(c.pp_scoring - rv.pp_sc) },
        { label: `playmaking: (${num2(c.pp_playmaking)} − ${num2(rv.pp_pm)}) × κ`,
          val: sg2(kap * (c.pp_playmaking - rv.pp_pm)) },
        { label: "PP Goals Created /60", val: sg2(A.pp_ga60.v), kind: "res" },
      ]} />
      {kapNote}
    </>
  ) : null;
  const pkDetail = c && rv && c.pk_defense != null ? (
    <>
      <p>Opponent chance value he erases on the penalty kill, minus what a replacement PK regular
        (8–12th percentile among players with real PK minutes) erases:</p>
      <Eq rows={[
        { label: `PK defense: (his ${num2(c.pk_defense)} − replacement ${num2(rv.pk_df)}) × κ`,
          val: sg2(A.pk_ga60.v), kind: "res" },
      ]} />
      {kapNote}
    </>
  ) : null;
  const typShots = c?.shots60 != null && A.shooting.v != null && A.shooting.v > -100
    ? c.shots60 / (1 + A.shooting.v / 100) : null;
  const scoringDetail = (
    <>
      {c?.shots60 != null && c?.goals_per_shot != null && A.scoring.v != null && (
        <Eq rows={[
          { label: "inferred shot volume /60", val: num1(c.shots60) },
          { label: "× goals per shot, his shot mix", val: num3(c.goals_per_shot) },
          { label: "Scoring, goals /60 at 5-on-5", val: num2(A.scoring.v), kind: "res" },
        ]} />
      )}
      <p>
        The goal rate you&apos;d expect from him at 5-on-5 with league-average linemates and
        opponents. Volume is his shooting skill ({sg1(A.shooting.v)}% vs an average {posN});
        goals per shot comes from his shot locations and finishing ({sg2(A.finishing.v)} per 100
        shots). An absolute rate, not a difference from a baseline player.
      </p>
    </>
  );
  const shootingDetail = (
    <>
      {c?.shots60 != null && typShots != null && A.shooting.v != null && (
        <Eq rows={[
          { label: "his inferred shots /60", val: num1(c.shots60) },
          { label: `÷ an average ${posN}`, val: num1(typShots) },
          { label: "Shooting, % shot volume", val: `${sg1(A.shooting.v)}%`, kind: "res" },
        ]} />
      )}
      <p>
        His unblocked-shot rate vs the average {posN}, after the model removes linemates,
        competition, score state, and arena.
      </p>
      <p className="modal-note">
        His age is part of his skill, not the baseline: comparisons are never age-adjusted.
      </p>
    </>
  );
  const finAvg = meta?.fin_avg_p100?.[g.pos];
  const finishingDetail = (
    <>
      {finAvg != null && A.finishing.v != null && (
        <Eq rows={[
          { label: "his goals per 100 average-location shots", val: num2(finAvg + A.finishing.v) },
          { label: `− an average ${posN}`, val: num2(finAvg) },
          { label: "Finishing, goals /100 shots", val: sg2(A.finishing.v), kind: "res" },
        ]} />
      )}
      <p>
        Whether his shots beat goalies more often than an average {posN}&apos;s from the same
        locations. {A.finishing.se != null &&
          `The ±${(1.96 * A.finishing.se).toFixed(2)} range is a 95% interval. `}Finishing is a
        small, slow-to-reveal skill, and estimates are shrunk toward typical when shots are few.
      </p>
      <p className="modal-note">
        His age is part of his skill, not the baseline: comparisons are never age-adjusted.
      </p>
    </>
  );
  const playmakingDetail = (
    <>
      {c?.pm_shots60 != null && c?.pm_xg_per_shot != null && A.playmaking.v != null && (
        <Eq rows={[
          { label: "extra teammate shots /60 with him on ice", val: sg2(c.pm_shots60) },
          { label: "× chance quality, xG per shot", val: num3(c.pm_xg_per_shot) },
          { label: "Playmaking, xG /60", val: num2(A.playmaking.v), kind: "res" },
        ]} />
      )}
      <p>
        The extra chances his teammates get because he&apos;s on the ice, on an average team. The
        creation volume is his own (split from linemates by the model); the chance quality is
        priced at his position&apos;s average rate.
      </p>
      <p className="modal-note">
        Sharpest for players whose linemates vary; long-lived fixed units leave more of the split
        to assist patterns.
      </p>
    </>
  );
  const defenseDetail = (
    <>
      {c?.def_allowed60 != null && rawDef != null && A.defense.v != null && (
        <Eq rows={[
          { label: "on-ice xGA /60 share, neutral defenders", val: num2(c.def_allowed60 + rawDef) },
          { label: "− with him defending", val: num2(c.def_allowed60) },
          { label: "xG he erases /60", val: sg2(rawDef), kind: "sum" },
          { label: `− what an average ${posN} erases`, val: sg2(rawDef - A.defense.v) },
          { label: "Defense, vs position average", val: sg2(A.defense.v), kind: "res" },
        ]} />
      )}
      <p>
        Opponent chance value he erases per 60 at 5-on-5, above the average {posN}: shots
        suppressed plus danger suppressed, in expected goals.
      </p>
      <p className="modal-note">
        Goals Prevented /60 prices the erased chances in goals (× κ) and re-zeroes at replacement
        level. His age is part of his skill, not the baseline: comparisons are never age-adjusted.
      </p>
    </>
  );
  const penD = p.individual.pen_drawn60?.v ?? null;
  const penT = p.individual.pen_taken60?.v ?? null;
  const penV = p.value.rates.pen_net60?.v ?? null;
  const penNet = penD != null && penT != null ? penD - penT : null;
  const penDetail = (
    <>
      {penNet != null && penV != null && Math.abs(penNet) >= 0.05 && (
        <Eq rows={[
          { label: "penalties drawn /60", val: num2(penD!) },
          { label: "− penalties taken /60", val: num2(penT!) },
          { label: "× goals per penalty, league PP conversion", val: num2(penV / penNet) },
          { label: "Penalties, goals /60", val: sg2(penV), kind: "res" },
        ]} />
      )}
      <p>
        Penalties drawn minus taken, per 60, priced at the league&apos;s power-play conversion rate.
        Counted from raw events by the production model; not part of WAR yet.
      </p>
    </>
  );

  return (
    <div className="panel">
      <h2>Player Card</h2>
      <p className="section-sub">
        Skills inferred from every shot and shift by our generative model, isolated from linemates,
        competition, and arena. Values are his skill now ({seasonLbl}), age included; value cards
        are vs a replacement-level player. Click any card for the full definition with his numbers.
      </p>
      <div className="metric-grid">
        <ValueBox name="WAR" group={p.group} v={A.war.v} pctile={A.war.pct} fmt={num2} unit="wins" detail={warDetail}
          explain="Estimated wins added above a replacement player playing his games this season, offense and defense combined." />
        <ValueBox name="Goals Added /60" group={p.group} v={A.ga60.v} pctile={A.ga60.pct} fmt={num2} unit="goals/60" detail={addedDetail}
          explain="Net goals added per 60 above a replacement player at 5-on-5, offense plus defense on one scale." />
        <ValueBox name="Goals Created /60" group={p.group} v={A.created60?.v ?? null} pctile={A.created60?.pct ?? null} fmt={num2} unit="goals/60" detail={createdDetail}
          explain="Offense per 60 above a replacement player at 5-on-5: his own scoring plus chances created for teammates." />
        <ValueBox name="Goals Prevented /60" group={p.group} v={A.prevented60?.v ?? null} pctile={A.prevented60?.pct ?? null} fmt={num2} unit="goals/60" detail={preventedDetail}
          explain="Opponent goals prevented per 60 above a replacement player, at 5-on-5." />
      </div>
      <div className="metric-grid" style={{ marginTop: 10 }}>
        <OniceBox name="Scoring" group={p.group} v={A.scoring.v} pctile={A.scoring.pct} fmt={num2} unit="goals/60" signed={false} detail={scoringDetail}
          explain="Inferred goal rate from his own shots per 60 at 5-on-5, as if on an average team." />
        <OniceBox name="Shooting" group={p.group} v={A.shooting.v} pctile={A.shooting.pct} fmt={num1} unit="% shot volume" signed detail={shootingDetail}
          explain="Shot volume vs an average player at his position." />
        <OniceBox name="Finishing" group={p.group} v={A.finishing.v} pctile={A.finishing.pct} fmt={num2} unit="goals/100 shots"
          se={A.finishing.se ?? undefined} signed detail={finishingDetail}
          explain="Goals per 100 shots above an average player at his position, from the same shot locations." />
        <OniceBox name="Playmaking" group={p.group} v={A.playmaking.v} pctile={A.playmaking.pct} fmt={num2} unit="xG/60"
          se={A.playmaking.se ?? undefined} signed={false} detail={playmakingDetail}
          explain="Inferred rate of extra chances he creates for teammates, in expected goals per 60 at 5-on-5, as if on an average team." />
        <OniceBox name="Defense" group={p.group} v={A.defense.v} pctile={A.defense.pct} fmt={num2} unit="xG/60" signed detail={defenseDetail}
          explain="Opponent chance value erased per 60 at 5-on-5, above an average player at his position." />
        <ValueBox name="PP Goals Created /60" group={p.group} v={A.pp_ga60.v} pctile={A.pp_ga60.pct} fmt={num2} unit="goals/60" detail={ppDetail}
          explain="Power-play offense per 60 above a replacement PP regular." />
        <ValueBox name="PK Goals Prevented /60" group={p.group} v={A.pk_ga60.v} pctile={A.pk_ga60.pct} fmt={num2} unit="goals/60" detail={pkDetail}
          explain="Penalty-kill goals prevented per 60 above a replacement PK regular." />
        <ValueBox name="Penalties" group={p.group} v={p.value.rates.pen_net60?.v ?? null} pctile={p.value.rates.pen_net60?.pct ?? null} fmt={num2} unit="goals/60" detail={penDetail}
          explain="Net goals from penalties drawn minus taken, per 60. Not yet in WAR." />
      </div>
    </div>
  );
}

// ── skill trajectory: inferred per-season skill + projection + league age reference + raw dots ──
const TRAJ_TABS = [
  { key: "ga60", label: "Goals Added /60", unit: "goals/60" },
  { key: "scoring", label: "Scoring", unit: "goals/60" },
  { key: "playmaking", label: "Playmaking", unit: "xG/60" },
  { key: "defense", label: "Defense", unit: "xG/60" },
] as const;
type TrajKey = (typeof TRAJ_TABS)[number]["key"];

function TrajectoryChart({ p }: { p: PlayerDetail }) {
  const g = p.gen!;
  const [attr, setAttr] = useState<TrajKey>("ga60");
  const data = useMemo(() => {
    const per = new Map(p.per_season.map((r) => [r.season, r]));
    const lg = (tp: Record<string, unknown>) => (tp[`lg_${attr}`] as number | null) ?? null;
    const pts = g.trajectory.map((tp) => ({
      season: seasonLabel(tp.season),
      age: tp.age,
      skill: tp[attr],
      league: lg(tp as unknown as Record<string, unknown>),
      raw: attr === "scoring" ? per.get(tp.season)?.scoring60 ?? null
        : attr === "playmaking" ? per.get(tp.season)?.playmaking60 ?? null : null,
      proj: null as number | null,
    }));
    if (g.projection?.season != null && pts.length) {
      pts[pts.length - 1].proj = pts[pts.length - 1].skill;      // connect the dashed segment
      pts.push({
        season: seasonLabel(g.projection.season), age: null, skill: null, raw: null,
        league: lg(g.projection as unknown as Record<string, unknown>),
        proj: (g.projection as unknown as Record<string, number | null>)[attr] ?? null,
      });
    }
    return pts;
  }, [g, attr, p]);
  const tab = TRAJ_TABS.find((t) => t.key === attr)!;
  return (
    <div>
      <div className="seg" style={{ marginBottom: 8 }}>
        {TRAJ_TABS.map((t) => (
          <button key={t.key} className={attr === t.key ? "active" : ""} onClick={() => setAttr(t.key)}>{t.label}</button>
        ))}
      </div>
      <ResponsiveContainer width="100%" height={240}>
        <ComposedChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: -8 }}>
          <CartesianGrid stroke="#e7f1fb" />
          <XAxis dataKey="season" tick={{ fontSize: 12, fill: "#6b7e90" }} />
          <YAxis tick={{ fontSize: 12, fill: "#6b7e90" }} width={48} domain={["auto", "auto"]} />
          <Tooltip formatter={(v: unknown, name: unknown) => [typeof v === "number" ? `${v.toFixed(2)} ${tab.unit}` : String(v), String(name)]}
            labelFormatter={(l, payload) => {
              const a = (payload?.[0]?.payload as { age?: number | null } | undefined)?.age;
              return a != null ? `${l} · age ${a}` : String(l);
            }} />
          <Line type="monotone" dataKey="league" name="typical for position + age" stroke="#b6c4d2"
            strokeDasharray="2 5" strokeWidth={1.5} dot={false} connectNulls />
          <Scatter dataKey="raw" name="single-season estimate" fill="#9db8d4" />
          <Line type="monotone" dataKey="skill" name="modeled skill" stroke="#2f6cb0" strokeWidth={2.5}
            dot={{ r: 3 }} connectNulls />
          <Line type="monotone" dataKey="proj" name="projection" stroke="#2f6cb0" strokeWidth={2}
            strokeDasharray="6 4" dot={{ r: 4, fill: "#fff", stroke: "#2f6cb0" }} connectNulls />
        </ComposedChart>
      </ResponsiveContainer>
      <p className="section-sub" style={{ marginTop: 4 }}>
        Solid line: skill the model infers each season (smoothed, deployment-free). Dashed: next-season
        projection along his position&apos;s aging curve. Grey dotted: a typical {p.group === "D" ? "defenseman" : "forward"} his
        age. Dots: unsmoothed single-season estimates.
      </p>
    </div>
  );
}

function SkaterView({ p }: { p: PlayerDetail }) {
  const [stat, setStat] = useState(p.gen ? "__traj__" : "points");
  const col = COLS.find((c) => c.key === stat) ?? COLS.find((c) => c.key === "points")!;
  const chartData = useMemo(
    () => p.per_season.map((r) => ({ season: seasonLabel(r.season), value: col.get(r) })),
    [p, col]
  );
  const last = p.seasons[p.seasons.length - 1];

  return (
    <div>
      <Link className="backlink" href="/players">← all players</Link>

      <div className="panel">
        <BioHero name={p.name} currentTeam={p.current_team} teams={p.teams} bio={p.bio}
          metaLine={<>{p.bio?.number != null ? `#${p.bio.number} · ` : ""}{p.pos} · {p.group === "D" ? "Defenseman" : "Forward"}</>} />
        <div className="statgrid">
          <div className="stat"><span className="v">{p.gp}</span><span className="k">games</span></div>
          <div className="stat"><span className="v">{p.g}</span><span className="k">goals</span></div>
          <div className="stat"><span className="v">{p.a}</span><span className="k">assists</span></div>
          <div className="stat"><span className="v">{p.points}</span><span className="k">points</span></div>
          <div className="stat"><span className="v">{seasonLabel(p.seasons[0])} – {seasonLabel(last)}</span><span className="k">regular seasons</span></div>
        </div>
      </div>

      {p.gen && <GenCards p={p} />}

      <div className="panel">
        <h2>Player stats</h2>
        <p className="section-sub">Counted straight from his game events, per 60 minutes, all situations.</p>
        <div className="metric-grid">
          {INDIV.filter((m) => STAT_KEYS.includes(m.key)).map((m) => {
            const d = p.individual[m.key];
            return <OniceBox key={m.key} name={m.name} explain={m.explain} v={d.v} pctile={d.pct} group={p.group} fmt={m.fmt} unit={m.unit} se={m.ci ? d.se : undefined} signed={m.signed} />;
          })}
        </div>
      </div>

      <div className="panel">
        <h2>On-Ice Team Rates</h2>
        <p className="section-sub">His team&apos;s rates while he was on the ice, not adjusted for teammates.</p>
        <div className="metric-grid">
          {ONICE.map((m) => {
            const d = p.onice[m.key];
            return <OniceBox key={m.key} name={m.name} explain={m.explain} v={d.v} pctile={d.pct} group={p.group} fmt={m.fmt} unit={m.unit} />;
          })}
        </div>
      </div>

      <div className="panel">
        <h2>{p.gen ? "Skill trajectory & seasons" : "By season: click a stat to chart it"}</h2>
        {p.gen && (
          <p className="section-sub">
            Where his skills have been and where they&apos;re heading, or click any stat in the table to
            chart it season by season.
          </p>
        )}
        <div className="chart-wrap">
          {p.gen && stat === "__traj__" ? (
            <TrajectoryChart p={p} />
          ) : (
            <>
              {p.gen && (
                <div className="seg" style={{ marginBottom: 8 }}>
                  <button onClick={() => setStat("__traj__")}>← skill trajectory</button>
                </div>
              )}
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={chartData} margin={{ top: 8, right: 16, bottom: 4, left: -8 }}>
                  <CartesianGrid stroke="#e7f1fb" />
                  <XAxis dataKey="season" tick={{ fontSize: 12, fill: "#6b7e90" }} />
                  <YAxis tick={{ fontSize: 12, fill: "#6b7e90" }} width={48} />
                  <Tooltip formatter={(v: unknown) => (typeof v === "number" && col.fmt ? col.fmt(v) : String(v))} />
                  <Line type="monotone" dataKey="value" name={col.label} stroke="#2f6cb0" strokeWidth={2} dot={{ r: 3 }} connectNulls />
                </LineChart>
              </ResponsiveContainer>
            </>
          )}
        </div>
        <table className="players season-table">
          <thead>
            <tr>
              <th>Season</th><th>Team</th>
              {COLS.map((c) => (
                <th key={c.key} className={`sortable ${stat === c.key ? "graphed" : ""}`} title={c.title} onClick={() => setStat(c.key)}>{c.label}</th>
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
                  return <td key={c.key} className={stat === c.key ? "graphed" : ""}>{v == null ? "—" : c.fmt ? c.fmt(v) : v}</td>;
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
              <p className="section-sub">Where he shoots from, both ends folded to one. Toggle shot volume, goals, shooting %, and average shot quality (xG); attacking net at right.</p>
              <ShotMap heat={p.heat} />
            </div>
          )}
          {p.gen && (
            <div className="viz-cell">
              <p className="section-sub">Percentile rank across the card attributes, within position ({p.group === "D" ? "defensemen" : "forwards"}). Further out is better.</p>
              <ProfileRadar p={p} />
            </div>
          )}
        </div>
      </div>

      <GameLog games={p.games ?? []} />
    </div>
  );
}

// ============================ GOALIE VIEW ============================

const GMETRICS: { key: GoalieMetricKey; name: string; explain: string; fmt: (v: number) => string; unit?: string; signed?: boolean; ci?: boolean }[] = [
  { key: "gsax_per100", name: "Goals Saved Above Expected /100", explain: "Goals he prevents above expected per 100 shots faced (shrunk for sample size). 0 = saved exactly what the shots' xG predicted. ± is a 95% range.", fmt: num2, unit: "goals/100 shots", signed: true, ci: true },
  { key: "gsax60", name: "GSAx per 60", explain: "Goals he prevents above expected per 60 minutes played. 0 = saved as expected.", fmt: num2, unit: "goals/60", signed: true },
  { key: "sv_pct", name: "Save %", explain: "Saves per shot on goal.", fmt: svFmt },
  { key: "gaa", name: "Goals-Against Average", explain: "Goals allowed per 60 minutes (empty-net excluded). Lower is better.", fmt: num2, unit: "goals/60" },
  { key: "hd_sv_pct", name: "High-Danger Save %", explain: "Save % on high-danger shots (xG ≥ 0.15).", fmt: svFmt },
  { key: "qs_pct", name: "Quality-Start %", explain: "Share of starts that were quality starts (Sv% ≥ .917, or ≤2 GA on <20 shots).", fmt: (v) => `${Math.round(v * 100)}%` },
];

function GoalieBox({ name, explain, m, fmt, signed: sgn, ci, unit }: {
  name: string; explain: string; m: { v: number | null; pct: number | null; se?: number };
  fmt: (v: number) => string; signed?: boolean; ci?: boolean; unit?: string;
}) {
  return (
    <BoxShell name={name} pctile={m.v != null ? m.pct : null} groupLabel="goalies" explain={explain}
      value={m.v != null
        ? <>{sgn && m.v >= 0 ? "+" : ""}{fmt(m.v)}{ci && m.se != null && <span className="mb-ci"> ± {(1.96 * m.se).toFixed(2)}</span>}{unit && <span className="mb-unit">{unit}</span>}</>
        : <span className="muted">—</span>} />
  );
}

function SplitTable({ head, rows }: { head: string; rows: { label: string; sa: number; sv_pct: number | null; gsax: number | null }[] }) {
  return (
    <table className="players">
      <thead><tr><th>{head}</th><th title="Shots on goal">SA</th><th title="Save %">Sv%</th><th title="Goals saved above expected">GSAx</th></tr></thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.label}>
            <td>{r.label}</td>
            <td className="num">{r.sa}</td>
            <td className="num">{r.sv_pct != null ? svFmt(r.sv_pct) : "—"}</td>
            <td className="num">{r.gsax != null ? signed(r.gsax, num1) : "—"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

const DANGER_LABEL: Record<string, string> = { ld: "Low danger", md: "Medium danger", hd: "High danger" };
const SIT_LABEL: Record<string, string> = { ev: "Even strength", pk: "Shorthanded (PK)", pp: "On the power play" };
const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

type GCol = { key: string; label: string; title: string; get: (r: GoalieSeasonRow) => number | null; fmt?: (v: number) => string };
const GCOLS: GCol[] = [
  { key: "gp", label: "GP", title: "Games played", get: (r) => r.gp },
  { key: "starts", label: "GS", title: "Starts", get: (r) => r.starts },
  { key: "toi_min", label: "TOI", title: "Minutes", get: (r) => r.toi_min, fmt: (v) => String(Math.round(v)) },
  { key: "sv_pct", label: "Sv%", title: "Save %", get: (r) => r.sv_pct, fmt: svFmt },
  { key: "gaa", label: "GAA", title: "Goals-against average", get: (r) => r.gaa, fmt: num2 },
  { key: "gsax", label: "GSAx", title: "Goals saved above expected", get: (r) => r.gsax, fmt: num1 },
  { key: "gsax_per100", label: "/100", title: "GSAx per 100 shots (shrunk)", get: (r) => r.gsax_per100, fmt: num2 },
  { key: "sa", label: "SA", title: "Shots on goal against", get: (r) => r.sa },
  { key: "ga", label: "GA", title: "Goals against", get: (r) => r.ga },
  { key: "shutouts", label: "SO", title: "Shutouts", get: (r) => r.shutouts },
  { key: "hd_sv_pct", label: "HDSv%", title: "High-danger save %", get: (r) => r.hd_sv_pct, fmt: svFmt },
  { key: "qs", label: "QS", title: "Quality starts", get: (r) => r.qs },
];

function GoalieGameLog({ games }: { games: GoalieDetail["games"] }) {
  const router = useRouter();
  if (!games?.length) return null;
  return (
    <div className="panel">
      <h2>Game log</h2>
      <p className="section-sub">Every appearance, most recent first. Decision is credited to the starter; relief outings show no decision.</p>
      <div className="gamelog-wrap">
        <table className="players gamelog">
          <thead>
            <tr>
              <th>Date</th><th>Team</th><th>Opp</th><th>Dec</th>
              <th title="Time on ice">TOI</th><th title="Shots on goal against">SA</th>
              <th title="Saves">SV</th><th title="Goals against">GA</th>
              <th title="Expected goals against">xGA</th><th title="Goals saved above expected">GSAx</th>
              <th title="Quality start">QS</th>
            </tr>
          </thead>
          <tbody>
            {games.map((g) => (
              <tr key={g.game_id} className="rowlink" onClick={() => router.push(`/game/${g.game_id}`)}>
                <td>{g.date ?? "—"}</td>
                <td><Link href={`/team/${g.team}`} prefetch={false} onClick={(e) => e.stopPropagation()}>{g.team}</Link></td>
                <td className="muted">{g.home ? "vs" : "@"} <Link href={`/team/${g.opp}`} prefetch={false} onClick={(e) => e.stopPropagation()}>{g.opp}</Link></td>
                <td className={`result result-${(g.decision ?? "").toLowerCase()}`}>{g.decision ?? "—"}</td>
                <td className="num">{mmss(g.toi_s)}</td>
                <td className="num">{g.sog_against}</td>
                <td className="num">{g.saves}</td>
                <td className="num">{g.ga}{g.shutout ? " ⬤" : ""}</td>
                <td className="num">{g.xga != null ? g.xga.toFixed(1) : "—"}</td>
                <td className="num">{g.gsax != null ? signed(g.gsax, num1) : "—"}</td>
                <td className="num">{g.qs ? "✓" : g.rbs ? "✗" : ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function GoalieView({ p }: { p: GoalieDetail }) {
  const [gstat, setGstat] = useState("gsax");
  const gcol = GCOLS.find((c) => c.key === gstat)!;
  const chartData = useMemo(
    () => p.per_season.map((r) => ({ season: seasonLabel(r.season), value: gcol.get(r) })),
    [p, gcol]
  );
  const last = p.seasons[p.seasons.length - 1];

  return (
    <div>
      <Link className="backlink" href="/players?pos=G">← all goalies</Link>

      <div className="panel">
        <BioHero name={p.name} currentTeam={p.current_team} teams={p.teams} bio={p.bio}
          metaLine={<>{p.bio?.number != null ? `#${p.bio.number} · ` : ""}G · Goaltender</>} />
        <div className="statgrid">
          <div className="stat"><span className="v">{p.gp}</span><span className="k">games</span></div>
          <div className="stat"><span className="v">{p.starts}</span><span className="k">starts</span></div>
          <div className="stat"><span className="v">{p.sv_pct != null ? svFmt(p.sv_pct) : "—"}</span><span className="k">save %</span></div>
          <div className="stat"><span className="v">{p.gaa != null ? p.gaa.toFixed(2) : "—"}</span><span className="k">GAA</span></div>
          <div className="stat"><span className="v">{p.gsax != null ? signed(p.gsax, num1) : "—"}</span><span className="k">GSAx</span></div>
          <div className="stat"><span className="v">{p.shutouts}</span><span className="k">shutouts</span></div>
          <div className="stat"><span className="v">{seasonLabel(p.seasons[0])} – {seasonLabel(last)}</span><span className="k">regular seasons</span></div>
        </div>
      </div>

      <div className="panel">
        <h2>Save performance</h2>
        <p className="section-sub">How he stops the puck vs. the expected-goal value of the shots he faced. Percentiles are among goalies; ± is a 95% range.</p>
        <div className="metric-grid">
          {GMETRICS.map((m) => (
            <GoalieBox key={m.key} name={m.name} explain={m.explain} m={p.metric[m.key]} fmt={m.fmt} unit={m.unit} signed={m.signed} ci={m.ci} />
          ))}
        </div>
      </div>

      <div className="panel">
        <h2>Where the GSAx comes from</h2>
        <p className="section-sub">Save % is over shots on goal; GSAx is over all unblocked shots, so the three breakdowns each sum to his total GSAx.</p>
        <div className="split-row">
          <div className="split-cell">
            <SplitTable head="Shot danger" rows={p.danger.map((d) => ({ label: DANGER_LABEL[d.bucket] ?? d.bucket, sa: d.sa, sv_pct: d.sv_pct, gsax: d.gsax }))} />
          </div>
          <div className="split-cell">
            <SplitTable head="Strength" rows={p.situation.map((s) => ({ label: SIT_LABEL[s.sit] ?? s.sit, sa: s.sa, sv_pct: s.sv_pct, gsax: s.gsax }))} />
          </div>
          <div className="split-cell">
            <SplitTable head="Shot type" rows={p.shot_types.slice(0, 7).map((s) => ({ label: cap(s.shot_type), sa: s.sa, sv_pct: s.sv_pct, gsax: s.gsax }))} />
          </div>
        </div>
      </div>

      <div className="panel">
        <h2>By season: click a stat to chart it</h2>
        <div className="chart-wrap">
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={chartData} margin={{ top: 8, right: 16, bottom: 4, left: -8 }}>
              <CartesianGrid stroke="#e7f1fb" />
              <XAxis dataKey="season" tick={{ fontSize: 12, fill: "#6b7e90" }} />
              <YAxis tick={{ fontSize: 12, fill: "#6b7e90" }} width={48} />
              <Tooltip formatter={(v: unknown) => (typeof v === "number" && gcol.fmt ? gcol.fmt(v) : String(v))} />
              <Line type="monotone" dataKey="value" name={gcol.label} stroke="#2f6cb0" strokeWidth={2} dot={{ r: 3 }} connectNulls />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <table className="players season-table">
          <thead>
            <tr>
              <th>Season</th><th>Team</th>
              {GCOLS.map((c) => (
                <th key={c.key} className={`sortable ${gstat === c.key ? "graphed" : ""}`} title={c.title} onClick={() => setGstat(c.key)}>{c.label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {[...p.per_season].reverse().map((r) => (
              <tr key={r.season}>
                <td>{seasonLabel(r.season)}</td>
                <td>{r.team}</td>
                {GCOLS.map((c) => {
                  const v = c.get(r);
                  return <td key={c.key} className={gstat === c.key ? "graphed" : ""}>{v == null ? "—" : c.fmt ? c.fmt(v) : v}</td>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {p.heat && (
        <div className="panel">
          <h2>Shots faced</h2>
          <div className="viz-row">
            <div className="viz-cell">
              <p className="section-sub">Where shots came from against him, both ends folded to one. Toggle shot volume, goals allowed, goal % (chance a shot from there beat him), and average shot quality; net at right.</p>
              <ShotMap heat={p.heat} variant="goalie" />
            </div>
          </div>
        </div>
      )}

      <GoalieGameLog games={p.games} />
    </div>
  );
}

export default function Player() {
  const params = useParams<{ id: string }>();
  const id = params.id;
  const [p, setP] = useState<AnyPlayerDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setP(null);
    setError(null);
    fetch(`/data/player/${id}.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then(setP)
      .catch((e) => setError(String(e)));
  }, [id]);

  if (error) return <div className="loading">Failed to load player {id} ({error}).</div>;
  if (!p) return <div className="loading">Loading player…</div>;
  return isGoalie(p) ? <GoalieView p={p} /> : <SkaterView p={p} />;
}
