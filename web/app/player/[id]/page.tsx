"use client";
import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import {
  CartesianGrid, Line, LineChart, PolarAngleAxis, PolarGrid, PolarRadiusAxis,
  Radar, RadarChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
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
  { key: "ev_xgf60", name: "Expected Goals For", explain: "His team's expected goals per 60 at 5-on-5 while he's on the ice — a team rate, not isolated to him.", fmt: num2, unit: "xG/60" },
  { key: "ev_xga60", name: "Expected Goals Against", explain: "His team's expected goals allowed per 60 at 5-on-5 while he's on the ice — a team rate, not isolated. Lower is better.", fmt: num2, unit: "xG/60" },
  { key: "ev_xgshare", name: "Expected Goals Share", explain: "His team's share of the expected goals (chance quality) at 5-on-5 while he's on the ice — xGF ÷ (xGF + xGA). Above 50% means out-chancing opponents.", fmt: pctFmt, unit: "%" },
  { key: "ev_cf60", name: "Shot Attempts For", explain: "His team's shot attempts (Corsi) per 60 at 5-on-5 while he's on the ice — a team rate.", fmt: num1, unit: "attempts/60" },
  { key: "ev_ca60", name: "Shot Attempts Against", explain: "Opponent shot attempts per 60 at 5-on-5 while he's on the ice — a team rate. Lower is better.", fmt: num1, unit: "attempts/60" },
  { key: "ev_cfshare", name: "Shot Attempt Share", explain: "His team's share of all shot attempts at 5-on-5 while he's on the ice — Corsi for ÷ (for + against). Above 50% means controlling play.", fmt: pctFmt, unit: "%" },
  { key: "pp_xgf60", name: "Power-Play Expected Goals For", explain: "His team's expected goals per 60 on the power play while he's on the ice — a team rate.", fmt: num2, unit: "xG/60" },
  { key: "pk_xga60", name: "Penalty-Kill Expected Goals Against", explain: "His team's expected goals allowed per 60 on the penalty kill while he's on the ice — a team rate. Lower is better.", fmt: num2, unit: "xG/60" },
];

const INDIV: { key: IndividualKey; name: string; explain: string; fmt: (v: number) => string; unit: string; signed?: boolean; ci?: boolean }[] = [
  { key: "shots60", name: "Shot Rate", explain: "His own unblocked shots per 60.", fmt: num1, unit: "shots/60" },
  { key: "xg_per_shot", name: "Shot Quality", explain: "Average danger of his shots (expected goals per shot).", fmt: num3, unit: "xG/shot" },
  { key: "ixg60", name: "Expected Goal Rate", explain: "Expected goals from his own shots per 60.", fmt: num2, unit: "xG/60" },
  { key: "fin_per100", name: "Finishing", explain: "Goals he scores above expected on his own shots, per 100 shots.", fmt: num2, unit: "goals/100 shots", signed: true, ci: true },
  { key: "g60", name: "Goal Rate", explain: "His goals per 60 (all situations).", fmt: num2, unit: "goals/60" },
  { key: "a60", name: "Assist Rate", explain: "His assists per 60 (all situations).", fmt: num2, unit: "assists/60" },
  { key: "a1_60", name: "Primary Assist Rate", explain: "First assists per 60 — the pass that directly set up the goal. More repeatable than total assists.", fmt: num2, unit: "assists/60" },
  { key: "pen_drawn60", name: "Penalty Draw Rate", explain: "Penalties he drew per 60.", fmt: num2, unit: "drawn/60" },
  { key: "pen_taken60", name: "Penalty Take Rate", explain: "Penalties he took per 60.", fmt: num2, unit: "taken/60" },
  { key: "fo_win", name: "Faceoff Win %", explain: "Share of his faceoffs won. Shown only for players who take enough draws.", fmt: pctFmt, unit: "%" },
  { key: "ozs", name: "Off. Zone Start %", explain: "Share of his 5-on-5 shifts that began with an offensive-zone faceoff (vs defensive). Deployment context — higher means more sheltered usage — not a rating of skill.", fmt: pctFmt, unit: "%" },
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

// shared box: a colored bar with the metric NAME and its percentile, then the value + a description.
function BoxShell({ name, pctile, groupLabel, explain, value, footer }: {
  name: string; pctile: number | null; groupLabel: string; explain: string;
  value: React.ReactNode; footer?: React.ReactNode;
}) {
  const has = pctile != null;
  return (
    <div className="metric-box">
      <div className="mb-bar" style={{ background: has ? pctColor(pctile) : "var(--accent-softer)" }}>
        <span className="mb-head">{name}</span>
        <span className="mb-rank">
          {has ? <><span className="mb-pctile">{Math.round(pctile!)}%</span><span className="mb-vs">among {groupLabel}</span></>
               : <span className="mb-vs">not enough volume</span>}
        </span>
      </div>
      <div className="mb-body">
        <div className="mb-val">{value}{footer}</div>
        <div className="mb-blurb">{explain}</div>
      </div>
    </div>
  );
}

// descriptive / attributed-share box: a value with a unit, optionally signed, optionally with a 95% CI
function OniceBox({ name, explain, v, pctile, group, fmt, se, signed: sgn, unit }: {
  name: string; explain: string; v: number | null; pctile: number | null; group: string;
  fmt: (v: number) => string; se?: number; signed?: boolean; unit?: string;
}) {
  return (
    <BoxShell name={name} pctile={v != null ? pctile : null} groupLabel={group === "D" ? "defensemen" : "forwards"} explain={explain}
      value={v != null
        ? <>{sgn && v >= 0 ? "+" : ""}{fmt(v)}{se != null && <span className="mb-ci"> ± {(1.96 * se).toFixed(2)}</span>}{unit && <span className="mb-unit">{unit}</span>}</>
        : <span className="muted">—</span>} />
  );
}

// player-value box: a goals number (per-60 share, per-game, or season total) + unit, colored by
// percentile. `signed` (default true) prepends + on differential metrics (net, penalties).
function ValueBox({ name, explain, v, pctile, group, fmt, footer, unit, signed: sgn = true }: {
  name: string; explain: string; v: number | null; pctile: number | null; group: string;
  fmt: (v: number) => string; footer?: React.ReactNode; unit?: string; signed?: boolean;
}) {
  return (
    <BoxShell name={name} pctile={v != null ? pctile : null} groupLabel={group === "D" ? "defensemen" : "forwards"} explain={explain}
      value={v != null ? <>{sgn && v >= 0 ? "+" : ""}{fmt(v)}{unit && <span className="mb-unit">{unit}</span>}</> : <span className="muted">—</span>} footer={footer} />
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

// percentile radar across the headline skills (0 = no qualifying ice time / bottom, 100 = best).
// Axis names match the cards above (modeled impact + player stats).
function ProfileRadar({ p }: { p: PlayerDetail }) {
  const data = [
    { axis: "Scoring", v: p.value.rates.scoring60?.pct ?? null },
    { axis: "Playmaking", v: p.value.rates.playmaking60?.pct ?? null },
    { axis: "Power-Play Scoring", v: p.value.rates.pp_scoring60?.pct ?? null },
    { axis: "Power-Play Playmaking", v: p.value.rates.pp_playmaking60?.pct ?? null },
    { axis: "Penalties", v: p.value.rates.pen_net60?.pct ?? null },
    { axis: "Penalty-Kill Defense", v: p.value.rates.pk_allow60?.pct ?? null },
    { axis: "Defense", v: p.value.rates.allow60?.pct ?? null },
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
      <p className="section-sub">Every game, most recent first — regular season and playoffs.</p>
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

function SkaterView({ p }: { p: PlayerDetail }) {
  const [stat, setStat] = useState("points");
  const col = COLS.find((c) => c.key === stat)!;
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

      <div className="panel">
        <h2>Modeled Impact</h2>
        <p className="section-sub">
          What our models credit him with, adjusted for the linemates and competition he played with.
        </p>

        <div className="metric-grid">
          <ValueBox name="Net Goals Added per Game" group={p.group} v={p.value.actual.net_pg} pctile={p.value.actual.net_pg_pct} fmt={num2} unit="goals/game"
            explain="Net goals created (scoring + playmaking) minus goals allowed, per game." />
          <OniceBox name="Scoring" group={p.group} v={p.value.rates.scoring60?.v ?? null} pctile={p.value.rates.scoring60?.pct ?? null} fmt={num2} unit="goals/60" signed={false}
            explain="Goals from his own shots — their expected value plus his finishing." />
          <OniceBox name="Playmaking" group={p.group} v={p.value.rates.playmaking60?.v ?? null} pctile={p.value.rates.playmaking60?.pct ?? null} fmt={num2} unit="goals/60" signed={false}
            explain="Expected goals he creates for teammates at 5-on-5 — the part of his offense that isn't his own shots." />
          <OniceBox name="Defense" group={p.group} v={p.value.rates.allow60?.v ?? null} pctile={p.value.rates.allow60?.pct ?? null} fmt={num2} unit="goals/60" se={p.impact.ev_def.se} signed={false}
            explain="His share of the expected goals allowed at 5-on-5 while he's on the ice. Lower is better." />
          <OniceBox name="Power-Play Scoring" group={p.group} v={p.value.rates.pp_scoring60?.v ?? null} pctile={p.value.rates.pp_scoring60?.pct ?? null} fmt={num2} unit="goals/60" signed={false}
            explain="Goals from his own shots on the power play — their expected value plus his finishing." />
          <OniceBox name="Power-Play Playmaking" group={p.group} v={p.value.rates.pp_playmaking60?.v ?? null} pctile={p.value.rates.pp_playmaking60?.pct ?? null} fmt={num2} unit="goals/60" signed={false}
            explain="Expected goals he creates for teammates on the power play — the part of his offense that isn't his own shots." />
          <OniceBox name="Penalty-Kill Defense" group={p.group} v={p.value.rates.pk_allow60?.v ?? null} pctile={p.value.rates.pk_allow60?.pct ?? null} fmt={num2} unit="goals/60" se={p.impact.pk_def.se} signed={false}
            explain="His share of the expected goals allowed on the penalty kill while he's on the ice. Lower is better." />
          <ValueBox name="Penalties" group={p.group} v={p.value.rates.pen_net60?.v ?? null} pctile={p.value.rates.pen_net60?.pct ?? null} fmt={num2} unit="goals/60"
            explain="Net goals from penalties he draws minus takes." />
        </div>
      </div>

      <div className="panel">
        <h2>Player stats</h2>
        <p className="section-sub">Counted straight from his game events — per 60 minutes, all situations.</p>
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
        <p className="section-sub">How he stops the puck vs. the expected-goal value of the shots he faced — percentiles are among goalies. ± is a 95% range.</p>
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
        <h2>By season — click a stat to chart it</h2>
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
              <p className="section-sub">Where shots came from against him, both ends folded to one — toggle shot volume, goals allowed, goal % (chance a shot from there beat him), and average shot quality. Net at right.</p>
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
