"use client";
import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { IndividualKey, MetricKey, OniceKey, PlayerDetail, SeasonRow } from "@/lib/types";
import { pctColor } from "@/lib/format";

const seasonLabel = (s: number) => `${s}-${String(s + 1).slice(-2)}`;

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
        <div className="player-head">
          <h2 className="player-name">{p.name}</h2>
          <span className="player-meta">
            {p.pos} · {p.group === "D" ? "Defenseman" : "Forward"} · {p.teams.join(", ")}
          </span>
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
            {p.per_season.map((r) => (
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
    </div>
  );
}
