"use client";
import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { MetricKey, OniceKey, PlayerDetail, SeasonRow } from "@/lib/types";
import { pctColor } from "@/lib/format";

const seasonLabel = (s: number) => `${s}-${String(s + 1).slice(-2)}`;

const IMPACT: { key: MetricKey; label: string; blurb: string }[] = [
  { key: "ev_off", label: "EV Offense", blurb: "5v5 expected goals added per 60" },
  { key: "ev_def", label: "EV Defense", blurb: "5v5 expected goals suppressed per 60" },
  { key: "pp_off", label: "PP Offense", blurb: "Power-play expected goals added per 60" },
  { key: "pk_def", label: "PK Defense", blurb: "Penalty-kill expected goals suppressed per 60" },
];

// on-ice (raw, descriptive) metrics — the team's rate while the player is on the ice
const pct1 = (v: number) => `${(v * 100).toFixed(1)}%`;
const num2 = (v: number) => v.toFixed(2);
const num1 = (v: number) => v.toFixed(1);
const ONICE: { key: OniceKey; label: string; blurb: string; fmt: (v: number) => string }[] = [
  { key: "ev_xgf60", label: "EV xGF/60", blurb: "5v5 on-ice expected goals for", fmt: num2 },
  { key: "ev_xga60", label: "EV xGA/60", blurb: "5v5 on-ice expected goals against", fmt: num2 },
  { key: "ev_xgshare", label: "EV xGF%", blurb: "Share of on-ice 5v5 expected goals", fmt: pct1 },
  { key: "ev_cf60", label: "EV CF/60", blurb: "5v5 on-ice shot attempts (Corsi) for", fmt: num1 },
  { key: "ev_ca60", label: "EV CA/60", blurb: "5v5 on-ice shot attempts against", fmt: num1 },
  { key: "ev_cfshare", label: "EV CF%", blurb: "Share of on-ice 5v5 shot attempts (Corsi %)", fmt: pct1 },
  { key: "pp_xgf60", label: "PP xGF/60", blurb: "Power-play on-ice expected goals for", fmt: num2 },
  { key: "pk_xga60", label: "PK xGA/60", blurb: "Penalty-kill on-ice expected goals against", fmt: num2 },
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
];

// shared percentile header + body shell for both metric families
function BoxShell({ pctile, group, label, blurb, children }: {
  pctile: number | null; group: string; label: string; blurb: string; children: React.ReactNode;
}) {
  const has = pctile != null;
  return (
    <div className="metric-box">
      <div className="mb-pct" style={{ background: has ? pctColor(pctile) : "var(--accent-softer)" }}>
        {has ? (
          <>
            <span className="mb-pctile">{Math.round(pctile!)}%</span>
            <span className="mb-vs">percentile · {group === "D" ? "defensemen" : "forwards"}</span>
          </>
        ) : (
          <span className="mb-vs">no qualifying ice time</span>
        )}
      </div>
      <div className="mb-body">
        <div className="mb-label">{label}</div>
        {children}
        <div className="mb-blurb">{blurb}</div>
      </div>
    </div>
  );
}

// modeled, isolated impact: signed per-60 delta with a 95% CI
function MetricBox({ label, blurb, v, se, toi, pctile, group }: {
  label: string; blurb: string; v: number | null; se: number | null; toi: number | null; pctile: number | null; group: string;
}) {
  const has = v != null && pctile != null;
  return (
    <BoxShell pctile={has ? pctile : null} group={group} label={label} blurb={blurb}>
      <div className="mb-val">
        {has ? <>{v! >= 0 ? "+" : ""}{v!.toFixed(3)} <span className="mb-ci">± {(1.96 * (se ?? 0)).toFixed(3)}</span></> : <span className="muted">—</span>}
      </div>
      <div className="mb-toi">{toi ? `${Math.round(toi)} min` : ""}</div>
    </BoxShell>
  );
}

// raw, descriptive on-ice rate: plain value, no CI
function OniceBox({ label, blurb, v, pctile, group, fmt }: {
  label: string; blurb: string; v: number | null; pctile: number | null; group: string; fmt: (v: number) => string;
}) {
  const has = v != null && pctile != null;
  return (
    <BoxShell pctile={has ? pctile : null} group={group} label={label} blurb={blurb}>
      <div className="mb-val">{v != null ? fmt(v) : <span className="muted">—</span>}</div>
    </BoxShell>
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
        <div className="metric-grid">
          {IMPACT.map((m) => {
            const d = p.impact[m.key];
            return <MetricBox key={m.key} label={m.label} blurb={m.blurb} v={d.v} se={d.se} toi={d.toi} pctile={d.pct} group={p.group} />;
          })}
        </div>
        <p className="muted card-note">
          Even-strength &amp; special-teams impact on expected goals, <strong>adjusted</strong> for linemates and
          competition (ridge, regular season). Percentiles are within position group; ± is a 95% confidence interval.
        </p>
      </div>

      <div className="panel">
        <h2>On-ice rates</h2>
        <div className="metric-grid">
          {ONICE.map((m) => {
            const d = p.onice[m.key];
            return <OniceBox key={m.key} label={m.label} blurb={m.blurb} v={d.v} pctile={d.pct} group={p.group} fmt={m.fmt} />;
          })}
        </div>
        <p className="muted card-note">
          Raw, <strong>un-adjusted</strong> team rates while the player was on the ice (5v5 unless noted),
          regular season. Unlike isolated impact these aren&apos;t separated from linemates or competition —
          comparing the two shows how much of a player&apos;s on-ice results he himself drives.
        </p>
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
