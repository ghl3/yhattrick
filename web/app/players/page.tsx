"use client";
import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { type ColumnDef, type SortingState } from "@tanstack/react-table";
import type { GoalieRow, PlayerRow } from "@/lib/types";
import { pctColor } from "@/lib/format";
import { DataTable } from "@/components/DataTable";

type View = "value" | "impact" | "onice" | "individual";
type Pos = "SKATERS" | "F" | "D" | "G";
const num3 = (v: number) => v.toFixed(3);
const num2 = (v: number) => v.toFixed(2);
const num1 = (v: number) => v.toFixed(1);
const svFmt = (v: number) => v.toFixed(3).replace(/^0/, ""); // .915

// metric columns per view: modeled (isolated) impact vs raw on-ice rates
type MetricCol = { key: keyof PlayerRow; label: string; title: string; fmt: (v: number) => string };
const VIEWS: Record<View, MetricCol[]> = {
  value: [
    { key: "gnet_pg", label: "Net G/G", title: "Net goals added per game (top-line modeled value)", fmt: num2 },
    { key: "g_net", label: "Net G", title: "Net goals added — window total, all situations", fmt: num1 },
    { key: "scoring60", label: "Scoring", title: "Scoring: goals from his own shots per 60 (expected value + finishing)", fmt: num2 },
    { key: "playmaking60", label: "Playmaking", title: "Playmaking: expected goals he creates for teammates per 60 (on-ice offense − own shots)", fmt: num2 },
    { key: "allow60", label: "Defense", title: "Defense: expected goals allowed per 60 at 5-on-5, his share (lower is better)", fmt: num2 },
    { key: "pk_allow60", label: "PK Def", title: "Penalty-kill defense: expected goals allowed per 60, his share (lower is better)", fmt: num2 },
    { key: "pen_net60", label: "Pen", title: "Net penalty goals per 60 (drawn − taken × penalty value)", fmt: num2 },
  ],
  impact: [
    { key: "ev_off", label: "EV O", title: "Even-strength offense impact, isolated (xGF/60 added)", fmt: num2 },
    { key: "ev_def", label: "EV D", title: "Even-strength defense impact, isolated (xGA/60 suppressed)", fmt: num2 },
    { key: "pp_off", label: "PP", title: "Power-play offense impact, isolated (xGF/60)", fmt: num2 },
    { key: "pk_def", label: "PK", title: "Penalty-kill defense impact, isolated (xGA/60 suppressed)", fmt: num2 },
  ],
  onice: [
    { key: "ev_xgf60", label: "xGF/60", title: "5v5 on-ice expected goals for / 60", fmt: num2 },
    { key: "ev_xga60", label: "xGA/60", title: "5v5 on-ice expected goals against / 60", fmt: num2 },
    { key: "ev_cf60", label: "CF/60", title: "5v5 on-ice Corsi for / 60", fmt: num1 },
    { key: "ev_ca60", label: "CA/60", title: "5v5 on-ice Corsi against / 60", fmt: num1 },
  ],
  individual: [
    { key: "fin_per100", label: "Finishing", title: "Finishing: goals above expected per 100 shots (shrunk)", fmt: num2 },
    { key: "shots60", label: "Sh/60", title: "Unblocked shots per 60", fmt: num1 },
    { key: "xg_per_shot", label: "xG/sh", title: "Average shot quality (xG per shot)", fmt: num3 },
    { key: "ixg60", label: "ixG/60", title: "Individual expected goals per 60", fmt: num2 },
    { key: "g60", label: "G/60", title: "Goals per 60", fmt: num2 },
    { key: "a60", label: "A/60", title: "Assists per 60", fmt: num2 },
  ],
};

// goalie leaderboard columns (single pool; percentiles among goalies)
type GMetricCol = { key: keyof GoalieRow; label: string; title: string; fmt: (v: number) => string };
const GVIEW: GMetricCol[] = [
  { key: "gsax_per100", label: "GSAx/100", title: "Goals saved above expected per 100 shots (regressed)", fmt: num2 },
  { key: "gsax60", label: "GSAx/60", title: "Goals saved above expected per 60 minutes", fmt: num2 },
  { key: "sv_pct", label: "Sv%", title: "Save %", fmt: svFmt },
  { key: "gaa", label: "GAA", title: "Goals-against average", fmt: num2 },
  { key: "hd_sv_pct", label: "HDSv%", title: "High-danger save %", fmt: svFmt },
  { key: "qs_pct", label: "QS%", title: "Quality-start %", fmt: (v) => `${Math.round(v * 100)}%` },
];

// cell: value tinted by its percentile
function metricCell(value: number | null, pctile: number | null, fmt: (v: number) => string) {
  if (value == null) return <span className="metric-cell muted">—</span>;
  return (
    <span className="metric-cell" style={{ background: pctColor(pctile) }} title={`${Math.round(pctile ?? 0)}th pctile`}>
      {fmt(value)}
    </span>
  );
}

function teamCell(t: string) {
  return t ? <Link href={`/team/${t}`} onClick={(e) => e.stopPropagation()}>{t}</Link> : "—";
}

function buildColumns(view: View): ColumnDef<PlayerRow>[] {
  return [
    { header: "Player", accessorKey: "name", cell: (c) => c.getValue<string>() },
    { header: "Team", accessorKey: "team", cell: (c) => teamCell(c.getValue<string>()) },
    { header: "Pos", accessorKey: "pos" },
    { header: "EV TOI", accessorKey: "ev_toi", cell: (c) => <span className="num">{Math.round(c.getValue<number>())}</span> },
    ...VIEWS[view].map(
      (m): ColumnDef<PlayerRow> => ({
        header: m.label,
        accessorKey: m.key,
        meta: { title: m.title },
        cell: (c) => metricCell(
          c.getValue<number | null>(),
          c.row.original[`${m.key}_pct` as keyof PlayerRow] as number | null,
          m.fmt,
        ),
      })
    ),
  ];
}

function goalieColumns(): ColumnDef<GoalieRow>[] {
  return [
    { header: "Goalie", accessorKey: "name", cell: (c) => c.getValue<string>() },
    { header: "Team", accessorKey: "team", cell: (c) => teamCell(c.getValue<string>()) },
    { header: "GP", accessorKey: "gp", cell: (c) => <span className="num">{c.getValue<number>()}</span> },
    ...GVIEW.map(
      (m): ColumnDef<GoalieRow> => ({
        header: m.label,
        accessorKey: m.key,
        meta: { title: m.title },
        cell: (c) => metricCell(
          c.getValue<number | null>(),
          c.row.original[`${m.key}_pct` as keyof GoalieRow] as number | null,
          m.fmt,
        ),
      })
    ),
  ];
}

export default function Players() {
  const [rows, setRows] = useState<PlayerRow[] | null>(null);
  const [goalies, setGoalies] = useState<GoalieRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [globalFilter, setGlobalFilter] = useState("");
  const [posFilter, setPosFilter] = useState<Pos>("SKATERS");
  const [view, setView] = useState<View>("value");
  const [sorting, setSorting] = useState<SortingState>([{ id: "gnet_pg", desc: true }]);
  const [gsorting, setGsorting] = useState<SortingState>([{ id: "gsax_per100", desc: true }]);

  useEffect(() => {
    Promise.all([
      fetch(`/data/players.json`).then((r) => (r.ok ? r.json() : Promise.reject(r.status))),
      fetch(`/data/goalies.json`).then((r) => (r.ok ? r.json() : Promise.reject(r.status))),
    ])
      .then(([p, g]) => { setRows(p); setGoalies(g); })
      .catch((e) => setError(String(e)));
    // deep-link from a goalie page (/players?pos=G)
    const q = new URLSearchParams(window.location.search).get("pos");
    if (q === "G" || q === "F" || q === "D") setPosFilter(q);
  }, []);

  const isG = posFilter === "G";
  const columns = useMemo(() => buildColumns(view), [view]);
  const gcolumns = useMemo(() => goalieColumns(), []);
  const switchView = (v: View) => {
    setView(v);
    setSorting([{ id: VIEWS[v][0].key as string, desc: true }]); // sort by the view's first metric
  };

  const data = useMemo(
    () => (rows ?? []).filter((r) => posFilter === "SKATERS" || r.group === posFilter),
    [rows, posFilter]
  );

  if (error)
    return <div className="loading">Failed to load index ({error}). Run <code>make players goalies</code>.</div>;
  if (!rows || !goalies) return <div className="loading">Loading players…</div>;

  return (
    <div className="panel">
      <div className="toolbar">
        <input className="search" placeholder={isG ? "Search goalie…" : "Search player…"} value={globalFilter} onChange={(e) => setGlobalFilter(e.target.value)} />
        <div className="seg">
          {(["SKATERS", "F", "D", "G"] as const).map((g) => (
            <button key={g} className={posFilter === g ? "active" : ""} onClick={() => setPosFilter(g)}>
              {g === "SKATERS" ? "Skaters" : g}
            </button>
          ))}
        </div>
        {!isG && (
          <div className="seg">
            {(["value", "impact", "individual", "onice"] as const).map((v) => (
              <button key={v} className={view === v ? "active" : ""} onClick={() => switchView(v)}>
                {v === "value" ? "Value" : v === "impact" ? "Isolated impact" : v === "individual" ? "Individual rates" : "Team rates"}
              </button>
            ))}
          </div>
        )}
        <span className="muted">
          {isG ? `${goalies.length} goalies` : `${data.length} skaters`} · color = percentile vs. {isG ? "goalies" : "position"}
        </span>
      </div>
      {isG ? (
        <DataTable
          data={goalies}
          columns={gcolumns}
          sorting={gsorting}
          onSortingChange={setGsorting}
          globalFilter={globalFilter}
          rowHref={(r) => `/player/${r.id}`}
          className="games ptable"
        />
      ) : (
        <DataTable
          data={data}
          columns={columns}
          sorting={sorting}
          onSortingChange={setSorting}
          globalFilter={globalFilter}
          rowHref={(r) => `/player/${r.id}`}
          className="games ptable"
        />
      )}
    </div>
  );
}
