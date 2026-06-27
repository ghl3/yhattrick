"use client";
import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  flexRender,
  getCoreRowModel,
  getFilteredRowModel,
  getSortedRowModel,
  useReactTable,
  type ColumnDef,
  type SortingState,
} from "@tanstack/react-table";
import type { PlayerRow } from "@/lib/types";
import { pctColor } from "@/lib/format";

type View = "impact" | "onice" | "individual";
const num3 = (v: number) => v.toFixed(3);
const num2 = (v: number) => v.toFixed(2);
const num1 = (v: number) => v.toFixed(1);

// metric columns per view: modeled (isolated) impact vs raw on-ice rates
type MetricCol = { key: keyof PlayerRow; label: string; title: string; fmt: (v: number) => string };
const VIEWS: Record<View, MetricCol[]> = {
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
    { key: "fin_per100", label: "Fin/100", title: "Finishing: goals above expected per 100 shots (regressed)", fmt: num2 },
    { key: "shots60", label: "Sh/60", title: "Unblocked shots per 60", fmt: num1 },
    { key: "xg_per_shot", label: "xG/sh", title: "Average shot quality (xG per shot)", fmt: num3 },
    { key: "ixg60", label: "ixG/60", title: "Individual expected goals per 60", fmt: num2 },
    { key: "g60", label: "G/60", title: "Goals per 60", fmt: num2 },
    { key: "a60", label: "A/60", title: "Assists per 60", fmt: num2 },
  ],
};

// cell: value tinted by its within-position percentile
function metricCell(value: number | null, pctile: number | null, fmt: (v: number) => string) {
  if (value == null) return <span className="metric-cell muted">—</span>;
  return (
    <span className="metric-cell" style={{ background: pctColor(pctile) }} title={`${Math.round(pctile ?? 0)}th pctile`}>
      {fmt(value)}
    </span>
  );
}

function buildColumns(view: View): ColumnDef<PlayerRow>[] {
  return [
    {
      header: "Player",
      accessorKey: "name",
      cell: (c) => <Link href={`/player/${c.row.original.id}`}>{c.getValue<string>()}</Link>,
    },
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

export default function Players() {
  const [rows, setRows] = useState<PlayerRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [globalFilter, setGlobalFilter] = useState("");
  const [posFilter, setPosFilter] = useState<"ALL" | "F" | "D">("ALL");
  const [view, setView] = useState<View>("impact");
  const [sorting, setSorting] = useState<SortingState>([{ id: "ev_off", desc: true }]);

  useEffect(() => {
    fetch(`/data/players.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then(setRows)
      .catch((e) => setError(String(e)));
  }, []);

  const columns = useMemo(() => buildColumns(view), [view]);
  const switchView = (v: View) => {
    setView(v);
    setSorting([{ id: VIEWS[v][0].key as string, desc: true }]); // sort by the view's first metric
  };

  const data = useMemo(
    () => (rows ?? []).filter((r) => posFilter === "ALL" || r.group === posFilter),
    [rows, posFilter]
  );

  const table = useReactTable({
    data,
    columns,
    state: { sorting, globalFilter },
    onSortingChange: setSorting,
    onGlobalFilterChange: setGlobalFilter,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
  });

  if (error)
    return <div className="loading">Failed to load players.json ({error}). Run <code>uv run python -m yhattrick.export_players</code>.</div>;
  if (!rows) return <div className="loading">Loading players…</div>;

  return (
    <div className="panel">
      <div className="toolbar">
        <input className="search" placeholder="Search player…" value={globalFilter} onChange={(e) => setGlobalFilter(e.target.value)} />
        <div className="seg">
          {(["ALL", "F", "D"] as const).map((g) => (
            <button key={g} className={posFilter === g ? "active" : ""} onClick={() => setPosFilter(g)}>
              {g === "ALL" ? "All" : g}
            </button>
          ))}
        </div>
        <div className="seg">
          {(["impact", "individual", "onice"] as const).map((v) => (
            <button key={v} className={view === v ? "active" : ""} onClick={() => switchView(v)}>
              {v === "impact" ? "Isolated impact" : v === "individual" ? "Individual rates" : "Team rates"}
            </button>
          ))}
        </div>
        <span className="muted">{data.length} players · color = percentile vs. position</span>
      </div>
      <table className="games ptable">
        <thead>
          {table.getHeaderGroups().map((hg) => (
            <tr key={hg.id}>
              {hg.headers.map((h) => (
                <th key={h.id} title={(h.column.columnDef.meta as { title?: string })?.title} onClick={h.column.getToggleSortingHandler()}>
                  {flexRender(h.column.columnDef.header, h.getContext())}
                  {{ asc: " ▲", desc: " ▼" }[h.column.getIsSorted() as string] ?? ""}
                </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody>
          {table.getRowModel().rows.map((row) => (
            <tr key={row.id}>
              {row.getVisibleCells().map((cell) => (
                <td key={cell.id}>{flexRender(cell.column.columnDef.cell, cell.getContext())}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
