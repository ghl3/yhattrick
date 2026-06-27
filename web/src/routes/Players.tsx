import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import {
  flexRender,
  getCoreRowModel,
  getFilteredRowModel,
  getSortedRowModel,
  useReactTable,
  type ColumnDef,
  type SortingState,
} from "@tanstack/react-table";
import type { MetricKey, PlayerRow } from "../lib/types";
import { pctColor } from "../lib/format";

const METRICS: { key: MetricKey; label: string; title: string }[] = [
  { key: "ev_off", label: "EV O", title: "Even-strength offense impact (xGF/60)" },
  { key: "ev_def", label: "EV D", title: "Even-strength defense impact (xGA/60 suppressed)" },
  { key: "pp_off", label: "PP", title: "Power-play offense impact (xGF/60)" },
  { key: "pk_def", label: "PK", title: "Penalty-kill defense impact (xGA/60 suppressed)" },
];

// cell that shows the value, tinted by its within-position percentile
function metricCell(value: number | null, pctile: number | null) {
  if (value == null) return <span className="metric-cell muted">—</span>;
  return (
    <span className="metric-cell" style={{ background: pctColor(pctile) }} title={`${Math.round(pctile ?? 0)}th pctile`}>
      {value.toFixed(2)}
    </span>
  );
}

const columns: ColumnDef<PlayerRow>[] = [
  {
    header: "Player",
    accessorKey: "name",
    cell: (c) => <Link to={`/player/${c.row.original.id}`}>{c.getValue<string>()}</Link>,
  },
  { header: "Pos", accessorKey: "pos" },
  { header: "EV TOI", accessorKey: "ev_toi", cell: (c) => <span className="num">{Math.round(c.getValue<number>())}</span> },
  ...METRICS.map(
    (m): ColumnDef<PlayerRow> => ({
      header: m.label,
      accessorKey: m.key,
      meta: { title: m.title },
      cell: (c) => metricCell(c.getValue<number | null>(), c.row.original[`${m.key}_pct` as keyof PlayerRow] as number | null),
    })
  ),
];

export default function Players() {
  const [rows, setRows] = useState<PlayerRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [globalFilter, setGlobalFilter] = useState("");
  const [posFilter, setPosFilter] = useState<"ALL" | "F" | "D">("ALL");
  const [sorting, setSorting] = useState<SortingState>([{ id: "ev_off", desc: true }]);

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data/players.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then(setRows)
      .catch((e) => setError(String(e)));
  }, []);

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
    return <div className="loading">Failed to load players.json ({error}). Run <code>uv run python -m hockeywar.export_model</code>.</div>;
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
