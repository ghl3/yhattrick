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
import type { GameIndexRow } from "../lib/types";
import { pct, seasonLabel } from "../lib/format";

const columns: ColumnDef<GameIndexRow>[] = [
  {
    header: "Date",
    accessorKey: "date",
    cell: (c) => (
      <Link to={`/game/${c.row.original.game_id}`}>{c.getValue<string>() ?? "—"}</Link>
    ),
  },
  { header: "Season", accessorKey: "season", cell: (c) => seasonLabel(c.getValue<number>()) },
  {
    header: "Matchup",
    id: "matchup",
    accessorFn: (r) => `${r.away} @ ${r.home}`,
    cell: (c) => (
      <Link to={`/game/${c.row.original.game_id}`}>
        {c.row.original.away} @ {c.row.original.home}
      </Link>
    ),
  },
  {
    header: "Score",
    id: "score",
    accessorFn: (r) => `${r.away_score ?? ""}-${r.home_score ?? ""}`,
    cell: (c) => (
      <span className="num">
        {c.row.original.away_score}–{c.row.original.home_score}
      </span>
    ),
  },
  { header: "Stints", accessorKey: "n_stints", cell: (c) => <span className="num">{c.getValue<number>()}</span> },
  { header: "Shots", accessorKey: "n_shots", cell: (c) => <span className="num">{c.getValue<number>()}</span> },
  { header: "Events", accessorKey: "n_events", cell: (c) => <span className="num">{c.getValue<number>()}</span> },
  {
    header: "On-ice match",
    accessorKey: "onice_exact",
    cell: (c) => <span className="num">{pct(c.getValue<number | null>())}</span>,
  },
];

export default function GamesIndex() {
  const [rows, setRows] = useState<GameIndexRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [globalFilter, setGlobalFilter] = useState("");
  const [sorting, setSorting] = useState<SortingState>([{ id: "date", desc: false }]);

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data/games.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then(setRows)
      .catch((e) => setError(String(e)));
  }, []);

  const table = useReactTable({
    data: rows ?? [],
    columns,
    state: { sorting, globalFilter },
    onSortingChange: setSorting,
    onGlobalFilterChange: setGlobalFilter,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
  });

  const subtitle = useMemo(() => (rows ? `${rows.length} games processed` : ""), [rows]);

  if (error) return <div className="loading">Failed to load games.json ({error}). Run <code>make games</code>.</div>;
  if (!rows) return <div className="loading">Loading games…</div>;

  return (
    <div className="panel">
      <div className="toolbar">
        <input
          className="search"
          placeholder="Search team, date, season…"
          value={globalFilter}
          onChange={(e) => setGlobalFilter(e.target.value)}
        />
        <span className="muted">{subtitle}</span>
      </div>
      <table className="games">
        <thead>
          {table.getHeaderGroups().map((hg) => (
            <tr key={hg.id}>
              {hg.headers.map((h) => (
                <th key={h.id} onClick={h.column.getToggleSortingHandler()}>
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
