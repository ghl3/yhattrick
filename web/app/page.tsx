"use client";
import { useEffect, useMemo, useState } from "react";
import { type ColumnDef, type SortingState } from "@tanstack/react-table";
import type { GameIndexRow } from "@/lib/types";
import { seasonLabel } from "@/lib/format";
import { DataTable } from "@/components/DataTable";

const columns: ColumnDef<GameIndexRow>[] = [
  { header: "Date", accessorKey: "date", cell: (c) => c.getValue<string>() ?? "—" },
  { header: "Season", accessorKey: "season", cell: (c) => seasonLabel(c.getValue<number>()) },
  {
    header: "Matchup",
    id: "matchup",
    accessorFn: (r) => `${r.away} @ ${r.home}`,
    cell: (c) => `${c.row.original.away} @ ${c.row.original.home}`,
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
];

export default function GamesIndex() {
  const [rows, setRows] = useState<GameIndexRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [globalFilter, setGlobalFilter] = useState("");
  const [sorting, setSorting] = useState<SortingState>([{ id: "date", desc: true }]);

  useEffect(() => {
    fetch(`/data/games.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then(setRows)
      .catch((e) => setError(String(e)));
  }, []);

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
      <DataTable
        data={rows}
        columns={columns}
        sorting={sorting}
        onSortingChange={setSorting}
        globalFilter={globalFilter}
        rowHref={(r) => `/game/${r.game_id}`}
      />
    </div>
  );
}
