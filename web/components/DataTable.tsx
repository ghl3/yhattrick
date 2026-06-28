"use client";
import {
  flexRender,
  getCoreRowModel,
  getFilteredRowModel,
  getSortedRowModel,
  useReactTable,
  type ColumnDef,
  type OnChangeFn,
  type SortingState,
} from "@tanstack/react-table";
import { useRouter } from "next/navigation";

// Shared sortable table used by the games index, players index, and team roster. Columns sort on
// header click; if `rowHref` is given, the whole row highlights and navigates on click (inner
// <Link>s should call stopPropagation). Filtering is controlled via the optional `globalFilter`.
export function DataTable<T>({
  data,
  columns,
  sorting,
  onSortingChange,
  globalFilter,
  rowHref,
  className = "games gtable",
}: {
  data: T[];
  columns: ColumnDef<T>[];
  sorting: SortingState;
  onSortingChange: OnChangeFn<SortingState>;
  globalFilter?: string;
  rowHref?: (row: T) => string;
  className?: string;
}) {
  const router = useRouter();
  const table = useReactTable({
    data,
    columns,
    state: { sorting, globalFilter: globalFilter ?? "" },
    onSortingChange,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
  });

  return (
    <table className={className}>
      <thead>
        {table.getHeaderGroups().map((hg) => (
          <tr key={hg.id}>
            {hg.headers.map((h) => (
              <th
                key={h.id}
                className="sortable"
                title={(h.column.columnDef.meta as { title?: string } | undefined)?.title}
                onClick={h.column.getToggleSortingHandler()}
              >
                {flexRender(h.column.columnDef.header, h.getContext())}
                {{ asc: " ▲", desc: " ▼" }[h.column.getIsSorted() as string] ?? ""}
              </th>
            ))}
          </tr>
        ))}
      </thead>
      <tbody>
        {table.getRowModel().rows.map((row) => {
          const href = rowHref?.(row.original);
          return (
            <tr
              key={row.id}
              className={href ? "rowlink" : undefined}
              onClick={href ? () => router.push(href) : undefined}
            >
              {row.getVisibleCells().map((cell) => (
                <td key={cell.id}>{flexRender(cell.column.columnDef.cell, cell.getContext())}</td>
              ))}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
