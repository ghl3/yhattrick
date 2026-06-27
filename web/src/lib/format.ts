export const mmss = (sec: number) => `${Math.floor(sec / 60)}:${String(sec % 60).padStart(2, "0")}`;

export const seasonLabel = (s: number) => `${s}-${String(s + 1).slice(-2)}`;

export const pct = (x: number | null) => (x == null ? "—" : `${(x * 100).toFixed(1)}%`);
