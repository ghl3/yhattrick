export const mmss = (sec: number) => `${Math.floor(sec / 60)}:${String(sec % 60).padStart(2, "0")}`;

export const seasonLabel = (s: number) => `${s}-${String(s + 1).slice(-2)}`;

export const pct = (x: number | null) => (x == null ? "—" : `${(x * 100).toFixed(1)}%`);

// Diverging cell tint for a 0–100 percentile: red (low) → neutral (~50) → blue (high).
export function pctColor(p: number | null | undefined): string {
  if (p == null) return "transparent";
  const d = (p - 50) / 50; // -1..1
  return d >= 0 ? `rgba(47,108,176,${(0.1 + 0.62 * d).toFixed(3)})` : `rgba(207,79,79,${(0.1 + 0.62 * -d).toFixed(3)})`;
}

export const ordinal = (p: number) => {
  const n = Math.round(p);
  const s = ["th", "st", "nd", "rd"], v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
};
