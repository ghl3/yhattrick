import type { TimelineEvent } from "@/lib/types";

// Full NHL rink. Coords: x in [-100,100] ft, y in [-42.5,42.5]; goal lines at x=±89, blue at ±25.
const FT = 2;
const W = 200 * FT; // 400
const H = 85 * FT; // 170
const px = (x: number) => (x + 100) * FT;
const py = (y: number) => (y + 42.5) * FT;

const RED = "#e2a3a3";
const BLUE = "#aecbe8";

function xgColor(xg?: number) {
  const t = Math.min(1, (xg ?? 0) / 0.5);
  return `rgb(${Math.round(74 + t * 133)},${Math.round(144 - t * 65)},${Math.round(217 - t * 138)})`;
}
const TYPE_COLOR: Record<string, string> = {
  faceoff: "#6b7e90",
  hit: "#c98a1c",
  giveaway: "#cf4f4f",
  takeaway: "#3a9d6a",
  penalty: "#c98a1c",
  "blocked-shot": "#4a90d9",
};
function dotColor(e: TimelineEvent) {
  if (e.type === "goal") return "#cf4f4f";
  if (e.xg != null) return xgColor(e.xg);
  return TYPE_COLOR[e.type] ?? "#4a90d9";
}

function FaceoffCircle({ x, y }: { x: number; y: number }) {
  return (
    <>
      <circle cx={px(x)} cy={py(y)} r={15 * FT} fill="none" stroke={RED} strokeWidth={1} opacity={0.7} />
      <circle cx={px(x)} cy={py(y)} r={2} fill={RED} opacity={0.8} />
    </>
  );
}

function Crease({ side }: { side: 1 | -1 }) {
  const gx = px(89 * side);
  const dir = -side * 6 * FT; // bulge toward center
  return (
    <path
      d={`M ${gx} ${py(-4)} Q ${gx + dir} ${py(0)} ${gx} ${py(4)} Z`}
      fill="#e6f1fb"
      stroke={BLUE}
      strokeWidth={1}
    />
  );
}

function Rink({ e }: { e: TimelineEvent }) {
  const hasPt = e.x != null && e.y != null;
  return (
    <svg className="rink" viewBox={`0 0 ${W} ${H}`} width={W} height={H}>
      <rect x={1} y={1} width={W - 2} height={H - 2} rx={28 * FT} ry={28 * FT} fill="#f7fbff" stroke="#cfe0f0" />
      {/* blue lines + center red line */}
      <line x1={px(-25)} y1={1} x2={px(-25)} y2={H - 1} stroke={BLUE} strokeWidth={2} />
      <line x1={px(25)} y1={1} x2={px(25)} y2={H - 1} stroke={BLUE} strokeWidth={2} />
      <line x1={px(0)} y1={1} x2={px(0)} y2={H - 1} stroke={RED} strokeWidth={2} />
      {/* goal lines */}
      <line x1={px(-89)} y1={py(-37)} x2={px(-89)} y2={py(37)} stroke={RED} strokeWidth={1} />
      <line x1={px(89)} y1={py(-37)} x2={px(89)} y2={py(37)} stroke={RED} strokeWidth={1} />
      <Crease side={-1} />
      <Crease side={1} />
      {/* nets */}
      <rect x={px(-89) - 8} y={py(-2)} width={8} height={py(2) - py(-2)} fill="none" stroke={RED} strokeWidth={1} />
      <rect x={px(89)} y={py(-2)} width={8} height={py(2) - py(-2)} fill="none" stroke={RED} strokeWidth={1} />
      {/* center circle */}
      <circle cx={px(0)} cy={py(0)} r={15 * FT} fill="none" stroke={BLUE} strokeWidth={1} opacity={0.7} />
      <circle cx={px(0)} cy={py(0)} r={2.5} fill={BLUE} />
      {/* zone faceoff circles + neutral dots */}
      <FaceoffCircle x={69} y={22} />
      <FaceoffCircle x={69} y={-22} />
      <FaceoffCircle x={-69} y={22} />
      <FaceoffCircle x={-69} y={-22} />
      {[-20, 20].map((x) =>
        [-22, 22].map((y) => <circle key={`${x},${y}`} cx={px(x)} cy={py(y)} r={2} fill={RED} opacity={0.7} />)
      )}
      {/* the event */}
      {hasPt && (
        <circle
          cx={px(e.x!)}
          cy={py(e.y!)}
          r={e.type === "goal" ? 8 : 6}
          fill={dotColor(e)}
          stroke="#1b2a3a"
          strokeWidth={e.type === "goal" ? 2 : 1}
        />
      )}
    </svg>
  );
}

export default function EventCard({ e }: { e: TimelineEvent }) {
  const isShot = e.xg != null;
  const rows: [string, string][] = [
    ["Event", e.type.replace(/-/g, " ")],
    ["Player", e.player ?? "—"],
    ["Team", e.team ?? "—"],
    ["Zone", e.zone ?? "—"],
    ["Coords", e.x != null && e.y != null ? `(${e.x}, ${e.y})` : "—"],
    ["Clock", e.clock],
  ];
  if (e.detail) rows.push(["Detail", e.detail]);
  if (isShot) {
    rows.push(
      ["xG", e.xg!.toFixed(3)],
      ["Shot type", e.shot_type ?? "—"],
      ["Distance", e.distance != null ? `${e.distance} ft` : "—"],
      ["Angle", e.angle != null ? `${e.angle}°` : "—"],
      ["Rush / Rebound", `${e.rush ? "rush" : "—"} / ${e.rebound ? "rebound" : "—"}`]
    );
  }
  return (
    <div className="shot-card">
      <Rink e={e} />
      <table className="shot-meta">
        <tbody>
          {rows.map(([k, v]) => (
            <tr key={k}>
              <th>{k}</th>
              <td>{v}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
