// Loading placeholder for the game view. Pure markup (no hooks), so it works both as the Next.js
// route-level `loading.tsx` fallback (shown instantly on navigation, before the page JS/data load)
// and as the in-component fallback while the per-game JSON streams in from object storage.

function TimelineRows({ n }: { n: number }) {
  return (
    <div style={{ marginTop: 14 }}>
      {Array.from({ length: n }).map((_, i) => (
        <div className="sk-row" key={i}>
          <span className="sk" style={{ width: 90, height: 13 }} />
          <span className="sk" style={{ width: 46, height: 13 }} />
          <span className="sk" style={{ width: `${30 + ((i * 17) % 45)}%`, height: 13 }} />
          <span className="sk" style={{ width: 60, height: 13, marginLeft: "auto" }} />
        </div>
      ))}
    </div>
  );
}

export function TimelineSkeleton() {
  return <TimelineRows n={6} />;
}

export default function GameSkeleton() {
  return (
    <>
      <span className="backlink sk" style={{ width: 70, height: 14 }} />
      <div className="panel">
        <div className="gv-header">
          <span className="sk" style={{ width: 260, height: 26 }} />
        </div>
        <div className="statgrid" style={{ marginTop: 14 }}>
          {[0, 1, 2].map((i) => (
            <div className="stat" key={i}>
              <span className="sk" style={{ width: 90, height: 18 }} />
              <span className="sk" style={{ width: 110, height: 11, marginTop: 6 }} />
            </div>
          ))}
        </div>
      </div>
      <div className="panel">
        <span className="sk" style={{ width: 220, height: 18 }} />
        <TimelineRows n={8} />
      </div>
    </>
  );
}
