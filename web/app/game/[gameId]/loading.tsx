// Next.js App Router shows this instantly during navigation to /game/[id] — i.e. while the route's
// JS and data are still loading — giving immediate feedback before GameView mounts.
import GameSkeleton from "./GameSkeleton";

export default function Loading() {
  return <GameSkeleton />;
}
