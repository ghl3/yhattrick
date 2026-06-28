"use client";
import { useParams } from "next/navigation";
import GameContent from "./GameContent";

// loading.tsx covers the navigation/route-load phase; GameContent fetches its JSON via SWR and
// renders its own skeleton while that's in flight (then streams the heavy timeline in deferred).
export default function GamePage() {
  const { gameId } = useParams<{ gameId: string }>();
  return <GameContent gameId={gameId} />;
}
