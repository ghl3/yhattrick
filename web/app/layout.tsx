import type { Metadata } from "next";
import "./globals.css";
import Navbar from "@/components/Navbar";

export const metadata: Metadata = {
  title: {
    default: "ŷTrick — isolated NHL player impact",
    template: "%s · ŷTrick",
  },
  description:
    "ŷTrick: a from-scratch NHL player-impact model. Even-strength and special-teams " +
    "expected-goals ratings isolated from linemates and competition, with a browsable, " +
    "shift-by-shift view of every game. Regular season, built from raw NHL data.",
  applicationName: "ŷTrick",
  keywords: ["NHL", "hockey analytics", "WAR", "GAR", "expected goals", "RAPM", "player impact"],
  openGraph: {
    title: "ŷTrick — isolated NHL player impact",
    description:
      "Even-strength & special-teams expected-goals ratings, isolated from linemates and " +
      "competition, with a shift-by-shift view of every game.",
    type: "website",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Navbar />
        <div className="container">{children}</div>
      </body>
    </html>
  );
}
