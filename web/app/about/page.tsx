import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "About",
  description:
    "An independent NHL analytics site: an in-house expected-goals model, player cards inferred by a generative player model, descriptive stats, and a shift-by-shift view of every game — all from public NHL data.",
};

export default function About() {
  return (
    <div className="prose">
      <div className="panel about-hero">
        <div>
          <h1><span className="brand-y">ŷ</span>Trick</h1>
          <p className="lede">
            An independent NHL analytics site: an in-house expected-goals model, player cards
            inferred by a generative player model, descriptive stats, and a shift-by-shift view of
            every game — all built from public NHL data.
          </p>
        </div>
      </div>

      <div className="panel">
        <h2>What&apos;s here</h2>
        <ul>
          <li>
            <strong>Expected Goals Model.</strong> An in-house, per-shot model that scores every
            unblocked shot by its chance of going in, using shot geometry and pre-shot context.
          </li>
          <li>
            <strong>Player cards.</strong>{" "}
            For each skater, the skills a generative model infers from every shot and shift —
            playmaking, shooting, scoring, finishing, defense — isolated from linemates,
            competition, and arena scorekeeping, plus two aggregates: <strong>Net Goals Added
            per 60</strong> (skill vs a replacement-level player on equal footing) and{" "}
            <strong>WAR</strong> (wins added over his actual season vs a replacement player). Each
            card shows units and a within-position percentile.
          </li>
          <li>
            <strong>Descriptive stats.</strong> Box score and individual rates — shot volume and
            quality, scoring, penalties.
          </li>
          <li>
            <strong>Per-player views.</strong> Shot maps, a percentile profile, a full game log, and
            bio.
          </li>
          <li>
            <strong>Game inspection.</strong> Every game broken into <em>stints</em> — intervals of
            constant on-ice personnel — with the exact players, strength state, shot events, and an
            expected-goals tally for each.
          </li>
        </ul>
      </div>

      <div className="panel">
        <h2>The pages</h2>
        <ul>
          <li><strong>Players</strong> — a sortable leaderboard; click through to a player&apos;s full card.</li>
          <li><strong>Teams</strong> — each team&apos;s roster and schedule.</li>
          <li><strong>Games</strong> — every game, newest first; open one for its stint-by-stint timeline.</li>
          <li><strong>Models</strong> — the expected-goals model: its danger map, calibration, and what drives it.</li>
          <li><strong>About</strong> — this page.</li>
        </ul>
      </div>

      <div className="panel">
        <h2>The models</h2>

        <h3>Expected Goals Model</h3>
        <p>
          A gradient-boosted model scores each unblocked shot using its location and pre-shot context
          — rebound, rush, shot type, and strength and score state, among others.
        </p>

        <h3>The Player Model</h3>
        <p>
          The player cards come from a <strong>generative model</strong>: it writes down how a shift
          produces shots and goals — who shoots, who creates for teammates, who suppresses, how
          dangerous the shots are, who converts them — and fits one latent skill per player for each
          of those verbs, jointly across five seasons. Every skill is isolated from linemates,
          competition, and arena scorekeeping (the model carries explicit terms for each), and each
          player gets a per-season skill trajectory with a shared aging curve and a next-season
          projection. Displayed skills always include the player&apos;s age — the aging curve is for
          understanding and projection, never a normalizer. Assists (primary and secondary) anchor playmaking; the model validates against
          held-out seasons before we publish it.
        </p>
        <p>
          <strong>Net Goals Added /60</strong> plugs a player&apos;s inferred skills into the model&apos;s
          own production equations and differences against a replacement-level player — an
          equal-footing skill read on the same zero as WAR. <strong>WAR</strong> replays his actual season, swapping him for
          a replacement-level player in every real stint, and converts the goal difference to wins.
          The full method, formulas, and honest caveats are in <code>docs/metrics.md</code>; the model
          itself is specified in <code>docs/generative_model.md</code>.
        </p>
      </div>

      <div className="panel">
        <h2>Data &amp; credit</h2>
        <p>
          Everything here is built from <strong>public NHL data</strong>: the play-by-play API, shift
          charts (with the NHL HTML time-on-ice reports as a fallback), player landing (bios and
          handedness), and the league schedule and standings. Team logos and player headshots are
          served from the NHL&apos;s asset CDN.
        </p>
        <p>
          Everything derived — the xG model, the generative player model behind the cards, and the
          box, on-ice, and individual stats — is computed in-house from those raw feeds.
        </p>
        <p>
          The ratings and analysis here are free to use. The underlying NHL data remains the
          NHL&apos;s and stays subject to its terms of use, so please respect those when reusing it.
        </p>
        <p className="muted card-note">
          ŷTrick is an independent project and is not affiliated with or endorsed by the National
          Hockey League.
        </p>
      </div>
    </div>
  );
}
