import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "About",
  description:
    "An independent NHL analytics site: an in-house expected-goals model, isolated player-impact ratings, descriptive stats, and a shift-by-shift view of every game — all from public NHL data.",
};

export default function About() {
  return (
    <div className="prose">
      <div className="panel about-hero">
        <div>
          <h1><span className="brand-y">ŷ</span>Trick</h1>
          <p className="lede">
            An independent NHL analytics site: an in-house expected-goals model, isolated
            player-impact ratings, descriptive stats, and a shift-by-shift view of every game — all
            built from public NHL data.
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
            <strong>Player Impact Models.</strong>{" "}
            For each skater, how much they raise their team&apos;s expected-goal rate and suppress the
            opponent&apos;s — at even strength, on the power play, and on the penalty kill — separated
            from linemates and competition, plus finishing (goals scored above what the shots were
            worth), with within-position percentiles.
          </li>
          <li>
            <strong>Descriptive stats.</strong> Box score, on-ice team rates (xG, Corsi, shares), and
            individual rates — shot volume and quality, scoring, penalties.
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

        <h3>Player Impact Models</h3>
        <p>
          The impact ratings are a regularized adjusted plus-minus model. Every stint becomes an
          observation: the response is a team&apos;s expected goals per 60 minutes of that ice time,
          and the predictors are indicators for who is on the ice — an offense term for each attacker
          and a defense term for each defender. Ridge regression untangles teammates from opponents;
          even strength and special teams are fit separately, regular season only, and pooled across
          seasons, with a 95% confidence interval on every rating.
        </p>
        <p>
          Finishing is a related player model: a skater&apos;s goals above what their shots were
          expected to yield, shrunk toward zero until the sample is large enough to trust.
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
          Everything derived — the xG model, impact ratings, finishing, and the box, on-ice, and
          individual stats — is computed in-house from those raw feeds.
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
