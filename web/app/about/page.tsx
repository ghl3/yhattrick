import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "About",
  description:
    "What ŷTrick is, how the isolated-impact model works, where the data comes from, and how to read the numbers.",
};

export default function About() {
  return (
    <div className="prose">
      <div className="panel about-hero">
        <div>
          <h1><span className="brand-y">ŷ</span>Trick</h1>
          <p className="lede">
            A from-scratch NHL player-value project: isolated even-strength and special-teams
            impact ratings, plus a browsable, shift-by-shift view of every game.
          </p>
        </div>
      </div>

      <div className="panel">
        <h2>The name</h2>
        <p>
          In statistics, a <em>hat</em> marks a predicted value — <strong>ŷ</strong> (“y-hat”) is the
          model’s estimate of an outcome. In hockey, a <strong>hat trick</strong> is three goals. The
          two share that hat: the <strong>ŷ</strong> supplies it, so <strong>ŷ</strong> + “Trick” said
          aloud is exactly “y-hat-trick.” Hence the mark <strong>ŷ&#8202;Trick</strong> — and the
          domain spells it out as <strong>yhattrick.com</strong>.
        </p>
      </div>

      <div className="panel">
        <h2>What it does</h2>
        <p>Two things, from the same underlying data:</p>
        <ul>
          <li>
            <strong>Isolated impact ratings.</strong> For every skater, an estimate of how much they
            raise their team’s expected-goal rate (offense) and suppress the opponent’s (defense),
            separated from the quality of their linemates and competition.
          </li>
          <li>
            <strong>Game inspection.</strong> Each game broken into <em>stints</em> — intervals of
            constant on-ice personnel — with the exact players, strength state, shot events, and an
            expected-goals tally for each, on a rink map.
          </li>
        </ul>
      </div>

      <div className="panel">
        <h2>How the model works</h2>
        <p>
          The ratings are a regularized adjusted plus-minus (RAPM). Every stint becomes an
          observation: the response is a team’s expected goals per 60 minutes of that ice time, and
          the predictors are indicators for which players are on the ice — an offense term for each
          attacker and a defense term for each defender. Solving the whole season at once with{" "}
          <strong>ridge regression</strong> untangles teammates from opponents, because players who
          appear in many different combinations can be separated.
        </p>
        <ul>
          <li>
            <strong>Strength states.</strong> Even strength (5v5) and special teams (5v4 power play,
            4v5 penalty kill) are fit separately so power-play offense and penalty-kill defense stay
            clean.
          </li>
          <li>
            <strong>Context adjustment.</strong> Shared covariates — home ice, offensive/defensive
            zone start, score state, period, and season — absorb deployment, score, and era effects
            so player coefficients better reflect skill.
          </li>
          <li>
            <strong>Regular season only,</strong> pooled across multiple seasons for a stronger
            signal. Playoffs and shootouts are tracked separately and never enter the model.
          </li>
          <li>
            <strong>Uncertainty is shown.</strong> Each rating carries a 95% confidence interval that
            widens with low ice time and with linemates who rarely separate; percentiles are computed
            within position group.
          </li>
        </ul>
      </div>

      <div className="panel">
        <h2>Where the data comes from</h2>
        <p>
          Everything is computed from raw NHL sources — play-by-play and shift charts — joined
          shift-to-shot in-house: box scores, on-ice personnel, shot volume (Corsi/Fenwick/shots on
          goal), score and zone context, the impact model, and our own per-shot{" "}
          <em>expected-goals</em> model. No third-party model outputs.
        </p>
      </div>

      <div className="panel">
        <h2>Caveats</h2>
        <ul>
          <li>
            Linemates who almost never play apart are hard to separate; their shared impact can land
            unevenly on one of them. Pooling seasons helps; a hierarchical model will help more.
          </li>
          <li>
            The most recent season is still filling in (not every game has shift data yet), so its
            sample is partial.
          </li>
          <li>
            These are impact components, not a finished Wins Above Replacement number. Finishing,
            penalties, and the goals-to-wins conversion are still to come.
          </li>
        </ul>
        <p className="muted card-note">
          Expected goals are a model estimate, not actual goals. Treat every rating as an estimate
          with a margin of error, not a verdict.
        </p>
      </div>
    </div>
  );
}
