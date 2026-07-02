# 07 — Metrics: how each player-card number is calculated

Reader-facing reference for the player page. The skater cards are powered by the **generative
player model** (spec: [`generative_model.md`](generative_model.md)); this doc explains what the
inferred parameters mean, how they become card metrics — including **Goals Added /60** and
**WAR** — and exactly which formulas run. The previous RAPM-based card definitions live on in
[`modeling.md`](modeling.md); that model still runs in the pipeline (validation, penalties,
per-season table rows) but no longer defines the headline cards.

## Philosophy — inferred skills, then two honest aggregates

The generative model writes down how a shift **produces** shots and goals — who shoots, who
creates for teammates, who suppresses, how dangerous the shots are, who converts them — and fits
one latent skill per player for each of those verbs. Cards are those **broad latent qualities**;
observed events (assists, blocks) are evidence that grounds them, never cards themselves.

Every skill is isolated from **linemates, competition, arena scorekeeping, and age** (the model
carries explicit terms for each), and is a **current-skill** read: the player's latest-season
drift state + his position's baseline + the league aging curve at his age today. Projections are
shown separately and never blended in.

Two aggregates answer the two different questions people ask:

- **Goals Added /60 (GA/60)** — *how good is he, on equal footing?* His net goal impact per 60
  vs a league-average player at his position, on a baseline team. Deployment-free; powers the
  percentiles.
- **WAR** — *how much did he actually add this season?* Expected goals with him vs a
  replacement-level player in his slot, accumulated over his **real shifts, linemates, and
  opposition**, converted to wins. Deployment-in; the counting stat.

## The inferred parameters

Per player (fit on the pooled multi-season window; details in the model doc):

| Parameter | Plain meaning | Feeds |
|---|---|---|
| `shoot` | own-shot volume vs position/age baseline (per-season drift states) | Shooting, Scoring |
| `qshoot` | danger of his own shots vs baseline | Scoring |
| `fin` | converts above what his shot locations predict (heavily shrunk — small skill) | Finishing, Scoring |
| `create` | how much more his teammates shoot with him on the ice (anchored by primary + secondary assists; per-season states) | Playmaking |
| `qcreate` (position-level) | danger a creator's setups add — estimable only per position (F ≈ neutral, D setups less dangerous) | Playmaking |
| `def`, `qdef` | opponent shot volume and danger he suppresses | Defense |
| `gsave` | goalie saves above expected (goalie pages) | — |
| aging curves + position offsets + arena states | shared context the skills are isolated FROM | everything |

PP/PK have their own volume parameters (`pp_shoot/pp_create/pp_def`); shot-quality and finishing
skills are shared across strengths.

## The value formulas (parameters → goals)

Cards are computed by **plugging the parameters into the model's own closed-form production
equations**. With the fitted intercepts (`mu_rate` = baseline log shot rate, `mu_qual` = baseline
logit shot quality, `a`,`b` = the xG→goal conversion map) and a player's effective parameters θ:

```
shots60(θ)   = exp(mu_rate + shoot)                       his own unblocked shots per 60
q_own(θ)     = sigmoid(mu_qual + qshoot)                  mean xG of his shots
p_goal(θ)    = sigmoid(a·logit(q_own) + b + fin)          goals per shot (quality × finishing)
Scoring      = shots60 · p_goal                                       [goals/60]

Playmaking   = 4 · exp(mu_rate) · (e^create − 1) · sigmoid(mu_qual + qcreate_pos)   [xG/60]
Defense      = 5 · [exp(mu_rate)·sigmoid(mu_qual) − exp(mu_rate + def)·sigmoid(mu_qual + qdef)]
                                                                       [xG erased/60]
```

**Goals Added /60** evaluates those equations for the player and differences against his
position's TOI-weighted average (the "baseline team" — so a league-average F or D is exactly 0),
pricing xG into goals with κ = the league goals-per-xG from the conversion fit (≈ 1 by
calibration, applied explicitly):

```
GA/60 = [Scoring − Scorinḡ_pos] + κ·[Playmaking − Playmakinḡ_pos] + κ·[Defense − Defensē_pos]
```

PP GA/60 and PK GA/60 are the same construction on the power-play/penalty-kill bucket (PP = own
scoring + creation above position average; PK = chance value erased above position average).

## WAR — wins above replacement, over his actual season

For every real stint he played, the model computes the expected goals for and against **with him
on the ice** (his linemates', opponents', and goalie's actual parameters, the stint's actual zone
start / score / venue context), then recomputes with him swapped for a **replacement-level player
at his position** — and accumulates the difference:

```
GAR  = Σ_stints  [ E(GF − GA | actual lineup, him) − E(GF − GA | him → replacement) ]
WAR  = GAR ÷ goals-per-win
```

The swap is closed-form (the rate model is exponential-additive), computed per season with his
per-season drift states, across EV + PP + PK.

**Calibration knobs (explicit and revisable):**
- **Replacement level**: the TOI-weighted average parameters of players in the **8th–12th
  percentile of GA/60** within his position — an empirical "freely available player" archetype.
- **Goals-per-win = 6.0** (the standard rule of thumb; a season-estimated value is a listed
  follow-up).

**Honest caveats:** penalties are *not* in WAR yet (the production penalty value is shown as its
own card until the penalties stage joins the generative model); defender shot-quality suppression
enters as a multiplicative factor (an approximation documented in the exporter); WAR ships without
confidence intervals until the parametric bootstrap lands.

**Scale note — our WAR runs hotter than public WAR models (audited 2026-07).** Top seasons here
land around 9–12 wins where Evolving-Hockey-style models top out near 5–6; the league sum is
≈ 1,390 WAR vs the ~700 public models normalize to. An explicit audit (per-team joint
counterfactuals vs summed player marginals, 2025 season) decomposed the gap:

- **+13% level inflation — a known approximation, fix planned.** The closed-form WAR engine
  converts shots at each shooter's *mean* shot quality; over a skewed xG distribution that
  overstates goals (model ΣE[GF] 7,714 vs 6,811 actual in the fitted shot set). A per-bucket
  reconciliation factor (Σp = Σy, as the conversion stage already does) removes it.
- **×1.16 from summing marginals.** Replacing a whole team with replacements (joint) is worth
  ~16% less than the sum of its players' individual swaps — the multiplicative lineup synergy is
  counted in every member's marginal. Real but modest; not the main driver.
- **The rest is a genuinely deeper replacement level / wider skill spread** than public RAPM-style
  models (replacement ≈ −0.48 GA/60 for F, −0.33 for D below position average). The rate-skill
  spread behind it is validated out-of-sample (holdout corr 0.84, calibrated totals).

**Known distortion — PP credit inside long-lived units.** Within a five-man unit that plays together
for years, the count data only pins the unit's *total* creation; individual shares are identified
mostly by the assist anchor. Assist-light net-front players on elite PP1s become residual sinks
(fitted `pp_create` strongly negative), and the swap then claims replacing them would *raise* the
unit's shot volume — e.g. Hyman's PP WAR is not credible, and star PP reads absorb the mirror-image
inflation. Treat PP WAR components on long-fixed units with caution until the roadmap fix
(Shapley-style / identification-aware unit attribution) lands. Team-level accounting is sound:
joint team GAR correlates 0.85 with actual goal differential and 0.79 with standings points.

Rankings, the replacement zero, and EV reads are unaffected. Compare WAR **within this site**, not
against other models' numbers.

## Card-by-card

Headline row — the complete value summary:

| Card | What it is | Unit | Zero means |
|---|---|---|---|
| **WAR** (latest season) | wins added vs replacement over his actual season | wins | replacement level |
| **Goals Added /60** | net skill vs a position-average player, baseline team, 5v5 | goals/60 | league-average at position |
| **PP Goals Added /60** | power-play version | goals/60 | position-average PP player |
| **PK Goals Added /60** | penalty-kill version | goals/60 | position-average PK player |

Attribute rows — the qualities beneath the value:

| Card | Source | Unit | Notes |
|---|---|---|---|
| **Scoring** | `shots60 · p_goal` | goals/60 | volume × danger × finishing |
| **Shooting** | `(e^shoot − 1) × 100` | % shot volume | vs position/age-typical volume |
| **Finishing** | `fin` mapped to probability | goals/100 shots | **always shown with ± CI** — honest about a small skill |
| **Playmaking** | Playmaking formula | xG/60 | creation volume is his; setup danger priced at his position; ± CI |
| **Defense** | Defense formula | xG erased/60 | volume + danger suppression combined |
| **Projected GA/60** | next season: last state + aging curve | goals/60 | labeled projection; uncertainty widens |
| **WAR (window)** | GAR summed over all fitted seasons | wins | the career-window counting stat |
| **Penalties** | production model (drawn − taken) × V | goals/60 | not yet inside WAR |

All rate copy reads **per 60**. Every card shows a **within-position percentile** (forwards vs
forwards, defensemen vs defensemen) among players clearing the ice-time gates (5v5: 100 min;
PP/PK: 40 min); below the gate the card greys out. Confidence intervals (± = 1.96·SE) are shown
where the model computes them (Finishing always; Playmaking via its creation-volume SE).

## The trajectory chart

The by-season chart shows four layers per attribute (GA/60, Scoring, Playmaking, Defense):
**solid line** — the skill the model infers each season (the drift states + that season's age,
smoothed and deployment-free); **dashed segment** — the next-season projection (state held, age
advanced along the league curve; visually distinct so extrapolation is never mistaken for
observation); **grey dotted** — a position-average player *at his age* each season (the aging
context); **dots** — unsmoothed single-season estimates (the raw evidence). Clicking a stat in
the season table switches to the classic per-season chart.

## Goalies

Unchanged: **GSAx** = `xGA − GA` over the calibrated shot set — an attribution needing no
baseline; `gsax_per100` (shrunk) and `gsax60` are the headline rates. Sv%, GAA, high-danger Sv%,
quality-start % are conventional descriptive rates. (The generative model's `gsave` + goalie
aging curve are the planned upgrade path.)

## Descriptive (non-model) cards

- **On-ice team rates** (xGF/60, xGA/60, Corsi): the team's rate while he's on the ice, not
  isolated to him.
- **Player stats** (shot rate, goal/assist rates, penalties drawn/taken, faceoffs, zone starts):
  counted straight from events, per 60.

## Caveats

- Skills are **shrunk estimates** (empirical-Bayes priors): small samples pull toward the
  position average — that's the honest read, and it's why low-TOI cards grey out rather than
  showing noise as skill.
- GA/60 is a **vs-average** comparison and WAR a **vs-replacement** total — they answer different
  questions and won't rank identically (a durable average player can out-WAR a brilliant
  part-timer).
- A roster's WAR does not sum exactly to team wins — replacement level and goals-per-win are
  league-calibrated constants, not team accounting identities.
- The model validates out-of-sample (held-out-season harness, model doc §7): skill reads carry
  corr ≈ 0.84 to next-season shot rates with calibrated totals; projections are honest
  extrapolations, not guarantees.
