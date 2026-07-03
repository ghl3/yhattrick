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
q_own(θ, c)  = sigmoid(mu_qual + qshoot + qcreate_c)      mean xG of his shots, by creator class c
ḡ(θ)         = Σ_c π_c · E[ sigmoid(a·logit(x) + b + fin) ]   over the model's own shot-quality
                 distribution (x ~ Beta around q_own) and creator mix π (unassisted/F/D)
Scoring      = shots60 · ḡ                                            [goals/60]

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
per-season drift states, across EV + PP + PK. Each shooter's shots convert at the model's own
marginal goals-per-shot ḡ, with the creator mix taken from the stint's ACTUAL teammates — so the
engine's expected goals reconcile to the goals the model was fit on within ~1%, with no
correction factors (the WAR audit asserts this).

**Calibration knobs (explicit and revisable):**
- **Replacement level, context-matched**: the TOI-weighted average parameters of players in the
  **8th–12th percentile band** within his position — of GA/60 among EV regulars for EV slots, of
  PP GA/60 among PP regulars for PP slots, and of PK GA/60 among PK regulars for PK slots. (One
  EV-wide band would make the PP baseline a league-average PP player, because non-PP players'
  PP parameters shrink to zero = average.)
- **Goals-per-win = 6.0** (the standard rule of thumb; a season-estimated value is a listed
  follow-up).

**Honest caveats:** penalties are *not* in WAR yet (the production penalty value is shown as its
own card until the penalties stage joins the generative model); defender shot-quality suppression
enters as a multiplicative factor (an approximation documented in the exporter); WAR ships without
confidence intervals until the parametric bootstrap lands.

**Scale note (re-audited 2026-07 after the fixes).** The engine's expected goals reconcile to
the goals the model was fit on within **+0.8%, with no correction factors anywhere** — the audit
asserts this instead of enforcing it. League totals: Σ individual WAR ≈ 1,290; replacing whole
teams jointly prices at ≈ 1,160 (summed marginals count each lineup synergy in every member's
swap — ×1.11, a documented property of the multiplicative model, not an error). Top seasons land
around 7–9 wins. This still runs above public models' ~700-WAR normalization for one honest
reason: our replacement level is deeper and our skill spread wider — and the spread is the part
the held-out harness validates directly (rate-skill corr 0.84, per-block calibration slopes at
their drift ceilings; model doc §7). Compare WAR **within this site**, not across models.

**The PP credit distortion is fixed in the model (2026-07).** Within long-lived five-man units,
shot counts pin only the unit's total creation and assists split it — but PP assists reflect
role, not creation, so assist-light net-front players used to absorb large negative residuals
(and stars the mirror inflation). The PP/PK bucket's assist-anchor weight and priors are now
selected by out-of-sample calibration (creation honesty 0.31 → 0.78, PK-defense 0.74, held-out
PP prediction improved throughout — the full sweep is in the model doc §7). WAR, PP GA/60, and
Playmaking all read the corrected parameters at face value.

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

All rate copy reads **per 60**, and card units are **real-world rates**: the value formulas are
evaluated in the league-average environment (`value_env` — the TOI-weighted mean of teammate,
defender, and context effects over the fit's own latest-season rows), so a card's goals/60 is on
the same scale as the rates you'd count from the games (the audit asserts this against actual
league scoring). Every card shows a **within-position percentile** (forwards vs forwards,
defensemen vs defensemen) among players clearing the ice-time gates (5v5: 100 min; PP/PK: 40 min);
below the gate the card greys out. **WAR's percentile pool is that season's regulars** (≥200 EV
minutes in the season) — WAR is a counting stat, and ranking a full-timer against part-timers
whose totals are ≈0 by ice time alone would read as a skill ranking it isn't.

**What WAR doesn't price (yet):** goals scored shorthanded, into empty nets, with the goalie
pulled, at 5v3, or in 3v3 OT — the model prices 5v5 and the 5v4 power play. Players with such
goals get an explicit note on the WAR card (e.g. Hyman 2025-26: 14 of his 31 goals — six of them
shorthanded). Pricing these situations is roadmap 5d-TODO(3). Confidence intervals (± = 1.96·SE) are shown
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
