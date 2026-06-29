# 07 — Metrics: how each player-page card is calculated

This is the reader-facing reference for the player page. It explains the **philosophy** behind the
modeled cards, the one **key step** that makes them work (absorbing the model intercept), and then
gives a **card-by-card** definition. For the model internals (how the coefficients are fit), see
[`modeling.md`](modeling.md).

## Philosophy — goals attributed

Every modeled card is a **goal attribution**: "goals we credit to this player." The design goal is
that, across the whole league, the attributions **sum to actual goals**. That isn't a coincidence —
it's enforced by two facts:

- **Expected goals are calibrated**: `Σ xG ≈ Σ goals`. So if we credit each player his share of the
  on-ice expected goals, the shares add up to (about) all the goals.
- **Finishing is defined as `goals − xG`**: so `created (xG) + finishing` reconciles xG back to *actual*
  goals. A tiny league baseline `μ` is left unattributed.

This is why we avoid "vs. an average player" framing on the cards: a number like "+0.1 goals/60 above
average" answers a *comparison*, not "how many goals." We want the cards to answer **how many goals**.

## Absorbing the intercept (the key step)

The on-ice (RAPM) model fits, for each stint, the attacking team's expected-goal rate as a sum:

```
xGF/60  =  intercept  +  Σ (ev_off of the 5 attackers)  +  Σ (ev_def of the 5 defenders)  +  context
```

The **intercept is fit free** (unpenalized), so it absorbs the league baseline — the xGF/60 a stint of
all-average skaters would post (≈ 2.5 at 5v5, the whole 5-man unit). The player coefficients are
penalized toward zero, so the exported `ev_off` is the player's **deviation** from his slice of that
baseline. An average skater sits near 0; a below-average one goes **negative**. That's why the raw
coefficient reads as "vs. average."

To make it an absolute attribution, fold each player's share of the baseline back in — split the
baseline equally among that side's on-ice skaters and add the coefficient:

```
his xG created /60  =  baseline ÷ skaters  +  ev_off  =  2.5/5 + ev_off  =  0.5 + ev_off
```

Worked example (5v5), a good two-way forward with `ev_off = +0.3`, `ev_def = −0.2`:

| | formula | value |
|---|---|---|
| Goals created /60 | `0.5 + 0.3` | **0.8** |
| Goals allowed /60 | `0.5 + (−0.2)` | **0.3** |

Both are ≥ 0 and read as real expected goals. Summed across the five on-ice skaters, the created shares
add back to the stint's xGF — which is exactly what "the baseline split that reconciles to Σ xG" means.
The per-player deviations already encode position (forwards get systematically higher `ev_off` than
defensemen), so an **equal** split of the baseline is the clean, neutral choice.

Splits by situation: even strength **÷5** (5 skaters a side); power-play offense **÷5** (5 PP skaters);
penalty-kill defense **÷4** (4 PK skaters). The special-teams baseline is the PP team's xGF/60 (higher
than 5v5).

## Why "centered at 0" is NOT "vs average"

The **net** is `created − allowed`:

```
net = (0.5 + ev_off) − (0.5 + ev_def) = ev_off − ev_def        the 0.5 baselines cancel
```

It's tempting to call this "vs average" because an average player (ev_off = ev_def = 0) nets 0. But
that reasoning is backwards. A metric is "vs average" only if it is **defined** as (player − average
player). Net isn't: it's `created − allowed`, a difference of two **absolute** quantities. Its zero
means **break-even** — he creates as many goals as he's on the hook for allowing — which is a real,
physical reference point, not "the average player." Average players merely *happen* to land near
break-even, because league-wide xG-created equals xG-allowed.

This is exactly plus/minus: "goals for while I'm on" and "goals against while I'm on" are absolute
counts from zero, but their *difference* is centered, because the common baseline cancels. Offense can
genuinely be zero (you can create no chances); but the meaningful zero for a *net* is break-even, so the
net is a vs-zero **differential**, not a vs-average comparison.

## Card-by-card

All modeled cards show a **unit** in small text after the number, a **within-position percentile**
(among forwards or defensemen; goalies among goalies), and — where available — a 95% confidence range.

### Modeled Impact (skater value)

| Card | Formula | Unit | Scope | Zero means |
|---|---|---|---|---|
| **Net Goals Added per Game** | `(g_created + g_fin − g_allowed + g_pen) / GP` | goals/game | all situations, deployment-weighted | break-even |
| **Two-Way Rating** | `create60 + finishing − allow60` | goals/60 | 5v5 | break-even |
| **Even-Strength Offense** | `baseline/5 + ev_off` | goals/60 | 5v5, on-ice share (incl. his own shots) | created no chances |
| **Even-Strength Defense** | `baseline/5 + ev_def` | goals/60 | 5v5, on-ice share; **lower is better** | allowed no chances |
| **Power-Play Offense** | `pp_baseline/5 + pp_off` | goals/60 | power play | created no chances |
| **Penalty-Kill Defense** | `pp_baseline/4 + pk_def` | goals/60 | penalty kill; **lower is better** | allowed no chances |
| **Finishing** | `fin_per100` = goals − xG on his shots, per 100 | goals/100 shots | his own shots, all situations | finished as expected |
| **Penalties** | `(drawn − taken) × V`, `V ≈ 0.14` | goals/60 | all situations | neutral |

`Offense + Finishing − Defense + Penalties = Net` (summed across situations, deployment-weighted, per
game). Offense already contains his own shots' xG; **Finishing** adds the goals-above-xG residual on top
(no double count). There is no separate "Scoring" card — it would just be Offense's own-shot slice plus
Finishing, double-counting both.

Note on penalty-kill: a PK skater is on the ice for goals against, so his Penalty-Kill Defense number is
large by role; his *skill* shows as being **below** the PK baseline (a low number). Good penalty-killers
minimize an unavoidably negative-attribution role.

### Goalies

GSAx is already an attribution and needs no intercept: **Goals Saved Above Expected** = `xGA − GA` over
the calibrated shot set; zero = saved exactly what the shots' xG predicted (not "vs the average
goalie"). `gsax_per100` (per 100 shots, shrunk for sample) and `gsax60` (per 60) are the headline rates;
`gsax_saved` is the season total goals prevented (the goalie's `g_net`). Sv%, GAA, high-danger Sv%, and
quality-start % are conventional descriptive rates.

### Descriptive (non-attribution) cards

- **On-ice team rates** (xGF/60, xGA/60, Corsi for/against): the *team's* rate while he's on the ice,
  **not** isolated to him.
- **Individual rates** (Shot Rate, Shot Quality, Expected Goal Rate, Goal/Assist Rate, Penalty
  Draw/Take Rate): his own on-puck production, per 60 of all-situations ice time.

## Caveats

- Attribution is **approximate per stint** (ridge shrinkage pulls coefficients toward zero; the tiny
  league baseline `μ` is unattributed) but **calibrated in aggregate** — leaguewide the shares
  reconcile to actual goals.
- These are per-player attributions, so a **roster's `g_net` does not sum to the team's goal
  differential** — expected for marginal/share quantities, and distinct from the league-level
  `goal_accounting` identity in [`modeling.md`](modeling.md).
