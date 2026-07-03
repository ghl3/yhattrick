# WAR case studies — the Hyman night (2026-07-03)

Overnight investigation triggered by one player page: Zach Hyman, 31 goals in 58 games, reading
WAR −0.07 at the 1st percentile. Everything below is measured on the 2021–25 window; fixes shipped
tonight are marked ✅ (commits `37dbdf6`, `f902e52`).

## The anatomy of the Hyman card (all four causes were real)

1. ✅ **Card units ran ~2× low.** The value formulas priced players in the reference environment
   (teammates' effective create = 0) while the league-average teammate sits at +0.17. Fixed in the
   model (`value_env`: EV ×2.48, MA ×0.99, measured from the fit's own rows); audit now asserts
   model F-mean scoring vs actual (×1.07). His Scoring card: 0.31 → **0.78 goals/60 (69th pct)** —
   matching his real 5v5 rate.
2. ✅ **WAR percentile pooled him with part-timers.** 762 players with ≈0 WAR by ice time alone
   made −0.07 read "worst in hockey". Percentile now ranks among that season's regulars (≥200 EV
   min). He's still bottom-decile among regulars — honest for a replacement-level season.
3. ✅ **A quarter of his goals are invisible to WAR.** 8 of 31 (4 empty-net, 2 at 6v5, one 5v3,
   one 3v3). Now surfaced on the WAR card (`unpriced_goals`). League-wide: **15.5% of all goals
   are unpriced** — this is a real modeling gap, not a rounding error (TODO(3b)).
4. **The remainder is the model's honest read**: at 5v5 he under-converted (14 goals on 17.4 xG),
   his playmaking is genuinely minimal (creation ≈ 0 next to McDavid — 13 primary assists), and
   his defense grades below average. A 33-year-old finisher coming off injury whose season netted
   ≈ replacement at the priced strengths.

## Discovery: `strength` labels are HOME-oriented (not shooter-relative)

Draisaitl showing 54 "shorthanded" goals exposed it: an away-team PP goal reads "4v5". After
re-orienting by `is_home` + sit counts, everything reconciles with the real league (true SHG ≈
200/season; top window SHG: Kreider 13, Konecny 12, Reinhart 12). **The model's pools were
verified clean** — quality/conversion rows require the shooter's side to actually have 5 skaters
(pool 76,296 ≈ true-PP 75,763), and rate buckets pick sides by skater counts. The first version
of the unpriced-goals stat used the raw label and shipped wrong for ~an hour; fixed + documented
in metrics.md, the roadmap (TODO 4), and the data-sources memory.

## What the data says about pricing the unpriced strengths (TODO 3b design evidence)

- **SH shot volume repeats at only 0.33 year-over-year** (vs ~0.7+ for EV volume). Individual SH
  offense is mostly deployment + variance ⇒ the extension should price SH stints with **EV skills
  + a fitted global SH-environment offset**, NOT per-player SH parameters. (SH shots are rarer
  but not more dangerous on average than PP shots: mean xG 0.071 vs 0.096.)
- **Empty-net goals concentrate on star closers** (Ovechkin 7 ENGs in 2025; window unpriced
  leaders are Kucherov 13, MacKinnon 12, McDavid 12 — all already at WAR pct ≈ 99–100). Pricing
  EN/extra-attacker mostly stretches the top and rescues Hyman-type mid-cards; it will NOT
  reshuffle rankings much. EN shots must be kept out of Stages 2–3 (no goalie → xG/conversion
  semantics differ); price EN conversion empirically.
- The structural cost of TODO(3b) is the **attacker mask**: a 4-skater attack has 3 teammates,
  and today's row builders/fit assume 5/4. Load-bearing change → synthetic tests + holdout
  validation before shipping. Not attempted unattended.

## State of the WAR stack after tonight

| Check (audit) | Value |
|---|---|
| League E[GF] vs actual | +0.8%, no correction factors |
| Card scale (model F scoring vs actual) | ×1.07 |
| Team joint GAR vs goal diff / points | 0.83 / 0.78 |
| WAR-rate ↔ GA/60 (regulars) | 0.91 |
| PP creation honesty γ (held out) | 0.78 (EV benchmark 0.90) |
| Worst regular WAR | −1.3 (Kapanen) — no outliers |

## Recommended order for the next round

1. **TODO(3b)** SH/extra-attacker buckets (attacker mask + global offsets + empirical EN
   conversion) — recovers the 15.5% unpriced value; validated via holdout + audit.
2. Bootstrap CIs (roadmap #5) and penalties into WAR (#8) — the two remaining "honest caveats".
3. The 10-season window (roadmap #7) with per-era re-validation.
