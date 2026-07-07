# Generative Model — Experiment Log

Append-only, reverse-chronological record (newest first) of what we tried on the shooter-resolved
generative model and what we learned. **Entries are immutable once written** — if a later result
overturns an earlier one, add a new entry that says so rather than editing the old one. The value of
this file is that a future session can see the dead ends and the reasons, instead of re-deriving or
re-running them.

Division of labour across the three docs:
- **`generative_model.md`** — the *current* model spec, stated affirmatively (no history).
- **`generative_model_roadmap.md`** — the *forward* backlog and current-state rationale.
- **this file** — the *history*: hypotheses, outcomes, numbers, and lessons, dated.

Entry template:

```
### YYYY-MM-DD — <title> — <SHIPPED | REVERTED | INCONCLUSIVE | IN PROGRESS>
- **Goal / hypothesis:** what we expected and why.
- **What we did:** the change, in one or two lines.
- **Outcome:** the numbers that decided it.
- **Lesson:** what future work should carry forward.
- **Refs:** commit <hash> (if any), docs/notes/…, roadmap §…, model.md §…
```

---

### 2026-07-07 — Position-mean `create` prior (Kapanen-class residual, lever 3) — SHIPPED
- **Goal / hypothesis:** the EV `create` level ridge shrinks toward 0, but forwards genuinely live
  above 0, so weakly-identified forwards get pulled below their reference class. Re-center the ridge on
  the F/D position mean — the *correct* prior — to relax the residual negative tail (Kapanen still the
  most-negative regular forward at −0.059 after Step 1).
- **What we did:** changed the rate ridge `lcr·Σ fmj·cr²` → `lcr·Σ fmj·(cr − center)²`, `center` = the
  F/D raw-state mean computed two-pass (fit at 0 with `skip_se`, measure means, refit centered).
  `center` is a fixed constant, so the level stays pinned and the SE curvature is unchanged. Gated
  behind `create_prior_center` (default now `position-mean`), threaded through fit/run/CLIs/holdout.
- **Data check (pre-flight):** confirmed the re-centering is IDENTIFIED, not a gauge no-op — raw states
  differ by position (forwards **+0.174**, defensemen **−0.308**). Aggregate weak-ID bias looked small
  (weak-ID forwards +0.172 vs well-ID +0.236), which made me predict "marginal". That prediction was
  wrong (see outcome) — the ridge-at-0 biased the *whole* forward distribution down, not just the tail.
- **Outcome — a clear win on two held-out years.** Target 2025: tm-corr 0.690 → **0.719**, tm-MAE
  2.67 → 2.48, **ev:create γ 0.83 → 0.94**; target 2024: tm-corr 0.687 → **0.705**, γ 0.86 → **0.95**.
  In-sample (production window): well-identified forwards invariant (**corr 0.9989** — clean prior
  change, no re-gauging), spread preserved (std 0.082 → 0.077), negative-create forwards 0.8% → 0.2%,
  Kapanen **−0.059 → −0.007** (playmaking ≈ 0). Σp=Σgoals still exact; κ/WAR stable (0.951 / 1080).
  A ~+0.05 uniform level lift is the gauge direction (absorbed by the intercept; WAR reads differences).
- **Lesson:** for a PRIOR change the valid in-sample check IS well-identified invariance (opposite of
  Step 1's likelihood change). Also: the "aggregate weak-ID bias is small ⇒ low value" heuristic
  under-counted the win — a systematically-off *level* hurts calibration across the whole distribution,
  not just the visibly-broken tail. And this is exactly the prerequisite the reverted `assist_role`
  attempt named: with the prior now centered on the position mean, a future separated create/role
  decomposition is unblocked.
- **Refs:** model.md §7 (EV sweep / position-mean), §9; roadmap §5e lever 3; commit: pending.

### 2026-07-07 — EV assist-anchor down-weight (Step 1, Kapanen-class lever 1) — SHIPPED
- **Goal / hypothesis:** roadmap §5e lever 1. The EV assist anchor might over-weight assist-ROLE
  (who gets credited) relative to genuine creation, dragging fixed-unit finishers (the Kapanen class)
  to negative `create`. Expose an `ev_anchor_scale` and select it by held-out validation, like the MA
  bucket.
- **What we did:** plumbed `ev_anchor_scale` through `fit_all` / `run` / both CLIs / the holdout
  harness; swept {1.0, 0.5, 0.25, 0.1, 0.0} on two held-out target years.
- **Outcome:** lowering the anchor **improves held-out prediction on every axis**. Teammate-shots corr
  0.657 → 0.690 (target 2025) and 0.620 → 0.687 (target 2024, where the full-weight model had
  *underperformed* naive shot-counting, 0.620 < 0.639). Own-shots corr and row deviance improved too.
  Optimum is an interior 0.25 (below ~0.1 `create` loses identification and its held-out block flips
  sign). Shipped EV `anchor_scale = 0.25`. Population effect: forward `create` spread 0.13 → 0.08,
  negative-`create` forwards 2.6% → 0.8%; Kapanen −0.109 → −0.059. Conservation still exact; elite
  playmaking board intact (Crosby/Kucherov/Scheifele/MacKinnon top it).
- **Lesson (the important one):** the in-sample view *reversed* the truth. Lowering the anchor
  compressed spread and moved "well-identified" forwards (in-sample corr 0.88), which looked like signal
  destruction and matched the `assist_role` spread-collapse fingerprint — but held-out prediction says
  the compression was **shed overfitting, not lost signal**. **Judge these levers by held-out
  prediction, never by in-sample invariance against the current fit — the current fit is the thing under
  suspicion.** (Corollary, learned designing Step 2: this only applies to *likelihood* changes; a
  *prior* change should leave well-identified players put, so there in-sample invariance IS a valid
  check.)
- **Refs:** model.md §7 "EV anchor sweep", §9; roadmap §5e lever 1. Commit: pending.

### 2026-07-03 — `assist_role` per-player anchor offset (Kapanen fix) — REVERTED
- **Goal / hypothesis:** free a finisher's `create` from the anchor's within-unit drag by adding a
  per-player offset used ONLY inside the assist-credit logit (`create + assist_role`), never the rate
  term. Intent: rate pins real creation, `assist_role` absorbs "credited passer vs chance creator".
- **What we did:** implemented, unit-tested, synthetic-validated, ran a full warm refit at
  `assist_role_sd = 0.30`.
- **Outcome:** REVERTED. (1) Did not fire for Kapanen (create −0.109 → −0.098, role ≈ 0): with the
  `create` prior centered at 0, "low create + zero role" is cheaper than "average create + negative
  role", so the optimizer never moved him. (2) Well-identified players moved too much (corr old/new
  0.69) and `create`'s spread collapsed 0.142 → 0.077 because the looser role prior stole shared signal.
  (3) Fringe players over-lifted from noise.
- **Lesson:** with the `create` prior at 0 there is no single `role_sd` that both helps Kapanen and
  stays stable — the fix needs a position-mean-centered prior (→ Step 2). The synthetic missed the
  failure because it measured focal-minus-population, which cancels exactly the level degeneracy; future
  synthetics must check LEVEL and SPREAD stability and well-identified invariance.
- **Refs:** docs/notes/2026-07-03-assist-role-negative-result.md; roadmap §5e; dangling blobs d426cbe4
  (model) / 7bb4df89 (tests).

### 2026-07-03 — Hyman-case WAR fixes (value-env, percentile pool, unpriced goals) — SHIPPED
- **Goal / hypothesis:** Zach Hyman (31 goals = 31.3 xG) read WAR −0.07 at the 1st percentile —
  investigate the clash.
- **Outcome:** three fixes shipped: (1) value-environment alignment (cards evaluate in the average
  environment, not the reference one, ~2× units fix); (2) WAR percentile gated on ≥200 min EV TOI
  ("among regulars"); (3) surfaced per-player `unpriced_goals` (league-wide 15.5% of goals are unpriced:
  SHG/EN/6v5/5v3/3v3). Also found the `shots_onice.strength` label is home-oriented, not
  shooter-relative (documented; model pools verified clean).
- **Refs:** docs/notes/2026-07-03-war-case-studies.md; roadmap §5d. Commit 37dbdf6.

### 2026-07-02 — MA (PP/PK) anchor + prior sweep — SHIPPED
- **Goal / hypothesis:** on fixed PP units, shot counts pin only the unit's total creation; assists
  split it, but PP assists are ROLE not creation, so assist-light net-front players absorb their
  linemates' negative residual. Select the MA anchor weight and create/def priors by held-out
  calibration slope.
- **Outcome:** selected MA anchor ×0.25, create prior 0.04, def prior 0.10. γ_create 0.31 → 0.78,
  γ_def restored to 0.74, held-out PP deviance 233.8 → 230.0 — best on every axis simultaneously.
- **Lesson:** down-weighting the anchor monotonically improves held-out PP prediction; misallocation
  migrates to the loosest block (γ_def slid until def got the same prior treatment). This is the direct
  precedent that motivated the EV sweep (Step 1).
- **Refs:** model.md §7 "MA identification sweep"; roadmap §5c. Commit fed793b.

### 2026-07-02 — Secondary assists (A2) as a fitted mixture — SHIPPED
- **Goal / hypothesis:** do recorded second assists reflect creation, or noise? Model (A1, A2) as a
  partial ranking with a fitted mixture share `q` = P(A2 reflects creation), rather than a hand-tuned
  weight.
- **Outcome:** q̂ saturated at 1.00 (A2s fully concordant with the create ordering); held-out
  create-side tm-corr +0.008; the model now beats the naive teammate-rate bar (0.657 vs 0.612 — note
  Step 1 later widened this gap further). Own-shot metrics unchanged.
- **Refs:** model.md §5; roadmap §2. Commit e05871e.

### 2026-07 — Arena / scorer recording-bias states — SHIPPED
- **Goal / hypothesis:** NHL scorekeeping varies by building (count bias → rate; location shift →
  recorded xG → quality). Model per-(venue, season) nuisance states.
- **Outcome:** ridge-to-zero + season random-walk smoothing in the rate AND quality stages; excluded
  from player values; reported per run. Max venue drift ≈ ±8% rate, ±0.03 logit-xG.
- **Refs:** model.md §2 "Arena effects"; roadmap §3. (Open: port to the production RAPM.)
