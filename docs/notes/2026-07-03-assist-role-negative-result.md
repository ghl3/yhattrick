# `assist_role`: a model-level fix for finisher EV-offense that did NOT validate (2026-07-03)

A negative result, captured so a future attempt starts here instead of re-deriving it. The code was
implemented, unit-tested, synthetic-validated, run as a full warm refit, and **reverted** after the
production validation failed. Production model unchanged.

## Motivation

Oliver Kapanen (MTL, 22 G) read worst-in-league WAR (−1.32). A four-model comparison (our generative,
our RAPM, hockeyviz isolated-impact, Evolving-Hockey GAR) agreed **unanimously that his defense is
bad**; the disagreement was entirely on **EV-offense credit** — E-H graded it +6.5/+7.0 GAR, we and
hockeyviz graded it ≈ −3.6. So the question was whether our model *under-credits* the EV offense of
finishers who play with elite distributors.

## Diagnosis (this part holds up)

`create` (playmaking) does double duty in the rate stage: (i) the multiplicative lift on linemates'
shot **rate** (identified by seeing a player with *varied* linemates), and (ii) the credited-assister
propensity in a **conditional logit** over the on-ice unit (`softmax([psi0, create[teammates]])`,
generative_model.py ~1146). The softmax only constrains create *differences within a unit*.

For a **finisher glued to an elite distributor** with low linemate variation, the rate channel can't
pin his absolute `create`, so the assist anchor drags it *below* the distributor to explain why the
distributor gets the assists — and because `create` enters the team shot rate multiplicatively and
absolutely in the WAR swap, that within-unit relativity becomes an absolute "suppresses offense"
penalty. Evidence: negative `create` is a **weak-identification fingerprint** — 0% of 5-season
forwards, but 4–5% of 1–2-season and top-SE-quartile forwards. Kapanen: `create` −0.109 (≈0.39 below
the +0.28 forward mean), high SE, elite-distributor linemates (Demidov +0.40, Hutson +0.33), 2
seasons. `ev_atk` correlates more with playmaking (0.76) than scoring (0.69): a finisher's offensive
WAR is driven more by assist-derived `create` than by his goals.

## The attempted fix

`assist_role` — a per-player static offset added to `create` **only inside the assist-credit logit**
(`create + assist_role`), never the rate term (which WAR reads), with its own ridge prior. Intent:
the rate pins `create` (real chance-creation), and `assist_role` absorbs "he's a finisher, not the
credited passer," so `create` isn't dragged down. `assist_role_sd → 0` reduces exactly to today.

Synthetic (finisher: true rate-lift = population mean, low assist propensity, glued to elite
distributors) **recovered** the focal's create deficit from −0.41 to +0.05, with `assist_role`
correctly absorbing the role (−0.69). Unit tests: `test_assist_role_frees_finisher_create`,
`test_assist_role_off_is_unchanged`. Both passed.

## Why it FAILED in production (warm refit, `assist_role_sd = 0.30`)

1. **It didn't move the target.** Kapanen: `create` −0.109 → −0.098, `assist_role` ≈ 0. No effect.
   - **Root cause (penalty math):** the `create` prior shrinks toward **0**, so "low create + zero
     assist_role" is *cheaper* than "average create + negative assist_role." For Kapanen, option A
     (create=−0.1, ar=0) costs ≈ ½·(1/0.12²)·0.01 ≈ 0.34; option B (create=+0.28, ar=−0.4) costs
     ≈ ½·69·0.078 + ½·11·0.16 ≈ 3.6. The optimizer keeps him low. `assist_role` cannot win this
     trade unless the `create` prior is **re-centered on the population mean**.
2. **Well-identified players moved too much.** corr(create old,new)=0.69 (wanted ~0.99), mean|Δ|=0.23,
   mean|assist_role|=0.41. Because `assist_role`'s prior (sd 0.30) is *looser* than `create`'s (0.12),
   it preferentially absorbed the shared assist signal and **collapsed create's spread** (std
   0.142 → 0.077) — compressing discrimination.
3. **Fringe/low-sample players over-lifted.** Marginal NHLers (Othmann, Vesalainen) got assist_role
   −1.0…−1.5 from noise, lifting their `create` to ≈ average.

Directionally the mechanism is right (corr(assist_role, goals/assist ratio) = −0.42; finishers get
negative). But **no `assist_role_sd` both helps Kapanen and preserves stability** while the create
prior is centered at 0.

## Why the synthetic missed it

The synthetic measured **focal-minus-population** create, which cancels a global level shift, and it
did **not** check (a) that well-identified players stay put or (b) that the population create *spread*
survives. The real data has a create-level/spread degeneracy the synthetic didn't exercise.

## What a future attempt needs

- **Re-center the `create` prior on the position (F/D) mean**, not 0, so "average creator" is the
  cheap default and the anchor's drag is what must pay a penalty (fixes the Kapanen penalty-math).
  NB: in the pure-rate synthetic this looked like a no-op (degenerate with the intercept) — but that
  degeneracy is broken by the anchor/`psi0`, so it must be re-tested *with the anchor*.
- **Constrain `assist_role`** (mean-zero / sum-to-zero, or a prior *tighter* than `create`'s) so it
  captures *relative* role, not the global create level/spread it stole at sd 0.30.
- **Guard low-sample noise** (identification-aware shrinkage) so fringe players don't get wild values.
- **Validate on a synthetic that checks LEVEL and SPREAD stability and well-identified invariance**,
  plus the real-data two-failure-mode harness (`scratchpad/validate_assist_role.py` pattern:
  well-ID'd corr/|Δ|; weak-ID create<0 count + spread; assist_role shape; Kapanen).

## Status

Reverted. The generative model is unchanged; the WAR read for Kapanen stands (defense-driven, with a
real, unresolved offense-credit disagreement vs. Evolving-Hockey). See also
`docs/notes/2026-07-03-war-case-studies.md`.
