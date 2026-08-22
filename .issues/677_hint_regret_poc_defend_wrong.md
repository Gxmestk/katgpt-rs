# Issue 677: Defend-wrong PoC — modelless hint-regret curriculum vs alternatives

> **Source:** Research 496 (katgpt-rs) · Guide 340 (riir-ai) · Plan 576
> **Filed:** 2026-08-21
> **Blocks:** Guide 340 P2 (quest-center consumer), Plan 576 Phase 5 wiring
> **Class:** §3.6 quality-parity defense (mandatory before any "the modelless loop recovers SPADE's gains" claim)

## The claim to defend or refute

SPADE's +5.3/+13.9 are *trained-designer* results. The modelless claim is: **a selection loop over the same hint-regret signal (paired rollouts, band gate, triage) recovers a meaningful fraction of frontier-curriculum value without any gradient descent.** Architectural + latency evidence exists; quality evidence does NOT. Per §3.6, quality-parity claims need a head-to-head PoC — architectural reasoning is insufficient.

## The rig (existing — no new infrastructure)

The demonstration-teachable pets harness in riir-mmorpg-examples (`tests/pet_teaching_headless.rs` + the Bench 013 A/B rig) already runs falsifiable curriculum A/Bs at 32 seeds with real systems. Extend it to three arms:

1. **Regret-gated** — pet-training content (monster ranks / quest types) selected by the Plan 576 primitive: hint = one hero demonstration; `r̂` = with-demo vs without-demo closure gap; triage keeps frontier, freezes mastered, evicts intractable.
2. **Uniform** — content offered uniformly (the Bench 013 generic arm).
3. **Aggregate-difficulty** — CGSP-shaped reward `(1−solve_rate)·guide_score`, no hint arm (isolates the conflation: this arm should farm intractable content).

## Metrics

- Closure rate (the Bench 013 metric) per training budget.
- Time-to-frontier (ticks until the learner's learnable share ∈ [0.2, 0.8] band).
- **Wasted attempts on intractable content** (the discriminator metric — arm 3 should lose here, arm 1 should not).
- Learnable-share trajectory (must rise under arm 1 — the SPADE 0.16→0.31 signature).

## Gates

- **Defend:** arm 1 > arm 2 on closure/time-to-frontier at ≥ 28/32 seeds AND arm 1 wastes < half the intractable attempts of arm 3 → the loop ships to Guide 340 P2.
- **Refute:** arm 1 ≈ arm 2 (the paired-rollout 2× overhead buys nothing) → the primitive demotes to the triage gate alone (still fixes the CGSP conflation); the full loop is recorded as a negative result.
- Either way: raw numbers recorded in Research 496 §"PoC Addendum" and this issue closes with the commit hash.

## Notes

- Runs on M3 (CPU harness, no GPU needed).
- If Plan 576 is not yet landed, the estimator can be inline in the test (the math is ~100 LOC); the PoC must not block on the primitive.
- Seed discipline: 32 seeds pinned, bit-identical replays (the Bench 013 precedent).
