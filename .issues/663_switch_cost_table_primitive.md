# Issue 663 — `SwitchCostTable` modelless primitive (skill-entropy distillation)

**Source:** [Research 484](../.research/484_Skill_Entropy_Directed_Switch_Cost.md) — He et al. [arXiv:2608.05139 "Toward Skill-Native LLMs: Skill Entropy for Benchmarking and Training Long-Horizon Reasoning"]
**Filed:** 2026-08-16
**Resolved:** 2026-08-17 — ALL TASKS DONE, GOAT G1/G2/G3/G4 PASS ([Bench 660](../.benchmarks/660_switch_cost_table_goat.md)); stays opt-in as pre-registered
**Feature flag:** `switch_cost` (opt-in; GOAT gate before any default promotion)
**Verdict:** Gain (3/4 novelty gates; Q3 selling point honestly deferred — see Research 484 §6.5)

## Outcome (2026-08-17)

Shipped as `katgpt-core/src/switch_cost.rs` + `tests/switch_cost_663_poc.rs` +
`tests/switch_cost_alloc_check.rs` + `examples/switch_cost_demo.rs` behind the
opt-in `switch_cost` feature. GOAT ALL PASS: G1 (hand-computed formula
3.0/0.667, directionality pinned, record-order-independent bit-identical,
cold table exactly 1.0, monotone failure ladder, factorized ranking
Spearman ≥ 0.75 + identical argmax at the paper's own 82–86% fidelity bar),
G2 (3.08 ns/op `ske`, 34.04 ns `sequence_entropy(16)`), G3 (default +
no-default + clippy --all-targets clean), G4 (0 allocs across all hot paths).

Honest findings recorded in Bench 660: (1) the factorized variant
under-estimates hard pairs on heterogeneous families (demo: 1.99 vs 3.30
exact — ranking fidelity holds, magnitude dilutes; use the exact table for
bounded N); (2) cold-start neutrality (zero-trial → 0.5 prior → SkE exactly
1.0) is our design choice, not the paper's text — `ske_if_armed` is the
warm-up floor consumers should gate triggers on.

The healer-consumer branch (from Research 484 §7) was already measured dead
by riir-clippy Bench 032 — fix-ordering refuted by mechanism. Remaining
consumers: riir-ai F1 (the promotion-gating A/B), F3, riir-train Gap 6.

## Ship

- [x] T1 — `SwitchCostTable<const N: usize>` in katgpt-core (`switch_cost` feature, opt-in): solo/pair success counters, `ske(a,b)` hot lookup, `sequence_entropy(&[usize])`, Laplace α=0.1 default, `record_solo`/`record_switch` telemetry ingest. Fixed-size arrays, zero-alloc lookups, `#[repr(transparent)]`-friendly snapshot for freeze/thaw.
- [x] T2 — `FactorizedSwitchCost<const N, const F>` variant: `SkE(a,b) ≈ SkE(a, fam_b)·SkE(fam_a, b)` — O(N·F) storage for large mode sets (paper Eq. 7). `record_switch` routes into leave `(a, fam_b)` + land `(fam_a, b)` cells in one call.
- [x] T3 — `cdf_rank` util: empirical-CDF rank normalization over a sample set (generic; consumed by Gap 6 reward + any UQ-adjacent use). Scale-free-ness pinned by test (reward invariant under 10× corpus rescale).
- [x] T4 — G1 determinism: same counter state → bit-identical table + entropies; directionality pinned (`ske(a,b) != ske(b,a)` constructible test). Extended: u32 counters make results **record-order independent** — pinned bit-identically under forward vs reverse replay.
- [x] T5 — G4 alloc-free: table build + lookups under `CountingAllocator`, 0 steady-state allocs (1M exact+factorized lookups + 100k sequence entropies + 100k records + snapshot + cdf_rank).
- [x] T6 — G2 bench: lookup latency (target: single-digit ns, table-row SIMD-friendly) + warm-up behavior under α-smoothing with cold counters. **3.08 ns/op** measured; cold = exactly 1.0; monotone failure ladder pinned.
- [x] T7 — Example: toy 5-mode FSM telemetry → table → "which incoming switch is hardest" readout (the F1 trigger preview). Designed Flee→Hunt carry-over recovered at 3.30 vs ~1.1 baseline; calm vs panic routines separate 1.16 vs 1.53.
- [x] T8 — GOAT gate doc `.benchmarks/` + promote/demote verdict (expect: stay opt-in until a riir-ai consumer A/Bs F1). Bench 660; verdict = **stay opt-in** (pre-registered expectation confirmed).

## Consumers (filed separately, not this issue)

- riir-ai F1: SkE-gated preemptive re-estimation arm beside `ReestimationScheduler`'s coherence arm — the falsifiable A/B is stuck-rate in the Issue-054 scenario with vs without the preemptive arm.
- riir-ai F3: quest-grammar sequence-entropy rejection sampling (difficulty dial).
- riir-train Plan 319 Gap 6: skill-entropy GRPO reward on civ traces (consumes T1 + T3).

## Non-goals

- Skill²-Bench reproduction (LLM benchmark, out of scope).
- Any weight training (that's Gap 6, riir-train).
- Symmetric cost approximation (directionality is load-bearing — see Research 484 §6.3).
