# Issue 663 — `SwitchCostTable` modelless primitive (skill-entropy distillation)

**Source:** [Research 484](../.research/484_Skill_Entropy_Directed_Switch_Cost.md) — He et al. [arXiv:2608.05139 "Toward Skill-Native LLMs: Skill Entropy for Benchmarking and Training Long-Horizon Reasoning"]
**Filed:** 2026-08-16
**Feature flag:** `switch_cost` (opt-in; GOAT gate before any default promotion)
**Verdict:** Gain (3/4 novelty gates; Q3 selling point honestly deferred — see Research 484 §6.5)

## Problem

The stack has no measure of *how hard it is for an agent to switch between modes* — behavior-FSM states, quest objective kinds, LEO goal kinds, cognition-runtime selections. The failure class is real and shipped: Issue 054 (riir-mmorpg-examples border-piling = flee-mode carry-over at crowd scale) and Issue 057 (hero quest deadlock = Idle-mode carry-over) are both mode-switch failures detected only *reactively*. The paper's skill entropy is a directed pairwise difficulty ratio derivable from success-rate counters — pure measurement math, zero training.

## Ship

- [ ] T1 — `SwitchCostTable<const N: usize>` in katgpt-core (`switch_cost` feature, opt-in): solo/pair success counters, `ske(a,b)` hot lookup, `sequence_entropy(&[usize])`, Laplace α=0.1 default, `record_solo`/`record_switch` telemetry ingest. Fixed-size arrays, zero-alloc lookups, `#[repr(transparent)]`-friendly snapshot for freeze/thaw.
- [ ] T2 — `FactorizedSwitchCost<const N, const F>` variant: `SkE(a,b) ≈ SkE(a, fam_b)·SkE(fam_a, b)` — O(N·F) storage for large mode sets (paper Eq. 7).
- [ ] T3 — `cdf_rank` util: empirical-CDF rank normalization over a sample set (generic; consumed by Gap 6 reward + any UQ-adjacent use).
- [ ] T4 — G1 determinism: same counter state → bit-identical table + entropies; directionality pinned (`ske(a,b) != ske(b,a)` constructible test).
- [ ] T5 — G4 alloc-free: table build + lookups under `CountingAllocator`, 0 steady-state allocs.
- [ ] T6 — G2 bench: lookup latency (target: single-digit ns, table-row SIMD-friendly) + warm-up behavior under α-smoothing with cold counters.
- [ ] T7 — Example: toy 5-mode FSM telemetry → table → "which incoming switch is hardest" readout (the F1 trigger preview).
- [ ] T8 — GOAT gate doc `.benchmarks/` + promote/demote verdict (expect: stay opt-in until a riir-ai consumer A/Bs F1).

## Consumers (filed separately, not this issue)

- riir-ai F1: SkE-gated preemptive re-estimation arm beside `ReestimationScheduler`'s coherence arm — the falsifiable A/B is stuck-rate in the Issue-054 scenario with vs without the preemptive arm.
- riir-ai F3: quest-grammar sequence-entropy rejection sampling (difficulty dial).
- riir-train Plan 319 Gap 6: skill-entropy GRPO reward on civ traces (consumes T1 + T3).

## Non-goals

- Skill²-Bench reproduction (LLM benchmark, out of scope).
- Any weight training (that's Gap 6, riir-train).
- Symmetric cost approximation (directionality is load-bearing — see Research 484 §6.3).
