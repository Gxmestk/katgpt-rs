# Benchmark 469 — Hebbian Kernel Memory Promotion Review

**Date:** 2026-07-25
**Plan:** [559 Phase 3](../.plans/559_hebbian_kernel_memory_primitive.md)
**Research:** [455](../.research/455_Hebbian_Kernel_Memory_Fact_Storing_MLP.md)
**Feature:** `hebbian_kernel_memory` (katgpt-core)
**Verdict:** ✅ **PROMOTED TO DEFAULT-ON** (Plan 559 Phase 3, 2026-07-25)
**Authority:** This review + `bench_559_hebbian_kernel_memory_goat.rs` + [Bench 462 (riir-neuron-db)](../../riir-neuron-db/.benchmarks/462_hebbian_construction_quality_poc.md)

---

## What was promoted

`hebbian_kernel_memory` moved from opt-in → **DEFAULT-ON** in
`katgpt-rs/crates/katgpt-core/Cargo.toml`. The open primitive is the
generic, IP-free, closed-form Hebbian construction distilled from
arXiv:2607.10034 (Garcia et al., "MLPs are Hebbians", Stanford/UB
2026-07-10).

## GOAT gate evidence

| Gate | Status | Evidence |
|---|---|---|
| **G1 correctness** | ✅ PASS | `γ_min = 25.11 > 0` at D=64, F=128, m=128. Bit-identical across two runs (deterministic SeedRng). Forward-path interpolation err `‖MLP(k_0) − v_0‖_∞ = 8.33e-5 < 1e-3`. 18 unit tests. |
| **G2 perf** | ✅ PASS (two regimes) | HLA-scale (D=8, m=64): forward = 97 ns/query (target < 200). Shard-scale (D=64, m=512): construction = 44.8 µs/fact (target < 200); forward = 5.1 µs/query (target < 50). |
| **G3 no-regression** | ✅ PASS | `cargo test --features hebbian_kernel_memory --lib` → 1814 green (1796 default + 18 new). `cargo check --all-features` clean. Clippy clean. |
| **G4 alloc-free hot path** | ✅ PASS | `CountingAllocator` audit on `forward_into` + `retrieval_scores_into` (100 calls each, after warmup): **0 allocs / 0 deallocs** on both. |
| **G5 Super-GOAT quality axis** | ✅ PASS | Bench 462 (riir-neuron-db, 2026-07-25). Three-competitor defend-wrong PoC: Constructed = GD = **1.000 edit_score** at 2/5/10% edits; Frozen = 0.000 efficacy / 1.000 specificity (expected "didn't apply edit" — confirms test is discriminating). |

All five gates required for Super-GOAT confirmation PASSed modellessly.

## G5 honest caveat (recorded in Bench 462, repeated here)

The perfect 1.000 scores across BOTH Constructed and GD indicate the test
config (`m·d = 32,768` vs capacity bound `F·log(F) ≈ 896`, ~36× headroom)
is in the **easy-capacity regime**. At this ratio, the closed-form
construction provably achieves `γ_min > 0` (by Plan 559 Phase 1 G1), so
perfect retrieval follows by paper Thm 4.3. The GD variant converges to
the same B (convex MSE surface in B with A/G fixed). The harder regime
(smaller `m`, structured non-isotropic values, real NPC personality data)
remains unproven but is **non-blocking** for promotion:

- production shards operate in the easy regime by design (capacity is
  deliberately over-provisioned);
- the closed-form construction CANNOT fail while GD succeeds in this
  regime (mathematical necessity given `γ_min > 0`);
- the harder regime is tracked as a follow-up (Plan 322 Phase 2 T2.3
  real-shard test; possible ALS refinement in Plan 559 Phase 2 T1.5).

Per AGENTS.md §"Feature Flag Discipline" + the research skill §3.5:
promotion requires **modelless gain**, not perfect knowledge. The G5 PASS
is a modelless quality match (Constructed ≥ 0.95 AND within 5% of GD);
both criteria met. The easy-regime caveat is recorded for future audit
but does not block the modelless promotion.

## Modelless verification (AGENTS.md constraint #1)

All weight-mutation paths in the primitive are modelless:

1. **Freeze/thaw** — `HebbianSlot` swaps the entire snapshot atomically.
2. **Raw/lora hot-swap** — N/A (no LoRA in this primitive).
3. **Latent-space updates** — the construction is closed-form linear
   algebra (random Gaussian features + ridge-whitened least squares +
   optional alternating least-squares). No GD, no backprop, no gradient
   descent.

The data-dependent variant's two alternating least-squares solves for
`A, G` are linear (paper §B.2.5 Eq 17, 18) — modelless. The ALS
refinement stubs (`als_refine_a` / `als_refine_g`) are no-ops in Phase 1;
the Whitened-without-ALS path already achieves perfect retrieval at
d=64 shard scale, so ALS may be unnecessary (deferred per Plan 559 T1.5).

## Layer split (feature-gate-audit Defense 3)

The promotion is a **deliberate layer split**, NOT a missed propagation:

| Layer | Feature | Status | What it gates |
|---|---|---|---|
| `katgpt-rs` (open primitive) | `hebbian_kernel_memory` | **DEFAULT-ON** (this review) | Generic, IP-free, closed-form math. Zero runtime cost unless constructed. |
| `riir-neuron-db` (private bridge) | `hebbian_fact_store` | **OPT-IN** (unchanged) | IP-bearing bridge: `ShardFactSet` (NeuronShard layout knowledge) + `HebbianConstructedShard` wrapper + `ConstructionAuditSidecar` (BLAKE3-committed audit trail). |

The bridge STAYS opt-in because it gates a different concern
(shard-specific IP integration) than the primitive (generic math). The
comment in `riir-neuron-db/Cargo.toml` was updated to explain the split
and remove the stale "G5 PENDING" claim.

This matches the established codebase pattern: `npc_sleep_time`,
`conformal_predictive_intervals`, `cognitive_branches_runtime` — all
ship as engine-DEFAULT-ON + consumer-OPT-IN. Each layer's opt-in is a
separate concern and should NOT be propagated.

## 5-surface audit (feature-gate-audit Defense 2)

All five documentation surfaces were updated in one commit:

| # | Surface | Status |
|---|---|---|
| 1 | Source `.rs` (`crates/katgpt-core/src/hebbian_kernel_memory.rs`) | ✅ Clean — no stale gate-status claim in module doc |
| 2 | `crates/katgpt-core/src/lib.rs` (L1706-1719) | ✅ Updated — "Opt-in ... PENDING" → "DEFAULT-ON (Plan 559 Phase 3)" |
| 3 | `crates/katgpt-core/Cargo.toml` default list + feature comment | ✅ Updated — added to `default = [...]`; feature comment now says DEFAULT-ON |
| 4 | `riir-neuron-db/Cargo.toml` (downstream consumer) | ✅ Updated — stale "G5 PENDING" removed; layer-split rationale added; bridge STAYS opt-in |
| 5 | This file (`.benchmarks/469_hebbian_kernel_memory_promotion_review.md`) | ✅ Created |

## Re-gate after promotion

After the Cargo.toml default-list change, ran the full GOAT re-gate to
confirm no regression at default-on:

```
cargo test -p katgpt-core --lib               # G3 re-gate
cargo check --all-features                     # combo-regression check
cargo clippy -p katgpt-core --all-targets      # clippy clean
```

All PASS. The feature is additive — it was already transitively compiled
under `--all-features`; moving it to default changes only which tests
run by default (the 18 hebbian unit tests now run in the default test
suite), not the code paths exercised.

## Sibling plan state

- **Plan 559** (this repo): Phase 1 COMPLETE; Phase 2 **G5 PASS**
  (T2.3 real-shard test still open — separate concern, does not block
  primitive promotion); Phase 3 **DONE** (this promotion).
- **Plan 322** (riir-neuron-db): Phase 1 COMPLETE; Phase 2 **G5 PASS**
  (G2 perf + G4 alloc-free on the bridge still open — private bridge
  concerns, not primitive concerns); Phases 3+4 deferred.

The primitive promotion closes the Super-GOAT loop on the open half.
The private bridge remains opt-in until its own G2/G4 gates close.

## References

- Plan: [katgpt-rs/.plans/559](../.plans/559_hebbian_kernel_memory_primitive.md)
- Research: [katgpt-rs/.research/455](../.research/455_Hebbian_Kernel_Memory_Fact_Storing_MLP.md)
- Phase 1 GOAT bench: `crates/katgpt-core/benches/bench_559_hebbian_kernel_memory_goat.rs`
- Phase 2 G5 PoC (three-competitor): [riir-neuron-db/.benchmarks/462](../../riir-neuron-db/.benchmarks/462_hebbian_construction_quality_poc.md)
- Phase 2 G5 issue (closed): [riir-neuron-db/.issues/027](../../riir-neuron-db/.issues/027_hebbian_construction_quality_poc.md)
- Private Super-GOAT guide: [riir-neuron-db/.research/303](../../riir-neuron-db/.research/303_Hebbian_Fact_Storing_Shard_SuperGOAT_Guide.md)
- Private bridge plan: [riir-neuron-db/.plans/322](../../riir-neuron-db/.plans/322_hebbian_fact_storing_shard_bridge.md)
- Feature-gate-audit skill: [katgpt-rs/.agents/skills/feature-gate-audit/SKILL.md](../.agents/skills/feature-gate-audit/SKILL.md)
- HOPE sibling (Super-GOAT dual): [katgpt-rs/.benchmarks/468_hope_kernel_goat.md](468_hope_kernel_goat.md)
