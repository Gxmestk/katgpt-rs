# Plan 439: ANE Fused-Chain Cost Model — Dependency-Aware Overlap Prediction

**Date:** 2026-07-14
**Research:** [katgpt-rs/.research/423_GPU_Tile_Sim_ANE_Tile_Graph_Overlap.md](../.research/423_GPU_Tile_Sim_ANE_Tile_Graph_Overlap.md)
**Source paper:** [arXiv:2607.11262](https://arxiv.org/abs/2607.11262) — Ding et al., *GPU-Tile-Sim*, MICRO 2026
**Target:** `katgpt-rs/crates/katgpt-core/src/ane_roofline.rs` (extend) + Cargo feature `ane_fused_chain` (opt-in, gated on `ane_roofline`)
**Status:** Active — Phase 1 DONE (T1.1–T1.8 shipped). Phase 2 GOAT gate next.

---

## Goal

Extend Plan 379's single-op `ane_estimate` with a **fused-chain cost estimator** that models the two fusion benefits the current steady-state `max(compute, memory)` cannot capture:

1. **Eliminated intermediate DRAM traffic** — when op A feeds op B and the intermediate fits in the ANE's on-chip working set, B does not reload A's output from DRAM.
2. **Single dispatch overhead** — a fused chain pays one `dispatch_floor_ms`, not N.

This distills GTSim's core insight (§7.1: "kernel performance is governed by dependency structure, not individual instruction latency") into a modelless multi-op extension of the shipped ANE roofline. The full tile-graph DAG (data edges + order edges + warp-specialization scheduling) is Phase 3+ stretch — this plan ships the **minimal dependency-aware overlap** that closes the biggest gap: fused chains being misestimated as N independent `max()` calls.

**Why modelless:** Pure arithmetic over op shapes, data dependencies, and target chip peaks. No weights, no runtime state, no training. Extends Plan 379's existing pure-arithmetic cost model.

**Why GOAT, not Super-GOAT:** Provable routing gain over Plan 379's single-op model for fused chains — specifically, eliminates the systematic `max()` overestimate when intermediates stay on-chip — but does not create a new capability class. The shipped roofline already routes; this plan makes it fusion-aware.

**What this plan does NOT do:**
- Does NOT replace `ane_estimate` — it adds a sibling `ane_fused_estimate` for chains. Single-op routing is unchanged.
- Does NOT model cross-op tile-level pipeline overlap (the full GTSim tile-graph). That's Phase 3 stretch.
- Does NOT require private Apple APIs. Same public chip-family identifier as Plan 379.
- Does NOT redirect to riir-train. Pure inference-time cost modeling.

---

## GOAT Gate

| Gate | Criterion | Measurement |
|---|---|---|
| **G1 (correctness)** | For a 2-op fused chain (conv→relu, gemm→bias) where the intermediate fits in working set: `ane_fused_estimate` predicts runtime ≤ the sum of two individual `ane_estimate` calls minus the eliminated intermediate traffic. The fusion savings must be non-negative and bounded by `eliminated_bytes / bandwidth`. | Unit test: assert `fused.runtime_ms ≤ sequential.runtime_ms` and `fused.bytes_moved < sequential.bytes_moved` when intermediate fits |
| **G2 (routing)** | Fused chain routing verdicts: (a) chain with all ops fitting working set → `Compute` or `Memory` (not `WorkingSet`); (b) chain with one op exceeding working set → `WorkingSet` (fallback to sequential); (c) tiny fused chain below dispatch floor → `Dispatch` (CPU wins) | Unit test: 3 verdict cases |
| **G3 (no-regression)** | Plan 379's single-op `ane_estimate` tests (23/23) still pass; `ane_fused_estimate` with a single-op chain (no deps) returns the same `AneCost` as `ane_estimate` | `cargo test -p katgpt-core --features ane_fused_chain --lib ane_roofline` |
| **G4 (alloc-free)** | `ane_fused_estimate` is `#[inline(always)]`, zero allocations, ≤1 µs CPU for ≤8-op chains (same budget as Plan 379) | criterion bench: `ane_fused_estimate` p50 < 1 µs for 8-op chain |
| **G5 (feature isolation)** | Build clean with and without `--features ane_fused_chain`; `ane_fused_chain` implies `ane_roofline` | `cargo check` + `cargo check --features ane_fused_chain` |

**UQ check (Report the Floor rule):** This primitive does NOT claim a probability distribution or coverage guarantee. It is a deterministic cost model. Floor rule does not apply.

**Promotion rule:** If G1–G5 all PASS → promote `ane_fused_chain` to default (it implies `ane_roofline` which is already default). If G1 FAILS (fusion savings negative — model is worse than sequential) → keep opt-in, file issue. If G3 FAILS (single-op regression) → block promotion.

---

## Phase 1 — Fused-Chain Skeleton (CORE)

Goal: a compiling, tested, feature-gated extension of `ane_roofline.rs` that implements the minimal fused-chain estimator. No integration with `NpcBrainRouter` yet.

### Tasks

- [x] **T1.1** Add feature flag `ane_fused_chain = ["ane_roofline"]` to `katgpt-rs/crates/katgpt-core/Cargo.toml` `[features]` section. Implies `ane_roofline` (Plan 379).
- [x] **T1.2** Add `#[cfg(feature = "ane_fused_chain")] pub use fused_chain::*;` re-export from the `ane_roofline` module (or inline the code under `#[cfg]` — pick whichever keeps the file < 2048 lines per AGENTS.md).
- [x] **T1.3** Implement `AneDataDep` struct — a directed data dependency between two ops in a chain:
  ```rust
  /// A producer→consumer edge: `from_op`'s output feeds `to_op`'s input.
  /// `intermediate_bytes` is the size of the intermediate tensor.
  /// If the intermediate fits in the ANE's on-chip working set, it is
  /// eliminated from DRAM traffic in the fused estimate.
  #[derive(Clone, Copy, Debug)]
  pub struct AneDataDep {
      pub from_op: usize,           // index into the ops slice
      pub to_op: usize,             // index into the ops slice
      pub intermediate_bytes: u64,  // size of the intermediate tensor
  }
  ```
- [x] **T1.4** Implement `AneFusedCost` struct — the output of the fused estimator (extends `AneCost` with fusion metadata):
  ```rust
  #[derive(Clone, Copy, Debug)]
  pub struct AneFusedCost {
      pub base: AneCost,              // the underlying single-op-shape cost
      pub n_ops: usize,               // number of ops in the chain
      pub n_fused_deps: usize,        // number of data deps with on-chip intermediates
      pub eliminated_bytes: u64,      // total DRAM traffic eliminated by fusion
      pub sequential_runtime_ms: f64, // what N independent ane_estimate calls would predict
      pub fusion_savings_ms: f64,     // sequential - fused (must be ≥ 0)
  }
  ```
- [x] **T1.5** Implement the core fused estimator `ane_fused_estimate`:
  ```rust
  /// Dependency-aware fused-chain ANE cost model.
  ///
  /// Takes a chain of ops + their data dependencies and returns a single
  /// cost estimate that accounts for:
  /// 1. Eliminated intermediate DRAM traffic (on-chip intermediates).
  /// 2. Single dispatch overhead (one floor, not N).
  ///
  /// Falls back to the sum of individual `ane_estimate` calls (sequential)
  /// when intermediates do NOT fit in the working set.
  ///
  /// Distilled from GTSim (arXiv:2607.11262) §7.1 insight: kernel performance
  /// is governed by dependency structure, not individual instruction latency.
  #[inline(always)]
  pub fn ane_fused_estimate(
      ops: &[AneOpShape],
      deps: &[AneDataDep],
      dtype: Dtype,
      peaks: &AnePeaks,
  ) -> AneFusedCost {
      // 1. Compute the sequential baseline: sum of individual ane_estimate calls.
      //    This is what the current router would predict if it treated each op
      //    independently (the status quo).
      let sequential_cost: AneCost = ops.iter()
          .map(|&op| ane_estimate(op, dtype, peaks))
          .fold(AneCost::zero(), |acc, c| acc + c);
      
      // 2. Check which intermediates fit in the working set.
      let working_set_budget = peaks.working_set_bytes;
      let mut total_intermediate: u64 = 0;
      let mut eliminated_bytes: u64 = 0;
      let mut n_fused = 0usize;
      for dep in deps {
          if dep.intermediate_bytes <= working_set_budget
              && total_intermediate + dep.intermediate_bytes <= working_set_budget
          {
              // This intermediate fits → eliminated from DRAM traffic.
              total_intermediate += dep.intermediate_bytes;
              eliminated_bytes += dep.intermediate_bytes;
              n_fused += 1;
          }
      }
      
      // 3. If NO intermediates fit, return the sequential baseline (no fusion benefit).
      if n_fused == 0 {
          return AneFusedCost {
              base: sequential_cost,
              n_ops: ops.len(),
              n_fused_deps: 0,
              eliminated_bytes: 0,
              sequential_runtime_ms: sequential_cost.runtime_ms,
              fusion_savings_ms: 0.0,
          };
      }
      
      // 4. Model the fused chain: aggregate FLOPs and bytes, subtract eliminated traffic.
      let total_flops: u64 = ops.iter().map(|o| o.flops).sum();
      let total_bytes_raw: u64 = ops.iter().map(|o| o.bytes_moved).sum();
      let total_bytes_fused: u64 = total_bytes_raw.saturating_sub(eliminated_bytes);
      let largest_operand: u64 = ops.iter().map(|o| o.largest_operand_bytes).max().unwrap_or(0);
      let min_family: AneFamily = ops.iter()
          .map(|o| o.min_family)
          .max()  // chain is gated by the highest family requirement
          .unwrap_or(AneFamily::A13);
      
      // 5. Build a single fused AneOpShape and estimate it as one op.
      let fused_op = AneOpShape::new(total_flops, total_bytes_fused, largest_operand, min_family);
      let fused_cost = ane_estimate(fused_op, dtype, peaks);
      
      // 6. Compute fusion savings (must be ≥ 0 — the model says fusion never hurts).
      let fusion_savings_ms = (sequential_cost.runtime_ms - fused_cost.runtime_ms).max(0.0);
      
      AneFusedCost {
          base: fused_cost,
          n_ops: ops.len(),
          n_fused_deps: n_fused,
          eliminated_bytes,
          sequential_runtime_ms: sequential_cost.runtime_ms,
          fusion_savings_ms,
      }
  }
  ```
- [x] **T1.6** Implement `AneCost::zero()` and `AneCost + AneCost` (element-wise sum for the sequential baseline). These are internal helpers for `ane_fused_estimate`. Mark `#[inline(always)]`.
- [x] **T1.7** Implement convenience constructor `AneDataDep::chain(ops: &[AneOpShape]) -> Vec<AneDataDep>` — builds a simple linear chain (op[0]→op[1]→...→op[n]) where each intermediate is the output of the previous op. The caller provides intermediate sizes via a separate slice or the constructor infers them from `bytes_moved` deltas.
- [x] **T1.8** Write unit tests in `ane_roofline.rs` under `#[cfg(all(test, feature = "ane_fused_chain"))]`:
  - **T1.8a** Single-op chain (no deps): `ane_fused_estimate(&[op], &[], ...)` returns the same `base` as `ane_estimate(op, ...)`. (G3 hook.)
  - **T1.8b** Two-op chain, intermediate fits: `eliminated_bytes == intermediate_bytes`, `fusion_savings_ms > 0`. (G1 hook.)
  - **T1.8c** Two-op chain, intermediate exceeds working set: `eliminated_bytes == 0`, `fusion_savings_ms == 0.0`, falls back to sequential. (G1 negative case.)
  - **T1.8d** Three-op chain, two of three intermediates fit: `n_fused_deps == 2`, `eliminated_bytes == sum of two fitting intermediates`.
  - **T1.8e** Tiny fused chain below dispatch floor: `base.bound == AneBound::Dispatch`. (G2 hook.)
  - **T1.8f** Family-gated chain: one op requires F3, target is M1 (A13) → `base.bound == AneBound::FamilyGated`.
  - **T1.8g** Determinism: same inputs → same outputs (no RNG).
  - **T1.8h** Empty ops slice: returns `AneFusedCost::zero()` (graceful degenerate case).

### Phase 1 Exit Criteria

- [x] `cargo clippy -p katgpt-core --features ane_fused_chain` compiles clean.
- [x] `cargo test -p katgpt-core --features ane_fused_chain --lib ane_roofline` passes (Plan 379's 23 tests + 13 new fused tests = 36 total).
- [x] `cargo test -p katgpt-core --lib ane_roofline` still passes (Plan 379's 23 tests, no `ane_fused_chain` feature — G3 no-regression).
- [x] `cargo check --all-features` clean (combo-regression check).
- [x] `ane_fused_estimate` is `#[inline(always)]`, zero-alloc.

---

## Phase 2 — GOAT Gate Benchmarks

Goal: prove G1–G5 on synthetic fused chains. No real ANE measurement yet (that's Phase 2.5, gated on M-series hardware availability).

### Tasks

- [ ] **T2.1** Create `katgpt-rs/crates/katgpt-core/benches/bench_439_ane_fused_goat.rs` mirroring `bench_379_ane_roofline_goat.rs` structure.
- [ ] **T2.2** G1 test: two reference chains:
  - **Conv→ReLU** (256ch 3×3 conv 28×28 → elementwise ReLU): intermediate = activation tensor. If it fits in 2 MB → `eliminated_bytes > 0`, `fusion_savings_ms > 0`.
  - **GEMM→Bias** (1024² GEMM → elementwise bias add): intermediate = GEMM output. If it exceeds 2 MB → fallback to sequential, `fusion_savings_ms == 0`.
- [ ] **T2.3** G2 test: routing verdicts for three chain classes (fits / exceeds / tiny-below-floor).
- [ ] **T2.4** G3 test: single-op chain parity (same as T1.8a but in the bench binary).
- [ ] **T2.5** G4 test: `ane_fused_estimate` latency for 8-op chain, p50 < 1 µs. Use `black_box` on both inputs and the output sink to prevent LLVM constant-folding (same technique as `bench_379`).
- [ ] **T2.6** G4-alloc test: 1000 fused-estimate calls, 0 allocations.
- [ ] **T2.7** G5 test: `cargo check` + `cargo check --features ane_fused_chain` + `cargo check --all-features` all clean.

### Phase 2 Exit Criteria

- All 5 GOAT gates PASS on synthetic chains.
- If G1 FAILS (fusion savings negative) → the model's "fusion never hurts" assumption is wrong. Debug the eliminated-bytes accounting.
- If G4 FAILS (> 1 µs for 8-op chain) → the `ops.iter().sum()` loop is too slow. Optimize or reduce the max chain length.

---

## Phase 2.5 — Real ANE Validation (gated on M-series hardware)

Goal: validate the fused-chain model against real ANE measurements on M1/M2/M3. This phase is OPTIONAL for promotion — the GOAT gate (G1–G5) passes on synthetic chains. Real-hardware validation upgrades the confidence but is not a blocker.

**Skip condition:** If no M-series hardware is available, skip this phase. The synthetic GOAT gate is sufficient for promotion (matching Plan 379's precedent, which also validated on synthetic shapes).

### Tasks

- [ ] **T2.5.1** Compile two CoreML models on the target Mac:
  - **Unfused:** conv → (DRAM round-trip) → relu, as two separate CoreML ops.
  - **Fused:** conv+relu as a single fused CoreML ML Program op.
- [ ] **T2.5.2** Measure actual latency of both models on the ANE (use `MLComputePlan` to verify ANE placement). Record: unfused_ms, fused_ms.
- [ ] **T2.5.3** Compare against model predictions:
  - `sequential_runtime_ms` should be ≈ unfused_ms (within Plan 379's ±30% or ~2× tolerance).
  - `base.runtime_ms` should be ≈ fused_ms.
  - `fusion_savings_ms` should be ≈ (unfused_ms - fused_ms).
- [ ] **T2.5.4** If the model's fusion savings diverge > 2× from measured savings → the eliminated-bytes accounting is wrong. File issue, adjust the model.

---

## Phase 3 — Stretch: Tile-Level Cross-Op Overlap (GTSim full distillation)

Goal: the FULL GTSim distillation — model cross-op tile-level pipeline overlap, not just eliminated DRAM traffic. This is the speculative part from Research 423.

**This phase is DEFERRED `[-]`** — it requires modeling the ANE firmware's tile scheduling behavior, which is opaque. Ship Phases 1–2 first; revisit Phase 3 if the eliminated-traffic model (Phase 1) is insufficient for routing decisions.

### Deferred tasks

- [-] **T3.1** Model DMA↔MAC overlap at tile granularity: when op A's tile[i] finishes computing, op B's tile[i] can start if the intermediate tile is on-chip. This requires a tile-graph DAG (GTSim's full abstraction) rather than the aggregate-FLOPs approach of Phase 1.
- [-] **T3.2** Model double-buffer occupancy: N-buffer pipelining in the 2 MB working set. This is GTSim's WS 2-buffer / WS 3-stage analysis applied to the ANE.
- [-] **T3.3** Validate against real ANE measurements on multi-stage fused kernels (conv→bn→relu→pool, 4-stage).

---

## Phase 4 — Consumer Integration (riir-ai)

Goal: wire `ane_fused_estimate` into `NpcBrainRouter` so fused CoreML model chains get fusion-aware routing.

### Tasks

- [ ] **T4.1** In `riir-ai/crates/riir-engine/src/npc_brain_router.rs`, add a method that accepts a fused chain (list of op shapes + deps) and consults `ane_fused_estimate` instead of summing individual `ane_estimate` calls.
- [ ] **T4.2** The `npc_brain.mlpackage` (Plan 255, shipped) is a 3-fused-op CoreML ML Program (sense/emotion/zone projection). Model it as a 3-op chain with 2 data deps and use `ane_fused_estimate` to get a fusion-aware routing threshold.
- [ ] **T4.3** Benchmark: compare the old threshold (sum of 3 individual estimates) vs the new fused estimate. The fused estimate should predict lower latency → potentially shift the ANE-vs-GPU threshold.

---

## Implementation Notes

### Why aggregate-FLOPs (Phase 1) is the right starting point

GTSim's Table 4 ablation shows that going from "data deps only" (42% MAPE) to "data + order constraints" (5.9% MAPE) is the biggest accuracy jump. But Plan 379 is at "no deps at all" (single-op steady-state) — it's BELOW the "data deps only" tier. The first improvement is to model data deps at all, which the aggregate-FLOPs approach does:

- **Current (Plan 379):** treats N fused ops as N independent `max()` calls → `sum(max(C_i, M_i))`.
- **Phase 1 (this plan):** treats N fused ops as one aggregate `max(sum(C_i), sum(M_i) - eliminated_bytes/bw)` → captures the single-dispatch + eliminated-traffic benefit.
- **Phase 3 (stretch):** adds tile-level overlap → captures the cross-op pipeline benefit (the remaining gap to GTSim's accuracy).

Each phase is independently useful and independently shippable. Phase 1 closes the biggest gap (eliminated DRAM traffic + single dispatch) with the least complexity.

### The "fusion never hurts" invariant

The model assumes `fused_runtime ≤ sequential_runtime` (T1.8b asserts `fusion_savings_ms ≥ 0`). This is true because:
1. Eliminated bytes reduce memory time (monotonic).
2. Single dispatch floor ≤ N dispatch floors (the aggregate op pays one floor).
3. `max(sum_C, sum_M) ≤ sum(max(C_i, M_i))` — the max of sums is always ≤ the sum of maxes.

If the invariant fails in practice (real ANE measurements show fusion hurts), it means the ANE compiler chose a bad fusion — the model's prediction is correct but the compiler's decision is wrong. That's a CoreML compiler bug, not a model bug.

### Connection to GTSim's §7.1 case study

GTSim's pipeline-organization insight (WS 3-stage > WS ping-pong for large tiles, WS 2-stage > WS ping-pong for small tiles) maps to the ANE's regime-dependent fusion benefit:
- **Memory-bound chains** (small tiles, below ridge): fusion helps mainly via eliminated DRAM traffic (Phase 1 captures this).
- **Compute-bound chains** (large tiles, above ridge): fusion helps mainly via cross-op pipeline overlap (Phase 3 captures this).

Phase 1 is sufficient for memory-bound chains (the ANE's most common regime, given its 141 FLOP/byte ridge point). Phase 3 adds compute-bound chain accuracy.

---

## Validation Plan

```bash
# Phase 1 exit
CARGO_TARGET_DIR=/tmp/plan439 cargo clippy -p katgpt-core --features ane_fused_chain
CARGO_TARGET_DIR=/tmp/plan439 cargo test -p katgpt-core --features ane_fused_chain --lib ane_roofline
CARGO_TARGET_DIR=/tmp/plan439 cargo test -p katgpt-core --lib ane_roofline  # no feature
CARGO_TARGET_DIR=/tmp/plan439 cargo check --all-features

# Phase 2 GOAT gate
CARGO_TARGET_DIR=/tmp/plan439 cargo run -p katgpt-core --features ane_fused_chain \
  --bench bench_439_ane_fused_goat --release -- --nocapture

# Cleanup
rm -rf /tmp/plan439
```

---

## References

- [Research 423](../.research/423_GPU_Tile_Sim_ANE_Tile_Graph_Overlap.md) — the distillation note (Gain → GOAT after this plan)
- [Plan 379](379_ane_aware_roofline_cost_model.md) — the single-op ANE roofline being extended
- [Research 377](../.research/377_Apple_Neural_Engine_Architecture_Programming_Performance.md) — ANE architecture (the substrate)
- [arXiv:2607.11262](https://arxiv.org/abs/2607.11262) — GTSim paper (the source technique)
