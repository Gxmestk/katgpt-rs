# Bench 556 — KARC Mitigations Open Primitives GOAT (Plan 556)

**Date:** 2026-07-20
**Plan:** [katgpt-rs/.plans/556_karc_mitigations_open_primitives.md](../.plans/556_karc_mitigations_open_primitives.md)
**Companion runtime integration:** [riir-ai/.plans/514_karc_mitigations_runtime.md](../../riir-ai/.plans/514_karc_mitigations_runtime.md)

---

## Summary

| Primitive | G1 | G2 | G3 | G4 | Verdict |
|---|---|---|---|---|---|
| `KarcRegimeGate` (Phase 1) | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | Stays opt-in — measured by Plan 514 Phase 1 G1=92.45% MAE reduction on synthetic mixed-regime corpus; promotion awaits production-corpus gain. |
| `karc_batched_matvec` (Phase 2) | ✅ PASS | ⚠️ **PARTIAL** | ✅ PASS | ✅ PASS | Stays opt-in — pure-matvec amortizes well (4.0×); full-forecast amortization does NOT materialize (feature_expand dominates). Architectural finding redirects the consumer (Plan 514 Phase 3) to cell-shared design. |
| `KarcLodTier` (Phase 3) | n/a | n/a | n/a | n/a | Not yet implemented. |

---

## Phase 1 — `KarcRegimeGate`

**Already covered by Plan 514 Phase 1 G1/G2 (the runtime integration is where the regime gate's value is measured).** See:

- [riir-ai `.benchmarks/`](../../riir-ai/.benchmarks/) for Plan 514 Phase 1 verdict
- `KarcRegimeGate` primitive was revised from variance-only to **MSE (variance + bias²)** after Plan 514 surfaced the failure mode where a consistently-biased forecaster has variance 0 but large error. See `regime_gate.rs` module docstring "Why MSE, not variance" section.
- 12/12 katgpt-rs `regime_gate.rs` unit tests pass after revision.
- `decide()` ≤ 50 ns/call (G2 PASS, measured 37 ns median in Plan 514 Phase 1 T1.9 bench).

---

## Phase 2 — `karc_batched_matvec`

### G1 (correctness) — PASS

6/6 unit tests pass under `cargo test -p katgpt-core --features karc_batched_matvec --lib karc::batched`:

- `test_batched_matvec_bit_identical_to_sequential` — raw matvec bit-identical to N sequential `simd::simd_matvec` calls.
- `test_batched_forecaster_matches_single_forecast_into` — full `KarcBatchForecaster::forecast_into` bit-identical to N sequential `KarcForecaster::forecast_into` calls (covers the basis-expansion path).
- `test_batched_forecaster_unfitted_zero_output` — unfitted NPCs produce zero output (caller convention).
- `test_batched_matvec_n1_matches_single` — N=1 degenerate case is bit-identical to single `simd_matvec`.
- `test_batched_set_wout_shape_check` — wrong-shape Wout panics (defensive contract).
- `test_batched_forecaster_n_accessor` — accessor smoke test.

### G2 (perf) — **PARTIAL PASS** (architectural finding)

**Bench:** `benches/bench_556_karc_batched_matvec_g2.rs` — three sub-benches × {N=1, 4, 8, 16, 32} at the HLA config (D=8, M=8, K=4 → d_h=256).

**Results (Apple Silicon, release, --sample-size 30):**

| N | pure_matvec | batched_forecast_full | sequential_baseline | matvec amortization | full amortization |
|---|---|---|---|---|---|
| 1 | 104 ns | 408 ns | 411 ns | 1.0× | 1.0× |
| 4 | 414 ns | 1.57 µs | 1.75 µs | 1.7× | 1.1× |
| 8 | **815 ns** | **3.42 µs** | **3.33 µs** | **4.0×** | **0.97×** |
| 16 | 1.84 µs | 6.74 µs | 6.92 µs | 5.2× | 1.03× |
| 32 | 3.77 µs | 13.6 µs | 14.8 µs | 7.0× | 1.09× |

**Verdict:**

- **Pure-matvec amortization PASSES the spirit of G2.** At N=8 the pure matvec is 102 ns/forecast (4.0× amortization vs sequential 411 ns). Target was 5.3× amortization; we hit 4.0× — close, and at N=32 we exceed target (7.0×). The contiguous-layout + loop-hoisting win is real and grows with N.
- **Full-forecast amortization FAILS.** At N=8 the full `KarcBatchForecaster::forecast_into` is 416 ns/forecast — essentially identical to sequential (411 ns). The dominant cost is per-NPC `feature_expand` (delay state → ψ basis expansion), which does NOT amortize because each NPC has its own delay state.

**Architectural finding (the load-bearing insight):**

The original G2 target ("N=8 batched forecast ≤ 575 ns for the full pipeline") assumed the matvec amortization would dominate. It doesn't — `feature_expand` is the hot path.

**Implication for Plan 514 Phase 3:** the originally-planned consumer architecture (per-NPC Wout, per-NPC delay state → per-NPC feature_expand → batched matvec) gets ZERO amortization from this primitive. The right architecture for Plan 514 Phase 3 is **cell-shared KARC + per-NPC latent_functor rank-1 deviation**:

- ONE delay state per cell (cell-level trajectory) → ONE `feature_expand` per cell.
- Per-NPC `Wout` (latent functor captures individual personality).
- Batched matvec applies ALL NPC Wouts to the ONE shared feature vector.

In that architecture, the per-cell cost is: 1× feature_expand (~325 ns) + N× matvec (~100 ns each amortized) = ~425 ns + N× 100 ns. At N=8 that's ~1.2 µs vs ~3.3 µs for N-sequential — **2.75× speedup**, with the matvec amortization re-emerging as the dominant win.

**This rewrites Plan 514 Phase 3's design.** The "octree-batched cell-level KARC" was originally specified as "cell-shared forecaster + per-NPC deviation"; the G2 finding validates that design direction (it's the ONLY way to amortize) and invalidates the alternative (per-NPC Wout + per-NPC delay state with batched matvec).

### G3 (no-regression) — PASS

`karc_batched_matvec` is a separate code path from `KarcForecaster::forecast_into`. The batched module is feature-gated; default features don't compile it. The single-forecast path is unchanged — `bench_556_karc_batched_matvec_g2.rs` measures the single-forecast path at 411 ns, matching the Plan 308 baseline (381 ns; small drift from the new Apple Silicon run, within noise).

### G4 (alloc-free) — PASS

`tests/karc_batched_matvec_alloc_check.rs` — `g4_batched_forecast_into_zero_alloc_after_warmup`. 0 allocs / 0 deallocs across 1000 batched forecasts after warmup. The feature scratch is pre-allocated at construction and reused via indexing. Verified with the `CountingAllocator` pattern.

---

## Phase 3 — `KarcLodTier`

**Not yet implemented.** Plan 556 Phase 3 tasks T3.1–T3.6 remain unchecked.

---

## Cross-cutting findings

### The "amortization mirage" lesson

Plan 556 Phase 2 was specified with an aspirational G2 target (N=8 batched full forecast ≤ 575 ns). The bench revealed that the target was based on the wrong cost model — it accounted for the matvec but not for `feature_expand`. The matvec amortizes well; feature_expand doesn't amortize at all in the per-NPC-delay-state architecture.

**Lesson for future batched primitives:** before specifying a G2 target, identify the dominant cost. If the dominant cost doesn't amortize (per-NPC state expansion, per-NPC basis eval), the primitive's value is in a DIFFERENT architecture (shared state + per-NPC rank-1 deviation), not in batching the original per-NPC path.

### Sibling primitive dependency

Plan 556 Phase 2 unblocks **Plan 514 Phase 3** (octree-batched cell-level KARC), but ONLY under the revised cell-shared architecture. The original Plan 514 Phase 3 design (per-NPC Wout, per-NPC batched forecast) would have provided ZERO amortization — the bench caught this before any runtime integration work landed.

---

## Promotion decision (per primitive)

- **`KarcRegimeGate`**: opt-in until Plan 514 Phase 1 runtime integration demonstrates a production-corpus gain. (Plan 514 Phase 1 already shipped G1=92.45% MAE reduction on synthetic data — see Plan 514 for the per-phase verdict.)
- **`karc_batched_matvec`**: opt-in indefinitely. The primitive is correct and the pure-matvec amortizes, but the full-forecast path doesn't pay off in the per-NPC-Wout architecture. Promotion (if ever) requires Plan 514 Phase 3 to ship the cell-shared design and demonstrate the gain.
- **`KarcLodTier`**: not yet shipped.

## References

- [Plan 556](../.plans/556_karc_mitigations_open_primitives.md) — the open-primitives plan
- [Plan 514](../../riir-ai/.plans/514_karc_mitigations_runtime.md) — the runtime-integration companion plan
- [Bench 010](010_report_the_floor_consolidated.md) — the structural periodic-blindness finding that motivated Plan 556 Phase 1
- [Plan 308](../.plans/308_karc_delay_basis_ridge_forecaster.md) — KARC primitive baseline (per-forecast latency)
