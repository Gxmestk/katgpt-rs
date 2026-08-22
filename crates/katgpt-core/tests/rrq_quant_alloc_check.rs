//! Plan 568 — RRQ G4 zero-alloc GOAT gate (separate test binary).
//!
//! The hot-path methods `RrqStage::dot_acc_into` and `RrqWeights::prefix_dot_into`
//! MUST NOT allocate. They are per-tick, in-place computations that dequantize
//! codes on the fly and accumulate into caller-provided `&mut [f32]` buffers.
//! This is the G4 gate: 0 allocations over 1000 steady-state calls.
//!
//! Separate binary from the lib unit tests so the `CountingAllocator` global
//! picks up only the hot-path allocations (matches the `linking_fold_alloc_check.rs`
//! / `sleep_time_alloc_check.rs` / `karc_alloc_check.rs` / `analytic_lattice_alloc_check.rs`
//! pattern — a CountingAllocator in the bench binary would skew the
//! `Instant::now()` timing loops, and parallel lib tests would corrupt the
//! deltas).
//!
//! Single test function so both checks run serially against the shared global
//! `CountingAllocator` (parallel test execution would corrupt the deltas — the
//! alloc counter is process-wide).
//!
//! The constructor (`from_weights_rtn`) and the full-reconstruction helper
//! (`prefix_reconstruct_into`) are cold-path (model load) and explicitly
//! allowed to allocate — they build the owned stage Vecs. They are NOT gated
//! here.

#![cfg(feature = "rrq_quant")]

use katgpt_core::rrq_quant::RrqWeights;
use std::hint::black_box;
use std::sync::atomic::Ordering;

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

const N_WARMUP: usize = 10;
const N_CALLS: usize = 1000;

/// G4 zero-alloc gate for `prefix_dot_into` (the headline hot path) and
/// `dot_acc_into` (the per-stage kernel it composes).
///
/// Single test function so both checks run serially against the shared global
/// `CountingAllocator` (parallel test execution would corrupt the deltas — the
/// alloc counter is process-wide).
#[test]
fn g4_zero_alloc_after_warmup_prefix_dot() {
    // Fixture: 8x16 weight matrix, 2 residual stages, group_size=8.
    // Constructed ONCE before the measured window (construction allocates —
    // that's fine, it's cold-path).
    let rows = 8;
    let cols = 16;
    let n = rows * cols;
    let weights: Vec<f32> = (0..n).map(|i| (i as f32) * 0.03 - 1.2).collect();
    let rrq = RrqWeights::from_weights_rtn(&weights, rows, cols, 2, 8);

    let x: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.1 - 0.8).collect();
    let mut out = vec![0.0_f32; rows];
    let mut scratch = vec![0.0_f32; rows];

    // Warmup — rule out first-call lazy init.
    for _ in 0..N_WARMUP {
        rrq.prefix_dot_into(2, &x, &mut out, &mut scratch);
        black_box(&mut out);
    }

    // Measure: N_CALLS calls, sum the alloc delta.
    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    for _ in 0..N_CALLS {
        rrq.prefix_dot_into(2, &x, &mut out, &mut scratch);
        black_box(&mut out);
    }
    let after = ALLOC_COUNT.load(Ordering::Relaxed);
    let alloc_count = after - before;

    assert!(
        alloc_count == 0,
        "G4 FAIL: prefix_dot_into allocated {alloc_count} times over {N_CALLS} steady-state calls (expected 0). \
         The hot path dequants codes into caller-provided &mut [f32]; nothing should allocate."
    );

    // Also test the per-stage kernel directly (dot_acc_into).
    let mut stage_out = vec![0.0_f32; rows];
    for _ in 0..N_WARMUP {
        rrq.base.dot_acc_into(cols, &x, &mut stage_out);
        black_box(&mut stage_out);
    }
    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    for _ in 0..N_CALLS {
        rrq.base.dot_acc_into(cols, &x, &mut stage_out);
        black_box(&mut stage_out);
    }
    let after = ALLOC_COUNT.load(Ordering::Relaxed);
    let alloc_count = after - before;

    assert!(
        alloc_count == 0,
        "G4 FAIL: dot_acc_into allocated {alloc_count} times over {N_CALLS} steady-state calls (expected 0)."
    );
}
