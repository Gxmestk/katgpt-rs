#![cfg(feature = "ternary_group_scale")]
//! G4 alloc check for `simd_ternary_group_matvec_parallel` (Issue 594 pre-flight).
//!
//! The row-parallel ternary matvec is 7.21× the serial kernel on real
//! Ternary-Bonsai-27B shapes (riir-ai `bench_594_ternary_bonsai_throughput`),
//! which is the difference between a feasible and an infeasible CPU decode. But
//! `forward_ternary` carries a **0-allocations-per-token** gate
//! (`forward_ternary_g4_alloc_free`), so it can only adopt the parallel kernel
//! if rayon's `par_chunks_mut` split is itself alloc-free in steady state.
//!
//! That is not obvious either way — rayon allocates for job queues in the
//! general case — so it is measured here rather than assumed. The precedent
//! suggesting it holds: `forward_llama` already calls the dense
//! `matmul_parallel` for its LM head and still passes its own 0-alloc gate.
//!
//! Run:
//! ```text
//! cargo test -p katgpt-core --features ternary_group_scale \
//!   --test ternary_group_parallel_alloc_check -- --nocapture
//! ```

use katgpt_core::{TernaryGroupWeights, simd_ternary_group_matvec_parallel};
#[path = "common/mod.rs"]
mod common;
counting_allocator!();

/// Rows must exceed `PARALLEL_ROW_MIN` (256) so the rayon path is actually
/// taken — below it the function delegates to the serial kernel and the test
/// would pass vacuously.
const ROWS: usize = 1024;
const COLS: usize = 512;

fn fixture() -> TernaryGroupWeights {
    let mut w = TernaryGroupWeights::new(ROWS, COLS);
    let mut s = 0x594u64;
    for r in 0..ROWS {
        for c in 0..COLS {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let v = match s % 3 {
                0 => -1i8,
                1 => 0,
                _ => 1,
            };
            if v != 0 {
                w.set(r, c, v);
            }
        }
        for g in 0..w.groups_per_row {
            w.set_scale(r, g, 0.05);
        }
    }
    w
}

/// G4 — the parallel matvec allocates nothing per call in steady state.
#[test]
fn parallel_ternary_matvec_is_alloc_free() {
    let w = fixture();
    let x = vec![0.01f32; COLS];
    let mut y = vec![0.0f32; ROWS];

    // Warm up: spins up rayon's thread pool and pages in the weights. Any
    // one-time pool allocation happens here, outside the measured region.
    for _ in 0..4 {
        simd_ternary_group_matvec_parallel(&w, &x, &mut y);
    }

    const N_CALLS: usize = 50;
    let (_, allocs) = alloc_delta(|| {
        for _ in 0..N_CALLS {
            simd_ternary_group_matvec_parallel(&w, &x, &mut y);
        }
    });

    println!(
        "simd_ternary_group_matvec_parallel × {N_CALLS} calls ({ROWS}×{COLS}) = {allocs} allocations"
    );

    assert_eq!(
        allocs, 0,
        "expected 0 allocations over {N_CALLS} calls, got {allocs}. \
         If rayon's par_chunks_mut allocates here, `forward_ternary` CANNOT adopt \
         the parallel kernel without breaking its G4 gate — the 7.21x speedup would \
         then be available only to prefill/batch paths, not per-token decode.",
    );
}
