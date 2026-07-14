//! Plan 438 T4.3 — G4 zero-alloc audit for FORE's inner KL-projection loop.
//!
//! `LinearLogRatioClass::fit_and_evaluate` is the hot-path inner loop: it's
//! called K times per `OccupancyRatioEstimator::fit` call, and each call runs
//! up to MAX_NEWTON_ITERS Newton iterations with up to MAX_LM_RETRIES LM
//! retries. All scratch buffers are pre-allocated in `KlProjectionScratch`;
//! the inner loop must be allocation-free (G4).
//!
//! This test calls `fit_and_evaluate` directly (bypassing `fit`) to isolate
//! the inner loop from the outer-loop allocations (`ratio`, `next_ratio`,
//! `params`, `scratch`). After warmup, 100 calls must produce 0 allocations.

#![cfg(feature = "occupancy_ratio")]

use katgpt_core::occupancy::{
    InitialMoments, KlProjectionScratch, LinearLogRatioClass, LogRatioClass, TransitionBatch,
};
use std::sync::atomic::Ordering;

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

#[test]
fn g4_fit_and_evaluate_zero_alloc_after_warmup() {
    const N: usize = 10_000;
    const D: usize = 8;
    const K_RETRIES: usize = 100;

    // Build synthetic 8-dim transition data (deterministic — no PRNG needed).
    // Two clusters: states 0..N/2 have feature vector [1,0,0,0,0,0,0,0],
    // states N/2..N have [0,1,0,0,0,0,0,0]. Successors swap the clusters.
    let mut states = vec![0.0_f32; N * D];
    let mut successors = vec![0.0_f32; N * D];
    for i in 0..N {
        let cluster = if i < N / 2 { 0 } else { 1 };
        states[i * D + cluster] = 1.0;
        // Successor: opposite cluster.
        successors[i * D + (1 - cluster)] = 1.0;
    }
    let transitions = TransitionBatch {
        states: &states,
        successors: &successors,
        rewards: None,
        n: N,
        state_dim: D,
    };

    // Initial state = cluster 0.
    let mut init = vec![0.0_f32; 100 * D];
    for row in init.chunks_mut(D) {
        row[0] = 1.0;
    }
    let initial = InitialMoments {
        initial_states: &init,
        n_init: 100,
        state_dim: D,
    };

    let class = LinearLogRatioClass::new(D);
    let mut scratch = KlProjectionScratch::new(N, D);
    scratch.compute_initial_mean(&initial);

    let mut ratio = vec![1.0_f32; N];
    let mut next_ratio = vec![0.0_f32; N];
    let mut params = class.new_params();

    // Warmup: settle lazy allocations (SIMD dispatch, etc.).
    for _ in 0..5 {
        class.fit_and_evaluate(
            &transitions,
            &initial,
            &ratio,
            0.95,
            &mut params,
            &mut next_ratio,
            &mut scratch,
        );
        std::mem::swap(&mut ratio, &mut next_ratio);
    }

    // Measure: 100 inner-loop calls must allocate 0 bytes.
    let alloc_before = ALLOC_COUNT.load(Ordering::Relaxed);
    let dealloc_before = DEALLOC_COUNT.load(Ordering::Relaxed);

    let mut sink = 0.0_f32;
    for _ in 0..K_RETRIES {
        class.fit_and_evaluate(
            &transitions,
            &initial,
            &ratio,
            0.95,
            &mut params,
            &mut next_ratio,
            &mut scratch,
        );
        sink += next_ratio[0];
        std::mem::swap(&mut ratio, &mut next_ratio);
    }
    std::hint::black_box(sink);

    let alloc_after = ALLOC_COUNT.load(Ordering::Relaxed);
    let dealloc_after = DEALLOC_COUNT.load(Ordering::Relaxed);
    let alloc_delta = alloc_after - alloc_before;
    let dealloc_delta = dealloc_after - dealloc_before;

    assert_eq!(
        alloc_delta, 0,
        "G4 FAIL: fit_and_evaluate allocated {alloc_delta} times in {K_RETRIES} calls"
    );
    assert_eq!(
        dealloc_delta, 0,
        "G4 FAIL: fit_and_evaluate deallocated {dealloc_delta} times in {K_RETRIES} calls"
    );
}
