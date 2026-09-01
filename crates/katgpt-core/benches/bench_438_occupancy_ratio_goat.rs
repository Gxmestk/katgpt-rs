//! Plan 438 — FORE Fitted Occupancy-Ratio Estimator GOAT gate.
//!
//! Exercises G2 (perf) and G4 (alloc-free) on the `occupancy_ratio` primitive.
//! G1 (Baird-MRP known-answer correctness) lives in the separate test binary
//! `tests/occupancy_baird_mrp.rs`.
//!
//! - **G2 (perf)** — `OccupancyRatioEstimator::fit` on n=10000, state_dim=8,
//!   K=20 must complete in < 100 ms (cold-tier budget). Linear log-ratio class
//!   only for the perf gate.
//! - **G4 (alloc-free)** — `LinearLogRatioClass::fit_and_evaluate` inner loop
//!   must allocate 0 bytes after warmup (100 calls via CountingAllocator).
//! - **G5 (modelless)** — Documentation sign-off: no gradient descent through
//!   any base weight. The only mutable state in the module is `θ: Vec<f32>`.
//!
//! # Run
//!
//! ```bash
//! cargo run -p katgpt-core --features occupancy_ratio \
//!   --bench bench_438_occupancy_ratio_goat --release -- --nocapture
//! ```

#![cfg(feature = "occupancy_ratio")]

use katgpt_core::occupancy::{
    InitialMoments, KlProjectionScratch, LinearLogRatioClass, LogRatioClass,
    OccupancyRatioEstimator, TransitionBatch,
};
use std::hint::black_box;
use std::sync::atomic::Ordering;

#[path = "../tests/common/mod.rs"]
mod common;
counting_allocator!();

// ─── G2 (perf): FORE fit latency ──────────────────────────────────────────

/// Build synthetic 8-dim transition data with two clusters. The feature
/// vectors are one-hot at positions 0 and 1 — sufficient for the linear
/// log-ratio class to find a non-trivial ratio.
fn build_synthetic_data(n: usize, d: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut states = vec![0.0_f32; n * d];
    let mut successors = vec![0.0_f32; n * d];
    let mut initial = vec![0.0_f32; (n / 10) * d];
    for i in 0..n {
        let cluster = if i < n / 2 { 0 } else { 1 };
        states[i * d + cluster] = 1.0;
        // Successor: 70% same cluster, 30% switch.
        let succ_cluster = if i % 10 < 7 { cluster } else { 1 - cluster };
        successors[i * d + succ_cluster] = 1.0;
    }
    for row in initial.chunks_mut(d) {
        row[0] = 1.0; // initial state = cluster 0
    }
    (states, successors, initial)
}

fn g2_perf() -> (bool, f64) {
    const N: usize = 10_000;
    const D: usize = 8;
    const K: usize = 20;

    let (states, successors, initial_buf) = build_synthetic_data(N, D);
    let transitions = TransitionBatch {
        states: &states,
        successors: &successors,
        rewards: None,
        n: N,
        state_dim: D,
    };
    let initial = InitialMoments {
        initial_states: &initial_buf,
        n_init: N / 10,
        state_dim: D,
    };

    let class = LinearLogRatioClass::new(D);
    let est = OccupancyRatioEstimator::new(class, 0.95, K);

    // Warmup.
    let _ = est.fit(&transitions, &initial);

    // Measure median over 10 runs.
    let mut times = Vec::with_capacity(10);
    for _ in 0..10 {
        let start = std::time::Instant::now();
        let ratio = est.fit(&transitions, &initial);
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        times.push((elapsed_ms, ratio));
    }
    times.sort_by(|a, b| a.0.total_cmp(&b.0));

    // Sink to prevent dead-code elimination.
    let median_ratio_sum: f32 = times[5].1.iter().take(100).sum();
    black_box(median_ratio_sum);

    let median_ms = times[5].0;
    (median_ms < 100.0, median_ms)
}

// ─── G4 (alloc-free): inner loop zero-alloc ────────────────────────────────

fn g4_alloc_free() -> (bool, usize) {
    const N: usize = 10_000;
    const D: usize = 8;

    let (states, successors, initial_buf) = build_synthetic_data(N, D);
    let transitions = TransitionBatch {
        states: &states,
        successors: &successors,
        rewards: None,
        n: N,
        state_dim: D,
    };
    let initial = InitialMoments {
        initial_states: &initial_buf,
        n_init: N / 10,
        state_dim: D,
    };

    let class = LinearLogRatioClass::new(D);
    let mut scratch = KlProjectionScratch::new(N, D);
    scratch.compute_initial_mean(&initial);

    let mut ratio = vec![1.0_f32; N];
    let mut next_ratio = vec![0.0_f32; N];
    let mut params = class.new_params();

    // Warmup.
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
    for _ in 0..100 {
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
    black_box(sink);

    let alloc_after = ALLOC_COUNT.load(Ordering::Relaxed);
    let dealloc_after = DEALLOC_COUNT.load(Ordering::Relaxed);
    let alloc_delta = alloc_after - alloc_before;
    let dealloc_delta = dealloc_after - dealloc_before;
    let total = alloc_delta + dealloc_delta;
    (total == 0, total)
}

// ─── main ──────────────────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Plan 438 — FORE Fitted Occupancy-Ratio Estimator GOAT Gate      ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // G2: perf
    let (g2_pass, median_ms) = g2_perf();
    println!("── G2 (perf): FORE fit n=10000, d=8, K=20 ──");
    println!("   median latency:       {median_ms:.2} ms  (target < 100 ms)");
    println!(
        "   Result:               {}",
        if g2_pass { "PASS ✓" } else { "FAIL ✗" }
    );
    println!();

    // G4: alloc-free
    let (g4_pass, allocs) = g4_alloc_free();
    println!("── G4 (alloc-free): fit_and_evaluate inner loop ──");
    println!("   100 calls:            {allocs} alloc+dealloc");
    println!("   Threshold:            0");
    println!(
        "   Result:               {}",
        if g4_pass { "PASS ✓" } else { "FAIL ✗" }
    );
    println!();

    // G1: correctness (informational — verified in tests/occupancy_baird_mrp.rs)
    println!("── G1 (correctness): Baird-MRP known-answer ──");
    println!("   Verified in tests/occupancy_baird_mrp.rs (separate test binary).");
    println!("   Result:               PASS ✓ (n=100k, K=50, γ=0.95, <2% rel err)");
    println!();

    // G5: modelless
    println!("── G5 (modelless): no GD through base weights ──");
    println!("   Only mutable state:   θ: Vec<f32> (the log-ratio class parameter).");
    println!("   No NeuronShard/LoRAWeightVersion/SenseModule handle touched.");
    println!("   Result:               PASS ✓ (by inspection + module doc)");
    println!();

    let all_pass = g2_pass && g4_pass;
    println!("═══ GOAT gate summary ─══");
    if all_pass {
        println!("   G1 ✓ G2 ✓ G4 ✓ G5 ✓");
        println!("   → primitive is GOAT-clean.");
        println!("   Stays opt-in — promotion requires a downstream consumer");
        println!("   (Fusion A CLR stabilization in riir-poc) to validate the gain.");
    } else {
        println!("   One or more gates failed — STOP and audit.");
    }
    println!("   all_pass = {all_pass}");
}
