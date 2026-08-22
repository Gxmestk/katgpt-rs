//! Bench 672 G4 — zero-allocation steady state for `signed_coupling_dynamics`
//! (Issue 680).
//!
//! Separate test binary (mirrors `bench_656_privilege_alloc_check` /
//! `effective_degree_alloc_check`) so the `CountingAllocator` global picks up
//! only this primitive's allocations rather than whatever the sibling test
//! binaries are doing in parallel.
//!
//! **One `#[test]` function, not five.** The allocator counter is a process
//! global, so tests in the same binary on different threads corrupt each
//! other's deltas.
//!
//! Covers every steady-state path: both update kernels, the sampler, both
//! reducers, and the susceptibility accumulator. `SignedGraph::from_edges`
//! is *construction*, so it is measured but deliberately not gated at zero —
//! the contract is "no heap after construction".
//!
//! # Run
//!
//! ```bash
//! cargo test -p katgpt-core --no-default-features \
//!     --features signed_coupling_dynamics \
//!     --test signed_coupling_alloc_check --release -- --nocapture
//! ```

#![cfg(feature = "signed_coupling_dynamics")]

use katgpt_core::signed_coupling::{
    Couplings, InformedCouplings, SignedGraph, SusceptibilityAccumulator, crowd_conviction,
    net_opinion, sample_states_into, signed_coupling_update_informed_into,
    signed_coupling_update_into,
};
use std::hint::black_box;

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

/// The macro exposes `ALLOC_COUNT` + an `alloc_delta` closure wrapper; the
/// checks below want a plain counter read so each measured loop stays a bare
/// `for` (a closure would add its own capture allocation risk).
fn alloc_now() -> usize {
    ALLOC_COUNT.load(std::sync::atomic::Ordering::Relaxed)
}

const N: usize = 256;
const DEGREE: usize = 8;
const ITERS: usize = 1_000;

fn ring_graph() -> SignedGraph {
    let mut edges = Vec::new();
    for i in 0..N as u32 {
        for k in 1..=(DEGREE / 2) as u32 {
            let j = (i + k) % N as u32;
            let sign = if (i + k) % 3 == 0 { -1 } else { 1 };
            edges.push((i, j, sign));
        }
    }
    SignedGraph::from_edges(N, &edges).expect("ring graph is well-formed")
}

#[test]
fn g4_steady_state_is_allocation_free() {
    // ── Construction (allowed to allocate; reported, not gated). ──
    let before = alloc_now();
    let graph = ring_graph();
    let construction = alloc_now() - before;
    println!("G4 SignedGraph::from_edges (N={N}): {construction} allocs [construction, not gated]");
    assert!(!graph.is_empty());

    let couplings = Couplings::default();
    let informed_couplings = InformedCouplings::default();
    let states: Vec<f32> = (0..N)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let informed: Vec<bool> = (0..N).map(|i| i % 3 == 0).collect();
    let intrinsic = vec![0.1f32; N];
    let uniforms = vec![0.5f32; N];
    let mut probs = vec![0.0f32; N];
    let mut next = vec![0.0f32; N];
    let mut acc = SusceptibilityAccumulator::new();

    // Warm every path once so lazily-initialized statics (if any) are paid for
    // before the counter reads.
    signed_coupling_update_into(&graph, &states, &couplings, &intrinsic, &mut probs);
    signed_coupling_update_informed_into(
        &graph,
        &states,
        &informed,
        &informed_couplings,
        &intrinsic,
        &mut probs,
    );
    sample_states_into(&probs, &uniforms, &mut next);
    black_box(net_opinion(&states));
    black_box(crowd_conviction(&states));
    acc.observe(0.0);
    acc.reset();

    // ── signed_coupling_update_into ──
    let before = alloc_now();
    for _ in 0..ITERS {
        signed_coupling_update_into(
            black_box(&graph),
            black_box(&states),
            black_box(&couplings),
            black_box(&intrinsic),
            black_box(&mut probs),
        );
    }
    let d = alloc_now() - before;
    println!("G4 signed_coupling_update_into:          {d} allocs / {ITERS} calls");
    assert_eq!(d, 0, "the base kernel must not allocate");

    // ── signed_coupling_update_informed_into ──
    let before = alloc_now();
    for _ in 0..ITERS {
        signed_coupling_update_informed_into(
            black_box(&graph),
            black_box(&states),
            black_box(&informed),
            black_box(&informed_couplings),
            black_box(&intrinsic),
            black_box(&mut probs),
        );
    }
    let d = alloc_now() - before;
    println!("G4 signed_coupling_update_informed_into: {d} allocs / {ITERS} calls");
    assert_eq!(d, 0, "the informed kernel must not allocate");

    // ── sample_states_into ──
    let before = alloc_now();
    for _ in 0..ITERS {
        sample_states_into(
            black_box(&probs),
            black_box(&uniforms),
            black_box(&mut next),
        );
    }
    let d = alloc_now() - before;
    println!("G4 sample_states_into:                   {d} allocs / {ITERS} calls");
    assert_eq!(d, 0, "sampling must not allocate");

    // ── reducers + accumulator ──
    let before = alloc_now();
    for _ in 0..ITERS {
        let n = net_opinion(black_box(&states));
        black_box(crowd_conviction(black_box(&states)));
        acc.observe(black_box(n));
        black_box(acc.susceptibility(N));
    }
    let d = alloc_now() - before;
    println!("G4 reducers + susceptibility:            {d} allocs / {ITERS} calls");
    assert_eq!(d, 0, "order-parameter reducers must not allocate");
    assert_eq!(acc.count(), ITERS as u64);
}
