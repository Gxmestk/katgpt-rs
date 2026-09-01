//! Issue 837 wiring G4 — zero steady-state allocations for the wired
//! [`EngramTripwire`] detector (the detector-half companion of the Issue-656
//! privileged-fuse alloc check).
//!
//! **Own test binary, one `#[test]` function** — the counting allocator is a
//! process global, so sharing a binary with sibling tests on parallel threads
//! corrupts the deltas (the `bench_656_privilege_alloc_check` lesson, first
//! draft read 28 spurious allocs that way).
//!
//! What must not allocate once the ring is at capacity:
//! - `observe_benign` — metrics on the caller's scratch, ring push within
//!   pre-reserved capacity (evict-oldest keeps `len <= cap`), threshold
//!   recompute into the pre-reserved sort scratch (in-place sort).
//! - `check` — read-only metrics into the caller's scratch.
//!
//! Run:
//! ```bash
//! cargo test -p katgpt-core --features engram_tripwire \
//!     --test engram_tripwire_alloc_check --release -- --nocapture
//! ```

#![cfg(feature = "engram_tripwire")]

use katgpt_core::engram::{EngramTripwire, EngramTripwireConfig};
use katgpt_core::evidence_tripwire::TripwireMetrics;
use std::hint::black_box;

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

use std::sync::atomic::Ordering;

const D: usize = 32;
const K: usize = 8;
const CAP: usize = 256;
const STEADY_ITERS: usize = 1_000;

struct Xs(u64);

impl Xs {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn jitter(&mut self, span: f32) -> f32 {
        (self.f32() * 2.0 - 1.0) * span
    }
}

#[test]
fn engram_tripwire_steady_state_is_alloc_free() {
    let mut rng = Xs(0xA110_C);
    let mut q = [0.0f32; D];
    for x in q.iter_mut() {
        *x = rng.f32();
    }

    let cfg = EngramTripwireConfig {
        alpha: 0.05,
        benign_pool_capacity: CAP,
    };
    let mut tw = EngramTripwire::new(cfg);
    let mut m = TripwireMetrics {
        n: 0,
        h_norm: 0.0,
        top1_share: 0.0,
        tau: 0.0,
        top1_consumer_rank: 0.0,
    };

    // Deterministic benign world: retrieval order == gate order (rank 1).
    let retrieval: Vec<f32> = (0..K).map(|i| 0.9 - 0.03 * i as f32 + rng.jitter(0.01)).collect();
    let gates: Vec<f32> = (0..K).map(|i| 0.95 - 0.03 * i as f32 + rng.jitter(0.005)).collect();

    // Warm-up: fill the ring past capacity so evictions + full-length sort
    // scratch are both engaged before the measured window.
    for _ in 0..(CAP * 2) {
        tw.observe_benign(&retrieval, &gates, &mut m);
    }
    assert_eq!(tw.pool_len(), CAP);

    // Steady state — the measured window.
    ALLOC_COUNT.store(0, Ordering::Relaxed);
    for _ in 0..STEADY_ITERS {
        black_box(tw.observe_benign(black_box(&retrieval), black_box(&gates), &mut m));
        let v = tw.check(black_box(&retrieval), black_box(&gates), &mut m);
        black_box(v.fired);
    }
    let allocs = ALLOC_COUNT.load(Ordering::Relaxed);
    assert_eq!(allocs, 0, "steady-state observe+check allocations");
    assert_eq!(tw.pool_len(), CAP, "ring stays capacity-bounded");
    assert!(tw.is_calibrated());
}
