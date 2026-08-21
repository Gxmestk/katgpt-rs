//! tether latency benchmark (Issue 675 T6 — G2 perf gate).
//!
//! Reports median latency for:
//! - `TetherBlend::observe` at the two representative window cadences
//!   (K = 16 and K = 64). Target: ≤ 100 ns per observe at K ≤ 64 (the issue's
//!   G2 gate); the closure boundary is included by construction — every K-th
//!   observe closes the window, so the amortized close+EMA cost is inside the
//!   measurement.
//! - `TetherBlend::blend` (the per-decision hot path).
//! - `horizon_decay` (the LUT lookup shape).
//!
//! # Run
//!
//! ```bash
//! cargo bench -p katgpt-core --features tether --bench tether_bench \
//!   -- --warm-up-time 0.5 --measurement-time 1.5 --sample-size 30
//! ```

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use katgpt_core::tether::{TetherBlend, horizon_decay};

fn bench_observe(c: &mut Criterion) {
    for k in [16u32, 64] {
        let mut g = c.benchmark_group(format!("tether_observe_K{k}"));
        g.throughput(criterion::Throughput::Elements(1));
        g.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, &k| {
            let mut t = TetherBlend::with_params(0.2, 0.95, k);
            // A varied but non-degenerate stream (p1 != p2 everywhere).
            let mut i = 0u32;
            b.iter(|| {
                let r = if i.is_multiple_of(3) { 1.0 } else { 0.0 };
                i = i.wrapping_add(1);
                black_box(t.observe(black_box(r), black_box(0.4), black_box(0.6)))
            });
        });
        g.finish();
    }
}

fn bench_blend(c: &mut Criterion) {
    let t = TetherBlend::with_params(0.37, 0.95, 64);
    c.bench_function("tether_blend", |b| {
        b.iter(|| black_box(t.blend(black_box(0.31), black_box(0.72))))
    });
}

fn bench_horizon_decay(c: &mut Criterion) {
    c.bench_function("tether_horizon_decay", |b| {
        b.iter(|| black_box(horizon_decay(black_box(0.4), black_box(8192))))
    });
}

criterion_group!(benches, bench_observe, bench_blend, bench_horizon_decay);
criterion_main!(benches);
