//! Quantile Balancing MoE Router — N/M/k sweep benchmark (Plan 455 Phase 2).
//!
//! Uses `std::time::Instant` (NOT criterion — matches Plan 279
//! `manifold_power_iter_router_bench.rs` style).
//!
//! Run:
//! ```bash
//! cargo run --release --bench quantile_balance_router_bench \
//!            --features quantile_balance_router -p katgpt-spectral
//! ```
//!
//! Sweeps `N ∈ {8, 32, 64, 256}` experts × `M ∈ {64, 256, 1024}` calibration
//! tokens × `k ∈ {1, 2, 4}` and measures:
//! - Total `β` compute time (per-snapshot cost; should be sub-ms at game scale).
//! - `route_with_bias` per-token cost (should match vanilla top-k).
//! - `MaxVio(s) → MaxVio(s − β)` for each `(N, M, k)`.

#![cfg(feature = "quantile_balance_router")]

use katgpt_spectral::quantile_balance_router::{
    QbConfig, QbScratch, compute_balance_violation, quantile_balance_router, route_with_bias,
};
use std::time::{Duration, Instant};

// ── Helpers ──────────────────────────────────────────────────────────────

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9E3779B97F4A7C15 } else { seed },
        }
    }
    /// Uniform in `[lo, hi)`.
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        let u01 = ((self.state >> 11) as f32) / ((1u64 << 53) as f32);
        lo + (hi - lo) * u01
    }
}

/// Build a deliberately-skewed `m × n` router-score matrix (expert 0 hot,
/// expert n-1 starved) so vanilla top-k produces large MaxVio.
fn build_skewed_scores(seed: u64, m: usize, n: usize) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let mut s = Vec::with_capacity(m * n);
    let denom = n.saturating_sub(1).max(1) as f32;
    for _ in 0..m {
        for j in 0..n {
            let offset = 2.0 - (j as f32) * (4.0 / denom);
            let jitter = rng.uniform(-0.3, 0.3);
            s.push(offset + jitter);
        }
    }
    s
}

/// Best-of-N wall-clock microseconds for a closure.
fn bench_us(warmup: usize, iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let mut best = Duration::from_secs(60);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        let dt = t0.elapsed();
        if dt < best {
            best = dt;
        }
    }
    best.as_secs_f64() * 1e6
}

// ── Main sweep ───────────────────────────────────────────────────────────

fn main() {
    println!("=== Quantile Balancing MoE Router Benchmark (Plan 455) ===\n");

    let n_values: &[usize] = &[8, 32, 64, 256];
    let m_values: &[usize] = &[64, 256, 1024];
    let k_values: &[usize] = &[1, 2, 4];

    println!(
        "{:>5} {:>5} {:>3} {:>12} {:>12} {:>12} {:>12}",
        "N", "M", "k", "qb_us", "route_us", "MaxVio_pre", "MaxVio_post"
    );
    println!("{}", "-".repeat(80));

    let cfg = QbConfig::default();

    for &n in n_values {
        for &m in m_values {
            for &k in k_values {
                if k > n {
                    continue;
                }
                let s = build_skewed_scores(1000 + (n * m * k) as u64, m, n);

                // Baseline MaxVio (no bias).
                let beta_zero = vec![0.0f32; n];
                let maxvio_pre =
                    compute_balance_violation(&s, m, n, k, &beta_zero);

                // QB compute (best-of-N timing — this is the per-snapshot cost).
                let mut scratch = QbScratch::new(m, n);
                let qb_us = bench_us(3, 20, || {
                    let _res =
                        quantile_balance_router(&s, m, n, k, &cfg, &mut scratch);
                    std::hint::black_box(&_res);
                });

                // Capture one final β for MaxVio-after + per-token route timing.
                let res = quantile_balance_router(&s, m, n, k, &cfg, &mut scratch);
                let maxvio_post = res.final_balance_violation;

                // Per-token route_with_bias cost (should be ~flat across M,
                // linear in N — vanilla top-k shape).
                let s_row = &s[0..n];
                let mut out_scores = vec![0.0f32; n];
                let route_us = bench_us(5, 500, || {
                    let _topk =
                        route_with_bias(s_row, &res.beta, k, &mut out_scores);
                    std::hint::black_box(&_topk);
                });

                println!(
                    "{:>5} {:>5} {:>3} {:>9.2}us {:>9.2}us {:>12.4} {:>12.4}",
                    n, m, k, qb_us, route_us, maxvio_pre, maxvio_post
                );
            }
        }
        println!();
    }

    // ── Game-scale focus: N=8, M=256, k=2 — must be sub-ms (G4) ────────────
    println!("G4 game-scale focus (N=8, M=256, k=2):");
    let (n, m, k) = (8, 256, 2);
    let s = build_skewed_scores(42, m, n);
    let mut scratch = QbScratch::new(m, n);
    // Warmup once.
    let _ = quantile_balance_router(&s, m, n, k, &cfg, &mut scratch);
    let t0 = Instant::now();
    let res = quantile_balance_router(&s, m, n, k, &cfg, &mut scratch);
    let dt = t0.elapsed();
    println!("  β compute time = {:?}", dt);
    println!("  converged_iter = {}", res.converged_iter);
    println!(
        "  MaxVio pre  (no bias)  = {:.4}",
        compute_balance_violation(&s, m, n, k, &vec![0.0f32; n])
    );
    println!("  MaxVio post (with β)   = {:.4}", res.final_balance_violation);
    println!(
        "  G4 (sub-ms): {}",
        if dt.as_secs_f64() < 1e-3 {
            "PASS"
        } else {
            "FAIL"
        }
    );
}
