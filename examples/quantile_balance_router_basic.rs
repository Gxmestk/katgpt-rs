//! Quantile Balancing MoE Router — before/after demo (Plan 455).
//!
//! Run:
//! ```bash
//! cargo run --example quantile_balance_router_basic \
//!            --features quantile_balance_router --release
//! ```
//!
//! Demonstrates QB's MaxVio gains (Su blog Feb 2026 + Marin 32B-A5B
//! validation) on a synthetic MoE pool: N=8 experts, M=64 calibration tokens.
//! Shows:
//! - MaxVio before → after (paper target: high → ≈0 shape)
//! - expert selection counts before → after (skewed → balanced)
//! - timing (target: sub-ms for game-scale pool)
//! - `route_with_bias` on a sample token
//!
//! # Honest output interpretation
//!
//! Phase 1 debug showed that on small integer-count-constrained batches,
//! QB does NOT drive MaxVio to exactly 0 — the theoretical LP optimum is
//! unachievable when `m·k/n` is not an integer or when near-ties prevent
//! perfect separation. The demo prints the honest floor (typically MaxVio
//! ≈ 0.1-0.3 on small batches, lower on larger ones).

#![cfg(feature = "quantile_balance_router")]

use katgpt_spectral::quantile_balance_router::{
    QbConfig, QbScratch, compute_balance_violation, quantile_balance_router, route_with_bias,
};
use std::time::Instant;

/// Deterministic xorshift32 PRNG (no dep on rand crate).
fn seeded_scores(seed: u32, m: usize, n: usize) -> Vec<f32> {
    let mut s = seed;
    let mut v = Vec::with_capacity(m * n);
    for _ in 0..(m * n) {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        v.push((s as f32) / (u32::MAX as f32));
    }
    v
}

/// Expert selection counts under top-k(s − β). Length-n vector.
fn expert_counts(s: &[f32], m: usize, n: usize, k: usize, beta: &[f32]) -> Vec<usize> {
    let mut counts = vec![0usize; n];
    let mut out = vec![0.0f32; n];
    for i in 0..m {
        let row = &s[i * n..(i + 1) * n];
        let selected = route_with_bias(row, beta, k, &mut out);
        for &idx in &selected {
            counts[idx] += 1;
        }
    }
    counts
}

fn main() {
    // Game-scale config: 8 experts (typical NPC LoRA pool), 64 calibration
    // tokens, k=2 (each token picks 2 experts).
    let n = 8usize;
    let m = 64usize;
    let k = 2usize;
    let cfg = QbConfig::default(); // iters=5, tol=1e-6

    println!("=== Quantile Balancing MoE Router (Plan 455) ===");
    println!(
        "config: N={} experts, M={} calibration tokens, k={}, iters={}, tol={:.1e}",
        n, m, k, cfg.iters, cfg.tol
    );
    println!();
    println!("Source: Su blog Feb 2026 + Marin 32B-A5B / 1e22 FLOPs validation");
    println!("Sibling: Plan 279 Manifold Power Iteration Router (rows vs bias)");
    println!();

    // Construct a deliberately-skewed calibration batch: expert 0 has +0.5
    // score bonus on every token, expert 7 has -0.5 penalty. This creates
    // severe load imbalance under vanilla top-k.
    let mut s = seeded_scores(12345, m, n);
    // Inject skew: boost expert 0, penalize expert 7.
    for i in 0..m {
        s[i * n] += 0.5; // expert 0 always favored
        s[i * n + 7] -= 0.5; // expert 7 always disfavored
    }

    // Baseline (no bias): show the skew.
    let beta_zero = vec![0.0f32; n];
    let counts_before = expert_counts(&s, m, n, k, &beta_zero);
    let maxvio_before = compute_balance_violation(&s, m, n, k, &beta_zero);
    let ideal = (m as f32) * (k as f32) / (n as f32);

    println!("── Baseline (no bias) ──────────────────────────────────────");
    println!(
        "expert selection counts: {:?}",
        counts_before
    );
    println!(
        "ideal count per expert:  {:.1} (m·k/n = {}·{}/{})",
        ideal, m, k, n
    );
    println!("MaxVio (max relative deviation): {:.4}", maxvio_before);
    println!();

    // Run QB.
    let t_qb = Instant::now();
    let mut scratch = QbScratch::new(m, n);
    let result = quantile_balance_router(&s, m, n, k, &cfg, &mut scratch);
    let elapsed_us = t_qb.elapsed().as_secs_f64() * 1e6;

    let counts_after = expert_counts(&s, m, n, k, &result.beta);
    let maxvio_after = result.final_balance_violation;

    println!("── After Quantile Balancing ────────────────────────────────");
    println!("β (per-expert bias):     {:?}", result.beta);
    println!(
        "α (per-token, first 8):  {:?}…",
        &result.alpha[..8.min(result.alpha.len())]
    );
    println!(
        "expert selection counts: {:?}",
        counts_after
    );
    println!(
        "MaxVio (max relative deviation): {:.4}  (was {:.4}, reduction {:.1}×)",
        maxvio_after,
        maxvio_before,
        if maxvio_after > 0.0 {
            maxvio_before / maxvio_after
        } else {
            f32::INFINITY
        }
    );
    println!("converged iterations:    {}", result.converged_iter);
    println!(
        "β compute time:          {:.1} µs  (target: < 1000 µs for game scale)",
        elapsed_us
    );
    println!();

    // Route a sample token with the computed bias.
    println!("── Sample token routing ────────────────────────────────────");
    let sample_row = &s[0..n]; // first calibration token
    let mut biased_scores = vec![0.0f32; n];
    let selected = route_with_bias(sample_row, &result.beta, k, &mut biased_scores);
    println!("raw scores:    {:?}", sample_row);
    println!("biased scores: {:?}", biased_scores);
    println!(
        "selected top-{} experts (by biased score): {:?}",
        k, selected
    );
    println!();

    // Honest summary.
    println!("── Verdict ────────────────────────────────────────────────");
    let reduction = if maxvio_after > 0.0 {
        maxvio_before / maxvio_after
    } else {
        f32::INFINITY
    };
    if reduction >= 2.0 {
        println!(
            "✓ QB reduced MaxVio by {:.1}× (≥2× gate PASS)",
            reduction
        );
    } else {
        println!(
            "△ QB reduced MaxVio by {:.1}× (<2× — check calibration batch size)",
            reduction
        );
    }
    if elapsed_us < 1000.0 {
        println!(
            "✓ β compute time {:.1} µs < 1ms (G4 sub-ms gate PASS)",
            elapsed_us
        );
    } else {
        println!(
            "✗ β compute time {:.1} µs ≥ 1ms (G4 sub-ms gate FAIL)",
            elapsed_us
        );
    }
    println!();
    println!("Note: on small batches (m·k/n near integer constraints), MaxVio");
    println!("may not reach exactly 0 — the LP optimum is bounded by the");
    println!("score structure. Larger calibration batches achieve lower floors.");
}
