//! Plan 455 Phase 3 — Head-to-Head: QB Router (Plan 455) vs MPI Router
//! (Plan 279) on a deliberately-hard synthetic MoE fixture.
//!
//! Run:
//! ```bash
//! cargo test --features "quantile_balance_router manifold_power_iter_router" \
//!            -p katgpt-spectral \
//!            --test bench_455_phase3_head_to_head -- --nocapture
//! ```
//!
//! Research 447 §2.4 predicts these two primitives solve **orthogonal**
//! problems on the joint `(λ, MaxVio)` Pareto frontier:
//!   - **MPI** retracts router rows toward expert Gram principal directions →
//!     improves alignment λ. Does NOT touch the per-token score distribution.
//!   - **QB** computes a per-expert bias β from score quantiles → drives
//!     load-balance MaxVio → 0. Does NOT touch the router weight matrix.
//!
//! The predicted outcome is **Case C — composition strictly beats either
//! alone**. This test constructs a fixture where BOTH axes are broken (low λ
//! AND high MaxVio), runs all four variants (vanilla / MPI-only / QB-only /
//! composed), fills the decision matrix, and asserts Case C.
//!
//! ## Fixture design
//!
//! - `N=8` experts, `D=256`, `M=256` tokens, `k=2`.
//! - Expert `i`'s Gram: `M[i] = e_i·e_i^T + 0.1·I` (principal direction =
//!   standard basis vector `e_i`).
//! - Router row `i`: `R[i] = cos(θ)·e_i + sin(θ)·e_{i+N}` with `θ=1.0` rad.
//!   This is deliberately misaligned with `e_i` → vanilla λ ≈ 0.65 (low).
//!   MPI retraction moves `R[i]` toward `e_i` → λ ≈ 0.99 (high).
//! - Input batch: `X[j] = strength·(e_0 + e_1)/√2 + Gaussian noise`.
//!   The hot direction `(e_0 + e_1)/√2` systematically favors experts 0,1
//!   → vanilla MaxVio_load ≈ 3.0 (both experts 0,1 picked every token).
//!   MPI doesn't fix this (retraction toward `e_i` preserves the bias: `e_0`
//!   and `e_1` still align with the hot direction). QB is needed.
//!
//! ## Decision matrix (predicted)
//!
//! | Variant       | λ     | MaxVio_load | Verdict                       |
//! |---------------|-------|-------------|-------------------------------|
//! | Vanilla       | ~0.65 | ~3.0        | both axes broken              |
//! | MPI only      | ~0.99 | ~3.0        | fixes λ, MaxVio still high    |
//! | QB only       | ~0.65 | ~0.06       | fixes MaxVio, λ unchanged     |
//! | Composed      | ~0.99 | ~0.06       | fixes both → Case C           |

#![cfg(all(
    feature = "quantile_balance_router",
    feature = "manifold_power_iter_router"
))]

use katgpt_spectral::manifold_power_iter_router::{
    compute_diagnostics, manifold_power_iter_router,
};
use katgpt_spectral::quantile_balance_router::{
    QbConfig, QbScratch, compute_balance_violation, quantile_balance_router,
};
use katgpt_spectral::spectral_retract::PowerRetractScratch;

// ── Fixture parameters ───────────────────────────────────────────────────

/// Model dimension D. Matches the Plan 279 / Plan 455 game-scale point.
const D: usize = 256;
/// Number of experts N. Game-scale MoE pool.
const N: usize = 8;
/// Calibration tokens M. Matches Plan 455 G4 fixture.
const M: usize = 256;
/// Top-k experts per token.
const K: usize = 2;

/// Misalignment angle θ (radians). `R[i] = cos(θ)·e_i + sin(θ)·e_{i+N}`.
/// θ=1.0 rad (≈57°) gives vanilla λ ≈ 0.65 — low enough for MPI to improve.
const THETA: f32 = 1.0;

/// Hot-direction signal strength in the input batch.
/// `X[j] = strength·(e_0+e_1)/√2 + noise`. Creates MaxVio_load ≈ 3.0.
const HOT_STRENGTH: f32 = 2.0;

// ── Deterministic RNG ────────────────────────────────────────────────────

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9E3779B97F4A7C15 } else { seed },
        }
    }
    /// Uniform (0, 1) via xorshift64.
    fn u01(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        ((self.state >> 11) as f32) / ((1u64 << 53) as f32)
    }
    /// Standard normal via Box-Muller transform.
    fn gauss(&mut self) -> f32 {
        let u1 = self.u01().max(1e-10);
        let u2 = self.u01();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        r * theta.cos()
    }
}

// ── Fixture construction ─────────────────────────────────────────────────

/// Build expert Gram matrices. `M[i] = e_i·e_i^T + 0.1·I` (D×D, diagonal).
///
/// The principal eigenvector of `M[i]` is `e_i` with eigenvalue 1.1; all
/// other directions have eigenvalue 0.1. MPI retracts `R[i]` toward `e_i`.
fn build_grams(n: usize, d: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| {
            let mut m = vec![0.0f32; d * d];
            for j in 0..d {
                m[j * d + j] = if j == i { 1.1 } else { 0.1 };
            }
            m
        })
        .collect()
}

/// Build the router matrix `R` (N×D row-major).
///
/// `R[i] = cos(θ)·e_i + sin(θ)·e_{i+N}` — deliberately misaligned with the
/// Gram principal direction `e_i` by angle θ. Vanilla λ ≈ cos(θ)-ish metric.
fn build_router(n: usize, d: usize, theta: f32) -> Vec<f32> {
    // Need d >= 2*n so that e_{i+N} indices are valid (positions N..2N).
    assert!(
        d >= 2 * n,
        "D must be >= 2*N for the fixture, got D={d}, N={n}"
    );
    let mut r = vec![0.0f32; n * d];
    let ct = theta.cos();
    let st = theta.sin();
    for i in 0..n {
        r[i * d + i] = ct; // cos(θ) along e_i
        r[i * d + (i + n)] = st; // sin(θ) along e_{i+N}
    }
    r
}

/// Build the input batch `X` (M×D row-major).
///
/// `X[j] = HOT_STRENGTH·(e_0 + e_1)/√2 + Gaussian(0, 0.5²)` noise per dim.
/// The hot direction systematically favors experts 0,1 (whose router rows
/// have a component along `e_0`/`e_1`). Creates MaxVio_load ≈ 3.0 under
/// top-2 routing.
fn build_input_batch(seed: u64, m: usize, d: usize) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let mut x = vec![0.0f32; m * d];
    let hot_component = HOT_STRENGTH / 2.0f32.sqrt();
    for j in 0..m {
        // Hot direction (e_0 + e_1)/√2 + noise.
        x[j * d] = hot_component + 0.5 * rng.gauss();
        x[j * d + 1] = hot_component + 0.5 * rng.gauss();
        // Remaining dimensions: pure noise.
        for k in 2..d {
            x[j * d + k] = 0.5 * rng.gauss();
        }
    }
    x
}

/// Compute scores `s = X · R^T` (M×N row-major).
///
/// `s[j, i] = Σ_d X[j, d] · R[i, d]`. This is the router score matrix that
/// QB operates on (and that drives top-k selection for MaxVio_load).
fn compute_scores(x: &[f32], r: &[f32], m: usize, n: usize, d: usize) -> Vec<f32> {
    let mut s = vec![0.0f32; m * n];
    for j in 0..m {
        let x_row = &x[j * d..(j + 1) * d];
        for i in 0..n {
            let r_row = &r[i * d..(i + 1) * d];
            let mut dot = 0.0f32;
            for k in 0..d {
                dot += x_row[k] * r_row[k];
            }
            s[j * n + i] = dot;
        }
    }
    s
}

// ── Head-to-head test ────────────────────────────────────────────────────

#[test]
fn head_to_head_decision_matrix() {
    // ── Build fixture ──────────────────────────────────────────────────
    let grams = build_grams(N, D);
    let gram_refs: Vec<&[f32]> = grams.iter().map(|g| g.as_slice()).collect();
    let r_original = build_router(N, D, THETA);
    let x = build_input_batch(42, M, D);

    // MPI config (paper §1.4 defaults: c_prime=1.0, iters=1).
    let c_prime = 1.0f32;
    let target_norm = c_prime / (N as f32).sqrt();

    // QB config (Su blog defaults).
    let cfg = QbConfig::default();

    // ── Variant 1: Vanilla (no conditioning) ───────────────────────────
    let s_vanilla = compute_scores(&x, &r_original, M, N, D);
    let (lambda_v, _) = compute_diagnostics(&r_original, &gram_refs, N, D, target_norm);
    let zeros = vec![0.0f32; N];
    let maxvio_v = compute_balance_violation(&s_vanilla, M, N, K, &zeros);

    // ── Variant 2: MPI only ────────────────────────────────────────────
    let mut r_mpi = r_original.clone();
    let mut mpi_scratch = PowerRetractScratch::new(D);
    let _mpi_res = manifold_power_iter_router(
        &mut r_mpi,
        &gram_refs,
        N,
        D,
        c_prime,
        1, // paper default iters=1
        &mut mpi_scratch,
    );
    let s_mpi = compute_scores(&x, &r_mpi, M, N, D);
    let (lambda_mpi, _) = compute_diagnostics(&r_mpi, &gram_refs, N, D, target_norm);
    let maxvio_mpi = compute_balance_violation(&s_mpi, M, N, K, &zeros);

    // ── Variant 3: QB only (β computed on vanilla scores) ──────────────
    let mut qb_scratch = QbScratch::new(M, N);
    let qb_res = quantile_balance_router(&s_vanilla, M, N, K, &cfg, &mut qb_scratch);
    let lambda_qb = lambda_v; // QB doesn't change R, so λ is unchanged.
    let maxvio_qb = compute_balance_violation(&s_vanilla, M, N, K, &qb_res.beta);

    // ── Variant 4: Composed (MPI → recompute scores → QB on new scores) ─
    let mut r_comp = r_original.clone();
    let mut mpi_scratch2 = PowerRetractScratch::new(D);
    let _ =
        manifold_power_iter_router(&mut r_comp, &gram_refs, N, D, c_prime, 1, &mut mpi_scratch2);
    let s_comp = compute_scores(&x, &r_comp, M, N, D);
    let mut qb_scratch2 = QbScratch::new(M, N);
    let qb_comp_res = quantile_balance_router(&s_comp, M, N, K, &cfg, &mut qb_scratch2);
    let (lambda_comp, _) = compute_diagnostics(&r_comp, &gram_refs, N, D, target_norm);
    let maxvio_comp = compute_balance_violation(&s_comp, M, N, K, &qb_comp_res.beta);

    // ── Print decision matrix ──────────────────────────────────────────
    eprintln!();
    eprintln!("═══════════════════════════════════════════════════════════════════════");
    eprintln!("  Plan 455 Phase 3 — Head-to-Head Decision Matrix");
    eprintln!("  Fixture: N={N}, D={D}, M={M}, k={K}, θ={THETA:.3} rad");
    eprintln!("═══════════════════════════════════════════════════════════════════════");
    eprintln!(
        "  {:<22} {:>10}  {:>12}  {:>10}",
        "Variant", "λ", "MaxVio_load", "β‖∞"
    );
    eprintln!("  {}", "-".repeat(58));
    eprintln!(
        "  {:<22} {:>10.4}  {:>12.4}  {:>10.4}",
        "Vanilla", lambda_v, maxvio_v, 0.0
    );
    eprintln!(
        "  {:<22} {:>10.4}  {:>12.4}  {:>10.4}",
        "MPI only", lambda_mpi, maxvio_mpi, 0.0
    );
    eprintln!(
        "  {:<22} {:>10.4}  {:>12.4}  {:>10.4}",
        "QB only",
        lambda_qb,
        maxvio_qb,
        qb_res.beta.iter().cloned().fold(0.0f32, f32::max).abs()
    );
    eprintln!(
        "  {:<22} {:>10.4}  {:>12.4}  {:>10.4}",
        "Composed (MPI+QB)",
        lambda_comp,
        maxvio_comp,
        qb_comp_res
            .beta
            .iter()
            .cloned()
            .fold(0.0f32, f32::max)
            .abs()
    );
    eprintln!("═══════════════════════════════════════════════════════════════════════");
    eprintln!();

    // ── Precondition: the fixture has BOTH problems ────────────────────
    // If either axis is already good, the head-to-head is meaningless.
    assert!(
        lambda_v < 0.85,
        "PRECONDITION FAIL: vanilla λ = {lambda_v:.4} must be < 0.85 (low alignment for MPI to fix). \
         Fixture is broken — θ may be too small."
    );
    assert!(
        maxvio_v > 1.0,
        "PRECONDITION FAIL: vanilla MaxVio_load = {maxvio_v:.4} must be > 1.0 (high imbalance for QB to \
         fix). Fixture is broken — hot-direction signal too weak."
    );

    // ── Case determination ─────────────────────────────────────────────
    // Case C (predicted): composition strictly beats either alone because
    // MPI and QB fix orthogonal axes.
    let mpi_improves_lambda = lambda_mpi > lambda_v + 0.1;
    let qb_improves_maxvio = maxvio_qb < maxvio_v * 0.5;
    // Composed uses the same MPI step → λ_comp should equal λ_mpi exactly.
    let comp_preserves_lambda = (lambda_comp - lambda_mpi).abs() < 1e-4;
    // QB on the MPI-conditioned scores should still reduce MaxVio.
    let comp_reduces_maxvio_vs_mpi = maxvio_comp < maxvio_mpi * 0.5;
    // Composed beats QB-only on λ (because MPI improved it; QB didn't).
    let comp_beats_qb_on_lambda = lambda_comp > lambda_qb + 0.1;

    eprintln!("  Case determination:");
    eprintln!(
        "    G-P3-1  MPI improves λ:        {}  (λ {:.4} → {:.4}, Δ=+{:.4})",
        mpi_improves_lambda,
        lambda_v,
        lambda_mpi,
        lambda_mpi - lambda_v
    );
    eprintln!(
        "    G-P3-2  QB improves MaxVio:    {}  (MaxVio {:.4} → {:.4}, {:.1}× reduction)",
        qb_improves_maxvio,
        maxvio_v,
        maxvio_qb,
        if maxvio_qb > 1e-9 {
            maxvio_v / maxvio_qb
        } else {
            f32::INFINITY
        }
    );
    eprintln!(
        "    G-P3-3  Comp ≈ MPI on λ:       {}  (λ_comp {:.4} vs λ_mpi {:.4}, |Δ|={:.2e})",
        comp_preserves_lambda,
        lambda_comp,
        lambda_mpi,
        (lambda_comp - lambda_mpi).abs()
    );
    eprintln!(
        "    G-P3-4  Comp reduces MaxVio:   {comp_reduces_maxvio_vs_mpi}  (MaxVio_comp {maxvio_comp:.4} vs MaxVio_mpi {maxvio_mpi:.4})"
    );
    eprintln!(
        "    G-P3-5  Comp > QB on λ:        {}  (λ_comp {:.4} vs λ_qb {:.4}, Δ=+{:.4})",
        comp_beats_qb_on_lambda,
        lambda_comp,
        lambda_qb,
        lambda_comp - lambda_qb
    );
    eprintln!();

    let is_case_c = mpi_improves_lambda
        && qb_improves_maxvio
        && comp_preserves_lambda
        && comp_reduces_maxvio_vs_mpi
        && comp_beats_qb_on_lambda;

    if is_case_c {
        eprintln!("  ✓ CASE C CONFIRMED: composition strictly beats either alone.");
        eprintln!("    → Promote BOTH quantile_balance_router and manifold_power_iter_router");
        eprintln!(
            "      to DEFAULT-ON. They solve orthogonal problems on the (λ, MaxVio) Pareto frontier:"
        );
        eprintln!(
            "        MPI  fixes alignment λ  (router rows → expert Gram principal direction)."
        );
        eprintln!("        QB   fixes balance MaxVio (per-expert bias β → balanced top-k).");
        eprintln!();
    } else {
        eprintln!("  ✗ CASE C NOT CONFIRMED — inspect metrics above for Case A/B/D.");
        eprintln!();
    }

    // ── Assert Case C (the predicted outcome) ──────────────────────────
    // If any of these fail, the outcome is NOT Case C and the promotion
    // decision must be revisited per Plan 455 T3.6.
    assert!(
        mpi_improves_lambda,
        "G-P3-1 FAIL: MPI did not improve λ by ≥ 0.1 (got {:.4} → {:.4}, Δ=+{:.4}). \
         Either θ is too small or MPI is broken on this fixture.",
        lambda_v,
        lambda_mpi,
        lambda_mpi - lambda_v
    );
    assert!(
        qb_improves_maxvio,
        "G-P3-2 FAIL: QB did not halve MaxVio (got {:.4} → {:.4}, ratio {:.3}). \
         Either hot-direction signal is too weak or QB is broken on this fixture.",
        maxvio_v,
        maxvio_qb,
        maxvio_qb / maxvio_v.max(1e-9)
    );
    assert!(
        comp_preserves_lambda,
        "G-P3-3 FAIL: Composition lost MPI's λ gain (λ_comp {:.6} vs λ_mpi {:.6}, |Δ|={:.2e}). \
         MPI determinism violated — both composed and MPI-only should use identical R'.",
        lambda_comp,
        lambda_mpi,
        (lambda_comp - lambda_mpi).abs()
    );
    assert!(
        comp_reduces_maxvio_vs_mpi,
        "G-P3-4 FAIL: QB on MPI-conditioned scores did not halve MaxVio (got {maxvio_mpi:.4} → {maxvio_comp:.4})."
    );
    assert!(
        comp_beats_qb_on_lambda,
        "G-P3-5 FAIL: Composition doesn't beat QB-only on λ by ≥ 0.1 \
         (λ_comp {lambda_comp:.4} vs λ_qb {lambda_qb:.4})."
    );

    // ── Strict Pareto dominance (composed > either alone) ──────────────
    // Composed must be no-worse on both axes and strictly better on at least
    // one, compared to EACH alternative.
    let comp_dominates_mpi = lambda_comp >= lambda_mpi - 1e-6 && maxvio_comp < maxvio_mpi - 1e-6;
    let comp_dominates_qb = maxvio_comp <= maxvio_qb + 0.1 && lambda_comp > lambda_qb + 0.1;
    assert!(
        comp_dominates_mpi && comp_dominates_qb,
        "G-P3-6 FAIL: Composition is not strictly Pareto-better than both alternatives.\n\
         dominates MPI-only: {} (λ_comp {:.4} ≥ λ_mpi {:.4}={}, MaxVio_comp {:.4} < MaxVio_mpi {:.4}={})\n\
         dominates QB-only:  {} (MaxVio_comp {:.4} ≤ MaxVio_qb {:.4}+0.1={}, λ_comp {:.4} > λ_qb {:.4}={})",
        comp_dominates_mpi,
        lambda_comp,
        lambda_mpi,
        lambda_comp >= lambda_mpi - 1e-6,
        maxvio_comp,
        maxvio_mpi,
        maxvio_comp < maxvio_mpi - 1e-6,
        comp_dominates_qb,
        maxvio_comp,
        maxvio_qb,
        maxvio_comp <= maxvio_qb + 0.1,
        lambda_comp,
        lambda_qb,
        lambda_comp > lambda_qb + 0.1,
    );

    eprintln!("  G-P3-6 PASS: Composition strictly Pareto-dominates both alternatives.\n");
    eprintln!("  ═══ Phase 3 verdict: CASE C — promote BOTH to DEFAULT-ON ═══\n");
}
