//! MOP value-iteration solver — the paper's Eq. 7 fixed-point map in
//! log-space LSE form (Plan 573 / Research 478).
//!
//! The map (paper Eq. 7, arXiv:2205.10316):
//!
//! ```text
//! z_i^(n+1) = ( Σ_{k ∈ A(s_i)} exp(H̄_ik) · Π_j (z_j^(n))^p_ijk )^γ
//! H̄_ik     = (β/α) · H(S'|s_i, a_k) = −(β/α) · Σ_j p_ijk · ln p_ijk
//! ```
//!
//! In log space (ζ = ln z) the product becomes a dot product and the sum a
//! log-sum-exp — one matvec-shaped sweep per iteration:
//!
//! ```text
//! ζ_i^(n+1) = γ · LSE_k( H̄_ik + Σ_j p_ijk · ζ_j^(n) )
//! V*(s_i)   = (α/γ) · ζ_i^∞
//! ```
//!
//! Convergence is unconditional for any positive init (paper Theorem 3);
//! this solver inits ζ = 0 (z = 1). Absorbing states (single available
//! action AND that action is a deterministic self-loop) and terminal states
//! (zero available actions) are PINNED to ζ = 0 ⇒ `V = 0` exactly — the
//! paper's emergent-survival property made bit-exact.

use super::types::{MopConfig, MopConfigError, MopSolution};
use crate::cgsp::types::entropy_nats;
use crate::simd::simd_dot_f32;

/// H(S'|s,a) — conditional next-state entropy of one kernel row
/// `p(·|s,a)`.
///
/// Semantic seam over [`crate::cgsp::types::entropy_nats`] (the plan's
/// no-duplicate-entropy rule): names the MOP meaning at the consumer
/// boundary. Exact zeros contribute 0 (the substrate's epsilon floor).
#[inline]
pub fn state_conditional_entropy(p_row: &[f32]) -> f32 {
    entropy_nats(p_row)
}

/// H(A|s) — action entropy under the uniform-over-available convention
/// (`ln m_i` for `m_i` available actions).
///
/// Consumes [`entropy_nats`] by passing the availability mask as weights
/// (the substrate normalizes internally, so `[1,1,0]` → uniform over 2).
#[inline]
pub fn action_entropy_nats<const A: usize>(mask: &[u8; A]) -> f32 {
    let mut w = [0.0f32; A];
    for k in 0..A {
        w[k] = mask[k] as f32;
    }
    entropy_nats(&w)
}

/// Caller-provided scratch for [`MopSolver::solve`] — zero per-solve
/// allocation (the G4 contract). Reused across solves; fully rewritten
/// each call.
pub struct MopScratch<const N: usize, const A: usize> {
    /// Current iterate ζ = ln z.
    ln_z: [f32; N],
    /// Next iterate (double buffer).
    ln_z_next: [f32; N],
    /// Precomputed H̄[i][k] = (β/α)·H(S'|i,k) — iteration-independent.
    h_bar: [[f32; A]; N],
    /// Pin mask: 1 = absorbing/terminal, ζ held at exactly 0.
    pin: [u8; N],
}

impl<const N: usize, const A: usize> MopScratch<N, A> {
    pub fn new() -> Self {
        Self {
            ln_z: [0.0; N],
            ln_z_next: [0.0; N],
            h_bar: [[0.0; A]; N],
            pin: [0; N],
        }
    }
}

impl<const N: usize, const A: usize> Default for MopScratch<N, A> {
    fn default() -> Self {
        Self::new()
    }
}

/// The MOP value-iteration operator (paper Eq. 7, log-space LSE form).
///
/// # Example — one-state absorbing world
///
/// ```
/// use katgpt_core::mop::{MopConfig, MopScratch, MopSolver};
///
/// // A single state whose only action deterministically stays.
/// // H(A|s) = 0 and H(S'|s,stay) = 0 → V* = 0 (the paper's survival
/// // property: absorbing states have zero occupancy value, bit-exact).
/// const N: usize = 1;
/// const A: usize = 1;
/// let p = [[[1.0f32; N]; A]; N];
/// let mask = [[1u8; A]; N];
/// let solver = MopSolver::<N, A>::new(MopConfig::paper_default()).unwrap();
/// let mut scratch = MopScratch::new();
/// let sol = solver.solve(&p, &mask, &mut scratch);
/// assert_eq!(sol.v_star[0], 0.0);
/// ```
pub struct MopSolver<const N: usize, const A: usize> {
    config: MopConfig,
}

impl<const N: usize, const A: usize> MopSolver<N, A> {
    /// Construct with a validated [`MopConfig`].
    pub fn new(config: MopConfig) -> Result<Self, MopConfigError> {
        config.validate()?;
        Ok(Self { config })
    }

    /// The solver's config.
    pub fn config(&self) -> &MopConfig {
        &self.config
    }

    /// Run the Eq. 7 fixed-point iteration to convergence.
    ///
    /// - `p`: frozen transition kernel `[N][A][N]` (rows should sum to ~1
    ///   over `j`; the entropy helper normalizes defensively).
    /// - `mask`: action availability `[N][A]`, `1` = admissible.
    /// - `scratch`: caller-provided (reused across solves).
    ///
    /// Returns the fixed point with `v_star`, `ln_z`, the materialized
    /// `lse_args`, and the convergence audit. Absorbing/terminal states are
    /// pinned: `V = 0` bit-exact.
    pub fn solve(
        &self,
        p: &[[[f32; N]; A]; N],
        mask: &[[u8; A]; N],
        scratch: &mut MopScratch<N, A>,
    ) -> MopSolution<N, A> {
        let cfg = &self.config;
        let beta_over_alpha = cfg.beta / cfg.alpha;

        // ── Prepare (iteration-independent): H̄ + pin detection ──────────
        for i in 0..N {
            let mut avail_count = 0usize;
            let mut sole_action = 0usize;
            for k in 0..A {
                if mask[i][k] != 0 {
                    avail_count += 1;
                    sole_action = k;
                }
                scratch.h_bar[i][k] = beta_over_alpha * state_conditional_entropy(&p[i][k]);
            }
            // Pin rule: terminal (no actions) OR absorbing (exactly one
            // available action AND it deterministically self-loops). A state
            // with m > 1 self-loop actions is NOT pinned — repeated choice
            // among m random stays carries genuine ln m action entropy.
            let pin = if avail_count == 0 {
                true
            } else {
                avail_count == 1 && p[i][sole_action][i] >= 1.0 - 1e-6
            };
            scratch.pin[i] = pin as u8;
        }

        // ── Iterate: ζ⁺_i = γ · LSE_k( H̄_ik + Σ_j p_ijk · ζ_j ) ──────────
        scratch.ln_z = [0.0; N]; // z = 1 init (Theorem 3: any positive init)
        let mut solution = MopSolution::<N, A>::default();
        let mut sup_delta = f32::INFINITY;
        let mut iterations = 0u32;
        // Per-k LSE arguments for one state (stack, reused across the sweep).
        let mut args = [0.0f32; A];
        while iterations < cfg.max_iter {
            iterations += 1;
            sup_delta = 0.0f32;
            for i in 0..N {
                if scratch.pin[i] != 0 {
                    scratch.ln_z_next[i] = 0.0;
                    continue;
                }
                // LSE over available actions of (H̄ + Σ_j p·ζ_j) — single
                // dot-product pass (SIMD), args stashed on the stack.
                let mut max_arg = f32::NEG_INFINITY;
                for k in 0..A {
                    if mask[i][k] == 0 {
                        continue;
                    }
                    let dot = simd_dot_f32(&p[i][k], &scratch.ln_z, N);
                    let arg = scratch.h_bar[i][k] + dot;
                    args[k] = arg;
                    if arg > max_arg {
                        max_arg = arg;
                    }
                }
                // Stable LSE: max + ln Σ exp(arg − max).
                let mut sum_exp = 0.0f32;
                for k in 0..A {
                    if mask[i][k] == 0 {
                        continue;
                    }
                    sum_exp += (args[k] - max_arg).exp();
                }
                let lse = max_arg + sum_exp.ln();
                let next = cfg.gamma * lse;
                let d = (next - scratch.ln_z[i]).abs();
                if d > sup_delta {
                    sup_delta = d;
                }
                scratch.ln_z_next[i] = next;
            }
            // Swap double buffer.
            core::mem::swap(&mut scratch.ln_z, &mut scratch.ln_z_next);
            if sup_delta < cfg.tol {
                break;
            }
        }

        // ── Materialize solution at the final ζ ──────────────────────
        let alpha_over_gamma = cfg.alpha / cfg.gamma;
        for i in 0..N {
            solution.ln_z[i] = scratch.ln_z[i];
            solution.v_star[i] = alpha_over_gamma * scratch.ln_z[i];
            for k in 0..A {
                if mask[i][k] == 0 {
                    solution.lse_args[i][k] = f32::NEG_INFINITY;
                    continue;
                }
                let dot = simd_dot_f32(&p[i][k], &scratch.ln_z, N);
                solution.lse_args[i][k] = scratch.h_bar[i][k] + dot;
            }
        }
        solution.iterations = iterations;
        solution.sup_delta = sup_delta;
        solution
    }

    /// Extract the optimal policy at state `s`: a categorical distribution
    /// over actions, 0 on unavailable ones.
    ///
    /// ```text
    /// π*(a_k|s) = exp(lse_args[s][k]) / Σ_{k'} exp(lse_args[s][k'])
    /// ```
    ///
    /// The normalizer is `Z(s) = exp(LSE) = z_s^{1/γ}` at the fixed point —
    /// **not** `z_s^{-1}` (the Research 478 §2.1 pseudocode error, corrected
    /// per the riir-poc parity oracle; do not "simplify" it away).
    ///
    /// # Softmax exemption (house rule)
    ///
    /// π\* is a **categorical probability distribution** — the paper's exact
    /// Eq. 5 math requires it to sum to 1 over available actions. The house
    /// "sigmoid, never softmax" rule governs *semantic scalar projections*
    /// (emotion gates, attention boosts); this is not one. A future lint or
    /// review "fix" replacing the normalization would corrupt the math.
    ///
    /// Terminal/pinned states: returns all-zeros (the caller treats the
    /// state as having no decision).
    pub fn pi_star(
        &self,
        solution: &MopSolution<N, A>,
        s: usize,
        out: &mut [f32; A],
    ) {
        let mut max_arg = f32::NEG_INFINITY;
        for (k, o) in out.iter_mut().enumerate() {
            let a = solution.lse_args[s][k];
            *o = a;
            if a > max_arg {
                max_arg = a;
            }
        }
        if max_arg == f32::NEG_INFINITY {
            // No available action (terminal).
            for o in out.iter_mut() {
                *o = 0.0;
            }
            return;
        }
        let mut z = 0.0f32;
        for o in out.iter_mut() {
            *o = (*o - max_arg).exp();
            z += *o;
        }
        let inv = 1.0 / z;
        for o in out.iter_mut() {
            *o *= inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mop::arenas::{GRID_DEAD, GRID_N, RING_A, RING_DEAD, RING_N, four_room_gridworld, ring_world, ring_world_noisy};

    /// Structurally-different reference implementation of paper Eq. 7:
    /// direct z-space with explicit Π_j z_j^{p_ijk} (powf loop) — no
    /// log-space, no LSE. Deliberately the naive transcription (Plan 573
    /// T2.1: golden-parity via structural difference at same precision).
    fn reference_eq7<const N: usize, const A: usize>(
        p: &[[[f32; N]; A]; N],
        mask: &[[u8; A]; N],
        cfg: &MopConfig,
        init_z: f32,
    ) -> ([f32; N], u32) {
        let mut z = vec![init_z; N];
        let mut iters = 0u32;
        for _ in 0..cfg.max_iter {
            iters += 1;
            let mut z_next = vec![0.0f32; N];
            let mut sup = 0.0f32;
            for (i, ((p_i, mask_i), z_next_i)) in p
                .iter()
                .zip(mask.iter())
                .zip(z_next.iter_mut())
                .enumerate()
            {
                // Same pin rule (recomputed independently).
                let avail: Vec<usize> = (0..A).filter(|&k| mask_i[k] != 0).collect();
                let pin = avail.is_empty() || (avail.len() == 1 && p_i[avail[0]][i] >= 1.0 - 1e-6);
                if pin {
                    *z_next_i = 1.0; // ln z = 0
                    continue;
                }
                let mut sum = 0.0f32;
                for &k in &avail {
                    let p_ik = &p_i[k];
                    // H̄ = (β/α) · (−Σ p ln p)
                    let mut h = 0.0f32;
                    for &pj in p_ik.iter() {
                        if pj > 0.0 {
                            h -= pj * pj.ln();
                        }
                    }
                    let h_bar = (cfg.beta / cfg.alpha) * h;
                    // Π_j z_j^{p_ijk}
                    let mut prod = 1.0f32;
                    for (&pj, &zj) in p_ik.iter().zip(z.iter()) {
                        prod *= zj.powf(pj);
                    }
                    sum += h_bar.exp() * prod;
                }
                *z_next_i = sum.powf(cfg.gamma);
                let d = (z_next_i.ln() - z[i].ln()).abs();
                if d > sup {
                    sup = d;
                }
            }
            z = z_next;
            if sup < cfg.tol {
                break;
            }
        }
        let mut v = [0.0f32; N];
        for (v_i, &zi) in v.iter_mut().zip(z.iter()) {
            *v_i = (cfg.alpha / cfg.gamma) * zi.ln();
        }
        (v, iters)
    }

    #[test]
    fn config_validation_rejects_bad_params() {
        let base = MopConfig::paper_default();
        assert!(MopConfig { alpha: 0.0, ..base }.validate().is_err());
        assert!(MopConfig { alpha: -1.0, ..base }.validate().is_err());
        assert!(MopConfig { beta: -0.1, ..base }.validate().is_err());
        assert!(MopConfig { gamma: 0.0, ..base }.validate().is_err());
        assert!(MopConfig { gamma: 1.0, ..base }.validate().is_err());
        assert!(MopConfig { tol: 0.0, ..base }.validate().is_err());
        assert!(MopConfig { max_iter: 0, ..base }.validate().is_err());
        assert!(base.validate().is_ok());
    }

    /// T2.1 golden parity: solver (log-space LSE) vs reference (z-space
    /// powf product) on the 4-room gridworld. Plan gate: |ΔV| ≤ 1e-6 and
    /// V(absorbing) = 0 exactly.
    #[test]
    fn golden_parity_four_room_gridworld() {
        let (p, mask) = four_room_gridworld();
        let cfg = MopConfig::paper_default();
        let solver = MopSolver::<GRID_N, 4>::new(cfg).unwrap();
        let mut scratch = MopScratch::new();
        let sol = solver.solve(&p, &mask, &mut scratch);

        let (v_ref, _) = reference_eq7(&p, &mask, &cfg, 1.0);

        let mut max_rel = 0.0f32;
        for (i, (&v_sol, &v_r)) in sol.v_star.iter().zip(v_ref.iter()).enumerate() {
            let d = (v_sol - v_r).abs();
            // Relative gate (plan deviation, documented): V* reaches ~55 on
            // this arena at γ=0.95, where 1e-6 ABSOLUTE is sub-ulp (f32 eps
            // at 55 ≈ 3.3e-6) — two structurally-different f32 evaluation
            // orders cannot meet it. The achievable tight gate is relative:
            // |Δ| ≤ max(1e-6, 1e-6·|V_ref|) (≈ few ulp).
            let bound = (v_ref[i].abs() * 1e-6).max(1e-6);
            assert!(
                d <= bound,
                "state {i}: |ΔV| {d:e} > bound {bound:e} (V_ref {})",
                v_ref[i]
            );
            let rel = d / v_ref[i].abs().max(1.0);
            if rel > max_rel {
                max_rel = rel;
            }
        }
        assert!(max_rel <= 1e-6, "max relative diff {max_rel:e}");
        // Absorbing pinning: DEAD + the four trap/food cells, bit-exact.
        assert_eq!(sol.v_star[GRID_DEAD], 0.0);
        for &(r, c) in &[(2usize, 2usize), (6, 6), (2, 6), (6, 2)] {
            let s = r * 9 + c;
            assert_eq!(sol.v_star[s], 0.0, "trap/food ({r},{c}) must be V=0");
        }
        // Converged within the cap.
        assert!(sol.sup_delta < cfg.tol, "did not converge: {}", sol.sup_delta);
        // Reachable states have strictly positive value (occupancy).
        assert!(sol.v_star[10] > 0.0); // cell (1,1)
    }

    /// T2.2 invariant battery on both arenas.
    #[test]
    fn invariants_pi_and_init_independence() {
        let cfg = MopConfig::paper_default();
        // Gridworld.
        {
            let (p, mask) = four_room_gridworld();
            let solver = MopSolver::<GRID_N, 4>::new(cfg).unwrap();
            let mut scratch = MopScratch::new();
            let sol = solver.solve(&p, &mask, &mut scratch);
            // π* sums to 1 over available actions, 0 elsewhere.
            let mut pi = [0.0f32; 4];
            for s in [10usize /* (1,1) */, 30 /* (3,3) */, 70 /* (7,7) */, 31 /* door (3,4) */] {
                solver.pi_star(&sol, s, &mut pi);
                let sum: f32 = pi.iter().sum();
                assert!((sum - 1.0).abs() <= 1e-5, "s={s} π sum {sum}");
                for (&m, &p_k) in mask[s].iter().zip(pi.iter()) {
                    if m == 0 {
                        assert_eq!(p_k, 0.0);
                    }
                }
            }
            // Terminal state: all-zero π.
            solver.pi_star(&sol, GRID_DEAD, &mut pi);
            assert!(pi.iter().all(|&x| x == 0.0));
        }
        // Ring.
        {
            let (p, mask) = ring_world();
            let solver = MopSolver::<RING_N, RING_A>::new(cfg).unwrap();
            let mut scratch = MopScratch::new();
            let sol = solver.solve(&p, &mask, &mut scratch);
            assert!(sol.sup_delta < cfg.tol);
            // Analytic fixed point: deterministic ring, β term vanishes →
            // V* = α·ln 3 / (1−γ) uniformly, DEAD = 0.
            let expected = cfg.alpha * 3.0f32.ln() / (1.0 - cfg.gamma);
            for i in 0..16 {
                assert!(
                    (sol.v_star[i] - expected).abs() / expected < 1e-5,
                    "ring[{i}] = {} vs analytic {}",
                    sol.v_star[i],
                    expected
                );
            }
            assert_eq!(sol.v_star[RING_DEAD], 0.0);
        }
    }

    /// T2.2: Theorem-3 init-invariance — ones vs twos init converge to the
    /// same fixed point. The solver always inits ζ=0; exercise the theorem
    /// through the reference (init 1.0 vs 2.0) AND verify the solver's
    /// ζ=0 result matches both (any positive init converges to the same
    /// unique fixed point).
    #[test]
    fn theorem3_init_invariance() {
        let (p, mask) = ring_world_noisy(0.2);
        let cfg = MopConfig::paper_default();
        let (v1, _) = reference_eq7::<RING_N, RING_A>(&p, &mask, &cfg, 1.0);
        let (v2, _) = reference_eq7::<RING_N, RING_A>(&p, &mask, &cfg, 2.0);
        let solver = MopSolver::<RING_N, RING_A>::new(cfg).unwrap();
        let mut scratch = MopScratch::new();
        let sol = solver.solve(&p, &mask, &mut scratch);
        for i in 0..RING_N {
            assert!((v1[i] - v2[i]).abs() <= 1e-5, "init-dependence at {i}");
            assert!((sol.v_star[i] - v1[i]).abs() <= 1e-5, "solver vs ref at {i}");
        }
    }

    /// T2.3 edge cases: noisy ring exercises H(S'|s,a) > 0 (β term live);
    /// γ→1 stability (tolerance-scaled); deterministic kernel (β vanishes).
    #[test]
    fn edge_cases_noisy_ring_and_high_gamma() {
        // Noisy ring: β>0 must change the values vs β=0 (the risk knob
        // bites) and still converge + satisfy π invariants.
        let (p, mask) = ring_world_noisy(0.3);
        let cfg = MopConfig::paper_default();
        let solver = MopSolver::<RING_N, RING_A>::new(cfg).unwrap();
        let mut scratch = MopScratch::new();
        let sol_beta = solver.solve(&p, &mask, &mut scratch);
        assert!(sol_beta.sup_delta < cfg.tol);
        let cfg_b0 = MopConfig { beta: 0.0, ..cfg };
        let solver_b0 = MopSolver::<RING_N, RING_A>::new(cfg_b0).unwrap();
        let sol_b0 = solver_b0.solve(&p, &mask, &mut scratch);
        let mut diff = 0.0f32;
        for i in 0..16 {
            diff = diff.max((sol_beta.v_star[i] - sol_b0.v_star[i]).abs());
        }
        assert!(diff > 1e-3, "β must move the values on a noisy kernel");
        // π invariants hold on the stochastic kernel.
        let mut pi = [0.0f32; RING_A];
        solver.pi_star(&sol_beta, 5, &mut pi);
        let sum: f32 = pi.iter().sum();
        assert!((sum - 1.0).abs() <= 1e-5);

        // γ → 1: analytic value scales as 1/(1−γ); assert relative accuracy.
        let (pd, md) = ring_world();
        let cfg_hg = MopConfig { gamma: 0.99, tol: 1e-10, max_iter: 100_000, ..cfg };
        let solver_hg = MopSolver::<RING_N, RING_A>::new(cfg_hg).unwrap();
        let sol_hg = solver_hg.solve(&pd, &md, &mut scratch);
        let expected = cfg_hg.alpha * 3.0f32.ln() / (1.0 - cfg_hg.gamma);
        for i in 0..16 {
            let rel = (sol_hg.v_star[i] - expected).abs() / expected;
            assert!(rel < 1e-4, "γ=0.99 ring[{i}] rel err {rel:e}");
        }
    }

    /// T2.3: all-unavailable state (gridworld wall) returns terminal —
    /// V = 0, π all-zero.
    #[test]
    fn all_unavailable_state_is_terminal() {
        let (p, mask) = four_room_gridworld();
        let cfg = MopConfig::paper_default();
        let solver = MopSolver::<GRID_N, 4>::new(cfg).unwrap();
        let mut scratch = MopScratch::new();
        let sol = solver.solve(&p, &mask, &mut scratch);
        // Center cross cell (4,4) is a wall.
        let wall = 4 * 9 + 4;
        assert_eq!(mask[wall], [0, 0, 0, 0]);
        assert_eq!(sol.v_star[wall], 0.0);
        let mut pi = [0.0f32; 4];
        solver.pi_star(&sol, wall, &mut pi);
        assert!(pi.iter().all(|&x| x == 0.0));
    }

    /// Single-action state (DEAD) → H(A|s) = 0; deterministic kernel →
    /// H(S'|s,a) = 0 (the helpers' contracts).
    #[test]
    fn entropy_helpers_contract() {
        assert_eq!(action_entropy_nats(&[0u8; 3]), 0.0);
        assert!((action_entropy_nats(&[1, 1, 0]) - 2.0f32.ln()).abs() < 1e-6);
        assert_eq!(state_conditional_entropy(&[1.0, 0.0, 0.0]), 0.0);
        assert!((state_conditional_entropy(&[0.5, 0.5]) - 2.0f32.ln()).abs() < 1e-6);
    }
}
