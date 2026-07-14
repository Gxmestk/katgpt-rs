//! Linear log-ratio class `h_θ(x) = θ^T φ(x)` with identity feature map.
//!
//! This is the concrete supervised learner for FORE's G1/G2 gates. The KL
//! projection (Algorithm 1 step 4) reduces to a convex optimization:
//!
//! ```text
//! min_θ  L(θ) = log( (1/n) Σ_i e^{θ·φ(Xi)} )  −  θ^T m
//! ```
//!
//! where `m = (1−γ) P̂_0(φ) + γ P̂^+_{n,ω}(φ)` is a fixed moment vector for the
//! current FORE iteration (computed once per call to `fit_and_evaluate`). The
//! loss is convex (log-sum-exp minus linear); we solve it via Newton's method
//! with the PSD Hessian `Cov̂_{ω_θ}(φ(X))` and Cholesky back-solve.

use super::solve::{cholesky_inplace, cholesky_solve_into};
use super::types::{InitialMoments, KlProjectionScratch, TransitionBatch};
use super::LogRatioClass;

/// 3-iterator zip helper (avoids pulling the `itertools` crate for one fn).
/// Takes slices by reference; returns an iterator of `(&mut A, &B, &C)`.
#[inline]
fn izip3_mut<'a, A, B, C>(
    a: &'a mut [A],
    b: &'a [B],
    c: &'a [C],
) -> impl Iterator<Item = (&'a mut A, &'a B, &'a C)> {
    a.iter_mut().zip(b.iter()).zip(c.iter()).map(|((a, b), c)| (a, b, c))
}

/// Maximum Newton iterations per KL projection. Quadratic convergence makes
/// 50 ample even for ill-conditioned cases; the typical count is < 10.
const MAX_NEWTON_ITERS: usize = 50;
/// Newton convergence tolerance on `||∇L||_∞`. Right at the f32 precision
/// floor; the 1% G1 gate has enormous headroom above this.
const NEWTON_TOL: f32 = 1e-6;
/// Diagonal jitter added on Cholesky failure (singular Hessian from collinear
/// features). Mirrors the `funcattn` pattern.
const CHOLESKY_JITTER: f32 = 1e-6;

/// Linear log-ratio class `h_θ(x) = θ^T · x` (identity feature map).
///
/// The raw state slice IS the feature vector: `state_dim == feature_dim`.
/// Consumers with nonlinear features (Fourier, Random Kitchen Sinks) pre-
/// compute them and pass the result as `states` in [`TransitionBatch`].
///
/// # Modelless-ness (G5)
///
/// The only mutable state is `θ` (`Vec<f32>`). No `NeuronShard`,
/// `LoRAWeightVersion`, or `SenseModule` handle is touched anywhere in this
/// module. The primitive is modelless by construction.
#[derive(Debug, Clone)]
pub struct LinearLogRatioClass {
    /// Feature dimension `d` (= `state_dim` for identity features).
    pub feature_dim: usize,
}

impl LinearLogRatioClass {
    /// Construct a new linear class with `feature_dim` parameters.
    #[must_use]
    pub fn new(feature_dim: usize) -> Self {
        Self { feature_dim }
    }
}

impl LogRatioClass for LinearLogRatioClass {
    type Params = Vec<f32>;

    #[inline]
    fn feature_dim(&self) -> usize {
        self.feature_dim
    }

    fn new_params(&self) -> Self::Params {
        vec![0.0; self.feature_dim]
    }

    #[inline]
    fn evaluate(&self, params: &Self::Params, x: &[f32]) -> f32 {
        debug_assert_eq!(params.len(), self.feature_dim);
        debug_assert!(x.len() >= self.feature_dim);
        let mut s = 0.0_f32;
        for (p, &v) in params.iter().zip(x.iter()) {
            s += p * v;
        }
        s
    }

    fn fit_and_evaluate(
        &self,
        transitions: &TransitionBatch<'_>,
        initial: &InitialMoments<'_>,
        current_ratio: &[f32],
        gamma: f32,
        params: &mut Self::Params,
        next_ratio: &mut [f32],
        scratch: &mut KlProjectionScratch,
    ) {
        let n = transitions.n;
        let d = self.feature_dim;
        debug_assert_eq!(transitions.state_dim, d, "identity feature map: state_dim == feature_dim");
        debug_assert_eq!(initial.state_dim, d);
        debug_assert_eq!(current_ratio.len(), n);
        debug_assert_eq!(next_ratio.len(), n);
        debug_assert_eq!(params.len(), d);
        debug_assert_eq!(scratch.n, n);
        debug_assert_eq!(scratch.feature_dim, d);
        if n == 0 {
            return;
        }

        scratch.clear_iteration();

        // ── Step 1: compute the fixed moment vector m ──────────────────
        //
        // m = (1−γ) P̂_0(φ) + γ P̂^+_{n,ω}(φ)
        //
        // P̂_0(φ) is precomputed in scratch.initial_mean (set once before the
        // FORE loop by KlProjectionScratch::compute_initial_mean).
        //
        // P̂^+_{n,ω}(φ) = (Σ_i ω(Xi) φ(X^+_i)) / (Σ_i ω(Xi))   [self-normalized]

        let omega_sum: f32 = current_ratio.iter().sum();
        if omega_sum <= 0.0 {
            // Degenerate: all-zero ratio (shouldn't happen since ω̂^(0) ≡ 1).
            // Fall through with zero successor moment.
        } else {
            let inv_omega_sum = 1.0 / omega_sum;
            for (i, &omega) in current_ratio.iter().enumerate() {
                let w = omega * inv_omega_sum;
                let succ = &transitions.successors[i * d..(i + 1) * d];
                for (slot, &s) in scratch
                    .successor_weighted_sum
                    .iter_mut()
                    .zip(succ.iter())
                {
                    *slot += w * s;
                }
            }
        }

        for (m_slot, &p0, &succ) in izip3_mut(
            &mut scratch.moment,
            &scratch.initial_mean,
            &scratch.successor_weighted_sum,
        ) {
            *m_slot = (1.0 - gamma) * p0 + gamma * succ;
        }

        // ── Step 2: Newton iteration on L(θ) = log(1/n Σ e^{θ·φ(Xi)}) − θ^T m ─
        //
        // Warm-start from the current params (the previous FORE iteration's θ).
        // Near the fixed point this makes Newton converge in 1–3 steps.

        for _newton_iter in 0..MAX_NEWTON_ITERS {
            // 2a. Compute exp_buf[i] = e^{θ·φ(Xi) − max_score} (log-sum-exp trick).
            //     Also compute the empirical normalizer Z = (1/n) Σ exp_buf.
            let mut max_score = f32::NEG_INFINITY;
            for i in 0..n {
                let phi = &transitions.states[i * d..(i + 1) * d];
                let mut s = 0.0_f32;
                for (th, &v) in params.iter().zip(phi.iter()) {
                    s += th * v;
                }
                if s > max_score {
                    max_score = s;
                }
            }
            // Guard against all-zero scores (θ = 0, max_score = 0 → fine).
            if !max_score.is_finite() {
                max_score = 0.0;
            }

            let mut z_sum = 0.0_f32;
            for i in 0..n {
                let phi = &transitions.states[i * d..(i + 1) * d];
                let mut s = 0.0_f32;
                for (th, &v) in params.iter().zip(phi.iter()) {
                    s += th * v;
                }
                let e = (s - max_score).exp();
                scratch.exp_buf[i] = e;
                z_sum += e;
            }
            let inv_nz = if z_sum > 0.0 { 1.0 / (n as f32 * z_sum) } else { 0.0 };
            // w_i = e^{θ·φ(Xi)} / (n · Z_empirical) = exp_buf[i] / (n · z_sum)
            // Ê_ν[ω_θ(X)φ(X)] = (1/n) Σ_i w_i φ(Xi) = inv_nz · Σ_i exp_buf[i] · φ(Xi)

            // 2b. Compute weighted_feature_sum = Σ_i exp_buf[i] · φ(Xi) (length d).
            for slot in &mut scratch.weighted_feature_sum {
                *slot = 0.0;
            }
            for i in 0..n {
                let e = scratch.exp_buf[i];
                let phi = &transitions.states[i * d..(i + 1) * d];
                for (slot, &v) in scratch.weighted_feature_sum.iter_mut().zip(phi.iter()) {
                    *slot += e * v;
                }
            }
            // mean_phi[d] = inv_nz · weighted_feature_sum  (this is Ê_ν[ω_θ(X)φ(X)])
            // But we need the normalized version: ω_θ(Xi) = e^{...} / Z, so
            // mean_phi[k] = (1/n) Σ_i ω_θ(Xi) φ(Xi)[k]
            //             = inv_nz · Σ_i exp_buf[i] · φ(Xi)[k]
            //             = inv_nz · weighted_feature_sum[k]

            // 2c. Gradient: ∇L = mean_phi − m = inv_nz · weighted_feature_sum − moment
            let mut grad_inf_norm = 0.0_f32;
            for (g, &wfs, &m) in izip3_mut(
                &mut scratch.gradient,
                &scratch.weighted_feature_sum,
                &scratch.moment,
            ) {
                let mean_phi = inv_nz * wfs;
                *g = mean_phi - m;
                let abs_g = g.abs();
                if abs_g > grad_inf_norm {
                    grad_inf_norm = abs_g;
                }
            }

            // Convergence check: ||∇L||_∞ < tol.
            if grad_inf_norm < NEWTON_TOL {
                break;
            }

            // 2d. Hessian: H = Cov̂_{ω_θ}(φ(X)) = Ê_ν[ω_θ φ φ^T] − mean_phi mean_phi^T
            //     H[a*d+b] = inv_nz · Σ_i exp_buf[i] · φ(Xi)[a] · φ(Xi)[b]  −  mean_phi[a]·mean_phi[b]
            for slot in &mut scratch.hessian {
                *slot = 0.0;
            }
            for i in 0..n {
                let e = scratch.exp_buf[i];
                let phi = &transitions.states[i * d..(i + 1) * d];
                for a in 0..d {
                    if phi[a] == 0.0 {
                        continue;
                    }
                    let ea = e * phi[a];
                    let row = &mut scratch.hessian[a * d..(a + 1) * d];
                    for b in 0..d {
                        row[b] += ea * phi[b];
                    }
                }
            }
            // Scale by inv_nz and subtract the outer product mean_phi · mean_phi^T.
            let inv_nz_scaled = inv_nz;
            for a in 0..d {
                let mean_a = inv_nz_scaled * scratch.weighted_feature_sum[a];
                for b in 0..d {
                    let mean_b = inv_nz_scaled * scratch.weighted_feature_sum[b];
                    scratch.hessian[a * d + b] = inv_nz_scaled * scratch.hessian[a * d + b] - mean_a * mean_b;
                }
            }

            // 2e. Cholesky-factor H (with jitter fallback), solve H · Δθ = ∇L.
            let cholesky_ok = cholesky_inplace(&mut scratch.hessian, d);
            if !cholesky_ok {
                // Rebuild H with jitter (defensive — shouldn't trigger for
                // non-degenerate features). Re-add jitter to diagonal.
                // NOTE: H was overwritten by the failed cholesky_inplace, so
                // we must recompute. The cost is O(n·d²) — acceptable in the
                // rare fallback path.
                for slot in &mut scratch.hessian {
                    *slot = 0.0;
                }
                for i in 0..n {
                    let e = scratch.exp_buf[i];
                    let phi = &transitions.states[i * d..(i + 1) * d];
                    for a in 0..d {
                        let ea = e * phi[a];
                        let row = &mut scratch.hessian[a * d..(a + 1) * d];
                        for b in 0..d {
                            row[b] += ea * phi[b];
                        }
                    }
                }
                for a in 0..d {
                    let mean_a = inv_nz * scratch.weighted_feature_sum[a];
                    for b in 0..d {
                        let mean_b = inv_nz * scratch.weighted_feature_sum[b];
                        scratch.hessian[a * d + b] =
                            inv_nz * scratch.hessian[a * d + b] - mean_a * mean_b;
                    }
                    scratch.hessian[a * d + a] += CHOLESKY_JITTER;
                }
                if !cholesky_inplace(&mut scratch.hessian, d) {
                    // Even with jitter the Hessian is singular — features are
                    // collinear. Abort Newton and accept current θ.
                    break;
                }
            }
            cholesky_solve_into(
                &scratch.hessian,
                &scratch.gradient,
                d,
                &mut scratch.y_buf,
                &mut scratch.newton_step,
            );

            // 2f. Newton update: θ ← θ − Δθ.
            for (th, &delta) in params.iter_mut().zip(scratch.newton_step.iter()) {
                *th -= delta;
            }
        }

        // ── Step 3: evaluate the updated ratio ω̂^(k+1)(Xi) ─────────────
        //
        // ω̂^(k+1)(Xi) = e^{θ·φ(Xi)} / (1/n Σ_j e^{θ·φ(Xj)})
        // (Algorithm 1 step 5). Uses log-sum-exp for stability.
        let mut max_score = f32::NEG_INFINITY;
        for i in 0..n {
            let phi = &transitions.states[i * d..(i + 1) * d];
            let mut s = 0.0_f32;
            for (th, &v) in params.iter().zip(phi.iter()) {
                s += th * v;
            }
            if s > max_score {
                max_score = s;
            }
        }
        if !max_score.is_finite() {
            max_score = 0.0;
        }
        let mut z_sum = 0.0_f32;
        for (i, ratio_slot) in next_ratio.iter_mut().enumerate() {
            let phi = &transitions.states[i * d..(i + 1) * d];
            let mut s = 0.0_f32;
            for (th, &v) in params.iter().zip(phi.iter()) {
                s += th * v;
            }
            let e = (s - max_score).exp();
            *ratio_slot = e;
            z_sum += e;
        }
        let inv_z = if z_sum > 0.0 { (n as f32) / z_sum } else { 1.0 };
        for slot in next_ratio {
            *slot *= inv_z;
        }
    }
}
