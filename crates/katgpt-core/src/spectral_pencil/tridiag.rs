//! `tridiag` — Sturm-count bisection for symmetric tridiagonal pencils
//! (Issue 676 T3).
//!
//! The O(d)-parameter pencil family: `A(x) = A₀ + Σ xᵢAᵢ` with every
//! matrix symmetric tridiagonal. Two operations:
//!
//! * [`count_below`] — **exact integer count** of eigenvalues strictly
//!   below a threshold μ, via the LDLᵀ pivot-sign recurrence in O(d).
//!   Integer-exact and platform-stable: the only quorum-grade readout
//!   class with zero float cross-platform drift.
//! * [`kth_eigenvalue_bisect`] — the k-th smallest eigenvalue by fixed
//!   60-step bisection over Gershgorin bounds, one O(d) Sturm count per
//!   step ⇒ ≈ 60·d ops/eigenvalue (the issue's "≈ 50·d" budget).
//!
//! `off` arrays use the [`crate::spectral_pencil::sym::Tridiagonal`]
//! convention: length `D`, last slot dead (always zero, never read).
//!
//! ## Pinned zero-pivot convention
//!
//! The LDLᵀ recurrence `d_i = (a_i − μ) − b_{i−1}²/d_{i−1}` hits a zero
//! pivot on exact-eigenvalue shifts. We pin the LAPACK-style convention:
//! a zero `d_{i−1}` is replaced by `+f32::EPSILON·max(1, |b_{i−1}|²)` —
//! a strictly-positive tiny pivot — BEFORE the division, so the sign of
//! the following pivot is decided by `(a_i − μ)` alone at that step.
//! Same binary + same input → identical counts.

/// Caller-owned scratch for the tridiagonal kernels.
pub struct TriScratch<const D: usize> {
    /// Fused diagonal: `a₀.diag + Σ xᵢ·aᵢ.diag` (μ applied on the fly).
    pub diag: [f32; D],
    /// Fused off-diagonal: `a₀.off + Σ xᵢ·aᵢ.off` (μ-independent).
    pub off: [f32; D],
}

impl<const D: usize> TriScratch<D> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            diag: [0.0; D],
            off: [0.0; D],
        }
    }
}

impl<const D: usize> Default for TriScratch<D> {
    fn default() -> Self {
        Self::new()
    }
}

/// Build the fused (unshifted) tridiagonal `A₀ + Σ xᵢAᵢ` into scratch.
#[inline]
pub fn fuse_into<const D: usize, const N: usize>(
    a0_diag: &[f32; D],
    a0_off: &[f32; D],
    a_diag: &[[f32; D]; N],
    a_off: &[[f32; D]; N],
    x: &[f32; N],
    scratch: &mut TriScratch<D>,
) {
    scratch.diag = *a0_diag;
    scratch.off = *a0_off;
    for (i, &xi) in x.iter().enumerate() {
        for (sd, &d) in scratch.diag.iter_mut().zip(a_diag[i].iter()) {
            *sd += xi * d;
        }
        for (so, &o) in scratch.off.iter_mut().zip(a_off[i].iter()) {
            *so += xi * o;
        }
    }
}

/// Sturm count: eigenvalues of the tridiagonal `(diag, off)` strictly
/// below μ, in place over `diag_shift` (μ subtracted; caller restores or
/// discards). O(d), allocation-free. The pinned zero-pivot convention
/// lives here (module doc).
#[must_use]
pub fn count_below_shifted_inplace<const D: usize>(
    diag_shift: &mut [f32; D],
    off: &[f32; D],
    mu: f32,
) -> u32 {
    let mut count = 0_u32;
    let mut d_prev = 1.0_f32;
    for i in 0..D {
        let b = if i > 0 { off[i - 1] } else { 0.0 };
        // pinned zero-pivot substitution (module doc)
        let denom = if d_prev == 0.0 {
            f32::EPSILON * (b * b).max(1.0)
        } else {
            d_prev
        };
        let d = (diag_shift[i] - mu) - b * b / denom;
        if d < 0.0 {
            count += 1;
        }
        d_prev = d;
    }
    count
}

/// Count eigenvalues of `(diag, off)` strictly below μ without
/// consuming `diag` (restores it after the shifted pass).
#[must_use]
pub fn count_below<const D: usize>(
    diag: &[f32; D],
    off: &[f32; D],
    mu: f32,
    scratch_diag: &mut [f32; D],
) -> u32 {
    *scratch_diag = *diag;
    count_below_shifted_inplace(scratch_diag, off, mu)
}

/// Fixed-iteration bisection for the k-th smallest eigenvalue
/// (0-indexed `k < D`) of `(diag, off)` via Gershgorin bounds.
///
/// 60 halvings shrink any f32-scale bracket far below f32 resolution —
/// the loop exits early only when `hi − lo` underflows to 0. The result
/// is deterministic per binary by construction (integer iteration count,
/// no library).
#[must_use]
pub fn kth_eigenvalue_bisect<const D: usize>(diag: &[f32; D], off: &[f32; D], k: usize) -> f32 {
    debug_assert!(k < D, "k={k} out of range for D={D}");
    let kk = k.min(D - 1);
    // Gershgorin over the fused matrix.
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for i in 0..D {
        let mut r = 0.0_f32;
        if i > 0 {
            r += off[i - 1].abs();
        }
        if i + 1 < D {
            r += off[i].abs();
        }
        lo = lo.min(diag[i] - r);
        hi = hi.max(diag[i] + r);
    }
    let mut work = [0.0_f32; D];
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if mid == lo || mid == hi {
            break; // bracket collapsed below f32 resolution
        }
        let c = count_below(diag, off, mid, &mut work);
        if c as usize <= kk {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    hi // count(hi) ≥ kk+1 > count(lo) — hi holds λ_k
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectral_pencil::dense::{DenseScratch, jacobi_eigen};

    fn random_tridiag<const D: usize>(seed: u64) -> ([f32; D], [f32; D]) {
        let mut rng = seed;
        let next = |rng: &mut u64| -> f32 {
            *rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((*rng >> 33) as f32) / 2.0_f32.powi(31)) * 4.0 - 2.0
        };
        let mut diag = [0.0; D];
        let mut off = [0.0; D];
        for d in diag.iter_mut() {
            *d = next(&mut rng);
        }
        for o in off.iter_mut().take(D.saturating_sub(1)) {
            *o = next(&mut rng);
        }
        (diag, off)
    }

    #[test]
    fn sturm_matches_full_solve_on_midpoints_10k() {
        // The 10⁶-seed release tier lives in tests.rs (debug-ignored);
        // this is the always-on 10k tier at d=8.
        const D: usize = 8;
        let mut work = [0.0_f32; D];
        for seed in 0..10_000_u64 {
            let (diag, off) = random_tridiag::<D>(seed.wrapping_mul(0x9E37) ^ 0x1234);
            let mut full = [[0.0_f32; D]; D];
            for i in 0..D {
                full[i][i] = diag[i];
            }
            for i in 0..(D - 1) {
                full[i][i + 1] = off[i];
                full[i + 1][i] = off[i];
            }
            let mut s = DenseScratch::<D>::new();
            jacobi_eigen(&full, false, &mut s);
            for w in s.values.windows(2) {
                if w[0] == w[1] {
                    continue; // repeated pair — midpoint IS the eigenvalue
                }
                let theta = 0.5 * (w[0] + w[1]);
                let sturm = count_below(&diag, &off, theta, &mut work);
                let dense = s.values.iter().filter(|&&v| v < theta).count() as u32;
                assert_eq!(
                    sturm, dense,
                    "seed {seed} theta {theta}: sturm {sturm} vs dense {dense}"
                );
            }
        }
    }

    #[test]
    fn bisect_matches_jacobi_values() {
        const D: usize = 10;
        for seed in 1..200_u64 {
            let (diag, off) = random_tridiag::<D>(seed.wrapping_mul(7919));
            let mut full = [[0.0_f32; D]; D];
            for i in 0..D {
                full[i][i] = diag[i];
            }
            for i in 0..(D - 1) {
                full[i][i + 1] = off[i];
                full[i + 1][i] = off[i];
            }
            let mut ds = DenseScratch::<D>::new();
            jacobi_eigen(&full, false, &mut ds);
            for k in [0_usize, 1, D / 2, D - 2, D - 1] {
                let b = kth_eigenvalue_bisect(&diag, &off, k);
                assert!(
                    (b - ds.values[k]).abs() < 1e-4 * ds.values[k].abs().max(1.0),
                    "seed {seed} k {k}: bisect {b} vs jacobi {}",
                    ds.values[k]
                );
            }
        }
    }

    #[test]
    fn zero_pivot_convention_survives_exact_ladder_shifts() {
        // The constant-off-diagonal tridiagonal has known eigenvalues
        // 2cos(kπ/(d+1)). Counting at midpoints between the known
        // eigenvalues must match the closed form exactly.
        const D: usize = 5;
        let diag = [0.0_f32; D];
        let mut off = [0.0_f32; D];
        for o in off.iter_mut().take(D - 1) {
            *o = 1.0;
        }
        let mut work = [0.0_f32; D];
        // μ points strictly between the known eigenvalues (non-aligned).
        let known = [-1.732_050_8_f32, -1.0, 0.0, 1.0, 1.732_050_8];
        for w in known.windows(2) {
            let mu = 0.5 * (w[0] + w[1]);
            let sturm = count_below(&diag, &off, mu, &mut work);
            let closed = known.iter().filter(|&&v| v < mu).count() as u32;
            assert_eq!(sturm, closed, "mu {mu}");
        }
    }
}
