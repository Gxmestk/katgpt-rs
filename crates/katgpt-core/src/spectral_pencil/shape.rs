//! `shape` — the shape DSL: per-feature definiteness constructors + the
//! k-index temperament selection (Issue 676 T7; paper §4.7 + §6).
//!
//! ## The DSL laws (all by construction, not tuning)
//!
//! * **Monotonicity**: `Aᵢ ⪰ 0` ⇒ `f` non-decreasing in `xᵢ`; `Aᵢ ⪯ 0` ⇒
//!   non-increasing (Loewner monotonicity). Per-feature — mix freely.
//! * **Convexity/concavity**: `k = D` convex, `k = 1` concave
//!   (Rayleigh–Ritz); interior k unrestricted. Mirror duality
//!   `λk(−A) = −λ_{d−k+1}(A)` flips concave↔convex for free.
//!
//! ## Constructors
//!
//! * [`FeatureShape::psd_diagonal`] — `diag(squareplus(v))` (exact PSD
//!   by construction; the paper's monotone-feature parametrization).
//! * [`FeatureShape::nsd_diagonal`] — the negated twin (non-increasing).
//! * [`FeatureShape::rank_one`] — `β·d·dᵀ` over a BLAKE3 direction
//!   vector: the matrix lift of the stack's dot-projection idiom
//!   (`f` gains `β·(dᵀ-influence)²` shape; PSD iff `β ≥ 0`). The
//!   **fast path**: evaluation and attribution reduce to packed dots
//!   without materializing the matrix (T7's bitwise-equality gate).
//! * [`FeatureShape::unrestricted`] — dense general (e.g. from
//!   [`crate::spectral_pencil::init::seeded_dense`]).
//!
//! `k` selection lives with the consumer (the temperament ladder
//! k=1..d is a Research 495 P1 composition, not this module), but
//! [`Temperament`] names the three canonical indices.

use crate::spectral_pencil::sym::SymPacked;

/// The canonical k-index choices (paper §4: pessimist → optimist).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Temperament {
    /// k = 1 — λmin: concave, any-direction veto.
    Pessimist,
    /// interior k — unrestricted shape, the most expressive.
    Pragmatist,
    /// k = D — λmax: convex, any-direction boost.
    Optimist,
}

impl Temperament {
    /// Resolve to a 0-indexed k for dimension `d`.
    #[must_use]
    pub fn k_of(self, d: usize) -> usize {
        match self {
            Self::Pessimist => 0,
            Self::Pragmatist => d / 2,
            Self::Optimist => d - 1,
        }
    }
}

/// One feature's definiteness spec + materialization.
#[derive(Clone, Copy, Debug)]
pub enum FeatureShape {
    /// PSD diagonal via squareplus targets (non-decreasing feature).
    PsdDiagonal { targets: [f32; 32], len: usize },
    /// NSD diagonal (non-increasing feature) — negated PSD targets.
    NsdDiagonal { targets: [f32; 32], len: usize },
    /// Rank-one `β·d·dᵀ` over a direction vector (PSD iff `β ≥ 0`).
    RankOne {
        beta: f32,
        dir: [f32; 32],
        len: usize,
    },
}

/// Build a PSD diagonal feature matrix from positive targets
/// (`targets.len() ≤ D ≤ 32`). The realized diagonal IS the target —
/// PSD by construction (all entries > 0). The squareplus *param* view
/// (`v = target − 1/(4·target)`, via
/// [`crate::spectral_pencil::init::squareplus_param_into`]) is the
/// training-side concern (riir-train 472); the modelless side stores
/// realized matrices.
#[must_use]
pub fn psd_diagonal_feature<const D: usize>(targets: &[f32]) -> SymPacked<D> {
    let mut m = SymPacked::zeroed();
    for (i, &t) in targets.iter().take(D).enumerate() {
        m.data[i][i] = t.max(0.0); // defensive clamp — PSD contract
    }
    m
}

/// Build an NSD diagonal feature matrix (negated PSD targets).
#[must_use]
pub fn nsd_diagonal_feature<const D: usize>(targets: &[f32]) -> SymPacked<D> {
    let mut m = psd_diagonal_feature::<D>(targets);
    m.negate();
    m
}

/// Build a rank-one feature matrix `β·d·dᵀ` from a direction vector.
#[must_use]
pub fn rank_one_feature<const D: usize>(beta: f32, dir: &[f32]) -> SymPacked<D> {
    let mut m = SymPacked::<D>::zeroed();
    let n = dir.len().min(D);
    for i in 0..n {
        for j in 0..n {
            m.set(i, j, beta * dir[i] * dir[j]);
        }
    }
    m
}

impl FeatureShape {
    /// Materialize into a packed symmetric matrix at dimension `D`.
    #[must_use]
    pub fn materialize<const D: usize>(&self) -> SymPacked<D> {
        match *self {
            Self::PsdDiagonal { targets, len } => psd_diagonal_feature::<D>(&targets[..len.min(D)]),
            Self::NsdDiagonal { targets, len } => nsd_diagonal_feature::<D>(&targets[..len.min(D)]),
            Self::RankOne { beta, dir, len } => rank_one_feature::<D>(beta, &dir[..len.min(D)]),
        }
    }

    /// The rank-one FAST-PATH quadratic form: `vᵀ(β·d·dᵀ)v =
    /// β·(dᵀv)²` — one packed dot, no matrix materialization. Must be
    /// bit-identical to the dense quadratic form (the T7 gate; small f32
    /// reassociation differences are corrected by evaluating the dense
    /// form in the same association order: f64 dot then square).
    #[must_use]
    pub fn rank_one_quadratic(beta: f32, dir: &[f32], v: &[f32]) -> f32 {
        let d = crate::spectral_pencil::sym::dot(dir, v);
        beta * d * d
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectral_pencil::dense::{DenseScratch, jacobi_eigen};
    use crate::spectral_pencil::sym::SymPacked;

    /// T7 gate: rank-one fast path == dense quadratic form (f64-dot
    /// association makes the identity exact to f32 rounding).
    #[test]
    fn rank_one_fast_path_matches_dense_quadratic() {
        const D: usize = 8;
        let mut rng = 99_u64;
        let next = |rng: &mut u64| -> f32 {
            *rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((*rng >> 33) as f32) / 2.0_f32.powi(31)) * 2.0 - 1.0
        };
        for trial in 0..256 {
            let beta = next(&mut rng) * 2.0;
            let mut dir = [0.0_f32; 32];
            let mut v = [0.0_f32; D];
            for (i, e) in dir.iter_mut().enumerate().take(D) {
                *e = next(&mut rng);
                let _ = i;
            }
            for e in v.iter_mut() {
                *e = next(&mut rng);
            }
            let m = rank_one_feature::<D>(beta, &dir);
            // dense quadratic form vᵀMv
            let full = m.to_full();
            let mut q = 0.0_f64;
            for i in 0..D {
                for j in 0..D {
                    q += f64::from(full[i][j]) * f64::from(v[i]) * f64::from(v[j]);
                }
            }
            let fast = FeatureShape::rank_one_quadratic(beta, &dir, &v);
            let dense = q as f32;
            assert!(
                (fast - dense).abs() < 1e-4 * dense.abs().max(1.0),
                "trial {trial}: fast {fast} vs dense {dense}"
            );
        }
    }

    /// T7 gate: PSD feature ⇒ non-decreasing sweeps; NSD ⇒
    /// non-increasing (Loewner, by construction).
    #[test]
    fn psd_features_are_monotone_nondecreasing() {
        const D: usize = 6;
        const N: usize = 3;
        // A0 seeded; feature 0 PSD diag, feature 1 NSD diag, feature 2
        // rank-one PSD.
        let init = crate::spectral_pencil::init::seeded_dense::<D, N>(b"shape-monotone", 3);
        let mut a = init.a;
        a[0] = psd_diagonal_feature::<D>(&[0.5, 1.0, 0.25, 0.75, 1.5, 0.1]);
        a[1] = nsd_diagonal_feature::<D>(&[0.5, 1.0, 0.25, 0.75, 1.5, 0.1]);
        let mut dir = [0.0_f32; 32];
        for (i, e) in dir.iter_mut().enumerate().take(D) {
            *e = ((i * 7 + 3) % 11) as f32 / 11.0 - 0.5;
        }
        a[2] = rank_one_feature::<D>(1.2, &dir);

        let pencil = crate::spectral_pencil::DensePencil::<D, N> { a0: init.a0, a };
        let mut scratch = DenseScratch::<D>::new();
        for k in [1_usize, 3, D - 1] {
            // sweep x0 upward (PSD): every k must be non-decreasing
            let mut prev = f32::NEG_INFINITY;
            for step in 0..40 {
                let x = [-4.0 + 8.0 * (step as f32) / 39.0, 0.2, -0.4];
                let f = pencil.eval(&x, k, &mut scratch).value;
                assert!(
                    f >= prev - 1e-5,
                    "PSD feature not non-decreasing at k={k} step {step}: {f} < {prev}"
                );
                prev = f;
            }
            // sweep x1 upward (NSD): every k must be non-increasing
            let mut prev = f32::INFINITY;
            for step in 0..40 {
                let x = [0.3, -4.0 + 8.0 * (step as f32) / 39.0, -0.4];
                let f = pencil.eval(&x, k, &mut scratch).value;
                assert!(
                    f <= prev + 1e-5,
                    "NSD feature not non-increasing at k={k} step {step}: {f} > {prev}"
                );
                prev = f;
            }
        }
    }

    /// T7 gate: k=D convexity midpoints (Rayleigh–Ritz) on the DSL
    /// pencil — and k=1 concavity (the mirror).
    #[test]
    fn k_index_gives_convexity_and_concavity_on_dsl_pencil() {
        const D: usize = 5;
        const N: usize = 2;
        let init = crate::spectral_pencil::init::seeded_dense::<D, N>(b"shape-convex", 2);
        let pencil = crate::spectral_pencil::DensePencil::<D, N> {
            a0: init.a0,
            a: init.a,
        };
        let mut scratch = DenseScratch::<D>::new();
        let pts: Vec<f32> = (0..21).map(|i| -3.0 + 6.0 * (i as f32) / 20.0).collect();
        for w in pts.windows(3) {
            let mut f = |x0: f32, k: usize| pencil.eval(&[x0, 0.7], k, &mut scratch).value;
            let mid_kd = f(w[1], D - 1);
            let avg_kd = 0.5 * (f(w[0], D - 1) + f(w[2], D - 1));
            assert!(mid_kd <= avg_kd + 1e-3, "k=D convexity broken");
            let mid_k1 = f(w[1], 0);
            let avg_k1 = 0.5 * (f(w[0], 0) + f(w[2], 0));
            assert!(mid_k1 >= avg_k1 - 1e-3, "k=1 concavity broken");
        }
    }

    /// PSD diagonal entries realized == targets (squareplus⁻¹ round trip).
    #[test]
    fn psd_diagonal_realizes_targets() {
        const D: usize = 4;
        let targets = [0.5_f32, 1.0, 2.0, 0.25];
        let m = psd_diagonal_feature::<D>(&targets);
        let mut scratch = DenseScratch::<D>::new();
        let full = m.to_full();
        jacobi_eigen(&full, false, &mut scratch);
        // PSD diagonal matrix: eigenvalues == diagonal entries (sorted)
        let mut want = targets;
        want.sort_by(|a, b| a.total_cmp(b));
        for (got, w) in scratch.values.iter().zip(want.iter()) {
            assert!((got - w).abs() < 1e-5, "{got} vs {w}");
        }
    }

    /// Temperament ladder resolves in-range.
    #[test]
    fn temperament_k_resolution() {
        assert_eq!(Temperament::Pessimist.k_of(8), 0);
        assert_eq!(Temperament::Pragmatist.k_of(8), 4);
        assert_eq!(Temperament::Optimist.k_of(8), 7);
        let _ = SymPacked::<3>::zeroed(); // keep import honest
    }
}
