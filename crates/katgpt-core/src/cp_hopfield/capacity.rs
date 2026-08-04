//! Capacity measurement — the honest finite-`N` counterpart to the paper's
//! asymptotic `α_c`.
//!
//! The source paper's capacities (`α_c(d=3) ≈ 0.62`, `α_c(d=4) ≈ 2.41`,
//! `α_c(d=8) ≈ 40`) come from a replica calculation in the `N → ∞` limit on
//! Haar-random (uncorrelated) memories. Neither assumption holds for this
//! codebase: belief state is `N = 8`, per-shard style weights are `N = 64`, and
//! real memories — personality snapshots, related KG triples — are correlated.
//!
//! This module measures `α_c` under the conditions actually in use. It exists so
//! the capacity claim is a *measurement* rather than a citation; where the measured
//! value falls short of the asymptotic one, that gap is the result.

use super::complex::C32;
use super::recaller::CpHopfieldRecaller;

/// How to generate the memory ensemble for a capacity sweep.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryDistribution {
    /// Independent Haar-random qudits — the distribution the replica `α_c` assumes.
    Haar,
    /// Correlated memories: `ξ^μ = cos(θ_μ) · v_base + sin(θ_μ) · v_orth` with
    /// `θ_μ` spread over `[0, spread_radians]` about a shared per-neuron base
    /// direction. `spread` is stored in milliradians to keep the enum `Copy` and
    /// hashable; use [`MemoryDistribution::correlated`] to construct.
    ///
    /// Small spread ⇒ strongly correlated memories ⇒ expect materially lower
    /// `α_c` than Haar, plus the paper's "shadow" phenomenon where an un-cued but
    /// correlated memory leaks into the recall.
    Correlated {
        /// Angular spread in milliradians.
        spread_mrad: u32,
    },
}

impl MemoryDistribution {
    /// Correlated memories with the given angular spread in radians.
    ///
    /// `spread = π/2` is effectively uncorrelated (base and orthogonal component
    /// fully exchanged); `spread → 0` collapses all memories onto one direction.
    pub fn correlated(spread_radians: f32) -> Self {
        Self::Correlated {
            spread_mrad: (spread_radians.max(0.0) * 1000.0) as u32,
        }
    }
}

/// One measured point on a capacity curve.
#[derive(Clone, Copy, Debug)]
pub struct CapacityPoint {
    /// Load `α = P / N`.
    pub alpha: f32,
    /// Memories stored, `P`.
    pub n_memories: usize,
    /// Mean recall overlap `m̄_0` with the cued memory, averaged over realizations.
    pub mean_overlap: f32,
    /// Mean BBP relative gap `(λ_max − λ_2)/λ_max` of the memory kernel at recall
    /// time, averaged over realizations and neurons sampled.
    pub mean_bbp_gap: f32,
}

/// Result of a capacity sweep.
#[derive(Clone, Debug)]
pub struct CapacityCurve {
    /// Complex dimension `d`.
    pub d: usize,
    /// Neuron count `N`.
    pub n_neurons: usize,
    /// Which memory ensemble was swept.
    pub distribution: MemoryDistribution,
    /// Measured points, in the order the `alpha_range` supplied them.
    pub points: Vec<CapacityPoint>,
    /// Recall-quality threshold used to locate `α_c`.
    pub threshold: f32,
}

impl CapacityCurve {
    /// Estimated `α_c`: the load at which `mean_overlap` crosses below
    /// [`Self::threshold`], linearly interpolated between the bracketing points.
    ///
    /// Returns `None` if the curve never crosses — either because every load
    /// recalled successfully (`α_c` is above the swept range) or because none did
    /// (`α_c` is below it). A `None` is informative, not an error: it says the
    /// sweep range was wrong, and reporting it as a number would be a fabrication.
    pub fn alpha_c(&self) -> Option<f32> {
        for w in self.points.windows(2) {
            let (lo, hi) = (&w[0], &w[1]);
            if lo.mean_overlap >= self.threshold && hi.mean_overlap < self.threshold {
                let span = lo.mean_overlap - hi.mean_overlap;
                if span.abs() < 1e-9 {
                    return Some(lo.alpha);
                }
                let t = (lo.mean_overlap - self.threshold) / span;
                return Some(lo.alpha + t * (hi.alpha - lo.alpha));
            }
        }
        None
    }
}

/// Deterministic Gaussian-pair source for Haar sampling.
///
/// Box-Muller on a `fastrand` stream. Seeded per call so a sweep is reproducible.
struct GaussianSource {
    rng: fastrand::Rng,
}

impl GaussianSource {
    fn new(seed: u64) -> Self {
        Self {
            rng: fastrand::Rng::with_seed(seed),
        }
    }

    fn normal(&mut self) -> f32 {
        let u1 = self.rng.f32().max(f32::EPSILON);
        let u2 = self.rng.f32();
        let r = (-2.0 * u1.ln()).sqrt();
        r * (2.0 * std::f32::consts::PI * u2).cos()
    }

    /// A Haar-random qudit: `d` i.i.d. complex Gaussians, normalized.
    ///
    /// Normalizing an i.i.d. complex Gaussian vector is the standard exact Haar
    /// construction on `CP^(d-1)` — the Gaussian measure is unitarily invariant,
    /// so the induced measure on the unit sphere is uniform, and the `U(1)` phase
    /// quotient is automatic because the Bloch projection discards it.
    fn haar_qudit<const D: usize>(&mut self) -> [C32; D] {
        let mut q = [C32::ZERO; D];
        for z in q.iter_mut() {
            *z = C32::new(self.normal(), self.normal());
        }
        super::recaller::normalize_qudit(&q)
    }
}

/// Generate one memory pattern (a qudit per neuron) under the requested ensemble.
fn make_pattern<const D: usize>(
    n_neurons: usize,
    mu: usize,
    n_memories: usize,
    dist: MemoryDistribution,
    base: &[[C32; D]],
    orth: &[[C32; D]],
    src: &mut GaussianSource,
) -> Vec<[C32; D]> {
    match dist {
        MemoryDistribution::Haar => (0..n_neurons).map(|_| src.haar_qudit::<D>()).collect(),
        MemoryDistribution::Correlated { spread_mrad } => {
            let spread = spread_mrad as f32 / 1000.0;
            // Spread θ evenly across memories so the ensemble is deterministic
            // given (spread, n_memories) — the correlation structure is the
            // variable under test, so it should not also be noisy.
            let theta = if n_memories <= 1 {
                0.0
            } else {
                spread * (mu as f32 / (n_memories - 1) as f32)
            };
            let (c, s) = (theta.cos(), theta.sin());
            (0..n_neurons)
                .map(|i| {
                    let mut q = [C32::ZERO; D];
                    for a in 0..D {
                        q[a] = base[i][a].scale(c).add(orth[i][a].scale(s));
                    }
                    super::recaller::normalize_qudit(&q)
                })
                .collect()
        }
    }
}

/// Measure the capacity curve for `CP^(D-1)` at `n_neurons`.
///
/// For each `α` in `alpha_range`: build `P = round(α·N)` memories, set every neuron
/// to memory 0, corrupt a `corrupt_fraction` of them with fresh Haar-random states,
/// recall to a fixed point, and record `m̄_0`. Averaged over `realizations`
/// independent draws.
///
/// `D2` must equal `D² − 1`.
///
/// # Panics
/// Panics if `D2 != D*D - 1`, if `n_neurons == 0`, or if `realizations == 0`.
pub fn measure_capacity<const D: usize, const D2: usize>(
    n_neurons: usize,
    alpha_range: &[f32],
    realizations: usize,
    corrupt_fraction: f32,
    distribution: MemoryDistribution,
    threshold: f32,
    seed: u64,
) -> CapacityCurve {
    assert!(realizations > 0, "cp_hopfield: need at least 1 realization");
    let max_sweeps = 20;
    let mut points = Vec::with_capacity(alpha_range.len());

    for (ai, &alpha) in alpha_range.iter().enumerate() {
        let p = ((alpha * n_neurons as f32).round() as usize).max(1);
        let mut overlap_acc = 0.0f32;
        let mut gap_acc = 0.0f32;

        for r in 0..realizations {
            let mut src = GaussianSource::new(
                seed ^ ((ai as u64) << 32) ^ (r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            );
            let mut rec = CpHopfieldRecaller::<D, D2>::new(n_neurons);

            // Shared base + orthogonal directions for the correlated ensemble.
            let base: Vec<[C32; D]> = (0..n_neurons).map(|_| src.haar_qudit::<D>()).collect();
            let orth: Vec<[C32; D]> = (0..n_neurons).map(|_| src.haar_qudit::<D>()).collect();

            for mu in 0..p {
                let pattern =
                    make_pattern::<D>(n_neurons, mu, p, distribution, &base, &orth, &mut src);
                rec.push_memory(&pattern);
            }

            // Cue memory 0, then corrupt a fraction of the neurons.
            for i in 0..n_neurons {
                let q = *rec.memory_qudit(0, i);
                rec.set_state_qudit(i, &q);
            }
            let n_corrupt = (corrupt_fraction * n_neurons as f32).round() as usize;
            for i in 0..n_corrupt.min(n_neurons) {
                let q = src.haar_qudit::<D>();
                rec.set_state_qudit(i, &q);
            }

            // Sample the BBP gap at recall time, before the state has settled —
            // that is the load-vs-protection relationship the claim rests on.
            gap_acc += rec.kernel_spectrum(0).relative_gap();

            rec.recall_to_fixed_point(1e-4, max_sweeps);
            overlap_acc += rec.mean_overlap(0);
        }

        let inv = 1.0 / realizations as f32;
        points.push(CapacityPoint {
            alpha,
            n_memories: p,
            mean_overlap: overlap_acc * inv,
            mean_bbp_gap: gap_acc * inv,
        });
    }

    CapacityCurve {
        d: D,
        n_neurons,
        distribution,
        points,
        threshold,
    }
}

/// Build a recaller preloaded with `p` memories over `n_neurons`, cued to memory 0
/// with `corrupt_fraction` of neurons randomized.
///
/// The shared fixture behind the correctness tests and the PoC arms.
pub fn distribution_fixture<const D: usize, const D2: usize>(
    n_neurons: usize,
    p: usize,
    corrupt_fraction: f32,
    distribution: MemoryDistribution,
    seed: u64,
) -> CpHopfieldRecaller<D, D2> {
    let mut src = GaussianSource::new(seed);
    let mut rec = CpHopfieldRecaller::<D, D2>::new(n_neurons);

    let base: Vec<[C32; D]> = (0..n_neurons).map(|_| src.haar_qudit::<D>()).collect();
    let orth: Vec<[C32; D]> = (0..n_neurons).map(|_| src.haar_qudit::<D>()).collect();
    for mu in 0..p {
        let pattern = make_pattern::<D>(n_neurons, mu, p, distribution, &base, &orth, &mut src);
        rec.push_memory(&pattern);
    }

    for i in 0..n_neurons {
        let q = *rec.memory_qudit(0, i);
        rec.set_state_qudit(i, &q);
    }
    let n_corrupt = (corrupt_fraction * n_neurons as f32).round() as usize;
    for i in 0..n_corrupt.min(n_neurons) {
        let q = src.haar_qudit::<D>();
        rec.set_state_qudit(i, &q);
    }
    rec
}

/// [`distribution_fixture`] with Haar-random memories.
pub fn haar_fixture<const D: usize, const D2: usize>(
    n_neurons: usize,
    p: usize,
    corrupt_fraction: f32,
    seed: u64,
) -> CpHopfieldRecaller<D, D2> {
    distribution_fixture::<D, D2>(
        n_neurons,
        p,
        corrupt_fraction,
        MemoryDistribution::Haar,
        seed,
    )
}
