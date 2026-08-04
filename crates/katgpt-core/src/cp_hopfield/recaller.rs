//! The `CP^(d-1)` top-eigenvector recaller.

use super::basis::{GellMannBasis, StructureConstants};
use super::complex::C32;
use super::{bloch_norm_sq, bloch_overlap, constraint_rhs};

/// Power-iteration cap. At `d ≤ 8` with a BBP-separated spike, convergence is
/// geometric in `λ_2/λ_max` and lands in well under 20 iterations; the cap only
/// matters near `α_c` where the gap closes and the top eigenvector is genuinely
/// ill-defined (which is the failure mode being measured, not a bug to hide).
const POWER_ITER_MAX: usize = 64;

/// Convergence threshold on the *relative eigenpair residual*
/// `‖Kv − λv‖ / |λ|`.
///
/// Deliberately not the more obvious `1 − |⟨v_new|v_old⟩|`: that quantity
/// saturates in `f32`, since `1.0 − x` for `x` near 1 cannot resolve below
/// ~6e-8, so tightening it past `1e-7` silently degenerates into "always run the
/// iteration cap". The residual norm is a difference of same-magnitude quantities
/// and stays meaningful far below that, which lets the projection be accurate
/// enough to be idempotent.
const POWER_ITER_TOL: f32 = 1e-6;

/// Fixed seed for the power-iteration start vector. Any value works; fixing it is
/// what makes recall bit-reproducible.
const POWER_ITER_SEED: u64 = 0x00CD_4667_0567;

/// Eigenvalue separation of a memory kernel `K_i` — the BBP diagnostic.
///
/// The whole capacity claim rests on `λ_max` being separated from the GUE
/// crosstalk bulk. When [`Self::relative_gap`] collapses toward zero the top
/// eigenvector is no longer pinned to the stored memory and recall degrades,
/// regardless of what the asymptotic `α_c` predicts. This is the G7 measurement.
#[derive(Clone, Copy, Debug)]
pub struct KernelSpectrum {
    /// Algebraically largest eigenvalue of `K_i` (the spike, below `α_c`).
    pub lambda_max: f32,
    /// Second-largest eigenvalue (the bulk edge, below `α_c`).
    pub lambda_2: f32,
}

impl KernelSpectrum {
    /// `(λ_max − λ_2) / |λ_max|` — the BBP protection margin.
    ///
    /// Returns 0.0 when `λ_max` is numerically zero (an unloaded kernel has no
    /// spike to protect).
    #[inline]
    pub fn relative_gap(&self) -> f32 {
        if self.lambda_max.abs() < 1e-12 {
            return 0.0;
        }
        (self.lambda_max - self.lambda_2) / self.lambda_max.abs()
    }
}

/// Deterministic PRNG for power-iteration start vectors.
///
/// Integer arithmetic only, so start vectors — and therefore recall results — are
/// bit-identical across platforms. Same rationale as `manifold_bandit`'s copy: a
/// non-uniform start avoids pathological convergence on degenerate spectra, but it
/// must not introduce run-to-run variation.
#[derive(Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f32_signed(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / ((1u64 << 24) as f32)) * 2.0 - 1.0
    }
}

/// Top-eigenvector associative memory recall on `CP^(d-1) = SU(d)/U(d-1)`.
///
/// `D` is the complex dimension `d`; `D2 = d² − 1` is the real Bloch dimension.
/// Holds `P` memories over `N` neurons (a `P × N` table of qudits) plus the
/// current `N` neuron states in Bloch coordinates.
///
/// Memories are stored in **both** representations: complex amplitudes (needed to
/// build the Hermitian kernel `K_i` as a sum of rank-1 projectors) and Bloch
/// vectors (needed for the Mattis overlaps and the mean field, which are pure real
/// dot products). The duplication trades `2×` memory for keeping the hot paths off
/// complex arithmetic.
///
/// See the [module docs](super) for the mechanism and the capacity table.
#[derive(Clone, Debug)]
pub struct CpHopfieldRecaller<const D: usize, const D2: usize> {
    n_neurons: usize,
    /// `P × N`, row-major by memory: `memories_c[mu * n + i] = |ξ_i^μ⟩`.
    memories_c: Vec<[C32; D]>,
    /// Same table in Bloch coordinates.
    memories_bloch: Vec<[f32; D2]>,
    /// Current neuron states, Bloch coordinates. Length `N`.
    states: Vec<[f32; D2]>,
    basis: GellMannBasis<D>,
    structure: StructureConstants,
}

impl<const D: usize, const D2: usize> CpHopfieldRecaller<D, D2> {
    /// Create a recaller over `n_neurons` neurons with no memories stored.
    ///
    /// All neuron states start at the first basis direction (an arbitrary but
    /// valid on-manifold point). Builds the SU(d) basis and contracts its
    /// structure constants — the only non-trivial cost in the whole type, and it
    /// is paid once.
    ///
    /// # Panics
    /// Panics if `D2 != D*D - 1`, if `D < 2`, or if `n_neurons == 0`.
    pub fn new(n_neurons: usize) -> Self {
        assert_eq!(
            D2,
            D * D - 1,
            "cp_hopfield: D2 must be d²−1; got d = {D}, D2 = {D2} (expected {})",
            D * D - 1
        );
        assert!(n_neurons > 0, "cp_hopfield: need at least one neuron");

        let basis = GellMannBasis::<D>::new();
        let structure = StructureConstants::new(&basis);

        let mut seed = [C32::ZERO; D];
        seed[0] = C32::ONE;
        let mut seed_bloch = [0.0f32; D2];
        basis.bloch_projection_into(&seed, &mut seed_bloch);

        Self {
            n_neurons,
            memories_c: Vec::new(),
            memories_bloch: Vec::new(),
            states: vec![seed_bloch; n_neurons],
            basis,
            structure,
        }
    }

    /// Number of neurons `N`.
    #[inline]
    pub fn n_neurons(&self) -> usize {
        self.n_neurons
    }

    /// Number of stored memories `P`.
    #[inline]
    pub fn n_memories(&self) -> usize {
        self.memories_c.len() / self.n_neurons
    }

    /// Load factor `α = P / N`.
    #[inline]
    pub fn load(&self) -> f32 {
        self.n_memories() as f32 / self.n_neurons as f32
    }

    /// The SU(d) basis.
    #[inline]
    pub fn basis(&self) -> &GellMannBasis<D> {
        &self.basis
    }

    /// The SU(d) structure constants.
    #[inline]
    pub fn structure(&self) -> &StructureConstants {
        &self.structure
    }

    /// Store one memory: a qudit per neuron. Normalizes each qudit on write.
    ///
    /// This is the entire "write" path — a deterministic Hebbian store, no
    /// optimization loop. Loading `NeuronShard::style_weights` or any other frozen
    /// snapshot as memories is exactly this call.
    ///
    /// # Panics
    /// Panics if `pattern.len() != n_neurons`.
    pub fn push_memory(&mut self, pattern: &[[C32; D]]) {
        assert_eq!(
            pattern.len(),
            self.n_neurons,
            "cp_hopfield: memory pattern must cover all {} neurons",
            self.n_neurons
        );
        for qudit in pattern {
            let normalized = normalize_qudit(qudit);
            let mut bloch = [0.0f32; D2];
            self.basis.bloch_projection_into(&normalized, &mut bloch);
            self.memories_c.push(normalized);
            self.memories_bloch.push(bloch);
        }
    }

    /// Read neuron `i`'s current Bloch state.
    #[inline]
    pub fn state(&self, i: usize) -> &[f32; D2] {
        &self.states[i]
    }

    /// All current Bloch states.
    #[inline]
    pub fn states(&self) -> &[[f32; D2]] {
        &self.states
    }

    /// Overwrite neuron `i`'s state from a qudit.
    pub fn set_state_qudit(&mut self, i: usize, qudit: &[C32; D]) {
        let normalized = normalize_qudit(qudit);
        self.basis
            .bloch_projection_into(&normalized, &mut self.states[i]);
    }

    /// Overwrite neuron `i`'s state from a Bloch vector, projecting it onto the
    /// manifold first so the stored state is always a valid `CP^(d-1)` point.
    pub fn set_state_bloch(&mut self, i: usize, bloch: &[f32; D2]) {
        let mut s = *bloch;
        self.project_to_manifold(&mut s);
        self.states[i] = s;
    }

    /// Read memory `mu`'s qudit for neuron `i`.
    #[inline]
    pub fn memory_qudit(&self, mu: usize, i: usize) -> &[C32; D] {
        &self.memories_c[mu * self.n_neurons + i]
    }

    /// Read memory `mu`'s Bloch vector for neuron `i`.
    #[inline]
    pub fn memory_bloch(&self, mu: usize, i: usize) -> &[f32; D2] {
        &self.memories_bloch[mu * self.n_neurons + i]
    }

    // ── Recall ───────────────────────────────────────────────────────────

    /// The Mattis overlap `O_μ^(i)`, excluding neuron `i`'s own contribution.
    ///
    /// ```text
    /// O_μ^(i) = (2/N) Σ_{j≠i} (|⟨s_j | ξ_j^μ⟩|² − 1/d)
    ///         = (1/N) Σ_{j≠i} s_j · s_j^μ
    /// ```
    ///
    /// The second form uses `|⟨u|v⟩|² = 1/d + (1/2) s_u · s_v`, keeping the whole
    /// computation in real Bloch arithmetic. Excluding `j = i` is what removes the
    /// self-interaction term that would otherwise pin a neuron to its own current
    /// state and make recall trivially self-confirming.
    ///
    /// `O(N · D2)`.
    pub fn mattis_overlap_excluding(&self, neuron_idx: usize, mu: usize) -> f32 {
        let base = mu * self.n_neurons;
        let mut acc = 0.0f32;
        for j in 0..self.n_neurons {
            if j == neuron_idx {
                continue;
            }
            let s = &self.states[j];
            let m = &self.memories_bloch[base + j];
            let mut dot = 0.0f32;
            for a in 0..D2 {
                dot += s[a] * m[a];
            }
            acc += dot;
        }
        acc / self.n_neurons as f32
    }

    /// Build the `d×d` Hermitian memory kernel
    /// `K_i = Σ_μ O_μ^(i) |ξ_i^μ⟩⟨ξ_i^μ|`.
    ///
    /// Near a stored memory this is a *spiked* random matrix: a rank-1 signal
    /// projector plus a GUE-like crosstalk sum. `O(P · d²)`.
    pub fn build_memory_kernel(&self, neuron_idx: usize) -> [[C32; D]; D] {
        let mut k = [[C32::ZERO; D]; D];
        let p = self.n_memories();
        for mu in 0..p {
            let o = self.mattis_overlap_excluding(neuron_idx, mu);
            if o == 0.0 {
                continue;
            }
            let xi = &self.memories_c[mu * self.n_neurons + neuron_idx];
            for r in 0..D {
                let xr = xi[r];
                for c in 0..D {
                    // |ξ⟩⟨ξ| has entries ξ_r · conj(ξ_c).
                    k[r][c] = k[r][c].add(xr.mul(xi[c].conj()).scale(o));
                }
            }
        }
        k
    }

    /// One top-eigenvector recall step for neuron `i`.
    ///
    /// Builds `K_i`, takes its algebraically-largest eigenvector, and returns that
    /// eigenvector's Bloch coordinates — the new state for neuron `i`.
    ///
    /// The neuron's **current state is not an input.** `K_i` is built from the
    /// *other* neurons' states only, so the recall target is determined by the
    /// spiked kernel's spectrum, not by where the neuron happens to be. That is
    /// precisely the "gapped" property that vector alignment on `S^(n-1)` lacks: a
    /// gapless update rule drifts with its own current state under any crosstalk,
    /// whereas this one snaps to the BBP-protected spike.
    ///
    /// Result is on-manifold by construction (it is the Bloch projection of a unit
    /// qudit), so no constraint projection is needed. `O(P·d² + d³)`.
    pub fn recall_step(&self, neuron_idx: usize) -> [f32; D2] {
        let k = self.build_memory_kernel(neuron_idx);
        let (evec, _) = hermitian_top_eigenvector(&k);
        let mut out = [0.0f32; D2];
        self.basis.bloch_projection_into(&evec, &mut out);
        out
    }

    /// Asynchronously apply [`Self::recall_step`] to every neuron in index order.
    ///
    /// Asynchronous (each neuron sees its predecessors' updated states, as in the
    /// source paper) rather than synchronous, which also makes the sweep
    /// allocation-free — no shadow state buffer is needed.
    ///
    /// Returns the mean Euclidean state displacement over the sweep, so callers can
    /// detect a fixed point.
    pub fn sweep(&mut self) -> f32 {
        let mut total = 0.0f32;
        for i in 0..self.n_neurons {
            let next = self.recall_step(i);
            let prev = self.states[i];
            let mut d2 = 0.0f32;
            for a in 0..D2 {
                let delta = next[a] - prev[a];
                d2 += delta * delta;
            }
            total += d2.sqrt();
            self.states[i] = next;
        }
        total / self.n_neurons as f32
    }

    /// Run [`Self::sweep`] until the mean displacement drops below `tol` or
    /// `max_sweeps` is reached. Returns the number of sweeps performed.
    pub fn recall_to_fixed_point(&mut self, tol: f32, max_sweeps: usize) -> usize {
        for n in 0..max_sweeps {
            if self.sweep() < tol {
                return n + 1;
            }
        }
        max_sweeps
    }

    /// Normalized mean Mattis overlap `m̄_μ` between the current states and memory
    /// `mu`.
    ///
    /// `1.0` is perfect recall; `0.0` is chance. This is the recall-quality metric
    /// the G1 and G2 gates measure.
    pub fn mean_overlap(&self, mu: usize) -> f32 {
        let base = mu * self.n_neurons;
        let mut acc = 0.0f32;
        for i in 0..self.n_neurons {
            acc += bloch_overlap(&self.states[i], &self.memories_bloch[base + i], D);
        }
        acc / self.n_neurons as f32
    }

    /// Measure the top two eigenvalues of `K_i` — the BBP gap diagnostic (G7).
    ///
    /// `λ_2` is found by Hotelling deflation after `λ_max`. See [`KernelSpectrum`].
    pub fn kernel_spectrum(&self, neuron_idx: usize) -> KernelSpectrum {
        let k = self.build_memory_kernel(neuron_idx);
        let (v1, lambda_max) = hermitian_top_eigenvector(&k);

        // Deflate: K' = K − λ_max v v†. The remaining top eigenvalue is λ_2.
        let mut deflated = k;
        for r in 0..D {
            for c in 0..D {
                let outer = v1[r].mul(v1[c].conj()).scale(lambda_max);
                deflated[r][c] = deflated[r][c].sub(outer);
            }
        }
        let (_, lambda_2) = hermitian_top_eigenvector(&deflated);
        KernelSpectrum {
            lambda_max,
            lambda_2,
        }
    }

    // ── Manifold constraint ──────────────────────────────────────────────

    /// Project a Bloch vector onto `CP^(d-1)`, enforcing both the norm constraint
    /// `|s|² = 2(1 − 1/d)` and the non-linear constraint
    /// `d_abc s_a s_b = (2 − 4/d) s_c`.
    ///
    /// # Exact, not iterative
    ///
    /// Plan 567 T2.1 specified alternating normalization and constraint projection
    /// until convergence. That is unnecessary: the projection has a **closed
    /// form**. The Bloch map `s ↦ ρ = (1/d)I + (1/2)s_a λ_a` satisfies
    /// `‖ρ − ρ'‖_F² = ‖s − s'‖²/2`, so it is a Euclidean similarity — minimizing
    /// distance in Bloch space is the same as minimizing Frobenius distance in
    /// density-matrix space. And the closest unit-trace rank-1 projector to a
    /// Hermitian `ρ` is `v v†` for `v` its top eigenvector (maximizing `v†ρv`).
    ///
    /// So: build `ρ`, take its top eigenvector, project back. One `O(d³)` power
    /// iteration, no convergence loop, and the result is the *exact* Euclidean
    /// projection rather than a fixed-point approximation of it. Both constraints
    /// are then satisfied identically, because the output is by construction the
    /// Bloch vector of a genuine pure state.
    ///
    /// # Panics
    /// Panics if `bloch.len() != D2`.
    pub fn project_to_manifold(&self, bloch: &mut [f32]) {
        assert_eq!(bloch.len(), D2, "cp_hopfield: bloch slice must be d²−1");
        let rho = self.basis.density_from_bloch(bloch);
        let (evec, _) = hermitian_top_eigenvector(&rho);
        self.basis.bloch_projection_into(&evec, bloch);
    }

    /// Worst-case violation of the non-linear constraint:
    /// `max_c |d_abc s_a s_b − (2 − 4/d) s_c|`.
    ///
    /// Zero (to round-off) for any on-manifold state. The G4 correctness check.
    pub fn constraint_residual(&self, bloch: &[f32]) -> f32 {
        assert_eq!(bloch.len(), D2, "cp_hopfield: bloch slice must be d²−1");
        let mut lhs = [0.0f32; D2];
        for t in self.structure.d_triples() {
            lhs[t.i as usize] += t.val * bloch[t.j as usize] * bloch[t.k as usize];
        }
        let rhs = constraint_rhs(D);
        (0..D2)
            .map(|c| (lhs[c] - rhs * bloch[c]).abs())
            .fold(0.0f32, f32::max)
    }

    /// Deviation of `|s|²` from the required `2(1 − 1/d)`.
    pub fn norm_residual(&self, bloch: &[f32]) -> f32 {
        let n2: f32 = bloch.iter().map(|x| x * x).sum();
        (n2 - bloch_norm_sq(D)).abs()
    }
}

/// Normalize a qudit to unit `L²` norm. Leaves a numerically-zero vector as
/// `|0⟩` so downstream projections stay on-manifold rather than producing NaN.
pub(crate) fn normalize_qudit<const D: usize>(q: &[C32; D]) -> [C32; D] {
    let n2: f32 = q.iter().map(|z| z.norm_sq()).sum();
    if n2 < 1e-30 {
        let mut out = [C32::ZERO; D];
        out[0] = C32::ONE;
        return out;
    }
    let inv = n2.sqrt().recip();
    let mut out = *q;
    for z in out.iter_mut() {
        *z = z.scale(inv);
    }
    out
}

/// Algebraically-largest eigenpair of a `d×d` Hermitian matrix, via shifted power
/// iteration.
///
/// Returns `(eigenvector, eigenvalue)` where the eigenvalue is the Rayleigh
/// quotient `v†Kv` of the *unshifted* matrix.
///
/// # Why the shift
///
/// The memory kernel's Mattis weights `O_μ` can be negative, so `K` is indefinite
/// and plain power iteration would converge to the largest eigenvalue *by
/// magnitude* — potentially the most-negative one, which is the wrong end of the
/// Rayleigh quotient. Shifting by the Gershgorin radius `c = max_r Σ_c |K_rc|`
/// makes `K + cI` positive semi-definite without moving its eigenvectors, so
/// largest-magnitude and algebraically-largest coincide.
pub(crate) fn hermitian_top_eigenvector<const D: usize>(k: &[[C32; D]; D]) -> ([C32; D], f32) {
    let shift = (0..D)
        .map(|r| (0..D).map(|c| k[r][c].norm()).sum::<f32>())
        .fold(0.0f32, f32::max);

    let mut shifted = *k;
    for i in 0..D {
        shifted[i][i] = shifted[i][i].add(C32::real(shift));
    }

    let mut rng = SplitMix64::new(POWER_ITER_SEED);
    let mut v = [C32::ZERO; D];
    for z in v.iter_mut() {
        *z = C32::new(rng.next_f32_signed(), rng.next_f32_signed());
    }
    v = normalize_qudit(&v);

    let mut next = [C32::ZERO; D];
    for _ in 0..POWER_ITER_MAX {
        for r in 0..D {
            let mut acc = C32::ZERO;
            for c in 0..D {
                acc = acc.mul_add(shifted[r][c], v[c]);
            }
            next[r] = acc;
        }
        let norm_sq: f32 = next.iter().map(|z| z.norm_sq()).sum();
        if norm_sq < 1e-30 {
            // Shifted matrix annihilated the iterate — spectrum is numerically
            // zero, so any unit vector is a valid top eigenvector.
            break;
        }
        // Rayleigh quotient of the shifted matrix, reusing the matvec just done.
        let lambda: f32 = (0..D).map(|i| v[i].conj().mul(next[i]).re).sum();
        let residual_sq: f32 = (0..D)
            .map(|i| next[i].sub(v[i].scale(lambda)).norm_sq())
            .sum();
        v = normalize_qudit(&next);
        if residual_sq.sqrt() <= POWER_ITER_TOL * lambda.abs().max(1e-20) {
            break;
        }
    }

    // Rayleigh quotient against the ORIGINAL (unshifted) matrix.
    let mut eigenvalue = 0.0f32;
    for r in 0..D {
        let mut acc = C32::ZERO;
        for c in 0..D {
            acc = acc.mul_add(k[r][c], v[c]);
        }
        eigenvalue += v[r].conj().mul(acc).re;
    }
    (v, eigenvalue)
}
