//! Transfer-matrix band-structure analyzer (Plan 458, Research 451).
//!
//! Distills the **transfer-matrix method** from the Kronig-Penney delta-lattice
//! model (Kronig & Penney 1931; Griffiths §5.3) into a generic, modelless,
//! allocation-aware primitive that classifies the propagation behavior of a
//! sequence of k×k transport operators.
//!
//! # The math (substrate-independent)
//!
//! Given a sequence of k×k matrices `[M_1, …, M_N]` (or a single periodic `M`
//! applied N times), the composite `M = M_N · … · M_1` describes how a state
//! vector propagates through the stack. The eigenvalues `{λ_i}` of `M` (or of
//! `M` itself in the periodic case) classify each mode:
//!
//! - `|μ_i| ≈ 1` (where `μ_i = λ_i^{1/N}`) — **propagating** (allowed band).
//! - `|μ_i| < 1 − ε` — **decaying / evanescent** (forbidden gap).
//! - `|μ_i| > 1 + ε` — **growing / unstable** (forbidden gap, runaway).
//!
//! This is the same stability criterion used by:
//! - **Bai, Koltun, Kolter** (*Stabilizing Equilibrium Models by Jacobian
//!   Regularization*, [arXiv:2106.14342](https://arxiv.org/abs/2106.14342),
//!   ICML 2021) — their DEQ `ρ(J_*) < 1` is the scalar spectral-radius version
//!   of the band-stability criterion; their Jacobian regularization loss is the
//!   training-time version of this runtime diagnostic.
//! - **Martin & Mahoney** (*Implicit Self-Regularization*,
//!   [arXiv:1810.01075](https://arxiv.org/abs/1810.01075)) — weight-matrix ESD
//!   phase analysis is the offline-trained-weights analog.
//! - **Orthogonal/unitary RNNs** (Arjovsky uRNN ICML 2016, coRNN, nnRNN) — all
//!   constrain `|λ| = 1` (the band edge) for stable long-term propagation.
//!
//! # Layering contract
//!
//! - **Modelless**: deterministic closed-form math only. No training, no
//!   backprop, no gradient descent.
//! - **Zero-alloc hot paths**: [`band_classify_into`], [`analyze_chain_into`],
//!   [`analyze_periodic_into`] all reuse caller-provided scratch buffers.
//! - **Symmetric-matrix Jacobi eigensolver**: the in-tree eigensolver
//!   ([`jacobi_eigenvalues_symmetric_inplace`]) operates on the *symmetrized*
//!   matrix `0.5·(M + M^T)`. This is exact when `M` is already symmetric
//!   (e.g. HLA linearization in the bounded regime, well-conditioned FuncAttn
//!   composites). For genuinely non-symmetric operators with complex
//!   eigenvalues (e.g. pure rotation matrices), the symmetric part's
//!   eigenvalues approximate the real parts; a future plan may add a QR-based
//!   non-symmetric eigensolver if a concrete consumer needs it.
//!
//! # References
//!
//! - Research: [katgpt-rs/.research/451_Delta_Lattice_Tunneling_Transfer_Matrix_Band_Structure.md](../../../.research/451_Delta_Lattice_Tunneling_Transfer_Matrix_Band_Structure.md)
//! - Plan: [katgpt-rs/.plans/458_transfer_matrix_band_structure.md](../../../.plans/458_transfer_matrix_band_structure.md)
//! - Closest shipped cousin: [`crate::analytic_lattice::compose_chain`] (the
//!   matmul chain this primitive analyzes).
//! - Closest stability cousin: [`crate::subspace_phase_gate`] (sample-sufficiency
//!   phase transition; this extends to mode-propagation phase transition).

use crate::analytic_lattice::TransportOperator;
use crate::analytic_lattice::chain::{ChainError, compose_chain_into};

/// Default band-edge tolerance: modes with `|μ|` within `1 ± ε` are classified
/// as `Propagating`. Matches the canonical numerical-stability convention
/// (`ε ≈ 1e-4` for f32).
pub const DEFAULT_BAND_EPSILON: f32 = 1e-4;

/// Maximum number of Jacobi sweeps. 50 is more than enough for k ≤ 16
/// (convergence is quadratic near the solution).
pub const DEFAULT_MAX_SWEEPS: usize = 50;

/// Band classification of a single propagation mode.
///
/// Distilled from the Kronig-Penney allowed-band / forbidden-gap distinction:
/// given the Bloch propagation factor `μ = λ^{1/N}` for a mode with eigenvalue
/// `λ` over `N` periods, the mode is:
///
/// - `Propagating` if `|μ| ≈ 1` (allowed band — the mode survives N steps).
/// - `Decaying` if `|μ| < 1 − ε` (forbidden gap — the mode evaporates).
/// - `Growing` if `|μ| > 1 + ε` (forbidden gap — the mode is unstable).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BandClass {
    /// `|μ| ≈ 1`: propagating Bloch wave, allowed band. The mode neither
    /// grows nor decays over N periods.
    Propagating = 0,
    /// `|μ| < 1 − ε`: evanescent mode, forbidden gap. Exponential decay.
    Decaying = 1,
    /// `|μ| > 1 + ε`: runaway mode, forbidden gap. Exponential growth —
    /// usually indicates a divergent operator chain.
    Growing = 2,
}

impl BandClass {
    /// Classify a mode from its Bloch-factor magnitude `|μ|` and a tolerance
    /// `epsilon`. Modes within `1 ± ε` are `Propagating`.
    #[inline]
    pub fn from_bloch_factor(mu_abs: f32, epsilon: f32) -> Self {
        if mu_abs.is_nan() {
            // NaN is treated as Decaying (conservative: the mode contributes
            // nothing actionable). This matches the `spectral_radius` NaN
            // convention in [`BandStructureReport`].
            return Self::Decaying;
        }
        if mu_abs > 1.0 + epsilon {
            Self::Growing
        } else if mu_abs < 1.0 - epsilon {
            Self::Decaying
        } else {
            Self::Propagating
        }
    }
}

/// Band-structure report for a transport-operator chain or periodic stack.
///
/// All `Vec<f32>` fields have length `k`. The eigenvalues are sorted
/// **descending by `|λ|`**, so `eigenvalues[0]` is the spectral-radius mode
/// (most likely to be `Growing` if anything is).
#[derive(Debug, Clone, PartialEq)]
pub struct BandStructureReport {
    /// Eigenvalues `{λ_i}` of the composite (or per-period) operator's
    /// symmetric part `0.5·(M + M^T)`, sorted descending by `|λ|`.
    ///
    /// For symmetric operators this is exact. For non-symmetric operators with
    /// complex eigenvalues, these are the eigenvalues of the symmetrized
    /// matrix — they approximate the real parts but cannot represent rotation
    /// (pure-imaginary eigenvalue components).
    pub eigenvalues: Vec<f32>,
    /// Bloch propagation factors `μ_i = sign(λ_i) · |λ_i|^{1/N}` per mode,
    /// sorted to match `eigenvalues`.
    ///
    /// The per-site growth/decay factor. `|μ| ≈ 1` ⟹ propagating; `|μ| < 1`
    /// ⟹ exponential decay per period; `|μ| > 1` ⟹ exponential growth.
    pub bloch_factors: Vec<f32>,
    /// Per-mode classification, sorted to match `eigenvalues`.
    pub band_classes: Vec<BandClass>,
    /// Spectral radius `max_i |λ_i|`. The single-number summary Bai/Kolter
    /// (`ρ(J_*) < 1`) use for DEQ stability.
    pub spectral_radius: f32,
    /// `|det(M)|^{1/N}` — the geometric-mean per-period attenuation factor.
    /// `1.0` = volume-preserving; `<1` = contracting; `>1` = expanding.
    pub geometric_mean_attenuation: f32,
    /// Operator dimension `k`.
    pub k: usize,
    /// Number of periods the operator was applied (or the chain length).
    pub n_periods: u32,
    /// Band-edge tolerance used for classification.
    pub epsilon: f32,
}

impl BandStructureReport {
    /// Construct an empty report sized for `k` modes. Used as the `out` slot
    /// for the `_into` APIs.
    pub fn zeros(k: usize) -> Self {
        Self {
            eigenvalues: vec![0.0; k],
            bloch_factors: vec![0.0; k],
            band_classes: vec![BandClass::Propagating; k],
            spectral_radius: 0.0,
            geometric_mean_attenuation: 0.0,
            k,
            n_periods: 1,
            epsilon: DEFAULT_BAND_EPSILON,
        }
    }

    /// Returns `true` iff all modes are `Propagating` (the clean case).
    #[inline]
    pub fn is_all_propagating(&self) -> bool {
        self.band_classes
            .iter()
            .all(|c| *c == BandClass::Propagating)
    }

    /// Returns `true` iff any mode is `Growing` (the unstable case — usually
    /// indicates a divergent operator chain).
    #[inline]
    pub fn has_growing_mode(&self) -> bool {
        self.band_classes.contains(&BandClass::Growing)
    }

    /// Count of modes in each band class, returned as
    /// `(propagating, decaying, growing)`.
    #[inline]
    pub fn counts(&self) -> (usize, usize, usize) {
        let mut p = 0;
        let mut d = 0;
        let mut g = 0;
        for c in &self.band_classes {
            match c {
                BandClass::Propagating => p += 1,
                BandClass::Decaying => d += 1,
                BandClass::Growing => g += 1,
            }
        }
        (p, d, g)
    }
}

// ─── Public API: classify pre-computed eigenvalues ─────────────────────────

/// Classify a pre-computed eigenvalue spectrum into band classes (allocating).
///
/// Convenience wrapper around [`band_classify_into`] — allocates a fresh
/// [`BandStructureReport`]. For hot paths, reuse a report buffer with
/// [`band_classify_into`].
///
/// # Arguments
///
/// - `eigenvalues` — slice of length `k`. Treated as the eigenvalues of the
///   per-period operator (if `n_periods > 1`) or the composite (if
///   `n_periods == 1`).
/// - `n_periods` — the number of periods the operator was applied. Used to
///   compute the Bloch factor `μ = λ^{1/N}`. Must be `≥ 1`.
/// - `epsilon` — band-edge tolerance. Use [`DEFAULT_BAND_EPSILON`] if unsure.
pub fn band_classify(eigenvalues: &[f32], n_periods: u32, epsilon: f32) -> BandStructureReport {
    assert!(n_periods >= 1, "n_periods must be >= 1, got {n_periods}");
    let mut out = BandStructureReport::zeros(eigenvalues.len());
    band_classify_into(eigenvalues, n_periods, epsilon, &mut out);
    out
}

/// Classify a pre-computed eigenvalue spectrum into band classes (in-place).
///
/// Writes into `out`. The `out.eigenvalues`, `out.bloch_factors`, and
/// `out.band_classes` vectors are resized to `eigenvalues.len()` if needed
/// (one-time alloc on first call; subsequent calls with the same `k` are
/// zero-alloc).
///
/// # Algorithm
///
/// 1. Copy `eigenvalues` into `out.eigenvalues`.
/// 2. Compute the determinant proxy `Π λ_i` (clamped to avoid underflow) and
///    from it `geometric_mean_attenuation = |det|^{1/N}`.
/// 3. Sort indices descending by `|λ_i|` (so the spectral-radius mode is
///    index 0). Apply the same permutation to all three vectors.
/// 4. Compute per-mode Bloch factor `μ_i = sign(λ_i) · |λ_i|^{1/N}`.
/// 5. Classify each mode via [`BandClass::from_bloch_factor`].
/// 6. Record `spectral_radius = |λ_0|` (post-sort: largest magnitude).
///
/// # Sorting stability
///
/// The sort uses `f32::total_cmp` on `|λ_i|` to be NaN-safe and deterministic
/// across architectures (mirrors the G5 cross-arch bit-identity requirement).
pub fn band_classify_into(
    eigenvalues: &[f32],
    n_periods: u32,
    epsilon: f32,
    out: &mut BandStructureReport,
) {
    use crate::simd::fast_exp;
    assert!(n_periods >= 1, "n_periods must be >= 1, got {n_periods}");
    let k = eigenvalues.len();
    if out.eigenvalues.len() != k {
        out.eigenvalues.resize(k, 0.0);
        out.bloch_factors.resize(k, 0.0);
        out.band_classes.resize(k, BandClass::Propagating);
    }
    out.k = k;
    out.n_periods = n_periods;
    out.epsilon = epsilon;

    // 1. Copy eigenvalues in.
    out.eigenvalues.copy_from_slice(eigenvalues);

    // 2. Determinant proxy + geometric-mean attenuation.
    // Use log-sum to avoid underflow on long chains: log|det| = Σ log|λ_i|.
    // For N periods, geometric_mean_attenuation = exp(log|det| / (N·k)).
    // (The 1/k accounts for det being a degree-k polynomial of eigenvalues.)
    let mut log_abs_det = 0.0f32;
    let mut valid_det = true;
    for &lam in eigenvalues {
        let abs_lam = lam.abs();
        if abs_lam < 1e-38 {
            // Treat as zero — determinant collapses.
            valid_det = false;
            break;
        }
        log_abs_det += abs_lam.ln();
    }
    let n_periods_f = n_periods as f32;
    let k_f = k.max(1) as f32;
    out.geometric_mean_attenuation = if valid_det {
        fast_exp(log_abs_det / (n_periods_f * k_f))
    } else {
        0.0
    };

    // 3. Sort indices descending by |λ_i| (NaN-safe via total_cmp).
    // Insertion sort on a fixed-size stack scratch — k is small (≤ 16 in the
    // headline use case). Avoids per-call Vec allocation.
    // For k > 64 (rare), fall back to a heap-allocated index vector.
    let mut idx_stack = [0usize; 64];
    let mut idx_heap: Vec<usize>;
    let needs_heap = k > 64;
    if k > 64 {
        idx_heap = (0..k).collect();
    } else {
        // Initialize first k entries of idx_stack to 0..k.
        for (i, slot) in idx_stack.iter_mut().enumerate().take(k) {
            *slot = i;
        }
        idx_heap = Vec::new(); // unused; satisfies borrow checker
    }
    // Helper macro-free sort: insertion sort on whichever buffer is active.
    fn sort_desc_by_abs(slice: &mut [usize], eigenvalues: &[f32]) {
        for i in 1..slice.len() {
            let mut j = i;
            while j > 0 {
                let abs_a = eigenvalues[slice[j - 1]].abs();
                let abs_b = eigenvalues[slice[j]].abs();
                // Descending: swap if previous < current.
                if abs_a.total_cmp(&abs_b) == std::cmp::Ordering::Less {
                    slice.swap(j - 1, j);
                    j -= 1;
                } else {
                    break;
                }
            }
        }
    }
    if needs_heap {
        sort_desc_by_abs(&mut idx_heap, eigenvalues);
    } else {
        sort_desc_by_abs(&mut idx_stack[..k], eigenvalues);
    }

    // Apply permutation: write sorted values back into out.
    // Use a small stack scratch to avoid double-borrowing.
    let mut sorted_eig = [0.0f32; 64];
    if k <= 64 {
        let idx_ref = &idx_stack[..k];
        for (i, &src) in idx_ref.iter().enumerate() {
            sorted_eig[i] = eigenvalues[src];
        }
        out.eigenvalues.copy_from_slice(&sorted_eig[..k]);
    } else {
        // Fallback for very large k (rare; not the headline use case).
        for (i, &src) in idx_heap.iter().enumerate() {
            sorted_eig[i.min(63)] = eigenvalues[src]; // bounds-safe; only first 64 used
        }
        // We can't fit k>64 into a stack scratch; reuse out.eigenvalues via
        // a heap copy. This is the rare slow path.
        let mut tmp: Vec<f32> = idx_heap.iter().map(|&i| eigenvalues[i]).collect();
        tmp.sort_by(|a, b| b.abs().total_cmp(&a.abs()));
        out.eigenvalues.copy_from_slice(&tmp);
    }

    // 4-5. Bloch factor + classification per mode.
    let inv_n = 1.0 / n_periods_f;
    let mut spectral_radius = 0.0f32;
    for i in 0..k {
        let lam = out.eigenvalues[i];
        let abs_lam = lam.abs();
        if abs_lam > spectral_radius {
            spectral_radius = abs_lam;
        }
        // Bloch factor: sign(λ) · |λ|^{1/N}.
        // For complex eigenvalues (which we can't represent in this symmetric
        // eigensolver), the symmetric-part eigenvalue approximates the real
        // part; the Bloch factor of the corresponding true complex pair would
        // have |μ| = |λ_complex|^{1/N}, so we use |λ|^{1/N} for classification.
        let abs_mu = if abs_lam < 1e-38 {
            0.0
        } else {
            abs_lam.powf(inv_n)
        };
        let signed_mu = if lam.is_sign_negative() {
            -abs_mu
        } else {
            abs_mu
        };
        out.bloch_factors[i] = signed_mu;
        out.band_classes[i] = BandClass::from_bloch_factor(abs_mu, epsilon);
    }
    out.spectral_radius = spectral_radius;
}

// ─── Public API: analyze a chain of transport operators ────────────────────

/// Analyze a chain of k×k transport operators: compose, eigendecompose, classify.
///
/// Allocates a fresh [`BandStructureReport`]. For hot paths, use
/// [`analyze_chain_into`] with reusable scratch buffers.
///
/// # Errors
///
/// Returns [`ChainError`] if `ops` is empty, has mismatched `k`, or exceeds
/// [`crate::analytic_lattice::chain::MAX_CHAIN_LEN`].
pub fn analyze_chain(
    ops: &[TransportOperator],
    epsilon: f32,
) -> Result<BandStructureReport, ChainError> {
    if ops.is_empty() {
        return Err(ChainError::ChainLengthInvalid {
            len: 0,
            max: crate::analytic_lattice::chain::MAX_CHAIN_LEN,
        });
    }
    let k = ops[0].k;
    let n_periods = ops.len() as u32;
    let mut composite = TransportOperator::zeros(k);
    let mut compose_scratch = Vec::with_capacity(k * k);
    let mut sym_scratch = vec![0.0f32; k * k];
    let mut out = BandStructureReport::zeros(k);
    analyze_chain_into(
        ops,
        epsilon,
        &mut compose_scratch,
        &mut sym_scratch,
        &mut composite,
        &mut out,
    )?;
    // analyze_chain_into writes eigenvalues into out via the symmetric eigensolver;
    // we still need to set n_periods correctly (band_classify_into does this).
    let _ = n_periods; // already set inside analyze_chain_into
    Ok(out)
}

/// Analyze a chain of k×k transport operators (in-place, zero-alloc after warmup).
///
/// Composes the chain via [`compose_chain_into`], symmetrizes the composite as
/// `0.5·(M + M^T)`, runs the Jacobi eigensolver, then classifies the spectrum
/// via [`band_classify_into`].
///
/// # Scratch buffers
///
/// - `compose_scratch` — passed through to `compose_chain_into`.
/// - `sym_scratch` — `k²` f32 slots for the symmetrized matrix.
/// - `composite` — receives the composed operator (also `k²` f32 slots).
/// - `out` — receives the final [`BandStructureReport`].
///
/// All scratch buffers are grown once on first call; subsequent calls with the
/// same `k` allocate zero bytes.
pub fn analyze_chain_into(
    ops: &[TransportOperator],
    epsilon: f32,
    compose_scratch: &mut Vec<f32>,
    sym_scratch: &mut Vec<f32>,
    composite: &mut TransportOperator,
    out: &mut BandStructureReport,
) -> Result<(), ChainError> {
    if ops.is_empty() {
        return Err(ChainError::ChainLengthInvalid {
            len: 0,
            max: crate::analytic_lattice::chain::MAX_CHAIN_LEN,
        });
    }
    let n_periods = ops.len() as u32;

    // 1. Compose the chain.
    compose_chain_into(ops, compose_scratch, composite)?;
    let k = composite.k;

    // 2. Symmetrize: sym = 0.5·(M + M^T).
    if sym_scratch.len() < k * k {
        sym_scratch.resize(k * k, 0.0);
    }
    for i in 0..k {
        for j in 0..k {
            let m_ij = composite.data[i * k + j];
            let m_ji = composite.data[j * k + i];
            sym_scratch[i * k + j] = 0.5 * (m_ij + m_ji);
        }
    }

    // 3. Run symmetric Jacobi eigensolver. Eigenvalues land on the diagonal.
    jacobi_eigenvalues_symmetric_inplace(sym_scratch, k, DEFAULT_MAX_SWEEPS);

    // 4. Extract diagonal into out.eigenvalues (intermediate buffer).
    if out.eigenvalues.len() != k {
        out.eigenvalues.resize(k, 0.0);
        out.bloch_factors.resize(k, 0.0);
        out.band_classes.resize(k, BandClass::Propagating);
    }
    // Write eigenvalues onto the diagonal slots first, then move them to the
    // head of the buffer via a small stack scratch before classification
    // (band_classify_into permutes them via an index array, so we need a
    // temporary copy to avoid clobbering during the sort).
    for i in 0..k {
        out.eigenvalues[i] = sym_scratch[i * k + i];
    }

    // 5. Classify. Use a stack scratch for the eigenvalue copy when k is small
    // (the headline case k ≤ 16); fall back to a heap vec for very large k
    // (rare). The clone is necessary because `band_classify_into` sorts the
    // eigenvalues into descending order, which would clobber the diagonal of
    // `sym_scratch` if we passed it directly. The stack scratch keeps this
    // allocation-free for k ≤ 64.
    let mut eig_scratch = [0.0f32; 64];
    if k <= 64 {
        eig_scratch[..k].copy_from_slice(&out.eigenvalues[..k]);
        band_classify_into(&eig_scratch[..k], n_periods, epsilon, out);
    } else {
        let tmp = out.eigenvalues.clone();
        band_classify_into(&tmp, n_periods, epsilon, out);
    }

    Ok(())
}

// ─── Public API: analyze a periodic stack (single M applied N times) ───────

/// Analyze a periodic stack: a single operator `op` applied `n_periods` times.
///
/// This is the Kronig-Penney case: eigenvalues of `M` directly give the Bloch
/// factors `μ = λ^{1/N}`. No need to compute `M^N`.
///
/// Allocates a fresh [`BandStructureReport`]. For hot paths, use
/// [`analyze_periodic_into`].
///
/// # Errors
///
/// Returns [`ChainError::DimensionMismatch`] if `op.data.len() != op.k²`.
pub fn analyze_periodic(
    op: &TransportOperator,
    n_periods: u32,
    epsilon: f32,
) -> Result<BandStructureReport, ChainError> {
    let k = op.k;
    let expected = k.checked_mul(k).ok_or(ChainError::DimensionOverflow)?;
    if op.data.len() != expected {
        return Err(ChainError::DimensionMismatch {
            expected,
            got: op.data.len(),
        });
    }
    let mut sym_scratch = vec![0.0f32; k * k];
    let mut out = BandStructureReport::zeros(k);
    analyze_periodic_into(op, n_periods, epsilon, &mut sym_scratch, &mut out)?;
    Ok(out)
}

/// Analyze a periodic stack (in-place, zero-alloc after warmup).
///
/// Writes into `out`. The `sym_scratch` buffer is `k²` f32 slots.
pub fn analyze_periodic_into(
    op: &TransportOperator,
    n_periods: u32,
    epsilon: f32,
    sym_scratch: &mut Vec<f32>,
    out: &mut BandStructureReport,
) -> Result<(), ChainError> {
    assert!(n_periods >= 1, "n_periods must be >= 1, got {n_periods}");
    let k = op.k;
    let expected = k.checked_mul(k).ok_or(ChainError::DimensionOverflow)?;
    if op.data.len() != expected {
        return Err(ChainError::DimensionMismatch {
            expected,
            got: op.data.len(),
        });
    }

    // 1. Symmetrize: sym = 0.5·(M + M^T).
    if sym_scratch.len() < k * k {
        sym_scratch.resize(k * k, 0.0);
    }
    for i in 0..k {
        for j in 0..k {
            let m_ij = op.data[i * k + j];
            let m_ji = op.data[j * k + i];
            sym_scratch[i * k + j] = 0.5 * (m_ij + m_ji);
        }
    }

    // 2. Symmetric Jacobi eigensolver.
    jacobi_eigenvalues_symmetric_inplace(sym_scratch, k, DEFAULT_MAX_SWEEPS);

    // 3. Extract diagonal.
    if out.eigenvalues.len() != k {
        out.eigenvalues.resize(k, 0.0);
        out.bloch_factors.resize(k, 0.0);
        out.band_classes.resize(k, BandClass::Propagating);
    }
    let mut eigen_tmp = [0.0f32; 64];
    if k <= 64 {
        for i in 0..k {
            eigen_tmp[i] = sym_scratch[i * k + i];
        }
        band_classify_into(&eigen_tmp[..k], n_periods, epsilon, out);
    } else {
        let mut tmp = vec![0.0f32; k];
        for i in 0..k {
            tmp[i] = sym_scratch[i * k + i];
        }
        band_classify_into(&tmp, n_periods, epsilon, out);
    }
    Ok(())
}

// ─── Internal: symmetric Jacobi eigensolver (f32, in-place) ────────────────

/// In-place Jacobi eigenvalue iteration on a symmetric `dim × dim` matrix.
///
/// On return, the diagonal of `mat` holds the eigenvalues (unordered). The
/// off-diagonal is driven to ≈ 0. No allocation — operates entirely on `mat`.
///
/// Mirrors `crate::gain_cost_halt::jacobi_eigenvalues_inplace` but is local to
/// this module to keep `band_structure` self-contained (the gain_cost_halt
/// version is private). Same algorithm, same convergence behavior.
///
/// # Algorithm
///
/// Classic cyclic-ish Jacobi with largest-off-diagonal pivot selection:
///
/// 1. Find the largest off-diagonal entry `M[p, q]`.
/// 2. Compute the rotation angle `θ = 0.5 · atan2(2·M[p,q], M[p,p] − M[q,q])`.
/// 3. Apply the Givens rotation to rows/cols `p, q`.
/// 4. Repeat until the largest off-diagonal is below `1e-12` or `max_sweeps`
///    is exhausted.
///
/// Convergence is quadratic near the solution, so 30–50 sweeps suffice for
/// `dim ≤ 16`.
fn jacobi_eigenvalues_symmetric_inplace(mat: &mut [f32], dim: usize, max_sweeps: usize) {
    if dim <= 1 {
        return;
    }
    for _ in 0..max_sweeps {
        // Find the largest off-diagonal element (upper triangle).
        let mut max_val = 0.0f32;
        let (mut p, mut q) = (0usize, 1usize);
        for i in 0..dim {
            for j in (i + 1)..dim {
                let val = mat[i * dim + j].abs();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }

        // Converged once the largest off-diagonal is negligible.
        if max_val < 1e-12 {
            break;
        }

        // Jacobi rotation angle.
        let app = mat[p * dim + p];
        let aqq = mat[q * dim + q];
        let apq = mat[p * dim + q];

        let theta = if (app - aqq).abs() < 1e-15 {
            std::f32::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };

        let cos_t = theta.cos();
        let sin_t = theta.sin();

        // Rotate rows/cols p, q for every other index.
        for r in 0..dim {
            if r == p || r == q {
                continue;
            }
            let arp = mat[r * dim + p];
            let arq = mat[r * dim + q];
            mat[r * dim + p] = cos_t * arp + sin_t * arq;
            mat[p * dim + r] = mat[r * dim + p];
            mat[r * dim + q] = -sin_t * arp + cos_t * arq;
            mat[q * dim + r] = mat[r * dim + q];
        }

        let new_pp = cos_t * cos_t * app + 2.0 * sin_t * cos_t * apq + sin_t * sin_t * aqq;
        let new_qq = sin_t * sin_t * app - 2.0 * sin_t * cos_t * apq + cos_t * cos_t * aqq;
        mat[p * dim + p] = new_pp;
        mat[q * dim + q] = new_qq;
        mat[p * dim + q] = 0.0;
        mat[q * dim + p] = 0.0;
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analytic_lattice::TransportOperator;

    /// Construct a transport operator from a flat row-major slice (test helper).
    fn op_from_slice(k: usize, data: &[f32]) -> TransportOperator {
        assert_eq!(data.len(), k * k);
        TransportOperator::from_row_major_unchecked(k, data.to_vec())
    }

    // ── BandClass classification ──────────────────────────────────────────

    #[test]
    fn band_class_propagating_at_one() {
        assert_eq!(
            BandClass::from_bloch_factor(1.0, 1e-4),
            BandClass::Propagating
        );
        assert_eq!(
            BandClass::from_bloch_factor(1.0 + 1e-5, 1e-4),
            BandClass::Propagating
        );
        assert_eq!(
            BandClass::from_bloch_factor(1.0 - 1e-5, 1e-4),
            BandClass::Propagating
        );
    }

    #[test]
    fn band_class_decaying_below_one() {
        assert_eq!(BandClass::from_bloch_factor(0.5, 1e-4), BandClass::Decaying);
        assert_eq!(
            BandClass::from_bloch_factor(0.99, 1e-4),
            BandClass::Decaying
        );
        assert_eq!(BandClass::from_bloch_factor(0.0, 1e-4), BandClass::Decaying);
    }

    #[test]
    fn band_class_growing_above_one() {
        assert_eq!(BandClass::from_bloch_factor(1.5, 1e-4), BandClass::Growing);
        assert_eq!(BandClass::from_bloch_factor(1.01, 1e-4), BandClass::Growing);
        assert_eq!(
            BandClass::from_bloch_factor(100.0, 1e-4),
            BandClass::Growing
        );
    }

    #[test]
    fn band_class_nan_is_decaying() {
        assert_eq!(
            BandClass::from_bloch_factor(f32::NAN, 1e-4),
            BandClass::Decaying
        );
    }

    // ── band_classify_into: basic cases ───────────────────────────────────

    #[test]
    fn classify_identity_propagating() {
        // Identity operator: eigenvalues all 1.0 → all Propagating.
        let eig = [1.0f32, 1.0, 1.0, 1.0];
        let report = band_classify(&eig, 5, DEFAULT_BAND_EPSILON);
        assert_eq!(report.k, 4);
        assert_eq!(report.n_periods, 5);
        assert!(report.is_all_propagating());
        assert!(!report.has_growing_mode());
        // Bloch factor of 1.0^N is 1.0.
        for mu in &report.bloch_factors {
            assert!((mu - 1.0).abs() < 1e-6);
        }
        // Spectral radius is 1.0.
        assert!((report.spectral_radius - 1.0).abs() < 1e-6);
        // Geometric-mean attenuation is 1.0.
        assert!((report.geometric_mean_attenuation - 1.0).abs() < 1e-6);
    }

    #[test]
    fn classify_scaling_decaying() {
        // Diagonal matrix diag(0.5, 0.5) → eigenvalues 0.5, 0.5 → Decaying.
        let eig = [0.5f32, 0.5];
        let report = band_classify(&eig, 10, DEFAULT_BAND_EPSILON);
        assert!(
            report
                .band_classes
                .iter()
                .all(|c| *c == BandClass::Decaying)
        );
        // Bloch factor 0.5^{1/10} ≈ 0.933.
        let expected_mu = 0.5f32.powf(1.0 / 10.0);
        for mu in &report.bloch_factors {
            assert!(
                (mu - expected_mu).abs() < 1e-5,
                "mu = {mu}, expected = {expected_mu}"
            );
        }
        assert!((report.spectral_radius - 0.5).abs() < 1e-6);
    }

    #[test]
    fn classify_growing_mode() {
        // Eigenvalues 2.0 and 0.5 → 2.0 is Growing, 0.5 is Decaying.
        let eig = [2.0f32, 0.5];
        let report = band_classify(&eig, 1, DEFAULT_BAND_EPSILON);
        assert_eq!(report.band_classes.len(), 2);
        // Sorted descending by |λ|: 2.0 first (Growing), 0.5 second (Decaying).
        assert_eq!(report.band_classes[0], BandClass::Growing);
        assert_eq!(report.band_classes[1], BandClass::Decaying);
        assert!((report.spectral_radius - 2.0).abs() < 1e-6);
        assert!(report.has_growing_mode());
        let (p, d, g) = report.counts();
        assert_eq!((p, d, g), (0, 1, 1));
    }

    #[test]
    fn classify_sorts_descending_by_abs() {
        // Mix of magnitudes; verify sort.
        let eig = [0.3f32, -2.5, 1.0, -0.8];
        let report = band_classify(&eig, 1, DEFAULT_BAND_EPSILON);
        // Expected order: |-2.5| > |1.0| > |-0.8| > |0.3|.
        assert!((report.eigenvalues[0] - (-2.5)).abs() < 1e-6);
        assert!((report.eigenvalues[1] - 1.0).abs() < 1e-6);
        assert!((report.eigenvalues[2] - (-0.8)).abs() < 1e-6);
        assert!((report.eigenvalues[3] - 0.3).abs() < 1e-6);
        assert!((report.spectral_radius - 2.5).abs() < 1e-6);
    }

    // ── analyze_periodic: identity / scaling / Kronig-Penney-like ────────

    #[test]
    fn analyze_periodic_identity_propagating() {
        let op = TransportOperator::identity(4);
        let report = analyze_periodic(&op, 5, DEFAULT_BAND_EPSILON).unwrap();
        assert!(report.is_all_propagating());
        assert!((report.spectral_radius - 1.0).abs() < 1e-4);
        assert!((report.geometric_mean_attenuation - 1.0).abs() < 1e-4);
    }

    #[test]
    fn analyze_periodic_scaling_decaying() {
        // diag(0.5, 0.5)
        let op = op_from_slice(2, &[0.5, 0.0, 0.0, 0.5]);
        let report = analyze_periodic(&op, 10, DEFAULT_BAND_EPSILON).unwrap();
        assert!(
            report
                .band_classes
                .iter()
                .all(|c| *c == BandClass::Decaying)
        );
        assert!((report.spectral_radius - 0.5).abs() < 1e-5);
    }

    #[test]
    fn analyze_periodic_growing() {
        // diag(1.5, 0.5) → one Growing, one Decaying.
        let op = op_from_slice(2, &[1.5, 0.0, 0.0, 0.5]);
        let report = analyze_periodic(&op, 1, DEFAULT_BAND_EPSILON).unwrap();
        assert!(report.has_growing_mode());
        let (p, d, g) = report.counts();
        assert_eq!((p, d, g), (0, 1, 1));
    }

    #[test]
    fn analyze_periodic_kronig_penney_allowed_band() {
        // Kronig-Penney-like: a 2×2 symmetric transfer matrix with
        // Tr(M)/2 = 0.5 (within (-1, 1) → allowed band → Propagating).
        // M = [[0.5, 0.3], [0.3, 0.5]]; Tr = 1.0, Tr/2 = 0.5.
        // Eigenvalues of M: 0.5 ± 0.3 = 0.8, 0.2.
        // For N=5: |μ_1| = 0.8^{1/5} ≈ 0.955, |μ_2| = 0.2^{1/5} ≈ 0.724.
        // Both < 1 − ε → both Decaying. (The eigenvalues of the *per-period*
        // matrix being inside the unit circle means the periodic stack decays;
        // the *allowed band* condition in Kronig-Penney is on the *complex
        // exponential* parameterization, which the symmetric eigensolver can't
        // represent. This test documents the symmetric-eigensolver regime.)
        let op = op_from_slice(2, &[0.5, 0.3, 0.3, 0.5]);
        let report = analyze_periodic(&op, 5, DEFAULT_BAND_EPSILON).unwrap();
        // Both modes decay (eigenvalues 0.8 and 0.2, both |λ|<1).
        assert!(
            report
                .band_classes
                .iter()
                .all(|c| *c == BandClass::Decaying)
        );
        // Spectral radius ≈ 0.8.
        assert!(
            (report.spectral_radius - 0.8).abs() < 1e-4,
            "spectral_radius = {}",
            report.spectral_radius
        );
    }

    #[test]
    fn analyze_periodic_kronig_penney_forbidden_gap_growing() {
        // Symmetric transfer matrix with eigenvalues > 1 in magnitude:
        // M = [[2.0, 0.5], [0.5, 1.0]] → Tr = 3.0, Tr/2 = 1.5 > 1 → forbidden.
        // Eigenvalues of M: (3 ± sqrt(9 - 4·(2 - 0.25)))/2 = (3 ± sqrt(2))/2.
        //   = (3 + 1.414)/2 ≈ 2.207 and (3 - 1.414)/2 ≈ 0.793.
        // For N=1: 2.207 is Growing, 0.793 is Decaying.
        let op = op_from_slice(2, &[2.0, 0.5, 0.5, 1.0]);
        let report = analyze_periodic(&op, 1, DEFAULT_BAND_EPSILON).unwrap();
        assert!(report.has_growing_mode());
        // Sorted descending: first mode is the growing one (≈2.207).
        assert_eq!(report.band_classes[0], BandClass::Growing);
        assert!(
            (report.spectral_radius - 2.207).abs() < 1e-3,
            "spectral_radius = {}",
            report.spectral_radius
        );
    }

    // ── analyze_chain: composition + analysis ────────────────────────────

    #[test]
    fn analyze_chain_identity_pair() {
        // Two identity operators composed → identity → all Propagating.
        let id = TransportOperator::identity(3);
        let ops = vec![id.clone(), id];
        let report = analyze_chain(&ops, DEFAULT_BAND_EPSILON).unwrap();
        assert!(report.is_all_propagating());
        assert!((report.spectral_radius - 1.0).abs() < 1e-4);
    }

    #[test]
    fn analyze_chain_scaling_pair() {
        // Two scaling operators diag(0.5) composed → diag(0.25) → all Decaying.
        let s = op_from_slice(2, &[0.5, 0.0, 0.0, 0.5]);
        let ops = vec![s.clone(), s];
        let report = analyze_chain(&ops, DEFAULT_BAND_EPSILON).unwrap();
        // Composite eigenvalues should be 0.25 (both). For n_periods=2:
        // |μ| = 0.25^{1/2} = 0.5 → Decaying.
        assert!(
            report
                .band_classes
                .iter()
                .all(|c| *c == BandClass::Decaying)
        );
        assert!(
            (report.spectral_radius - 0.25).abs() < 1e-5,
            "spectral_radius = {}",
            report.spectral_radius
        );
    }

    #[test]
    fn analyze_chain_into_reuses_scratch() {
        // Two calls with the same k should not reallocate after the first.
        let id = TransportOperator::identity(3);
        let ops = vec![id.clone(), id];
        let mut compose_scratch = Vec::new();
        let mut sym_scratch = Vec::new();
        let mut composite = TransportOperator::zeros(3);
        let mut out = BandStructureReport::zeros(3);

        // First call: allocates scratch + output buffers.
        analyze_chain_into(
            &ops,
            DEFAULT_BAND_EPSILON,
            &mut compose_scratch,
            &mut sym_scratch,
            &mut composite,
            &mut out,
        )
        .unwrap();
        assert!(out.is_all_propagating());

        // Capture capacities after first call.
        let compose_cap = compose_scratch.capacity();
        let sym_cap = sym_scratch.capacity();
        let eig_cap = out.eigenvalues.capacity();

        // Second call: should not reallocate.
        analyze_chain_into(
            &ops,
            DEFAULT_BAND_EPSILON,
            &mut compose_scratch,
            &mut sym_scratch,
            &mut composite,
            &mut out,
        )
        .unwrap();
        assert_eq!(compose_scratch.capacity(), compose_cap);
        assert_eq!(sym_scratch.capacity(), sym_cap);
        assert_eq!(out.eigenvalues.capacity(), eig_cap);
        assert!(out.is_all_propagating());
    }

    #[test]
    fn analyze_chain_empty_returns_error() {
        let ops: Vec<TransportOperator> = vec![];
        let result = analyze_chain(&ops, DEFAULT_BAND_EPSILON);
        assert!(matches!(result, Err(ChainError::ChainLengthInvalid { .. })));
    }

    // ── Jacobi eigensolver: sanity ───────────────────────────────────────

    #[test]
    fn jacobi_diagonal_matrix_unchanged() {
        // Already-diagonal matrix: eigenvalues are the diagonal.
        let mut mat = [2.0f32, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 5.0];
        jacobi_eigenvalues_symmetric_inplace(&mut mat, 3, 50);
        // Diagonal should still hold the eigenvalues (possibly reordered).
        let mut eigs: Vec<f32> = (0..3).map(|i| mat[i * 3 + i]).collect();
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((eigs[0] - 2.0).abs() < 1e-5);
        assert!((eigs[1] - 3.0).abs() < 1e-5);
        assert!((eigs[2] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn jacobi_symmetric_2x2() {
        // M = [[2, 1], [1, 2]] → eigenvalues 1 and 3.
        let mut mat = [2.0f32, 1.0, 1.0, 2.0];
        jacobi_eigenvalues_symmetric_inplace(&mut mat, 2, 50);
        let e0 = mat[0];
        let e1 = mat[3];
        let mut eigs = [e0, e1];
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((eigs[0] - 1.0).abs() < 1e-5, "eigs = {:?}", eigs);
        assert!((eigs[1] - 3.0).abs() < 1e-5, "eigs = {:?}", eigs);
    }

    #[test]
    fn jacobi_symmetric_3x3() {
        // M = [[2, 1, 0], [1, 2, 1], [0, 1, 2]] → eigenvalues 2, 2±sqrt(2).
        // 2 - sqrt(2) ≈ 0.586, 2, 2 + sqrt(2) ≈ 3.414.
        let mut mat = [2.0f32, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0];
        jacobi_eigenvalues_symmetric_inplace(&mut mat, 3, 100);
        let mut eigs: Vec<f32> = (0..3).map(|i| mat[i * 3 + i]).collect();
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(
            (eigs[0] - (2.0 - std::f32::consts::SQRT_2)).abs() < 1e-4,
            "eigs = {:?}",
            eigs
        );
        assert!((eigs[1] - 2.0).abs() < 1e-4, "eigs = {:?}", eigs);
        assert!(
            (eigs[2] - (2.0 + std::f32::consts::SQRT_2)).abs() < 1e-4,
            "eigs = {:?}",
            eigs
        );
    }

    // ── Report helpers ──────────────────────────────────────────────────

    #[test]
    fn report_counts_and_predicates() {
        let mut report = BandStructureReport::zeros(3);
        report.band_classes[0] = BandClass::Propagating;
        report.band_classes[1] = BandClass::Decaying;
        report.band_classes[2] = BandClass::Growing;
        assert_eq!(report.counts(), (1, 1, 1));
        assert!(!report.is_all_propagating());
        assert!(report.has_growing_mode());
    }
}
