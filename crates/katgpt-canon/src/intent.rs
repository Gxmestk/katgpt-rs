//! [`CanonicalIntent`] — architecture-neutral intent direction.
//!
//! A canonical intent is a unit-norm f32 direction in *canonical space*
//! (the architecture-neutral latent space defined by Proposal 009 /
//! Research 459). Each base model carries a [`crate::ModelAdapter`] that
//! projects canonical directions into its model-specific latent space.
//!
//! # Construction
//!
//! Construct via [`CanonicalIntent::new`] which:
//! 1. Computes the BLAKE3 tag of the human-readable `label` (sync/commit
//!    attestation + cross-node verify).
//! 2. Normalizes `direction` to unit L2 norm.
//!
//! # Composition
//!
//! Canonical directions compose by vector arithmetic in canonical space
//! (sum, scale, gate via sigmoid). Because every [`crate::ModelAdapter`]
//! is linear (Procrustes, subspace projection) or elementwise (mask),
//! composition commutes through the adapter: `F(d₁ + α·d₂) ==
//! F(d₁) + α·F(d₂)` for the linear adapters.
//!
//! # Modelless
//!
//! Both the type and the adapters are modelless — no training, no gradient
//! descent. The only weight mutations allowed are freeze/thaw swaps of the
//! adapter state itself (BLAKE3-committed).

use blake3::Hasher;

/// Architecture-neutral intent direction. Unit-norm f32 vector + BLAKE3 tag
/// for sync/commit. Lives in canonical space — never decoded directly.
///
/// The tag is the BLAKE3 hash of the `label` bytes (NOT of the direction
/// bytes — the label is the human-readable identity; two adapters that
/// project the same canonical intent share the same tag). The direction
/// bytes are committed separately by the adapter's [`crate::ModelAdapter::commitment`].
#[derive(Clone, Debug)]
pub struct CanonicalIntent {
    /// BLAKE3 of the `label` passed to [`CanonicalIntent::new`]. Stable
    /// across runs, architectures, and Rust versions (BLAKE3 is
    /// deterministic). Used for sync/commit attestation + cross-node verify.
    pub tag: [u8; 32],
    /// Unit-norm f32 direction in canonical space. Length = canonical dim.
    pub direction: Vec<f32>,
}

impl CanonicalIntent {
    /// Construct a canonical intent from a human-readable label + a direction
    /// vector. The direction is normalized to unit L2 norm in-place; the
    /// input is consumed. The tag is BLAKE3(label bytes).
    ///
    /// # Zero-length direction
    ///
    /// If `direction` is empty, returns an intent with an empty direction
    /// and the correct tag. Downstream code MUST handle `dim() == 0` as
    /// the identity / no-op (most adapters will panic on project_into
    /// with mismatched dims — this is the caller's contract).
    #[inline]
    pub fn new(label: &str, mut direction: Vec<f32>) -> Self {
        let mut hasher = Hasher::new();
        hasher.update(label.as_bytes());
        let tag = *hasher.finalize().as_bytes();
        normalize_into_unit(&mut direction);
        Self { tag, direction }
    }

    /// Construct from raw parts (already-normalized direction + precomputed tag).
    /// Bypasses the BLAKE3 hash — use only when the tag has been pre-attested
    /// (e.g. loading from a freeze/thaw snapshot where the tag was committed
    /// alongside the direction).
    ///
    /// # Safety contract (caller responsibility)
    ///
    /// The caller MUST ensure `direction` is unit L2 norm. This constructor
    /// does NOT re-normalize (it would change the BLAKE3-committed bytes).
    /// If the direction is not unit norm, downstream dot products will be
    /// scaled by the norm and the adapter commitment will mismatch.
    #[inline]
    pub fn from_parts(tag: [u8; 32], direction: Vec<f32>) -> Self {
        Self { tag, direction }
    }

    /// Canonical dimension (length of `direction`).
    #[inline]
    pub fn dim(&self) -> usize {
        self.direction.len()
    }

    /// Cosine similarity to another canonical intent. Because both directions
    /// are unit-norm, this is just the dot product. Returns NaN if either
    /// direction is empty or zero-norm (zero-norm can happen if the caller
    /// abused [`CanonicalIntent::from_parts`]).
    #[inline]
    pub fn dot(&self, other: &Self) -> f32 {
        debug_assert_eq!(
            self.dim(),
            other.dim(),
            "CanonicalIntent::dot: dim mismatch ({} vs {})",
            self.dim(),
            other.dim()
        );
        let mut s = 0.0f32;
        for (a, b) in self.direction.iter().zip(other.direction.iter()) {
            s += a * b;
        }
        s
    }

    /// Reference to the raw direction slice (unit-norm).
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.direction
    }
}

/// Normalize `v` to unit L2 norm in-place. No-op on zero-length or empty.
/// Uses two passes (compute norm, divide) — vectorized by LLVM auto-SIMD
/// for f32 slices. Allocation-free.
#[inline]
fn normalize_into_unit(v: &mut [f32]) {
    if v.is_empty() {
        return;
    }
    let mut sum_sq = 0.0f32;
    for x in v.iter() {
        sum_sq += x * x;
    }
    if sum_sq == 0.0 {
        return; // leave zero vector as-is (downstream dot returns 0)
    }
    let norm = sum_sq.sqrt();
    let inv = 1.0 / norm;
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// Projects a [`CanonicalIntent`] into a specific base model's latent space.
/// Modelless: zero training, deterministic given (adapter_state, base_model).
///
/// # Implementors
///
/// Three concrete impls ship in this crate:
/// - [`crate::ProcrustesAdapter`] — orthogonal rotation, bijective, same-dim.
///   The default for same-architecture model swap (e.g. Gemma-A ↔ Gemma-B).
/// - [`crate::SubspaceAdapter`] — project onto joint-SVD basis, lossy but
///   works cross-architecture. Bench 423 G5 PASS at k ∈ {2, 4}.
/// - [`crate::MaskAdapter`] — apply a precomputed lottery-ticket mask.
///
/// # Allocation contract
///
/// `project_into` MUST be zero-alloc after construction (G4 gate). The
/// caller-owned `out` buffer is the only output. `extract_from` is the
/// diagnostic inverse and MAY allocate (it's not on the hot path).
pub trait ModelAdapter: Send + Sync {
    /// Apply the adapter to `canonical`, writing into `out`.
    /// `out.len()` MUST equal [`ModelAdapter::target_dim`]. Panics on
    /// mismatched length (release) / debug-asserts (debug).
    ///
    /// Zero-alloc hot path: caller-owned buffer, no internal allocation
    /// after adapter construction.
    fn project_into(&self, canonical: &CanonicalIntent, out: &mut [f32]);

    /// Inverse: extract canonical coordinates from an observed model latent.
    /// Used for "what intent is this activation expressing?" diagnostics.
    /// Returns a vector of length = canonical dim.
    ///
    /// MAY allocate. Not on the hot path.
    fn extract_from(&self, model_latent: &[f32]) -> Vec<f32>;

    /// Latent dim of the target model (adapter output dim).
    fn target_dim(&self) -> usize;

    /// Canonical dim of this adapter (input dim — length of canonical
    /// directions this adapter accepts). For [`crate::ProcrustesAdapter`]
    /// this equals `target_dim` (square rotation). For [`crate::SubspaceAdapter`]
    /// this equals the joint-SVD `k` (< target_dim — lossy projection).
    fn canonical_dim(&self) -> usize;

    /// BLAKE3 commitment of the adapter state (rotation / basis / mask).
    /// For freeze/thaw attestation + cross-node verify. Two adapters with
    /// the same commitment produce bit-identical projections.
    fn commitment(&self) -> [u8; 32];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_normalizes_to_unit_norm() {
        let d = CanonicalIntent::new("rust_idiom", vec![3.0, 4.0]);
        let norm = d.direction.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "expected unit norm, got {norm}");
    }

    #[test]
    fn tag_is_blake3_of_label() {
        let d1 = CanonicalIntent::new("rust_idiom", vec![1.0, 0.0]);
        let d2 = CanonicalIntent::new("rust_idiom", vec![0.5, 0.5]);
        // Same label → same tag, even though directions differ.
        assert_eq!(d1.tag, d2.tag);

        let d3 = CanonicalIntent::new("python_idiom", vec![1.0, 0.0]);
        // Different label → different tag.
        assert_ne!(d1.tag, d3.tag);
    }

    #[test]
    fn tag_matches_manual_blake3() {
        let d = CanonicalIntent::new("test_label", vec![1.0]);
        let expected = *blake3::hash("test_label".as_bytes()).as_bytes();
        assert_eq!(d.tag, expected);
    }

    #[test]
    fn dot_is_cosine_for_unit_vectors() {
        // Two orthogonal unit vectors → dot 0.
        let a = CanonicalIntent::new("a", vec![1.0, 0.0]);
        let b = CanonicalIntent::new("b", vec![0.0, 1.0]);
        assert!(a.dot(&b).abs() < 1e-6);

        // Two parallel unit vectors → dot 1.
        let c = CanonicalIntent::new("c", vec![1.0, 0.0]);
        assert!((a.dot(&c) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn dot_handles_non_unit_input_via_from_parts() {
        // from_parts bypasses normalization; dot returns scaled product.
        let tag = [0u8; 32];
        let a = CanonicalIntent::from_parts(tag, vec![2.0, 0.0]);
        let b = CanonicalIntent::from_parts(tag, vec![3.0, 0.0]);
        // Not normalized → dot = 6 (2*3), not 1.
        assert!((a.dot(&b) - 6.0).abs() < 1e-6);
    }

    #[test]
    fn empty_direction_does_not_panic() {
        let d = CanonicalIntent::new("empty", vec![]);
        assert_eq!(d.dim(), 0);
        assert!(d.direction.is_empty());
    }

    #[test]
    fn zero_vector_stays_zero() {
        // Zero vector should not divide by zero.
        let d = CanonicalIntent::new("zero", vec![0.0, 0.0, 0.0]);
        assert!(d.direction.iter().all(|x| *x == 0.0));
    }
}
