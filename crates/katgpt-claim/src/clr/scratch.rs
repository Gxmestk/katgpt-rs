//! CLR per-cycle scratch buffers (Plan 284 T2.2).
//!
//! Reusable storage for the verdict, reliability, and cluster-id arrays used by
//! [`crate::clr::vote::clr_vote`] / [`crate::clr::vote::clr_vote_minimal`].
//!
//! ## Allocation discipline
//!
//! [`ClrScratch::new`] allocates exactly once per buffer (with `with_capacity`).
//! Subsequent [`ClrScratch::reset_for`] calls only `clear()` + `resize()` — and
//! because the capacities are already sized, `resize` will NOT reallocate
//! provided `K` and `M` do not grow beyond the values passed to `new`.
//!
//! **Rule of thumb:** size `ClrScratch::new(k, m)` to the max `(K, M)` the
//! caller will ever use (e.g. `new(32, 5)` for the paper default). After that,
//! zero heap allocation per CLR cycle.
//!
//! ## K limit
//!
//! `cluster_id: Vec<u8>` caps `K` at 256 trajectories. Plan 284 fixes `K ≤ 32`,
//! so this is fine. For larger `K`, switch to `Vec<u16>` or `Vec<usize>`
//! (not needed for Phase 2).

/// Reusable CLR scratch state.
///
/// All four buffers are sized to `(K, M)` and zeroed by [`Self::reset_for`].
/// The voter writes verdicts/reliability/cluster-id + claim-embeddings into
/// these in place rather than allocating fresh `Vec`s per cycle.
#[derive(Clone, Debug)]
pub struct ClrScratch {
    /// Flattened verdict matrix `[K * M]`, row-major.
    /// Row `k` holds the M per-direction verdicts for trajectory `k`.
    pub verdicts: Vec<f32>,
    /// Per-trajectory reliability score `[K]`.
    pub reliability: Vec<f32>,
    /// Per-trajectory cluster id `[K]`. `u8` caps K at 256 (see module docs).
    pub cluster_id: Vec<u8>,
    /// Flat claim-embedding buffer `[M * k]`, row-major.
    ///
    /// Written by `ClaimExtractor::extract_embeddings_into` once per trajectory
    /// in the zero-alloc hot path. Row `m` occupies `[m*k..(m+1)*k]`. The
    /// voter reads each row via `ClaimVerifier::verify_embedding` without
    /// constructing any `Claim<T>` values.
    ///
    /// Not generic over `T` — only raw f32 embeddings live here (the verifier
    /// ignores the claim payload on the vote path).
    pub claim_embeddings: Vec<f32>,
}

impl ClrScratch {
    /// Allocate scratch sized for `K * M` verdicts.
    ///
    /// Uses `with_capacity` so the first [`Self::reset_for`] is the only
    /// allocation; later `reset_for` calls with `k <= K, m <= M` do not
    /// reallocate.
    #[inline]
    pub fn new(k: usize, m: usize) -> Self {
        assert!(k > 0, "ClrScratch::new: k must be > 0");
        assert!(m > 0, "ClrScratch::new: m must be > 0");
        assert!(
            k <= 256,
            "ClrScratch::new: k={} exceeds Vec<u8> cluster_id limit (256)",
            k
        );
        Self {
            verdicts: Vec::with_capacity(k * m),
            reliability: Vec::with_capacity(k),
            cluster_id: Vec::with_capacity(k),
            claim_embeddings: Vec::with_capacity(m * k),
        }
    }

    /// Zero and size the buffers for one CLR cycle of shape `(k, m)`.
    ///
    /// After this call:
    ///   - `verdicts.len() == k * m`, all `0.0`
    ///   - `reliability.len() == k`, all `0.0`
    ///   - `cluster_id.len() == k`, all `0`
    ///   - `claim_embeddings.len() == m * k`, all `0.0`
    ///
    /// No allocation occurs if `k * m <= verdicts.capacity()` etc., which holds
    /// when `k, m` are within the values passed to [`Self::new`].
    #[inline]
    pub fn reset_for(&mut self, k: usize, m: usize) {
        self.verdicts.clear();
        self.verdicts.resize(k * m, 0.0);
        self.reliability.clear();
        self.reliability.resize(k, 0.0);
        self.cluster_id.clear();
        self.cluster_id.resize(k, 0);
        self.claim_embeddings.clear();
        self.claim_embeddings.resize(m * k, 0.0);
    }
}
