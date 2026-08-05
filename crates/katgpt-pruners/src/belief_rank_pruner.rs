//! Belief-State Rank Pruner — uses hidden state effective rank as screening signal.
//!
//! Low effective rank (peaked distribution) → high confidence → accept draft tokens.
//! High effective rank (flat distribution) → uncertain → reject → deeper search needed.
//!
//! The effective rank is approximated via the participation ratio (PR) of hidden state
//! vectors, avoiding full SVD. For a single vector h:
//!
//! ```text
//! flatness(h) = (Σ h_i²)² / (n_embd * Σ h_i⁴)
//! ```
//!
//! - flatness ≈ 1.0 → all dimensions equal → high effective rank → uncertain → reject
//! - flatness ≈ 0.0 → peaked on one dimension → low effective rank → confident → accept
//!
//! For a buffer of recent hidden states, we compute PR of the variance diagonal
//! (diagonal of covariance), which is O(n * k) instead of O(n² * k) for full SVD.
//!
//! Plan 217 Phase 3.

use katgpt_core::ScreeningPruner;

/// Pruner that uses hidden state effective rank (participation ratio) to gate draft acceptance.
///
/// The "belief rank" is computed from a sliding window of recent hidden states.
/// When the hidden state has low effective rank (peaked on a few dimensions),
/// the model is "confident" — draft tokens are likely correct → accept.
/// When high rank (spread across many dimensions), model is "uncertain" → reject.
///
/// # Relevance Gating
///
/// `relevance()` uses sigmoid smooth gating:
/// ```text
/// relevance = sigmoid(-k * (rank - threshold))
/// ```
/// - rank < threshold → sigmoid > 0.5 → accept
/// - rank > threshold → sigmoid < 0.5 → reject
///
/// # Performance
///
/// - `observe()`: O(n_embd) copy, amortized O(1) push with capacity pre-allocated
/// - `flatness()`: O(n_embd), two accumulators (L2², L4), branch-free inner loop
/// - `effective_rank()`: O(1) — recomputed once per `observe()`, then cached
/// - `relevance()`: O(1), a cache read plus a single sigmoid
pub struct BeliefRankPruner {
    /// Recent hidden states buffer [max_buffer_len][n_embd].
    hidden_buffer: Vec<Vec<f32>>,
    /// Scratch for the per-observation covariance mean row (reused).
    mean_scratch: Vec<f32>,
    /// Scratch for the per-observation covariance variance row (reused).
    var_scratch: Vec<f32>,
    /// Maximum number of hidden states to keep in buffer.
    max_buffer_len: usize,
    /// Flatness threshold above which to reject drafts.
    /// Range [0, 1]. Default: 0.7 (reject when >70% flat = high uncertainty).
    reject_threshold: f32,
    /// Cached effective rank of the current buffer.
    ///
    /// `relevance()` is called once per candidate token but the rank depends
    /// only on `hidden_buffer` / `initialized`, both of which are written
    /// **exclusively** by [`Self::observe`] (verified: no other `&mut self`
    /// method touches them). Recomputing it there instead of on every
    /// `relevance()` turns an O(n_embd · buffer_len) scan plus two n_embd-sized
    /// allocations per candidate into an O(1) field read. The stored value is
    /// produced by the unchanged covariance code at exactly the buffer state a
    /// subsequent `relevance()` would have seen, so the float result is
    /// identical.
    cached_rank: f32,
    /// Number of hidden state dimensions (n_embd).
    n_embd: usize,
    /// Whether the pruner has seen at least one hidden state.
    initialized: bool,
}

impl BeliefRankPruner {
    /// Create a new pruner.
    ///
    /// - `n_embd`: hidden state dimensionality
    /// - `max_buffer_len`: sliding window size for covariance estimation
    /// - `reject_threshold`: flatness above this → reject drafts (default: 0.7)
    pub fn new(n_embd: usize, max_buffer_len: usize, reject_threshold: f32) -> Self {
        Self {
            hidden_buffer: Vec::with_capacity(max_buffer_len),
            mean_scratch: Vec::new(),
            var_scratch: Vec::new(),
            max_buffer_len,
            reject_threshold,
            // Matches the "unknown → neutral" value `effective_rank` returns
            // before the first observation.
            cached_rank: 0.5,
            n_embd,
            initialized: false,
        }
    }

    /// Update the pruner with a new hidden state observation.
    ///
    /// Silently ignores vectors with wrong dimensionality.
    pub fn observe(&mut self, h: &[f32]) {
        if h.len() != self.n_embd {
            return;
        }

        if self.max_buffer_len > 0 && self.hidden_buffer.len() == self.max_buffer_len {
            // Steady state: recycle the evicted front row's allocation instead
            // of freeing it and allocating a fresh `h.to_vec()`. Ordering is
            // preserved (front removed, new row appended) — identical to
            // "push then remove(0)" — so the covariance loops below still see
            // the same element order and the same float accumulation order.
            let mut recycled = self.hidden_buffer.remove(0);
            recycled.clear();
            recycled.extend_from_slice(h);
            self.hidden_buffer.push(recycled);
        } else {
            self.hidden_buffer.push(h.to_vec());
            if self.hidden_buffer.len() > self.max_buffer_len {
                self.hidden_buffer.remove(0);
            }
        }
        self.initialized = true;

        // Refresh the cache once per observation (see `cached_rank`).
        let mut mean = std::mem::take(&mut self.mean_scratch);
        let mut var = std::mem::take(&mut self.var_scratch);
        self.cached_rank = self.compute_effective_rank(&mut mean, &mut var);
        self.mean_scratch = mean;
        self.var_scratch = var;
    }

    /// Compute the flatness score of a single hidden state vector.
    ///
    /// Returns value in [0, 1]:
    /// - 0.0 = peaked (confident, low effective rank)
    /// - 1.0 = flat (uncertain, high effective rank)
    ///
    /// Participation ratio: P = (Σ σ²)² / (n * Σ σ⁴).
    /// For a single vector: P = (Σ h²)² / (n * Σ h⁴).
    pub fn flatness(&self, h: &[f32]) -> f32 {
        match (h.len() == self.n_embd, self.n_embd == 0) {
            (false, _) | (_, true) => return 0.5,
            (true, false) => {}
        }

        let mut l2_sq: f32 = 0.0;
        let mut l4: f32 = 0.0;
        for &x in h {
            let x2 = x * x;
            l2_sq += x2;
            l4 += x2 * x2;
        }

        match l4 < 1e-12 {
            true => 0.0, // zero vector → peaked (trivially confident)
            false => {
                let pr = (l2_sq * l2_sq) / (self.n_embd as f32 * l4);
                pr.clamp(0.0, 1.0)
            }
        }
    }

    /// Compute the effective rank of the hidden state buffer.
    ///
    /// Uses participation ratio of the variance diagonal (diagonal of covariance),
    /// which is O(n * k) instead of O(n² * k) for full SVD.
    ///
    /// Returns value in [0, 1]:
    /// - 0.0 = low rank (confident, peaked)
    /// - 1.0 = high rank (uncertain, flat/diverse)
    #[inline]
    pub fn effective_rank(&self) -> f32 {
        // O(1): kept fresh by `observe`, the only writer of the inputs.
        self.cached_rank
    }

    /// The covariance/participation-ratio computation behind
    /// [`Self::effective_rank`]. Called once per [`Self::observe`].
    ///
    /// `mean` / `var` are caller-owned scratch (reused across observations, so
    /// the two `vec![0.0f32; n]` allocations are gone); `clear()` + `resize` to
    /// zero reproduces the old fresh-zeroed vectors exactly.
    fn compute_effective_rank(&self, mean: &mut Vec<f32>, var: &mut Vec<f32>) -> f32 {
        match (self.initialized, self.hidden_buffer.is_empty()) {
            (false, _) | (true, true) => return 0.5, // Unknown → neutral
            (true, false) => {}
        }

        let k = self.hidden_buffer.len();

        if k == 1 {
            return self.flatness(&self.hidden_buffer[0]);
        }

        let n = self.n_embd;

        // Compute mean of buffer
        mean.clear();
        mean.resize(n, 0.0f32);
        for h in &self.hidden_buffer {
            for i in 0..n {
                mean[i] += h[i];
            }
        }
        for m in mean.iter_mut() {
            *m /= k as f32;
        }

        // Compute diagonal of covariance: var[i] = Σ (h[j][i] - mean[i])² / (k-1)
        var.clear();
        var.resize(n, 0.0f32);
        for h in &self.hidden_buffer {
            for i in 0..n {
                let d = h[i] - mean[i];
                var[i] += d * d;
            }
        }
        let denom = (k - 1).max(1) as f32;
        for v in var.iter_mut() {
            *v /= denom;
        }

        // Participation ratio: (Σ var²)² / (n * Σ var⁴)
        let mut l2_sq: f32 = 0.0;
        let mut l4: f32 = 0.0;
        for &x in var.iter() {
            let x2 = x * x;
            l2_sq += x2;
            l4 += x2 * x2;
        }

        match l4 < 1e-20 {
            true => 0.0,
            false => {
                let pr = (l2_sq * l2_sq) / (n as f32 * l4);
                pr.clamp(0.0, 1.0)
            }
        }
    }

    /// Returns `true` if the pruner has observed at least one hidden state.
    #[inline]
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Returns the current buffer length.
    pub fn buffer_len(&self) -> usize {
        self.hidden_buffer.len()
    }
}

impl ScreeningPruner for BeliefRankPruner {
    fn relevance(&self, _depth: usize, _token_idx: usize, _parent_token: &[usize]) -> f32 {
        // Neutral relevance when no observations yet
        if !self.initialized {
            return 0.5;
        }

        let rank = self.effective_rank();

        // Sigmoid smooth gating: relevance = sigmoid(-k * (rank - threshold))
        // rank < threshold → sigmoid > 0.5 → accept (confident)
        // rank > threshold → sigmoid < 0.5 → reject (uncertain)
        let k = 10.0;
        let x = -k * (rank - self.reject_threshold);
        katgpt_core::simd::fast_sigmoid(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a peaked vector (one dominant dimension).
    fn peaked_vector(n: usize) -> Vec<f32> {
        let mut v = vec![0.1f32; n];
        v[0] = 10.0;
        v
    }

    /// Helper: create a uniform vector (all equal).
    fn uniform_vector(n: usize) -> Vec<f32> {
        vec![1.0f32; n]
    }

    #[test]
    fn test_flatness_zero_vector() {
        let pruner = BeliefRankPruner::new(4, 8, 0.7);
        let h = vec![0.0f32; 4];
        let f = pruner.flatness(&h);
        assert!(
            (f - 0.0).abs() < 1e-6,
            "zero vector should have flatness 0.0 (peaked/trivially confident), got {f}"
        );
    }

    #[test]
    fn test_flatness_peaked_vector() {
        let pruner = BeliefRankPruner::new(4, 8, 0.7);
        let h = peaked_vector(4);
        let f = pruner.flatness(&h);
        // [10.0, 0.1, 0.1, 0.1] → PR ≈ 0.25 (one dominant dim out of 4)
        assert!(f < 0.5, "peaked vector should have flatness < 0.5, got {f}");

        // Even "extreme" [100, 0, 0, 0] still has PR = 1/4 = 0.25
        // because PR = (Σ h²)² / (n * Σ h⁴), and one non-zero gives rank 1/n
        let extreme = vec![100.0f32, 0.0, 0.0, 0.0];
        let f2 = pruner.flatness(&extreme);
        assert!(
            (f2 - 0.25).abs() < 0.01,
            "single-dim peaked vector has PR = 1/n_embd = 0.25, got {f2}"
        );
    }

    #[test]
    fn test_flatness_uniform_vector() {
        let pruner = BeliefRankPruner::new(4, 8, 0.7);
        let h = uniform_vector(4);
        let f = pruner.flatness(&h);
        assert!(
            (f - 1.0).abs() < 0.01,
            "uniform vector should have flatness ≈ 1.0, got {f}"
        );
    }

    #[test]
    fn test_effective_rank_single_observation() {
        let mut pruner = BeliefRankPruner::new(4, 8, 0.7);
        let h = peaked_vector(4);
        let expected = pruner.flatness(&h);
        pruner.observe(&h);
        let rank = pruner.effective_rank();
        assert!(
            (rank - expected).abs() < 1e-6,
            "single observation rank should equal flatness, got rank={rank}, flatness={expected}"
        );
    }

    #[test]
    fn test_effective_rank_peaked_observations() {
        let mut pruner = BeliefRankPruner::new(4, 8, 0.7);
        // All peaked → low variance across dimensions → low rank
        for _ in 0..4 {
            pruner.observe(&peaked_vector(4));
        }
        let rank = pruner.effective_rank();
        assert!(
            rank < 0.1,
            "peaked observations should have low rank, got {rank}"
        );
    }

    #[test]
    fn test_effective_rank_diverse_observations() {
        let mut pruner = BeliefRankPruner::new(4, 8, 0.7);
        // Mix peaked + uniform → higher variance → higher rank than all-peaked
        let peaked = peaked_vector(4);
        let uniform = uniform_vector(4);
        pruner.observe(&peaked);
        pruner.observe(&uniform);
        pruner.observe(&peaked);
        pruner.observe(&uniform);
        let rank = pruner.effective_rank();
        assert!(
            rank > 0.1,
            "diverse observations should have higher rank, got {rank}"
        );
    }

    #[test]
    fn test_screening_pruner_relevance_confident() {
        let mut pruner = BeliefRankPruner::new(4, 8, 0.7);
        // All peaked → low rank → should be above threshold → high relevance
        for _ in 0..4 {
            pruner.observe(&peaked_vector(4));
        }
        let rel = pruner.relevance(0, 0, &[]);
        assert!(
            rel > 0.5,
            "confident (low rank) should give relevance > 0.5, got {rel}"
        );
    }

    #[test]
    fn test_screening_pruner_relevance_uncertain() {
        let mut pruner = BeliefRankPruner::new(4, 8, 0.3);
        // Use diverse one-hot vectors → variance is uniform across dims → high PR
        // Each dimension has equal variance → rank ≈ 1.0 → uncertain
        pruner.observe(&[1.0, 0.0, 0.0, 0.0]);
        pruner.observe(&[0.0, 1.0, 0.0, 0.0]);
        pruner.observe(&[0.0, 0.0, 1.0, 0.0]);
        pruner.observe(&[0.0, 0.0, 0.0, 1.0]);
        let rank = pruner.effective_rank();
        assert!(
            rank > 0.9,
            "diverse one-hot should have high rank, got {rank}"
        );
        let rel = pruner.relevance(0, 0, &[]);
        assert!(
            rel < 0.5,
            "uncertain (high rank) should give relevance < 0.5, got {rel}"
        );
    }

    #[test]
    fn test_observe_maintains_buffer_size() {
        let mut pruner = BeliefRankPruner::new(4, 3, 0.7);
        for i in 0..10 {
            pruner.observe(&[i as f32; 4]);
        }
        assert_eq!(
            pruner.buffer_len(),
            3,
            "buffer should not exceed max_buffer_len"
        );
    }

    #[test]
    fn test_relevance_uninitialized() {
        let pruner = BeliefRankPruner::new(4, 8, 0.7);
        let rel = pruner.relevance(0, 0, &[]);
        assert!(
            (rel - 0.5).abs() < 1e-6,
            "uninitialized pruner should return 0.5, got {rel}"
        );
    }
}
