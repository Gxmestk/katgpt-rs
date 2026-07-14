//! Data containers for the occupancy-ratio estimator.
//!
//! All containers are borrow-only (zero-copy over caller-owned buffers).
//! Scratch buffers are owned by the caller and reused across iterations to
//! keep the inner KL-projection loop allocation-free (G4).

/// One-step offline transition batch `(X_i, X^+_i)`.
///
/// States are stored as flattened `&[f32]` slices of length `n * state_dim`.
/// The successor `X^+_i` is sampled from the **target** policy's transition
/// kernel `P_π(·|X_i)` — not the behavior policy. For NPC consumers this
/// requires engram/delta_mem instrumentation to record
/// `(next-state, target-policy-action)` pairs (see module-level limitation
/// note).
///
/// `rewards` is optional; present when the caller intends to compute the
/// downstream value estimate `V̂^π = mean(ω · r)`.
#[derive(Debug, Clone, Copy)]
pub struct TransitionBatch<'a> {
    /// Flattened `[n * state_dim]` source states `X_i`.
    pub states: &'a [f32],
    /// Flattened `[n * state_dim]` successor states `X^+_i ∼ P_π(·|X_i)`.
    pub successors: &'a [f32],
    /// Optional `[n]` rewards `r_i` for downstream value estimation.
    pub rewards: Option<&'a [f32]>,
    /// Number of transitions.
    pub n: usize,
    /// Dimension of each state vector.
    pub state_dim: usize,
}

impl<'a> TransitionBatch<'a> {
    /// Get the `i`-th source state as a `&[f32]` slice of length `state_dim`.
    ///
    /// Returns `None` if `i >= n`.
    #[inline]
    #[must_use]
    pub fn state(&self, i: usize) -> Option<&'a [f32]> {
        if i < self.n {
            Some(&self.states[i * self.state_dim..(i + 1) * self.state_dim])
        } else {
            None
        }
    }

    /// Get the `i`-th successor state as a `&[f32]` slice of length `state_dim`.
    ///
    /// Returns `None` if `i >= n`.
    #[inline]
    #[must_use]
    pub fn successor(&self, i: usize) -> Option<&'a [f32]> {
        if i < self.n {
            Some(&self.successors[i * self.state_dim..(i + 1) * self.state_dim])
        } else {
            None
        }
    }
}

/// Empirical initial-state distribution moments for the `P̂_0` term.
///
/// The adjoint Bellman operator is `B^γ_π ω = (1−γ)ω_0 + γ · d((ων)P_π)/dν`,
/// where `ω_0 = d^π_0 / d^ν_0` is the ratio of the target policy's initial
/// distribution to the behavior policy's initial distribution. This struct
/// supplies the `ω_0` estimates at the initial-state subsample.
///
/// **Phase 1**: simple container. The exact fields may be refined in Phase 2
/// once the paper's Algorithm 1 `P̂_0 h` estimator has been verified.
#[derive(Debug, Clone, Copy)]
pub struct InitialMoments<'a> {
    /// Flattened `[n_init * state_dim]` initial states drawn from `d^ν_0`.
    pub initial_states: &'a [f32],
    /// `ω_0(X_i) = d^π_0(X_i) / d^ν_0(X_i)` at each initial state.
    pub initial_ratio: &'a [f32],
    /// Number of initial states.
    pub n_init: usize,
    /// Dimension of each state vector (must match [`TransitionBatch::state_dim`]).
    pub state_dim: usize,
}

/// Pre-allocated scratch buffers for the KL-projection inner loop.
///
/// Constructed once before the fitted-iteration loop and reused via `clear()`
/// on every iteration to keep the inner loop allocation-free (G4). The fields
/// are sized to `n` (the transition count) or `feature_dim` (the log-ratio
/// class parameter dimension) at construction time.
///
/// **Phase 1**: placeholder fields. The exact buffer set is finalized in
/// Phase 2 when the KL-projection solver (weighted normal equations) lands.
#[derive(Debug, Clone)]
pub struct KlProjectionScratch {
    /// `[n]` target weights from the adjoint-Bellman image of the current ratio.
    pub target_weights: Vec<f32>,
    /// `[n * feature_dim]` flattened design-matrix rows `Φ` (one row per transition).
    pub design_rows: Vec<f32>,
    /// `[feature_dim]` right-hand side of the weighted normal equations.
    pub normal_eq_rhs: Vec<f32>,
}

impl KlProjectionScratch {
    /// Allocate scratch buffers sized for `n` transitions and `feature_dim`
    /// parameters. The vectors are allocated once here and `clear()`-reused
    /// inside the fitted-iteration loop.
    #[must_use]
    pub fn new(n: usize, feature_dim: usize) -> Self {
        Self {
            target_weights: Vec::with_capacity(n),
            design_rows: Vec::with_capacity(n * feature_dim),
            normal_eq_rhs: Vec::with_capacity(feature_dim),
        }
    }

    /// Reset all buffers to empty (capacity preserved) for the next iteration.
    /// This is the allocation-free reuse path — no grow, no shrink.
    #[inline]
    pub fn clear(&mut self) {
        self.target_weights.clear();
        self.design_rows.clear();
        self.normal_eq_rhs.clear();
    }
}
