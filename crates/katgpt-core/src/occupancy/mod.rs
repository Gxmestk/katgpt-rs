//! `occupancy` — generic modelless fitted occupancy-ratio estimator (FORE).
//!
//! # Plan 438 / Research 423 / arXiv:2607.05375
//!
//! Source paper: van der Laan & Kallus, *"Fitted Occupancy-Ratio Evaluation
//! without Bellman Completeness"* (2026, [arXiv:2607.05375](https://arxiv.org/abs/2607.05375)).
//!
//! This module ships the **open engine primitive**: a generic fitted-iteration
//! estimator for the discounted occupancy ratio `ω_π,γ = d^π,γ / d_ν` in
//! offline policy evaluation. The substrate-independent contribution is the
//! **adjoint Bellman KL contraction** (paper Lemma 3.1) — the operator
//! `B^γ_π ω = (1−γ)ω_0 + γ · d((ων)P_π)/dν` contracts relative entropy by
//! factor `γ` per fitted iteration, so FORE converges under **realizability
//! alone** (no Bellman completeness of a value/critic class required).
//!
//! # Status — Phase 1 (skeleton)
//!
//! This module currently ships only the type/trait surface. The fitted-
//! iteration loop (`OccupancyRatioEstimator::fit`) and the linear log-ratio
//! class (`LinearLogRatioClass`) arrive in Phase 2 once the paper's
//! Algorithm 1 per-sample update rule has been verified against the source.
//! No math is implemented yet.
//!
//! # Softmax-vs-sigmoid carve-out (READ ME)
//!
//! The global `AGENTS.md` rule says "Use sigmoid not softmax". FORE's math
//! requires a **normalized exponential class** `ω_h(x) = exp(h(x) − Λ_ν(h))`,
//! which is structurally a softmax over the offline sample. This is in
//! principled tension with the sigmoid rule, and the resolution is:
//!
//! The sigmoid rule's intent is "don't use softmax where sigmoid-on-direction-
//! vectors suffices for projections onto learned directions" (the semantic-
//! domain rule). FORE's use of softmax is **not a projection onto learned
//! directions** — it is a **density-ratio normalization** over a discrete
//! offline sample, which is the correct mathematical operation (the log-
//! partition `Λ_ν(h)` is the cumulant-generating function of the empirical
//! distribution). The sigmoid rule does not apply to density-ratio
//! normalization; it applies to direction-vector projections.
//!
//! This is the same carve-out already documented in
//! [`crate::product_key_memory`]: "Deviation from the global sigmoid rule —
//! these are convex-combination coefficients over the k²-restricted candidate
//! set, not a probability/UQ claim."
//!
//! # Honest limitations
//!
//! - **Offline transition data is the binding input.** FORE requires one-step
//!   target-policy successor pairs `(X_i, X^+_i)` where `X^+_i ∼ P_π(·|X_i)`.
//!   For NPC consumers, this means the engram/delta_mem subsystem must record
//!   not just `(state, action, reward, next-state)` but also
//!   `(next-state, target-policy-action)`. This is additional instrumentation
//!   on the consumer side, not a free lunch.
//! - **Continuous high-dimensional state spaces** are the paper's acknowledged
//!   limitation (§7). The log-ratio class must approximate `ω_π,γ` in
//!   `L²(ν)`. For 8-dim HLA scalars or 64-dim `style_weights`, this is
//!   feasible. For raw pixel state or 1000+-d transformer activations, it is
//!   not. Do not over-promote.
//! - **The DEC codifferential isomorphism** (Research 423 §2.2 — "adjoint
//!   Bellman ≡ codifferential in the Markov-kernel cochain complex") is an
//!   architectural insight, not a proven theorem. It may yield a Lean 4
//!   theorem eventually; treat it as a research hypothesis until proven.
//!
//! See [`kl_contraction`] for the theorem statement.

mod types;

pub use types::{InitialMoments, KlProjectionScratch, TransitionBatch};

/// Theorem-statement module for the adjoint Bellman KL contraction
/// (paper Lemma 3.1). Doc-only — no implementation.
pub mod kl_contraction;

/// Occupancy-ratio estimator: fitted-iteration loop over a [`LogRatioClass`].
///
/// Holds the discount `gamma`, the iteration count `k_iterations`, and the
/// log-ratio class `H` (the supervised learner that realizes `log ω_π,γ`).
/// The `fit` method runs K rounds of KL projection; each round contracts
/// relative entropy by factor `gamma` (Lemma 3.1).
///
/// **Phase 1 status**: constructor only. `fit` and `value_estimate` arrive in
/// Phase 2.
pub struct OccupancyRatioEstimator<H: LogRatioClass> {
    /// The supervised learner realizing `h(x) = log ω_π,γ(x)` up to a constant.
    pub log_ratio_class: H,
    /// Discount factor `gamma ∈ [0, 1)`. KL contraction factor per iteration.
    pub gamma: f32,
    /// Number of fitted KL-projection iterations K.
    pub k_iterations: usize,
}

impl<H: LogRatioClass> OccupancyRatioEstimator<H> {
    /// Construct a new estimator. Panics if `gamma >= 1.0` or `gamma < 0.0`
    /// (the contraction guarantee requires `gamma ∈ [0, 1)`).
    #[must_use]
    pub fn new(log_ratio_class: H, gamma: f32, k_iterations: usize) -> Self {
        assert!(
            (0.0..1.0).contains(&gamma),
            "gamma must be in [0, 1) for KL contraction; got {gamma}"
        );
        Self {
            log_ratio_class,
            gamma,
            k_iterations,
        }
    }
}

/// Trait for the supervised learner realizing `h(x) = log ω_π,γ(x)` (up to a
/// normalization constant absorbed by the log-partition `Λ_ν(h)`).
///
/// The trait is generic over the parameterization (`Self::Params`) so that
/// any sufficiently rich function class can be plugged in: linear features,
/// Fourier features, or a frozen-direction dot-product class. The FORE
/// convergence guarantee (Theorems 4.1, 4.2) requires only that the class
/// **realizes** the target ratio `ω_π,γ` — no Bellman completeness needed.
///
/// **Phase 1 status**: trait surface only. The first concrete impl
/// (`LinearLogRatioClass`) arrives in Phase 2.
///
/// # Modelless constraint (G5)
///
/// `fit_kl_projection` may use gradient descent on its **own** parameters
/// (`Self::Params`), but must NOT touch any base weight: no `NeuronShard`,
/// no `LoRAWeightVersion`, no `SenseModule` handle may appear in the impl.
/// The primitive is modelless by construction.
pub trait LogRatioClass {
    /// The parameterization of the log-ratio function `h(x)`.
    type Params;

    /// Evaluate `h(x)` at a single point. Returns the (un-normalized) log-ratio
    /// score; the caller normalizes via the log-partition `Λ_ν(h)`.
    fn evaluate(&self, params: &Self::Params, x: &[f32]) -> f32;

    /// Fit `h` to the adjoint-Bellman image of the current ratio.
    ///
    /// Given the current ratio estimates `current_ratio[i] ≈ ω̂^(k)(X_i)` at
    /// each transition, compute the adjoint-Bellman target weights and project
    /// onto the log-ratio class via cross-entropy (KL) minimization.
    ///
    /// The `scratch` buffer is reused across iterations (G4 alloc-free inner
    /// loop); implementations must `clear()` rather than re-allocate.
    ///
    /// **Phase 2**: the exact target-weight formula is derived from the paper's
    /// Algorithm 1 (pending verification against the source).
    fn fit_kl_projection(
        &self,
        transitions: &TransitionBatch<'_>,
        initial_moments: &InitialMoments<'_>,
        current_ratio: &[f32],
        gamma: f32,
        scratch: &mut KlProjectionScratch,
    ) -> Self::Params;
}
