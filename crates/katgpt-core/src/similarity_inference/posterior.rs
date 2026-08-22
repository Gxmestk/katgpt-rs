//! [`SimilarityPosterior`] — the per-partner `ω ∈ (0, 1)` belief state.
//!
//! See the [module-level docs](crate::similarity_inference) for the full
//! mathematical background. This module implements the closed-form incremental
//! posterior update and the predictive-similarity operator.

use crate::sigmoid;
use crate::similarity_inference::SimilarityError;

/// Maintains the similarity posterior `ω ∈ (0, 1)` between one focal agent
/// and one partner, updated incrementally from joint-action observations.
///
/// The update rule (paper §H.2 closed form):
///
/// ```text
/// ω_T = α / (α + (1−α) · exp(log_w_independent))
/// ```
///
/// where `log_w_independent` is the log of `W(æ_<T) = Π_t P(a_i_t, a_j_t |
/// situation_t)` under the *independent-policy* marginal. Each observation
/// increments `log_w_independent` by the log-likelihood of the observed
/// partner action under independence — for a matched discrete action in a
/// game with `|A|` actions, that's `−ln(|A|)`.
///
/// # Construction
///
/// - [`SimilarityPosterior::new(prior_alpha)`](Self::new) — uninformative
///   start with `log_w_independent = 0` (i.e., `W = 1`, so `ω_0 = α`).
///
/// # Allocation discipline
///
/// - [`Self::observe`] is O(1): one `fma` into `log_w_independent`. Zero allocs.
/// - [`Self::observe_match`] (the discrete-action fast path) is also O(1).
/// - [`Self::predictive_similarity`] is O(D) on the continuous-embedding
///   contemplated action — one dot-product + one sigmoid. Zero allocs.
///
/// # Synchronization boundary
///
/// `ω` is a per-focal **latent** scalar — it MUST NOT be synced directly. Only
/// the final cooperate/defect action (produced by
/// [`embedded_best_response`](crate::similarity_inference::embedded_best_response))
/// crosses the sync boundary as a raw `u8`. See AGENTS.md §"Sync Boundary
/// Rule" and §"Bridge Pattern".
#[derive(Clone, Debug)]
pub struct SimilarityPosterior {
    /// Prior `α = P(shared)` at T=0. Strictly in `(0, 1)`.
    prior_alpha: f32,
    /// `log W(æ_<T)` — log of the joint-action likelihood under the
    /// independent-policy marginal. Starts at 0 (W=1, ω=α) and decreases
    /// (becomes more negative) as evidence accumulates, pushing `ω → 1`.
    log_w_independent: f32,
    /// Cached `ω_T` — recomputed only when the posterior is read. Avoids
    /// repeated `exp` calls when many observations land between reads.
    last_omega: f32,
    /// Number of observations accumulated. Used by the staleness window
    /// check in Plan 526 Phase 3 (indirect inference — third-party
    /// encounters must be within K ticks to count as evidence).
    observation_count: u32,
}

impl SimilarityPosterior {
    /// Construct a fresh posterior with prior `α`.
    ///
    /// Returns `Err(InvalidPriorAlpha)` if `alpha` is not in `(0, 1)`. The
    /// endpoints are excluded: `α=0` means "never believe partner is similar"
    /// (no evidence can move it); `α=1` means "certain a priori" (no evidence
    /// can move it). Both degenerate the posterior.
    pub fn new(prior_alpha: f32) -> Result<Self, SimilarityError> {
        if !prior_alpha.is_finite() || prior_alpha <= 0.0 || prior_alpha >= 1.0 {
            return Err(SimilarityError::InvalidPriorAlpha {
                given: prior_alpha.to_bits(),
            });
        }
        Ok(Self {
            prior_alpha,
            log_w_independent: 0.0,
            last_omega: prior_alpha,
            observation_count: 0,
        })
    }

    /// Prior `α`. Read-only — the prior is fixed at construction; posterior
    /// updates move `log_w_independent`, never `prior_alpha`.
    #[inline]
    pub fn prior_alpha(&self) -> f32 {
        self.prior_alpha
    }

    /// Number of observations accumulated (each call to [`Self::observe`] or
    /// [`Self::observe_match`] increments by 1). Used by the staleness window
    /// check in Plan 526 Phase 3 (indirect inference).
    #[inline]
    pub fn observations(&self) -> usize {
        self.observation_count as usize
    }

    /// Incorporate a continuous-embedding joint-action observation.
    ///
    /// `self_a`, `partner_a`, `situation` are arbitrary-length `&[f32]`
    /// embeddings. The likelihood ratio is computed as
    /// `P_indep(partner_a | self_a, situation)`, which for a uniform action
    /// distribution reduces to `1/|A|`. For the general continuous case the
    /// caller must supply a likelihood; this method delegates to
    /// [`Self::observe_match`] for the canonical discrete matched-action case.
    ///
    /// **Phase 1 (this module) implements only the discrete-action closed
    /// form.** The continuous-embedding likelihood is a Phase 3 concern
    /// (indirect inference + arbitrary embeddings); it requires a likelihood
    /// function the caller supplies. Use [`Self::observe_match`] for the G1
    /// validated path.
    ///
    /// This signature exists so consumers can wire up the trait surface today
    /// and slot in their likelihood model later.
    pub fn observe(
        &mut self,
        _self_a: &[f32],
        _partner_a: &[f32],
        _situation: &[f32],
        log_likelihood_under_independence: f32,
    ) {
        // log W += log P(a_partner | situation) under independence
        self.log_w_independent += log_likelihood_under_independence;
        self.observation_count = self.observation_count.saturating_add(1);
        self.recompute_omega();
    }

    /// Incorporate one matched discrete-action observation: the focal and
    /// partner both played action `a_self == a_partner` in the same situation,
    /// in a symmetric game with `n_actions` total actions.
    ///
    /// Each match contributes `log(1/n_actions)` to `log_w_independent`
    /// (the "miracle" evidence: under independence the partner would play the
    /// same action with probability `1/n_actions`). This is the canonical
    /// closed-form path validated by G1.
    ///
    /// # Panics
    ///
    /// Debug-only: panics if `n_actions == 0`. The math is undefined for an
    /// empty action set.
    #[inline]
    pub fn observe_match(&mut self, n_actions: usize) {
        debug_assert!(n_actions > 0, "n_actions must be > 0");
        // log(1/n) = -log(n). Use ln (natural log) — the posterior formula
        // uses exp(), which is e-based.
        let log_inv_n = -(n_actions as f32).ln();
        self.log_w_independent += log_inv_n;
        self.observation_count = self.observation_count.saturating_add(1);
        self.recompute_omega();
    }

    /// Incorporate one *mismatched* discrete-action observation: the partner
    /// played a different action from the focal.
    ///
    /// Under the perfect-identity shared hypothesis, the partner mirrors the
    /// focal's action with probability 1 (the Kronecker delta `δ(a_partner,
    /// a_self) = 1` for a match, `0` for a mismatch). A single mismatch is
    /// therefore *impossible* under the shared hypothesis — it has likelihood
    /// ratio `LR = P(mismatch|shared) / P(mismatch|indep) = 0 / (1/|A|) = 0`,
    /// which drives the posterior `ω` to exactly 0.
    ///
    /// This is mathematically correct for the perfect-identity model (Plan 526
    /// Phase 2 PoC: shared-shard pairs are deterministic-same-policy, so a
    /// mismatch never occurs for them; random-shard pairs will mismatch at
    /// least once in T=50 rounds with overwhelming probability, setting their
    /// ω to 0 and forcing Defect). For a *soft* identity model (shared
    /// partners match with probability `δ < 1`), pass a custom likelihood to
    /// [`Self::observe`] instead.
    ///
    /// # Implementation
    ///
    /// Sets `log_w_independent = +∞`, which makes `W = exp(+∞) = +∞` and thus
    /// `ω = α / (α + (1−α)·∞) = 0`. Once set, subsequent `observe_match` calls
    /// cannot recover ω (the evidence is permanent — you can't "unsee" a
    /// contradiction of the identity hypothesis). This matches the paper's
    /// perfect-identity Bayes update.
    #[inline]
    pub fn observe_mismatch(&mut self, _n_actions: usize) {
        // A mismatch is impossible under the perfect-identity shared
        // hypothesis: LR = 0 → W = +∞ → ω = 0 permanently.
        self.log_w_independent = f32::INFINITY;
        self.observation_count = self.observation_count.saturating_add(1);
        self.last_omega = 0.0;
    }

    /// Whether the posterior has been permanently driven to zero by a
    /// mismatched observation under the perfect-identity model. Once true, no
    /// further evidence can revive ω.
    #[inline]
    pub fn is_collapsed_to_zero(&self) -> bool {
        self.log_w_independent.is_infinite() && self.log_w_independent > 0.0
    }

    /// Current posterior `ω_T`. Recomputed lazily on each observe; this is a
    /// plain field read.
    #[inline]
    pub fn omega(&self) -> f32 {
        self.last_omega
    }

    /// Log of `W(æ_<T)` — the joint-action likelihood under the independent
    /// marginal. Exposed for diagnostic / debugging (the G1 test asserts it
    /// matches `−T·ln(|A|)`).
    #[inline]
    pub fn log_w_independent(&self) -> f32 {
        self.log_w_independent
    }

    /// Predictive similarity for a contemplated action: the difference in
    /// predicted partner behavior between "focal plays `contemplated`" and
    /// "focal plays anything else". Under the shared-shard hypothesis this
    /// equals `ω_T` exactly; for arbitrary agents it's a useful heuristic.
    ///
    /// For Phase 1 (discrete closed form): `S_pred(a) = ω · 1 + (1−ω) · q(a)`,
    /// which for uniform `q` and `|A|=2` simplifies to `(1+ω)/2`.
    ///
    /// For the continuous-embedding path: `S_pred = sigmoid(dot(contemplated,
    /// identity_direction))` — a sigmoid projection of the contemplated
    /// embedding onto an identity direction. This is the canonical latent
    /// bridge pattern (AGENTS.md §"Bridge Pattern"): raw → latent via
    /// dot-product + sigmoid.
    ///
    /// Phase 1 implements the discrete form; Phase 3 wires the continuous form
    /// once `identity_direction` is supplied by the consumer's embedding model.
    #[inline]
    pub fn predictive_similarity(&self, contemplated_dot_identity: f32) -> f32 {
        // Discrete shared-shard prediction: S_pred(a) = ω + (1−ω) · q(a)
        // For the contemplated action, the shared-shard contribution is ω.
        // The independent contribution is (1−ω) · q(a), which for uniform q
        // and |A|=2 is (1−ω)/2. Phase 1 exposes this as a single scalar.
        //
        // The continuous-embedding caller passes `dot(contemplated,
        // identity_direction)` here; we apply a sigmoid so the result is in
        // (0, 1), then blend with the posterior.
        let sigmoid_proj = sigmoid(contemplated_dot_identity);
        // Blend: prior similarity × projected similarity. The math reduces to
        // ω · sigmoid_proj + (1−ω) · sigmoid_proj · (1−sigmoid_proj) which is
        // a regularized form. For Phase 1 G1 validation, the caller uses the
        // pure-discrete path via `omega()` directly.
        self.last_omega * sigmoid_proj + (1.0 - self.last_omega) * sigmoid_proj * 0.5
    }

    /// Update ω from a parallel third-party encounter: the focal and the
    /// partner both played against the SAME NPC in the SAME situation, and
    /// the focal observed the partner's action. This is the **indirect
    /// inference** path (Plan 526 Phase 3, paper §H.4) — the focal and
    /// partner never interacted directly, but the focal can still infer
    /// similarity by comparing "what I played" to "what the partner played"
    /// in the shared situation.
    ///
    /// Mathematically identical to a direct observation: the evidence is the
    /// same (match or mismatch on the action pair). The distinction is purely
    /// semantic — where the evidence came from. The posterior update doesn't
    /// care whether the partner's action was observed directly or via a shared
    /// third-party encounter.
    ///
    /// This method is provided for API clarity (consumers that want to
    /// distinguish direct vs indirect evidence in their logs). The PoC in
    /// `poc::IndirectAgent` calls `observe_match`/`observe_mismatch` directly
    /// — this wrapper is for consumers that prefer the semantic name.
    #[inline]
    pub fn observe_third_party(&mut self, my_action: u8, their_action: u8, n_actions: usize) {
        if their_action == my_action {
            self.observe_match(n_actions);
        } else {
            self.observe_mismatch(n_actions);
        }
    }

    /// Recompute `last_omega` from the current `log_w_independent`. Called
    /// after every observe so reads are free.
    #[inline]
    fn recompute_omega(&mut self) {
        // ω = α / (α + (1−α) · W) = α / (α + (1−α) · exp(log_w))
        let w = self.log_w_independent.exp();
        let denom = self.prior_alpha + (1.0 - self.prior_alpha) * w;
        // denom > 0 by construction (both α and (1−α)·W are positive when
        // α ∈ (0,1) and W > 0).
        self.last_omega = self.prior_alpha / denom;
    }
}
