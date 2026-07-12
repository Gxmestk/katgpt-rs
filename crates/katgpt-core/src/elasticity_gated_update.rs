//! Elasticity-Gated Update — DSOM error-scaled neighborhood update primitive
//! (Plan 429, Research 415, Rougier & Boniface "Dynamic Self-Organising Map"
//! Neurocomputing 2011, ⟨inria-00495827⟩ + survey arXiv:2501.08416).
//!
//! A generic, modelless, MIT-licensed primitive: compute a DSOM-style
//! elasticity-gated update delta on any latent-state vector. The step size
//! scales with the local error (plasticity when far, stability when close),
//! and the neighborhood weights are error-gated (wide neighborhood when the
//! error is large, tight when small).
//!
//! # The math
//!
//! ```text
//! error   = ‖target − state‖ / support_diameter       // normalized L2
//! step    = ε · error                                   // error-scaled step
//! weightᵢ = exp(−dᵢ² / (η² · error²))                  // error-gated Gaussian
//! delta   = step · Σ(weightᵢ · (neighborᵢ − state)) / Σ(weightᵢ)
//! ```
//!
//! The caller applies: `state_new = state + delta`.
//!
//! # Properties
//!
//! - **Error-scaled step**: when the state is far from the target, the step is
//!   large (plasticity); when close, the step is small (stability).
//! - **Error-gated neighborhood**: when the error is large, the neighborhood
//!   expands (many neighbors contribute); when small, it contracts (only the
//!   nearest neighbors contribute).
//! - **Structure-matching** (the DSOM headline): the update maps the *support*
//!   of the distribution, not its density — rare regions get equal
//!   representation (unlike standard SOM which follows the magnification law
//!   `P(w) ∝ ρ(w)^α`). This property needs a PoC (Plan 429 Phase 3 / §3.6).
//! - **Time-invariant**: no decaying learning rate or neighborhood width. The
//!   same η governs plasticity-stability at all times — enables lifelong
//!   online adaptation without reset.
//!
//! # Why Gaussian, not sigmoid (AGENTS.md §2)
//!
//! The AGENTS.md "sigmoid not softmax" rule is about *normalization* — softmax
//! forces competition (Σ=1), sigmoid allows independent per-pair gating. The
//! DSOM neighborhood function is a Gaussian weight `exp(−d²/(η²·error²))`,
//! which is already independent per neighbor (no normalization to 1). Each
//! weight is in `(0, 1]` and decays smoothly with distance. This is the
//! standard SOM neighborhood function; using sigmoid here would change the
//! semantics (sigmoid is a step-like gate, Gaussian is a smooth decay).
//!
//! # Zero-allocation
//!
//! [`elasticity_gated_update_into`] takes only borrowed slices and writes into
//! a caller-provided output buffer. Weights are computed into a stack-
//! allocated `[f32; 32]` buffer (max 32 neighbors). No heap allocation in
//! steady state.
//!
//! # Modelless
//!
//! Pure closed-form math (exponential + weighted average). No training, no
//! backprop, no gradient descent. The only state mutation is the latent-space
//! update, permitted under the modelless mandate.
//!
//! # Sync boundary (AGENTS.md)
//!
//! The error and neighborhood weights are computed in latent space (L2 on
//! style_weights / HLA). Only the post-update `state_new` + BLAKE3 cross the
//! sync boundary — same as today's `neighbor_heal`.
//!
//! # Feature gate
//!
//! Gated behind the `elasticity_gated_update` Cargo feature (opt-in, no
//! default). Promotion to default-on requires the GOAT gate (G1–G6) to pass —
//! see Plan 429.
//!
//! # References
//!
//! - Plan: `katgpt-rs/.plans/429_elasticity_gated_update_dsom_primitive.md`
//! - Research: `katgpt-rs/.research/415_Dynamic_SOM_Elasticity_Gated_Latent_Update.md`
//! - Source paper: Rougier & Boniface, "Dynamic Self-Organising Map"
//!   (Neurocomputing 2011, ⟨inria-00495827⟩)
//! - Survey: Guérin et al., arXiv:2501.08416
//! - Closest shipped cousin: `riir-neuron-db::neighbor_heal` (fixed alpha,
//!   sigmoid-gated cosine weights — no error scaling, no neighborhood expansion)

#![allow(clippy::needless_range_loop)] // explicit indexing aids SIMD auto-vectorization

/// Maximum number of neighbors supported by the stack-allocated weights buffer.
const MAX_NEIGHBORS: usize = 32;

/// Threshold below which the error is considered zero (no heal needed).
/// When `error < ZERO_ERROR_THRESHOLD`, the output is all zeros.
const ZERO_ERROR_THRESHOLD: f32 = 1e-8;

/// Threshold below which the weight sum is considered zero (no neighborhood
/// signal). When `Σweight < ZERO_WEIGHT_THRESHOLD`, the output is all zeros.
const ZERO_WEIGHT_THRESHOLD: f32 = 1e-10;

// ─── Config ───────────────────────────────────────────────────────────────

/// Configuration for [`elasticity_gated_update_into`].
///
/// All fields are `Copy` so the config can be passed by value cheaply.
///
/// # Fields
///
/// - `eta` (η): elasticity parameter. Controls the plasticity-stability
///   tradeoff. Higher η → wider neighborhood (more plastic, adapts faster but
///   may overshoot). Lower η → tighter neighborhood (more stable, slower
///   adaptation). Default 1.0.
/// - `epsilon` (ε): base step size. The actual step is `ε · error`, so the
///   effective step is bounded by `ε` (since error ∈ [0, 1] after
///   normalization). Default 0.1 (matching `NeighborHealConfig::alpha`).
/// - `support_diameter` (Ω): normalization factor for the error. Typically the
///   max pairwise L2 distance in the latent space. The caller computes this
///   periodically (e.g. per sleep-cycle, not per tick). Default 1.0.
#[derive(Clone, Copy, Debug)]
pub struct ElasticityConfig {
    pub eta: f32,
    pub epsilon: f32,
    pub support_diameter: f32,
}

impl Default for ElasticityConfig {
    fn default() -> Self {
        Self {
            eta: 1.0,
            epsilon: 0.1,
            support_diameter: 1.0,
        }
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────

/// Compute the normalized error: `‖target − state‖ / support_diameter`.
///
/// This is the DSOM error signal — the distance from the current state to the
/// target, normalized by the support diameter so it's roughly in `[0, 1]`.
#[inline]
pub fn compute_error(state: &[f32], target: &[f32], support_diameter: f32) -> f32 {
    debug_assert_eq!(state.len(), target.len());
    let diff_sq: f32 = state
        .iter()
        .zip(target.iter())
        .map(|(s, t)| {
            let d = t - s;
            d * d
        })
        .sum();
    diff_sq.sqrt() / support_diameter
}

/// Compute the DSOM neighborhood weight for a single neighbor.
///
/// `weight = exp(−d² / (η² · error²))`
///
/// When the error is small, the denominator is small → the weight decays
/// quickly with distance (tight neighborhood). When the error is large, the
/// denominator is large → the weight decays slowly (wide neighborhood).
///
/// Returns 0.0 when `error < ZERO_ERROR_THRESHOLD` (the zero-error guard
/// prevents division by near-zero `error²`).
#[inline]
pub fn neighborhood_weight(lattice_distance: f32, error: f32, eta: f32) -> f32 {
    if error < ZERO_ERROR_THRESHOLD {
        return 0.0;
    }
    let d_sq = lattice_distance * lattice_distance;
    let eta_sq = eta * eta;
    let error_sq = error * error;
    let arg = -d_sq / (eta_sq * error_sq);
    arg.exp()
}

/// Compute the effective neighborhood size (participation ratio).
///
/// `effective_k = (Σw)² / Σ(w²)` — the inverse Herfindahl index.
///
/// - Returns `N` when all `N` weights are equal (maximum participation).
/// - Returns `1` when one weight dominates.
/// - Returns `0` when all weights are negligible (sum of squares below
///   [`ZERO_WEIGHT_THRESHOLD`]).
///
/// Useful for diagnosing neighborhood expansion: a higher `effective_k` means
/// more neighbors are contributing meaningfully.
#[inline]
pub fn effective_neighborhood_size(weights: &[f32]) -> f32 {
    let sum: f32 = weights.iter().copied().sum();
    let sum_sq: f32 = weights.iter().map(|w| w * w).sum();
    if sum_sq < ZERO_WEIGHT_THRESHOLD {
        0.0
    } else {
        (sum * sum) / sum_sq
    }
}

// ─── Core function ────────────────────────────────────────────────────────

/// Compute the DSOM elasticity-gated update delta.
///
/// Writes the update delta into `out`. The caller applies:
/// `state_new = state + out`.
///
/// # Arguments
///
/// - `state` — current state vector (e.g. damaged shard's `style_weights`).
/// - `target` — target to move toward (e.g. neighbor centroid or observation).
///   Only used to compute the error signal — the direction of movement is
///   toward the weighted neighbor centroid, NOT directly toward `target`.
/// - `neighbor_states` — neighboring state vectors (slices into their data).
/// - `lattice_distances` — lattice distance for each neighbor (e.g. cosine
///   distance, rank distance, or grid distance). Must have the same length as
///   `neighbor_states`. All distances MUST be `≥ 0`.
/// - `config` — elasticity parameters (η, ε, Ω).
/// - `out` — output buffer, same length as `state`. Filled with the update
///   delta.
///
/// # Math
///
/// ```text
/// error   = ‖target − state‖ / support_diameter
/// step    = ε · error
/// weightᵢ = exp(−dᵢ² / (η² · error²))
/// delta   = step · Σ(weightᵢ · (neighborᵢ − state)) / Σ(weightᵢ)
/// ```
///
/// This is equivalent to: `delta = step · (weighted_centroid − state)`, where
/// `weighted_centroid = Σ(weightᵢ · neighborᵢ) / Σ(weightᵢ)`.
///
/// # Guards
///
/// - **Zero-error guard**: when `error < 1e-8`, `out` is filled with zeros
///   (state is already at the target — no heal needed).
/// - **Zero-weight guard**: when `Σweight < 1e-10`, `out` is filled with zeros
///   (no neighborhood signal — all neighbors are too far given the error).
/// - **Empty neighbors**: when `neighbor_states` is empty, `out` is filled
///   with zeros.
///
/// # Zero-allocation
///
/// Weights are computed into a stack-allocated `[f32; 32]` buffer. No heap
/// allocation. The output is written into the caller-provided `out` buffer.
///
/// # Determinism
///
/// The FMA chain in the inner loop and the `exp` computation are bit-identical
/// across all nodes running the same binary (quorum bit-identity gate G4).
pub fn elasticity_gated_update_into(
    state: &[f32],
    target: &[f32],
    neighbor_states: &[&[f32]],
    lattice_distances: &[f32],
    config: &ElasticityConfig,
    out: &mut [f32],
) {
    let dim = state.len();
    debug_assert_eq!(target.len(), dim, "state and target must have the same length");
    debug_assert_eq!(out.len(), dim, "out must have the same length as state");
    debug_assert_eq!(
        neighbor_states.len(),
        lattice_distances.len(),
        "neighbor_states and lattice_distances must have the same length"
    );
    debug_assert!(
        neighbor_states.len() <= MAX_NEIGHBORS,
        "at most {MAX_NEIGHBORS} neighbors supported (got {})",
        neighbor_states.len()
    );
    debug_assert!(config.eta > 0.0, "eta must be positive");
    debug_assert!(
        config.support_diameter > 0.0,
        "support_diameter must be positive"
    );

    let n = neighbor_states.len();

    // ── Empty neighbors: no signal ───────────────────────────────────────
    if n == 0 {
        out.fill(0.0);
        return;
    }

    // ── Compute error ────────────────────────────────────────────────────
    let error = compute_error(state, target, config.support_diameter);

    // Zero-error guard: state is already at the target.
    if error < ZERO_ERROR_THRESHOLD {
        out.fill(0.0);
        return;
    }

    // ── Compute neighborhood weights (stack-allocated) ──────────────────
    let mut weights: [f32; MAX_NEIGHBORS] = [0.0; MAX_NEIGHBORS];
    for i in 0..n {
        weights[i] = neighborhood_weight(lattice_distances[i], error, config.eta);
    }
    let w_sum: f32 = weights[..n].iter().copied().sum();

    // Zero-weight guard: no neighborhood signal.
    if w_sum < ZERO_WEIGHT_THRESHOLD {
        out.fill(0.0);
        return;
    }

    // ── Accumulate weighted differences ─────────────────────────────────
    //
    // delta = step · Σ(wᵢ · (neighborᵢ − state)) / Σ(wᵢ)
    //       = (step / Σw) · Σ(wᵢ · (neighborᵢ − state))
    //
    // We pre-multiply each weight by `step / w_sum` so the inner loop is a
    // single FMA per lane: `out[lane] += scale · (neighbor[lane] − state[lane])`.
    let step = config.epsilon * error;
    let scale_factor = step / w_sum;

    out.fill(0.0);
    for (i, neighbor) in neighbor_states.iter().enumerate() {
        debug_assert_eq!(neighbor.len(), dim, "neighbor {} has wrong dimension", i);
        let scale = weights[i] * scale_factor;

        let mut lane = 0;
        // Chunk-8 unrolled for SIMD auto-vectorization (matches neighbor_heal_delta_into).
        while lane + 8 <= dim {
            out[lane] = (neighbor[lane] - state[lane]).mul_add(scale, out[lane]);
            out[lane + 1] = (neighbor[lane + 1] - state[lane + 1]).mul_add(scale, out[lane + 1]);
            out[lane + 2] = (neighbor[lane + 2] - state[lane + 2]).mul_add(scale, out[lane + 2]);
            out[lane + 3] = (neighbor[lane + 3] - state[lane + 3]).mul_add(scale, out[lane + 3]);
            out[lane + 4] = (neighbor[lane + 4] - state[lane + 4]).mul_add(scale, out[lane + 4]);
            out[lane + 5] = (neighbor[lane + 5] - state[lane + 5]).mul_add(scale, out[lane + 5]);
            out[lane + 6] = (neighbor[lane + 6] - state[lane + 6]).mul_add(scale, out[lane + 6]);
            out[lane + 7] = (neighbor[lane + 7] - state[lane + 7]).mul_add(scale, out[lane + 7]);
            lane += 8;
        }
        // Remainder.
        while lane < dim {
            out[lane] = (neighbor[lane] - state[lane]).mul_add(scale, out[lane]);
            lane += 1;
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── G1: Error-scaled step ────────────────────────────────────────────

    /// Verify that the step scales linearly with the error.
    ///
    /// `step(δ=0.5) / step(δ=0.01) ≈ 50` (within 5%).
    ///
    /// Uses equal lattice distances so the weighted centroid is the same for
    /// both errors, making the delta ratio equal to the step ratio = error ratio.
    #[test]
    fn g1_error_scaled_step_ratio() {
        let state = [0.0f32, 0.0, 0.0, 0.0];
        // target at distance 0.5 → error = 0.5 (support_diameter = 1.0)
        let target_large = [0.5f32, 0.0, 0.0, 0.0];
        // target at distance 0.01 → error = 0.01
        let target_small = [0.01f32, 0.0, 0.0, 0.0];

        let neighbor_data = [[1.0f32, 0.0, 0.0, 0.0], [0.0f32, 1.0, 0.0, 0.0]];
        let neighbors: Vec<&[f32]> = neighbor_data.iter().map(|n| n.as_slice()).collect();
        // Very small equal distances → weights ≈ equal for both errors,
        // so weighted centroid is the same → delta ratio = step ratio = error ratio.
        let distances = [0.001f32, 0.001];

        let config = ElasticityConfig {
            epsilon: 0.1,
            ..Default::default()
        };

        let mut delta_large = [0.0f32; 4];
        let mut delta_small = [0.0f32; 4];
        elasticity_gated_update_into(
            &state,
            &target_large,
            &neighbors,
            &distances,
            &config,
            &mut delta_large,
        );
        elasticity_gated_update_into(
            &state,
            &target_small,
            &neighbors,
            &distances,
            &config,
            &mut delta_small,
        );

        let norm_large = delta_large.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_small = delta_small.iter().map(|x| x * x).sum::<f32>().sqrt();

        assert!(norm_large > 0.0, "delta_large should be non-zero");
        assert!(norm_small > 0.0, "delta_small should be non-zero");

        let ratio = norm_large / norm_small;
        let expected = 50.0f32; // 0.5 / 0.01
        assert!(
            (ratio - expected).abs() / expected < 0.05,
            "step ratio should be ~{expected} (within 5%), got {ratio}"
        );
    }

    // ── G2: Neighborhood expansion ───────────────────────────────────────

    /// Verify that the neighborhood expands with error.
    ///
    /// At `error=0.01`, `effective_k ≤ 2` (tight neighborhood).
    /// At `error=0.5`, `effective_k ≥ 5` (wide neighborhood, all 5 neighbors).
    ///
    /// Uses equal lattice distances for the `≥ 5` case — equal distances →
    /// equal weights → `effective_k = 5` (the maximum for 5 neighbors).
    #[test]
    fn g2_neighborhood_expansion() {
        let distances = [0.1f32; 5];

        // error = 0.5: all weights ≈ exp(−0.01/0.25) ≈ 0.961 (equal) → ek ≈ 5
        let weights_large: Vec<f32> = distances
            .iter()
            .map(|&d| neighborhood_weight(d, 0.5, 1.0))
            .collect();
        let ek_large = effective_neighborhood_size(&weights_large);
        assert!(
            ek_large >= 5.0 - 0.01,
            "effective_k at error=0.5 should be ~5 (equal weights), got {ek_large}"
        );

        // error = 0.01: all weights ≈ exp(−100) ≈ 0 → ek = 0 (zero-weight guard)
        let weights_small: Vec<f32> = distances
            .iter()
            .map(|&d| neighborhood_weight(d, 0.01, 1.0))
            .collect();
        let ek_small = effective_neighborhood_size(&weights_small);
        assert!(
            ek_small <= 2.0,
            "effective_k at error=0.01 should be ≤ 2, got {ek_small}"
        );
    }

    /// Neighborhood expansion with varying distances — verifies the
    /// directional property (larger error → wider neighborhood) even when
    /// distances are not all equal.
    #[test]
    fn g2_neighborhood_expansion_varying_distances() {
        let distances = [0.001f32, 0.01, 0.1, 0.5, 1.0];

        let weights_large: Vec<f32> = distances
            .iter()
            .map(|&d| neighborhood_weight(d, 0.5, 1.0))
            .collect();
        let ek_large = effective_neighborhood_size(&weights_large);

        let weights_small: Vec<f32> = distances
            .iter()
            .map(|&d| neighborhood_weight(d, 0.01, 1.0))
            .collect();
        let ek_small = effective_neighborhood_size(&weights_small);

        assert!(
            ek_large > ek_small,
            "effective_k at error=0.5 ({ek_large}) should be > at error=0.01 ({ek_small})"
        );
        // At small error, only 1-2 nearest neighbors contribute.
        assert!(
            ek_small <= 2.0,
            "effective_k at error=0.01 should be ≤ 2, got {ek_small}"
        );
    }

    // ── G3: Zero-error guard ─────────────────────────────────────────────

    /// When the state equals the target (error = 0), the output must be all
    /// zeros — no heal needed.
    #[test]
    fn g3_zero_error_guard() {
        let state = [1.0f32, 2.0, 3.0];
        let target = [1.0f32, 2.0, 3.0]; // same as state → error = 0
        let neighbor_data = [[2.0f32, 3.0, 4.0]];
        let neighbors: Vec<&[f32]> = neighbor_data.iter().map(|n| n.as_slice()).collect();
        let distances = [0.1f32];
        let config = ElasticityConfig::default();
        let mut out = [0.0f32; 3];

        elasticity_gated_update_into(&state, &target, &neighbors, &distances, &config, &mut out);

        assert!(
            out.iter().all(|&x| x == 0.0),
            "zero error should produce zero delta, got {:?}",
            out
        );
    }

    /// When error is below the threshold but non-zero, the output must still
    /// be all zeros (the zero-error guard catches it).
    #[test]
    fn g3_near_zero_error_guard() {
        let state = [0.0f32, 0.0];
        // distance = 1e-9 < ZERO_ERROR_THRESHOLD (1e-8)
        let target = [1e-9f32, 0.0];
        let neighbor_data = [[1.0f32, 0.0]];
        let neighbors: Vec<&[f32]> = neighbor_data.iter().map(|n| n.as_slice()).collect();
        let distances = [0.1f32];
        let config = ElasticityConfig::default();
        let mut out = [0.0f32; 2];

        elasticity_gated_update_into(&state, &target, &neighbors, &distances, &config, &mut out);

        assert!(
            out.iter().all(|&x| x == 0.0),
            "near-zero error should produce zero delta, got {:?}",
            out
        );
    }

    // ── Zero-weight guard ────────────────────────────────────────────────

    /// When all weights are effectively zero (very large lattice distances
    /// relative to the error), the output must be all zeros — no neighborhood
    /// signal.
    #[test]
    fn zero_weight_guard() {
        let state = [0.0f32, 0.0];
        let target = [0.5f32, 0.0]; // error = 0.5
        let neighbor_data = [[1.0f32, 0.0]];
        let neighbors: Vec<&[f32]> = neighbor_data.iter().map(|n| n.as_slice()).collect();
        // Huge lattice distance → weight = exp(−10000/0.25) ≈ 0
        let distances = [100.0f32];
        let config = ElasticityConfig::default();
        let mut out = [0.0f32; 2];

        elasticity_gated_update_into(&state, &target, &neighbors, &distances, &config, &mut out);

        assert!(
            out.iter().all(|&x| x == 0.0),
            "zero-weight guard should produce zero delta, got {:?}",
            out
        );
    }

    // ── Empty neighbors ──────────────────────────────────────────────────

    /// When there are no neighbors, the output must be all zeros.
    #[test]
    fn empty_neighbors_guard() {
        let state = [0.5f32, 0.3];
        let target = [0.7f32, 0.2];
        let neighbors: Vec<&[f32]> = vec![];
        let distances: [f32; 0] = [];
        let config = ElasticityConfig::default();
        let mut out = [0.0f32; 2];

        elasticity_gated_update_into(&state, &target, &neighbors, &distances, &config, &mut out);

        assert!(
            out.iter().all(|&x| x == 0.0),
            "empty neighbors should produce zero delta, got {:?}",
            out
        );
    }

    // ── G4: Determinism (quorum bit-identity) ────────────────────────────

    /// Same input → bit-identical output across 100 runs.
    #[test]
    fn g4_determinism() {
        let state = [0.5f32, 0.3, 0.8, 0.1, 0.9];
        let target = [0.7f32, 0.2, 0.6, 0.3, 0.8];
        let neighbor_data = [
            [0.6f32, 0.4, 0.7, 0.2, 0.85],
            [0.4f32, 0.35, 0.75, 0.15, 0.88],
            [0.55f32, 0.25, 0.65, 0.25, 0.82],
        ];
        let neighbors: Vec<&[f32]> = neighbor_data.iter().map(|n| n.as_slice()).collect();
        let distances = [0.1f32, 0.15, 0.2];
        let config = ElasticityConfig::default();

        let mut out_ref = [0.0f32; 5];
        elasticity_gated_update_into(
            &state,
            &target,
            &neighbors,
            &distances,
            &config,
            &mut out_ref,
        );

        for run in 0..100 {
            let mut out = [0.0f32; 5];
            elasticity_gated_update_into(
                &state,
                &target,
                &neighbors,
                &distances,
                &config,
                &mut out,
            );
            for i in 0..5 {
                assert_eq!(
                    out_ref[i].to_bits(),
                    out[i].to_bits(),
                    "non-deterministic output at lane {i} on run {run}: ref={} got={}",
                    out_ref[i],
                    out[i]
                );
            }
        }
    }

    // ── Direction correctness ────────────────────────────────────────────

    /// The delta should point from state toward the weighted neighbor centroid.
    ///
    /// With equal weights (equal distances), the weighted centroid = unweighted
    /// centroid. For state=[0,0] and neighbors [[1,0],[0,1]], the centroid is
    /// [0.5, 0.5]. With error=0.5, ε=0.1: step = 0.05, and
    /// delta = step · (centroid − state) = 0.05 · [0.5, 0.5] = [0.025, 0.025].
    #[test]
    fn delta_points_toward_weighted_centroid() {
        let state = [0.0f32, 0.0];
        let target = [0.5f32, 0.0]; // error = 0.5
        let neighbor_data = [[1.0f32, 0.0], [0.0f32, 1.0]];
        let neighbors: Vec<&[f32]> = neighbor_data.iter().map(|n| n.as_slice()).collect();
        let distances = [0.1f32, 0.1]; // equal → equal weights → centroid = [0.5, 0.5]
        let config = ElasticityConfig::default();
        let mut out = [0.0f32; 2];

        elasticity_gated_update_into(&state, &target, &neighbors, &distances, &config, &mut out);

        // Both components should be positive (moving toward [0.5, 0.5] from [0, 0]).
        assert!(out[0] > 0.0, "delta[0] should be positive, got {}", out[0]);
        assert!(out[1] > 0.0, "delta[1] should be positive, got {}", out[1]);

        // With equal weights: scale = w * step / (2*w) = step/2 = 0.025
        // delta[lane] = 0.025 * (neighbor[lane] - state[lane])
        // delta[0] = 0.025 * (1.0 - 0.0) = 0.025  (from neighbor 0)
        //          + 0.025 * (0.0 - 0.0) = 0      (from neighbor 1)
        // delta[1] = 0.025 * (0.0 - 0.0) = 0      (from neighbor 0)
        //          + 0.025 * (1.0 - 0.0) = 0.025  (from neighbor 1)
        assert!(
            (out[0] - 0.025).abs() < 1e-5,
            "delta[0] should be ~0.025, got {}",
            out[0]
        );
        assert!(
            (out[1] - 0.025).abs() < 1e-5,
            "delta[1] should be ~0.025, got {}",
            out[1]
        );
    }

    // ── Config defaults ─────────────────────────────────────────────────

    #[test]
    fn config_default() {
        let config = ElasticityConfig::default();
        assert_eq!(config.eta, 1.0);
        assert_eq!(config.epsilon, 0.1);
        assert_eq!(config.support_diameter, 1.0);
    }

    // ── Larger dimension (STYLE_DIM=64 compatibility) ───────────────────

    /// Verify the chunk-8 unrolled loop works correctly at STYLE_DIM=64
    /// (the dimension used by `NeuronShard::style_weights`).
    #[test]
    fn works_at_style_dim_64() {
        let dim = 64;
        let state: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.01).collect();
        let target: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.01 + 0.5).collect();
        let neighbor_data: Vec<[f32; 64]> = (0..5)
            .map(|n| {
                let mut arr = [0.0f32; 64];
                for i in 0..64 {
                    arr[i] = (i as f32) * 0.01 + 0.1 * (n as f32 + 1.0);
                }
                arr
            })
            .collect();
        let neighbors: Vec<&[f32]> = neighbor_data.iter().map(|n| n.as_slice()).collect();
        let distances = [0.1f32, 0.2, 0.3, 0.4, 0.5];
        let config = ElasticityConfig::default();
        let mut out = vec![0.0f32; dim];

        elasticity_gated_update_into(&state, &target, &neighbors, &distances, &config, &mut out);

        // Verify not all zero (the error is non-trivial).
        let norm: f32 = out.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm > 0.0, "delta should be non-zero at dim=64");

        // Verify determinism.
        let mut out2 = vec![0.0f32; dim];
        elasticity_gated_update_into(&state, &target, &neighbors, &distances, &config, &mut out2);
        for i in 0..dim {
            assert_eq!(out[i].to_bits(), out2[i].to_bits(), "non-deterministic at lane {i}");
        }
    }

    // ── compute_error unit test ──────────────────────────────────────────

    #[test]
    fn compute_error_basic() {
        let state = [0.0f32, 0.0];
        let target = [3.0f32, 4.0]; // L2 distance = 5.0
        let error = compute_error(&state, &target, 10.0);
        assert!((error - 0.5).abs() < 1e-6, "error should be 0.5, got {}", error);
    }

    #[test]
    fn compute_error_zero() {
        let state = [1.0f32, 2.0, 3.0];
        let error = compute_error(&state, &state, 1.0);
        assert!(error < ZERO_ERROR_THRESHOLD, "error should be ~0, got {}", error);
    }

    // ── effective_neighborhood_size unit tests ───────────────────────────

    #[test]
    fn effective_k_equal_weights() {
        let weights = [0.5f32, 0.5, 0.5, 0.5, 0.5];
        let ek = effective_neighborhood_size(&weights);
        assert!((ek - 5.0).abs() < 1e-5, "equal weights → ek=5, got {}", ek);
    }

    #[test]
    fn effective_k_dominant_weight() {
        let weights = [1.0f32, 1e-6, 1e-6];
        let ek = effective_neighborhood_size(&weights);
        assert!(
            (ek - 1.0).abs() < 0.01,
            "dominant weight → ek≈1, got {}",
            ek
        );
    }

    #[test]
    fn effective_k_all_zero() {
        let weights = [0.0f32, 0.0, 0.0];
        let ek = effective_neighborhood_size(&weights);
        assert_eq!(ek, 0.0, "all-zero weights → ek=0, got {}", ek);
    }

    // ── neighborhood_weight unit tests ───────────────────────────────────

    #[test]
    fn neighborhood_weight_zero_error() {
        let w = neighborhood_weight(0.1, 0.0, 1.0);
        assert_eq!(w, 0.0, "zero error → weight=0");
    }

    #[test]
    fn neighborhood_weight_zero_distance() {
        // d=0 → weight = exp(0) = 1.0 (the winner itself)
        let w = neighborhood_weight(0.0, 0.5, 1.0);
        assert!((w - 1.0).abs() < 1e-6, "zero distance → weight=1, got {}", w);
    }

    #[test]
    fn neighborhood_weight_decreases_with_distance() {
        let error = 0.5;
        let eta = 1.0;
        let w1 = neighborhood_weight(0.1, error, eta);
        let w2 = neighborhood_weight(0.5, error, eta);
        assert!(w1 > w2, "closer neighbor should have higher weight: {} vs {}", w1, w2);
    }

    #[test]
    fn neighborhood_weight_increases_with_error() {
        let d = 0.1;
        let eta = 1.0;
        let w_small = neighborhood_weight(d, 0.01, eta);
        let w_large = neighborhood_weight(d, 0.5, eta);
        assert!(
            w_large > w_small,
            "larger error → wider neighborhood (higher weight): {} vs {}",
            w_large,
            w_small
        );
    }
}
