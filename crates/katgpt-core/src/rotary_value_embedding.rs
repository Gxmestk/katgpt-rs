//! RoVE — Rotary Value Embeddings Attention.
//!
//! Distilled from García-Castellanos, Weiler, Bekkers, *RoVE: Rotary Value
//! Embeddings Attention for Relative Position-dependent Value Pathways*
//! (arXiv:2606.11275, Jul 2026, [code](https://github.com/AGarciaCast/RoVE)).
//! See [Research 452](../../.research/452_RoVE_Rotary_Value_Embeddings_Attentive_Convolution.md)
//! for the full distillation and [Plan 557](../../.plans/557_rotary_value_embeddings.md)
//! for the execution tracker.
//!
//! # The asymmetry RoVE fixes
//!
//! Standard RoPE rotates queries `q_i` by `R_i` and keys `k_j` by `R_j` before
//! the inner product, making attention scores depend only on the relative
//! offset `δ = j − i`. The **value pathway (OV circuit) is position-blind** —
//! `W_V` is applied identically to every token regardless of its offset from
//! the query.
//!
//! RoVE additionally rotates each value `W_V·x_j` by `R_j` before aggregation,
//! then inverse-rotates the output by `R_{−i}` to put it in the query's frame:
//!
//! ```text
//! ỹ_i = R_{−i} · Σ_j A_ij · R_j · W_V · x_j
//!     = Σ_j A_ij · (R_{j−i} · W_V) · x_j
//!     = Σ_j A_ij · ψ_{j−i} · x_j    ← attentive convolution
//! ```
//!
//! The effective kernel `ψ_δ = R_δ·W_V` is an offset-indexed block-Toeplitz
//! family — converting the attention mixer from Kronecker structure
//! (`A ⊗ W_V`) to gated block-Toeplitz structure.
//!
//! # Three equivalent views
//!
//! 1. **Attentive convolution** — the value kernel becomes a function of the
//!    offset `δ = j − i`, not just the attention weight. This is the paper's
//!    headline framing (§3, Eq. 3).
//! 2. **Matrix mixer** — the attention output `Σ_j A_ij · V_j` is a
//!    matrix-vector product `A · V`. Standard attention factors `A` (RoPE-only
//!    on Q/K) from `V` (position-blind). RoVE couples them: `V` becomes a
//!    function of the row index via `R_j`, and the output is re-indexed by
//!    `R_{−i}`. The mixer structure is no longer `A ⊗ I_d` — it is a gated
//!    block-Toeplitz operator.
//! 3. **Local frame** — each value lives in its token's local frame; the
//!    inverse rotation `R_{−i}` maps the aggregated output back into the
//!    query's frame. This is the geometric view and is the most useful for
//!    FlashAttention compatibility (see below).
//!
//! # FlashAttention compatibility
//!
//! The rotations act on V (before the kernel call) and on the aggregated
//! output (after the kernel call). They **never** touch the `n×n` attention
//! score matrix. This means RoVE can be fused into any FlashAttention-style
//! kernel that already supports per-head V rotation — no `n×n` materialization
//! is needed. Plan 557 Phase 2 G5 verifies the output-equivalence algebraically.
//!
//! # Sibling relationship with [`crate::position_group_action`]
//!
//! RoVE is the **first hot-path consumer** of GRAPE's [`RopeAction`] (Issue 160,
//! Research 446). GRAPE shipped the `PositionGroupAction` trait as a
//! "vocabulary bridge for cold-path code"; RoVE turns it into a real attention
//! variant by calling [`RopeAction::apply_at`] on the V projection and
//! [`RopeAction::apply_inverse_at`] on the post-softmax output.
//!
//! # Inference-only caveat (the open question)
//!
//! The paper validates RoVE as a **training-time architectural choice** — the
//! model is trained from scratch with V rotation. The inference-time retrofit
//! onto RoPE-trained checkpoints is **unvalidated**. The structural argument
//! cuts both ways: RoVE makes the OV circuit offset-aware, but `W_V` was
//! trained under the offset-blind assumption. Plan 557 Phase 5 is a mandatory
//! PoC before any default-on promotion.
//!
//! # Parameter-free
//!
//! Zero new weights. The only config is `theta` (the RoPE base frequency,
//! default 10000.0), and that exists only to thread YaRN-style rescaling in
//! future work (paper Appendix C). The primitive itself is pure float
//! arithmetic over caller-owned buffers.

use crate::position_group_action::{PositionGroupAction, RopeAction};

// ── Config ──────────────────────────────────────────────────────

/// Configuration for RoVE.
///
/// RoVE is parameter-free — the only field is `theta` (the RoPE base
/// frequency), which exists for forward-compat with YaRN-style rescaling
/// (paper Appendix C). The default `10000.0` matches the paper's RoPE base
/// and our existing [`crate::position_group_action::RopeAction::new`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoVeConfig {
    /// RoPE base frequency. Default `10000.0`.
    pub theta: f32,
}

impl Default for RoVeConfig {
    #[inline]
    fn default() -> Self {
        Self { theta: 10000.0 }
    }
}

impl RoVeConfig {
    /// Construct with a custom base frequency.
    #[inline]
    pub const fn new(theta: f32) -> Self {
        Self { theta }
    }

    /// Build a [`RopeAction`] for the given head dimension.
    ///
    /// `dim` must be even and `>= 2` (RoPE operates on pairs). Odd `dim`
    /// panics with a clear message — see G7.
    #[inline]
    pub fn build_rope_action(&self, dim: usize) -> RopeAction {
        RopeAction::with_theta(dim, self.theta)
    }
}

// ── Per-token primitives (zero-allocation) ──────────────────────

/// Rotate the value projection at position `pos` into the global frame.
///
/// Computes `V_rot[pos] = R_pos · V[pos]` in place into `out`.
///
/// **Zero allocation.** Caller-owned `out` buffer (length `dim`).
#[inline]
pub fn rotate_values_into(
    action: &RopeAction,
    pos: usize,
    values: &[f32],
    out: &mut [f32],
) {
    action.apply_at(pos as f32, values, out);
}

/// Inverse-rotate the softmax-aggregated output at query position `pos` from
/// the global frame back into the query's local frame.
///
/// Computes `ỹ[pos] = R_{−pos} · aggregated[pos]` in place into `out`.
///
/// **Zero allocation.** Caller-owned `out` buffer (length `dim`).
#[inline]
pub fn inverse_rotate_output_into(
    action: &RopeAction,
    pos: usize,
    aggregated: &[f32],
    out: &mut [f32],
) {
    action.apply_inverse_at(pos as f32, aggregated, out);
}

// ── Batch primitives (zero-allocation in the loop) ──────────────

/// Batch value rotation for all tokens in a sequence.
///
/// `values` and `out` are flat `[n * dim]` row-major buffers. For each token
/// `t` at position `positions[t]`, computes `out[t] = R_{positions[t]} · values[t]`.
///
/// - **Zero allocation** in the loop (per-token slice borrows only).
/// - This is the API the attention forward path calls once per layer (Phase 3).
pub fn batch_rotate_values_into(
    action: &RopeAction,
    positions: &[usize],
    values: &[f32],
    out: &mut [f32],
    dim: usize,
) {
    debug_assert_eq!(values.len(), positions.len() * dim, "values buffer length mismatch");
    debug_assert_eq!(out.len(), positions.len() * dim, "out buffer length mismatch");
    for (t, &pos) in positions.iter().enumerate() {
        let start = t * dim;
        rotate_values_into(action, pos, &values[start..start + dim], &mut out[start..start + dim]);
    }
}

/// Batch inverse output rotation for all query positions.
///
/// `aggregated` and `out` are flat `[n * dim]` row-major buffers. For each
/// query `i` at position `positions[i]`, computes
/// `out[i] = R_{−positions[i]} · aggregated[i]`.
///
/// - **Zero allocation** in the loop.
pub fn batch_inverse_rotate_output_into(
    action: &RopeAction,
    positions: &[usize],
    aggregated: &[f32],
    out: &mut [f32],
    dim: usize,
) {
    debug_assert_eq!(
        aggregated.len(),
        positions.len() * dim,
        "aggregated buffer length mismatch"
    );
    debug_assert_eq!(out.len(), positions.len() * dim, "out buffer length mismatch");
    for (t, &pos) in positions.iter().enumerate() {
        let start = t * dim;
        inverse_rotate_output_into(
            action,
            pos,
            &aggregated[start..start + dim],
            &mut out[start..start + dim],
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_DIM: usize = 8;
    const TEST_THETA: f32 = 10000.0;

    fn make_action() -> RopeAction {
        RoVeConfig::new(TEST_THETA).build_rope_action(TEST_DIM)
    }

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() <= tol,
                "mismatch at index {i}: {x} vs {y} (tol {tol})"
            );
        }
    }

    // ── G1: identity at pos 0 ────────────────────────────────────

    /// Rotation by angle 0 is identity: `rotate_values_into(action, 0, v, out)`
    /// writes `v` to `out`.
    #[test]
    fn g1_identity_at_pos_zero() {
        let action = make_action();
        let v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut out = [0.0f32; TEST_DIM];
        rotate_values_into(&action, 0, &v, &mut out);
        approx_eq(&v, &out, 1e-6);
    }

    /// Same for the inverse direction.
    #[test]
    fn g1_inverse_identity_at_pos_zero() {
        let action = make_action();
        let v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut out = [0.0f32; TEST_DIM];
        inverse_rotate_output_into(&action, 0, &v, &mut out);
        approx_eq(&v, &out, 1e-6);
    }

    // ── G2: round-trip ───────────────────────────────────────────

    /// `rotate_values_into` then `inverse_rotate_output_into` at the same
    /// position recovers the original (to f32 precision).
    #[test]
    fn g2_round_trip() {
        let action = make_action();
        let v = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
        let mut rotated = [0.0f32; TEST_DIM];
        let mut recovered = [0.0f32; TEST_DIM];

        for pos in [0usize, 1, 5, 17, 100, 1023] {
            rotate_values_into(&action, pos, &v, &mut rotated);
            inverse_rotate_output_into(&action, pos, &rotated, &mut recovered);
            approx_eq(&v, &recovered, 1e-5);
        }
    }

    // ── G3: relativity check (the paper Eq. 3 claim) ─────────────

    /// The two-step composition (rotate at `j`, inverse at `i`) produces
    /// `R_{j−i} · v` — equivalent to a single `action.apply_at((j − i), v, out)`.
    /// This verifies the offset-indexed kernel `ψ_{j−i} = R_{j−i} · W_V` claim.
    #[test]
    fn g3_relativity_check() {
        let action = make_action();
        let v = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
        let mut v_at_j = [0.0f32; TEST_DIM];
        let mut v_at_i = [0.0f32; TEST_DIM];
        let mut expected = [0.0f32; TEST_DIM];

        // Test several (i, j) pairs with distinct offsets.
        for (i, j) in [(0usize, 5), (3, 10), (10, 3), (7, 50), (50, 7)] {
            rotate_values_into(&action, j, &v, &mut v_at_j);
            inverse_rotate_output_into(&action, i, &v_at_j, &mut v_at_i);

            // Expected: R_{j-i} · v  (note: offset can be negative when i > j)
            action.apply_at(j as f32 - i as f32, &v, &mut expected);

            approx_eq(&v_at_i, &expected, 1e-5);
        }
    }

    // ── G4: zero-degradation when feature off ────────────────────

    // G4 is architecturally unverifiable from within the feature-gated module:
    // when `rotary_value_embedding` is off, this module does not compile, so
    // no test inside it can run. The guarantee is structural (the feature gate
    // on the module declaration in lib.rs) and is verified by the Phase 1 Exit
    // Criterion: `cargo build` (default features) is unchanged. No test here.

    // ── G5: zero-allocation ──────────────────────────────────────

    /// G5 zero-alloc assertion via code-inspection pattern (matches
    /// `phase_rotation.rs` — we can't use `#[global_allocator]` in lib unit
    /// tests due to parallel test harness collisions). The empirical
    /// `CountingAllocator` audit lives in the Phase 2 bench
    /// (`benches/rotary_value_embedding_goat.rs`).
    ///
    /// This test constructs the primitives, runs both hot paths + both batch
    /// paths, and verifies by code inspection that no `Vec::new`, `vec![]`,
    /// `Vec::clone`, or `.resize` appears on the hot path. The smoke assertion
    /// here is that the functions run without panic on caller-owned buffers.
    #[test]
    fn g5_zero_alloc_smoke() {
        let action = make_action();
        let n = 16;
        let positions: Vec<usize> = (0..n).collect();
        let values = vec![0.5f32; n * TEST_DIM];
        let mut out = vec![0.0f32; n * TEST_DIM];
        let mut recovered = vec![0.0f32; n * TEST_DIM];

        // Both batch paths run without panic on caller-owned buffers.
        batch_rotate_values_into(&action, &positions, &values, &mut out, TEST_DIM);
        batch_inverse_rotate_output_into(&action, &positions, &out, &mut recovered, TEST_DIM);

        // Round-trip recovers the original.
        approx_eq(&values, &recovered, 1e-5);

        // Single-token paths also run on caller-owned buffers.
        let v = [1.0f32; TEST_DIM];
        let mut single_out = [0.0f32; TEST_DIM];
        let mut single_recovered = [0.0f32; TEST_DIM];
        rotate_values_into(&action, 5, &v, &mut single_out);
        inverse_rotate_output_into(&action, 5, &single_out, &mut single_recovered);
        approx_eq(&v, &single_recovered, 1e-5);
    }

    // ── G6: batch correctness ────────────────────────────────────

    /// `batch_rotate_values_into` produces identical results to per-token
    /// `rotate_values_into` for every token. Same for the inverse batch.
    #[test]
    fn g6_batch_correctness() {
        let action = make_action();
        let n = 32;
        let positions: Vec<usize> = (0..n).map(|t| t * 3 + 1).collect();
        let values: Vec<f32> = (0..n * TEST_DIM).map(|i| (i as f32) * 0.01 - 1.0).collect();

        // Batch path.
        let mut batch_out = vec![0.0f32; n * TEST_DIM];
        batch_rotate_values_into(&action, &positions, &values, &mut batch_out, TEST_DIM);

        // Per-token reference.
        let mut ref_out = vec![0.0f32; n * TEST_DIM];
        for (t, &pos) in positions.iter().enumerate() {
            let start = t * TEST_DIM;
            rotate_values_into(
                &action,
                pos,
                &values[start..start + TEST_DIM],
                &mut ref_out[start..start + TEST_DIM],
            );
        }
        approx_eq(&batch_out, &ref_out, 0.0); // bit-identical expected

        // Same for inverse batch.
        let mut batch_inv = vec![0.0f32; n * TEST_DIM];
        batch_inverse_rotate_output_into(&action, &positions, &batch_out, &mut batch_inv, TEST_DIM);

        let mut ref_inv = vec![0.0f32; n * TEST_DIM];
        for (t, &pos) in positions.iter().enumerate() {
            let start = t * TEST_DIM;
            inverse_rotate_output_into(
                &action,
                pos,
                &batch_out[start..start + TEST_DIM],
                &mut ref_inv[start..start + TEST_DIM],
            );
        }
        approx_eq(&batch_inv, &ref_inv, 0.0);
    }

    // ── G7: odd-dim safety ───────────────────────────────────────

    /// RoPE requires even `dim`. `RoVeConfig::build_rope_action` delegates to
    /// `RopeAction::with_theta`, which panics on odd dim. This test verifies
    /// the panic propagates through the RoVE API.
    #[test]
    #[should_panic(expected = "RoPE requires even dim >= 2")]
    fn g7_odd_dim_panics() {
        let config = RoVeConfig::default();
        let _ = config.build_rope_action(7); // odd dim
    }

    /// Dim 0 also panics (must be >= 2).
    #[test]
    #[should_panic(expected = "RoPE requires even dim >= 2")]
    fn g7_zero_dim_panics() {
        let config = RoVeConfig::default();
        let _ = config.build_rope_action(0);
    }

    /// Dim 2 (minimum valid) works.
    #[test]
    fn g7_min_valid_dim() {
        let config = RoVeConfig::default();
        let action = config.build_rope_action(2);
        let v = [1.0, 0.0];
        let mut out = [0.0f32; 2];
        rotate_values_into(&action, 1, &v, &mut out);
        // Non-trivial rotation at pos 1.
        assert!((out[0] - v[0]).abs() > 0.01 || (out[1] - v[1]).abs() > 0.01);
    }
}
