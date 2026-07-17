//! Depthwise causal 1D convolution (paper §2.3 eq 5).
//!
//! Plan 299 Phase 3 T3.7. The Engram paper applies a small depthwise causal
//! conv over the retrieved memory patterns `Ṽ` before the residual fuse:
//!
//! ```text
//! Y = SiLU(Conv1D(RMSNorm(Ṽ))) + Ṽ
//! ```
//!
//! This module implements the `Conv1D` term only — the caller does RMSNorm +
//! SiLU + residual add. Keeping the conv as a standalone zero-alloc primitive
//! lets the host compose it with the sigmoid gate and projection weights in
//! any order (paper §2.4 puts conv *after* the mHC shared-`V` projection;
//! other orderings are valid for ablation).
//!
//! # CRITICAL — never softmax
//!
//! Per AGENTS.md this module contains **no `softmax` symbol**. The conv is a
//! purely linear operator (weighted sum of past taps); the only nonlinearity
//! in the paper's recipe is `SiLU`, which the applied by the caller. The
//! sigmoid gate lives in [`crate::engram::kernel`].
//!
//! # Conv Zero Init
//!
//! Per the paper's "Conv Zero Init" hyperparameter, the default kernel is
//! [`IDENTITY_KERNEL`] = `[0, 0, 1, 0]`. With this kernel the conv is the
//! identity: `out == v_tilde`. The output then reduces to pure residual
//! (`Y = SiLU(RMSNorm(Ṽ)) + Ṽ`), matching the "no conv at init" training
//! stability trick used in the paper's pretrained checkpoints.
//!
//! # Indexing convention
//!
//! The kernel is laid out left-to-right from **oldest** to **newest** tap:
//! - `kernel[0]` = tap at `t - 3 * dilation` (oldest)
//! - `kernel[1]` = tap at `t - 2 * dilation`
//! - `kernel[2]` = tap at `t - 1 * dilation`
//! - `kernel[3]` = tap at `t - 0 * dilation` (current)
//!
//! This matches the standard CNN causal-conv convention. The spec's literal
//! `[0, 0, 1, 0]` activates `kernel[2]` (the 1-step-back tap), which is **not**
//! strictly identity under this convention. To honor the spec's intent —
//! "zero conv → pure residual" — we interpret "identity" as "the conv output
//! equals the input bit-identically", which requires the current-tap weight
//! to be 1. So [`IDENTITY_KERNEL`] = `[0, 0, 0, 1]` for strict identity.
//!
//! But the spec's literal `[0, 0, 1, 0]` is also exposed as
//! [`SPEC_KERNEL`] for direct paper-text reproduction. The unit test for
//! "identity kernel → out == v_tilde" uses [`IDENTITY_KERNEL`] (strict).
//!
//! # Layout
//!
//! `v_tilde` and `out` are flat slices treated as a 1D signal. For a true
//! depthwise conv across `D` channels, the caller loops:
//!
//! ```text
//! for d in 0..D {
//!     conv_causal_into(&v_tilde[d..], &mut out[d..], kernel, dilation);
//!     // strided by D — caller slices `v_tilde[d..n*D].step_by(D)`.
//! }
//! ```
//!
//! The flat-slice signature keeps the API simple and matches the spec.
//!
//! # Hot-path contract
//!
//! [`conv_causal_into`] is **zero-allocation**: caller provides `out` of size
//! `v_tilde.len()`. The kernel is a fixed-size `[f32; 4]` stack value. Inner
//! loop is `O(4n)` multiply-adds with at most 3 boundary checks per output.

/// Identity kernel — strict passthrough (`out == v_tilde`).
///
/// `kernel[3]` (current-position tap) is 1; all others are 0. With this
/// kernel the conv contributes nothing and the residual
/// `Y = SiLU(RMSNorm(Ṽ)) + Ṽ` is recovered. This is the operational form
/// of the paper's "Conv Zero Init" hyperparameter — training (when done in
/// riir-train) starts from a known-good baseline.
pub const IDENTITY_KERNEL: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

/// Spec-literal kernel — the paper text's `[0, 0, 1, 0]`.
///
/// Under our left-to-right oldest→newest convention, `[0, 0, 1, 0]` activates
/// `kernel[2]` (the 1-step-back tap). With dilation=1 this shifts `v_tilde`
/// forward by 1 (with a leading zero) — NOT identity. Kept here for
/// paper-text reproduction; use [`IDENTITY_KERNEL`] for strict passthrough.
pub const SPEC_KERNEL: [f32; 4] = [0.0, 0.0, 1.0, 0.0];

/// Zero kernel — true zero conv.
///
/// All taps are 0, so `out = 0` and `Y = SiLU(0) + Ṽ = Ṽ`. This is the
/// strictest reading of "Conv Zero Init".
pub const ZERO_KERNEL: [f32; 4] = [0.0; 4];

/// Apply a depthwise causal 1D convolution to `v_tilde`, writing into `out`.
///
/// Plan 299 Phase 3 T3.7. See the module docs for the layout and zero-init
/// convention.
///
/// # Formula
///
/// For each position `t ∈ [0, n)`:
/// ```text
/// out[t] = Σ_{j=0..4} kernel[j] * v_tilde[t - (3 - j) * δ]
/// ```
/// where `δ = max(dilation, 1)` and out-of-range indices contribute 0.
/// `kernel[3]` is the current-position tap; `kernel[0]` is the oldest
/// (3 × dilation positions back).
///
/// # Arguments
///
/// - `v_tilde` — input slice. Treated as a 1D signal of length `n`.
/// - `out` — output slice. MUST equal `v_tilde.len()` (debug_asserted).
/// - `kernel` — 4 tap weights. See [`IDENTITY_KERNEL`] for the passthrough.
/// - `dilation` — stride between taps. `dilation = 1` is a standard causal
///   conv; the paper uses `dilation = max N-gram order` (= 3 for trigram).
///   `dilation = 0` is treated as 1 (degenerate).
///
/// # Panics (debug only)
///
/// `debug_assert!` checks `out.len() == v_tilde.len()`. Zero-length input is
/// a no-op.
#[inline]
pub fn conv_causal_into(v_tilde: &[f32], out: &mut [f32], kernel: [f32; 4], dilation: usize) {
    let n = v_tilde.len();
    if n == 0 {
        return;
    }
    debug_assert_eq!(
        out.len(),
        n,
        "conv_causal_into: out.len() must equal v_tilde.len()"
    );

    let dil = dilation.max(1) as isize;
    for (t, out_slot) in out.iter_mut().enumerate().take(n) {
        let mut acc = 0.0f32;
        // kernel[0] = oldest tap (t - 3δ); kernel[3] = current (t - 0δ).
        // Out-of-range taps contribute 0 (zero-padding at the left edge).
        for (j, &k) in kernel.iter().enumerate() {
            let offset = (3 - j) as isize * dil;
            let tap_t = t as isize - offset;
            if tap_t >= 0 {
                acc += k * v_tilde[tap_t as usize];
            }
        }
        *out_slot = acc;
    }
}

/// Apply a depthwise causal 1D convolution with a **runtime-length** kernel.
///
/// This is the generalization of [`conv_causal_into`] to kernel sizes other
/// than 4. It exists to unblock multi-scale conv write-backs (paper §A
/// Table 6 uses kernels `{4, 8, 12}`) without forcing every caller to pay for
/// a heap allocation when they only need the common `k=4` case.
///
/// # Formula
///
/// For a kernel of length `K`, for each position `t ∈ [0, n)`:
/// ```text
/// out[t] = Σ_{j=0..K} kernel[j] * v_tilde[t - (K-1 - j) * δ]
/// ```
/// where `δ = max(dilation, 1)` and out-of-range indices contribute 0.
/// `kernel[K-1]` is the current-position tap; `kernel[0]` is the oldest
/// (`(K-1) × dilation` positions back). This convention matches
/// [`conv_causal_into`] exactly when `K == 4`.
///
/// # Identity kernels for arbitrary `K`
///
/// There is no `IDENTITY_KERNEL_K8` / `IDENTITY_KERNEL_K12` constant — the
/// identity kernel of length `K` is `&[0.0; K-1] + &[1.0]` (all zeros with a
/// trailing 1 at position `K-1`). For `K=4` this reduces to
/// [`IDENTITY_KERNEL`] = `[0, 0, 0, 1]`. Callers building multi-scale convs
/// should construct each scale's identity kernel inline, e.g.:\
/// `let id_k8: Vec<f32> = (0..8).map(|i| if i == 7 { 1.0 } else { 0.0 }).collect();`
///
/// # Equivalence with `conv_causal_into` at `K=4`
///
/// For `kernel.len() == 4` this function is bit-identical to
/// [`conv_causal_into`] with the same kernel values and dilation. The test
/// `dyn_matches_static_at_k4` verifies this on random inputs.
///
/// # Arguments
///
/// - `v_tilde` — input slice. Treated as a 1D signal of length `n`.
/// - `out` — output slice. MUST equal `v_tilde.len()` (debug_asserted).
/// - `kernel` — `K` tap weights, runtime length. `K` must be `>= 1`.
/// - `dilation` — stride between taps. `dilation = 0` is treated as 1.
///
/// # Panics (debug only)
///
/// `debug_assert!` checks `out.len() == v_tilde.len()` and `kernel.len() >= 1`.
/// Zero-length input is a no-op. Zero-length kernel is a panic (debug only).
///
/// # Hot-path contract
///
/// Same as [`conv_causal_into`]: zero-allocation, `O(K·n)` multiply-adds.
#[inline]
pub fn conv_causal_dyn_into(v_tilde: &[f32], out: &mut [f32], kernel: &[f32], dilation: usize) {
    let n = v_tilde.len();
    if n == 0 {
        return;
    }
    let k = kernel.len();
    debug_assert!(k >= 1, "conv_causal_dyn_into: kernel must be non-empty");
    debug_assert_eq!(
        out.len(),
        n,
        "conv_causal_dyn_into: out.len() must equal v_tilde.len()"
    );

    let dil = dilation.max(1) as isize;
    let last = (k - 1) as isize; // index of the current-position tap
    for (t, out_slot) in out.iter_mut().enumerate().take(n) {
        let mut acc = 0.0f32;
        // kernel[0] = oldest tap (t - (K-1)·δ); kernel[K-1] = current (t - 0·δ).
        // Out-of-range taps contribute 0 (zero-padding at the left edge).
        for (j, &kw) in kernel.iter().enumerate() {
            let offset = (last - j as isize) * dil;
            let tap_t = t as isize - offset;
            if tap_t >= 0 {
                acc += kw * v_tilde[tap_t as usize];
            }
        }
        *out_slot = acc;
    }
}

/// Per-channel temporal causal conv — current-timestep output from a
/// time-series of channel vectors.
///
/// Unlike [`conv_causal_dyn_into`] (which treats a single flat slice as a 1D
/// signal), this primitive operates on a **temporal sequence** of D-dim
/// channel vectors and produces **only the current-timestep output**. This is
/// the paper-faithful depthwise temporal conv: each channel is convolved
/// independently across time, and we emit one D-dim vector (the convolved
/// value at the current timestep).
///
/// # Formula
///
/// For a kernel of length `K`, history of `H = v_history.len()` past vectors,
/// and current vector `v_current` (all length `D`):
/// ```text
/// out[j] = Σ_{k=0..K} kernel[k] * v_signal[k][j]
/// ```
/// where `v_signal` is the zero-padded temporal signal of length `K`:
/// - `v_signal[K-1]` = `v_current` (the current tap, weighted by `kernel[K-1]`)
/// - `v_signal[K-1-i]` = `v_history[H-i]` for `1 <= i <= H` (past taps)
/// - `v_signal[k]` = 0 for `k < K-1-H` (before sequence start — zero padding)
///
/// This matches the causal convention of [`conv_causal_dyn_into`]:
/// `kernel[0]` is the oldest tap, `kernel[K-1]` is the current tap.
///
/// # Arguments
///
/// - `v_history` — past `v` vectors, ordered **oldest first**. Length `H`.
///   At sequence start `H = 0`; the conv degenerates to `out = kernel[K-1] * v_current`
///   (zero-padded). `H` MUST be `<= K-1` (callers ring-buffer to enforce this);
///   if `H > K-1` only the most recent `K-1` entries are read.
/// - `v_current` — the current-timestep `v` vector. Length `D`.
/// - `out` — output slice. MUST equal `v_current.len()` (debug_asserted).
/// - `kernel` — `K` tap weights, runtime length. `K >= 1`.
/// - `dilation` — stride between taps in timesteps. `dilation = 0` is treated
///   as 1. The history is sampled at `dilation`-strided intervals so a
///   `dilation = 2` kernel of length 4 reaches back 6 timesteps.
///
/// # Panics (debug only)
///
/// `debug_assert!` checks `out.len() == v_current.len()` and `kernel.len() >= 1`.
///
/// # Hot-path contract
///
/// Zero-allocation, `O(K·D)` multiply-adds. The history is borrowed (not cloned).
/// The caller maintains the ring buffer.
///
/// # Why this exists
///
/// [`conv_causal_dyn_into`] treats the hidden dimension `D` as the 1D signal —
/// a Tier A simplification that makes the conv a *spatial* filter across
/// hidden units. The paper's multi-scale conv is *temporal* (across sequence
/// positions). This primitive implements the paper-faithful temporal variant:
/// pass a per-layer ring buffer of past `v` vectors plus the current `v`, and
/// get the current-timestep conv output. Multi-scale (`{4, 8}` kernels) then
/// captures different temporal receptive fields — the actual paper mechanism.
#[inline]
pub fn conv_temporal_step_into(
    v_history: &[Vec<f32>],
    v_current: &[f32],
    out: &mut [f32],
    kernel: &[f32],
    dilation: usize,
) {
    let d = v_current.len();
    if d == 0 {
        return;
    }
    let k = kernel.len();
    debug_assert!(k >= 1, "conv_temporal_step_into: kernel must be non-empty");
    debug_assert_eq!(
        out.len(),
        d,
        "conv_temporal_step_into: out.len() must equal v_current.len()"
    );
    let h = v_history.len();

    let dil = dilation.max(1);
    let last = k - 1; // index of the current-position tap in kernel[]

    // Zero output first (all channels), then accumulate. Cheaper than
    // per-channel fill because the inner kernel loop is short (K small).
    for slot in out.iter_mut() {
        *slot = 0.0;
    }

    for j in 0..d {
        let mut acc = 0.0f32;
        for (ki, &kw) in kernel.iter().enumerate() {
            if kw == 0.0 {
                continue;
            }
            // kernel[ki] weights the signal at offset (last - ki) * dil timesteps ago.
            let timesteps_ago = (last - ki) * dil;
            if timesteps_ago == 0 {
                acc += kw * v_current[j];
            } else if timesteps_ago <= h {
                // v_history is oldest-first; the entry timesteps_ago back is at
                // index `h - timesteps_ago`.
                let hist_idx = h - timesteps_ago;
                acc += kw * v_history[hist_idx][j];
            }
            // else: before sequence start — zero-padding (contributes 0).
        }
        out[j] = acc;
    }
}

/// Backward of [`conv_temporal_step_into`] — flows the output gradient back
/// to the current-timestep input only (truncated BPTT).
///
/// This is the **truncated** temporal conv backward: gradients flow to
/// `v_current` (which feeds `v_proj @ input` → `v_proj` grad), but NOT to
/// `v_history` entries. Past `v` vectors are treated as constants for
/// gradient purposes. This is the standard truncated-BPTT approximation for
/// recurrent temporal convs — full BPTT would require storing per-timestep
/// caches and flowing grads across the entire sequence.
///
/// # Formula
///
/// Forward: `out[j] = Σ_k kernel[k] * v_signal[k][j]` (see
/// [`conv_temporal_step_into`]). Only `v_signal[K-1] = v_current` receives
/// gradient (truncation):
/// ```text
/// grad_v_current[j] += kernel[K-1] * grad_out[j]
/// ```
///
/// The kernel is frozen (not learned), so no `grad_kernel` is computed.
///
/// # Why truncated is acceptable
///
/// For frozen kernels (the Tier A/Phase 6' design), the only learnable
/// parameter affected by the conv is `v_proj` (which produces `v_current`).
/// Past `v` vectors were produced by past `v_proj` applications — but `v_proj`
/// is the SAME weight matrix across timesteps. Full BPTT would accumulate the
/// gradient contribution from each past timestep into `grad_v_proj`. The
/// truncated variant captures only the current-timestep contribution, which
/// is the dominant term when the kernel decays (e.g. averaging kernel: each
/// past tap contributes `1/K`, the current tap contributes `1/K`, so
/// truncation loses at most `(K-1)/K` of the gradient magnitude — a known,
/// bounded approximation that the Muon optimizer's momentum partially
/// compensates for).
///
/// # Arguments
///
/// - `grad_out` — gradient w.r.t. the conv output (`∂L/∂out`). Length `D`.
/// - `grad_v_current` — accumulated gradient w.r.t. `v_current`. Length `D`.
///   **Accumulates** (+=) so the caller can chain with other gradient sources.
/// - `kernel` — the frozen conv kernel. Only `kernel[K-1]` is read.
///
/// # Hot-path contract
///
/// Zero-allocation, `O(D)` scalar multiply-adds. Cheaper than the forward.
#[inline]
pub fn conv_temporal_step_backward_truncated_into(
    grad_out: &[f32],
    grad_v_current: &mut [f32],
    kernel: &[f32],
) {
    let d = grad_out.len();
    if d == 0 {
        return;
    }
    debug_assert_eq!(
        grad_v_current.len(),
        d,
        "conv_temporal_step_backward_truncated_into: len mismatch"
    );
    let k = kernel.len();
    debug_assert!(k >= 1, "conv_temporal_step_backward_truncated_into: kernel empty");
    // Only the current tap (kernel[K-1]) contributes under truncation.
    let current_weight = kernel[k - 1];
    for j in 0..d {
        grad_v_current[j] += current_weight * grad_out[j];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_kernel_is_strict_passthrough() {
        // IDENTITY_KERNEL → out == v_tilde, bit-identically, for any input.
        let v_tilde = [1.0f32, 2.0, 3.0, 4.0, 5.0, 7.0, 11.0, 13.0];
        let mut out = [0.0f32; 8];
        conv_causal_into(&v_tilde, &mut out, IDENTITY_KERNEL, 1);
        assert_eq!(out, v_tilde, "IDENTITY_KERNEL → out == v_tilde");

        // Also true for dilation > 1 (the current tap is always in range).
        let mut out2 = [0.0f32; 8];
        conv_causal_into(&v_tilde, &mut out2, IDENTITY_KERNEL, 3);
        assert_eq!(out2, v_tilde, "identity holds for any dilation");

        // And dilation = 0 (treated as 1).
        let mut out3 = [0.0f32; 8];
        conv_causal_into(&v_tilde, &mut out3, IDENTITY_KERNEL, 0);
        assert_eq!(out3, v_tilde, "identity holds for dilation=0 (→ 1)");
    }

    #[test]
    fn zero_kernel_produces_zero_output() {
        let v_tilde = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut out = [99.0f32; 5];
        conv_causal_into(&v_tilde, &mut out, ZERO_KERNEL, 1);
        assert!(out.iter().all(|&v| v == 0.0), "ZERO_KERNEL → all zeros");
    }

    #[test]
    fn zero_input_produces_zero_output() {
        // Any kernel × zero input = zero output.
        let v_tilde = [0.0f32; 8];
        let mut out = [99.0f32; 8];
        let kernel = [0.25, 0.25, 0.25, 0.25]; // averaging kernel
        conv_causal_into(&v_tilde, &mut out, kernel, 1);
        assert!(out.iter().all(|&v| v == 0.0), "zero input → zero output");
    }

    #[test]
    fn non_trivial_kernel_convolves() {
        // kernel = [0.0, 0.0, 0.5, 0.5] → out[t] = 0.5*v[t] + 0.5*v[t-1]
        // (current + previous, both with weight 0.5).
        // At t=0: only current tap in range → out[0] = 0.5 * v[0].
        // At t=1+: out[t] = 0.5 * (v[t] + v[t-1]).
        let v_tilde = [2.0f32, 4.0, 6.0, 8.0];
        let mut out = [0.0f32; 4];
        let kernel = [0.0, 0.0, 0.5, 0.5];
        conv_causal_into(&v_tilde, &mut out, kernel, 1);

        // Expected: out[0] = 0.5*2 = 1; out[1] = 0.5*4 + 0.5*2 = 3; etc.
        let expected = [1.0f32, 3.0, 5.0, 7.0];
        for i in 0..4 {
            assert!(
                (out[i] - expected[i]).abs() < 1e-6,
                "out[{i}] = {}, expected {}",
                out[i],
                expected[i]
            );
        }
    }

    #[test]
    fn averaging_kernel_smooths() {
        // kernel = [0.25; 4] → moving average over 4 taps.
        // At t=0: only current tap in range → 0.25 * v[0].
        // At t=1: current + 1-back → 0.25 * (v[1] + v[0]).
        // At t=3+: all 4 taps in range → 0.25 * sum.
        let v_tilde = [1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0];
        let mut out = [0.0f32; 6];
        let kernel = [0.25; 4];
        conv_causal_into(&v_tilde, &mut out, kernel, 1);

        // out[0..3] = 0.25, 0.5, 0.75 (partial window); out[3..] = 1.0 (full).
        assert!((out[0] - 0.25).abs() < 1e-6, "out[0] = {}", out[0]);
        assert!((out[1] - 0.5).abs() < 1e-6, "out[1] = {}", out[1]);
        assert!((out[2] - 0.75).abs() < 1e-6, "out[2] = {}", out[2]);
        for (i, oi) in out[3..6].iter().enumerate() {
            assert!((*oi - 1.0).abs() < 1e-6, "out[{}] = {}", i + 3, oi);
        }
    }

    #[test]
    fn dilation_stretches_tap_stride() {
        // kernel = [0.0, 0.0, 1.0, 0.0], dilation = 2 → out[t] = v[t-2].
        // At t < 2, the tap is out of range → out[t] = 0.
        let v_tilde = [10.0f32, 20.0, 30.0, 40.0, 50.0];
        let mut out = [0.0f32; 5];
        let kernel = [0.0, 0.0, 1.0, 0.0]; // tap at offset 1*δ = 2
        conv_causal_into(&v_tilde, &mut out, kernel, 2);

        // Expected: out = [0, 0, 10, 20, 30] (shift by 2).
        let expected = [0.0f32, 0.0, 10.0, 20.0, 30.0];
        for i in 0..5 {
            assert!(
                (out[i] - expected[i]).abs() < 1e-6,
                "out[{i}] = {}, expected {}",
                out[i],
                expected[i]
            );
        }
    }

    #[test]
    fn empty_input_is_noop() {
        let v_tilde: [f32; 0] = [];
        let mut out: [f32; 0] = [];
        conv_causal_into(&v_tilde, &mut out, IDENTITY_KERNEL, 1); // must not panic
    }

    #[test]
    fn spec_kernel_is_one_step_shift() {
        // SPEC_KERNEL = [0, 0, 1, 0] — paper text's literal value. Under our
        // convention this activates kernel[2] = tap at offset δ. The output
        // is v_tilde shifted forward by δ (with leading zeros). Document this
        // so the discrepancy between spec text and behavior is explicit.
        let v_tilde = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut out = [0.0f32; 5];
        conv_causal_into(&v_tilde, &mut out, SPEC_KERNEL, 1);
        // out[t] = v[t-1] (or 0 if t < 1)
        let expected = [0.0f32, 1.0, 2.0, 3.0, 4.0];
        for i in 0..5 {
            assert!(
                (out[i] - expected[i]).abs() < 1e-6,
                "SPEC_KERNEL out[{i}] = {}, expected {}",
                out[i],
                expected[i]
            );
        }
    }

    // ── Tests for conv_causal_dyn_into (runtime-length kernel) ───────────────

    /// Build the identity kernel of length K: all zeros with a trailing 1 at
    /// position K-1. Matches [`IDENTITY_KERNEL`] when K=4.
    fn identity_kernel_dyn(k: usize) -> Vec<f32> {
        let mut v = vec![0.0; k];
        if k > 0 {
            v[k - 1] = 1.0;
        }
        v
    }

    #[test]
    fn dyn_matches_static_at_k4() {
        // For kernel.len() == 4, conv_causal_dyn_into MUST be bit-identical to
        // conv_causal_into. This is the core equivalence guarantee.
        let v_tilde: Vec<f32> = (0..32).map(|i| (i as f32).sin() * 3.0).collect();
        let kernel_static = [0.1, -0.2, 0.3, 0.9];
        let kernel_dyn: Vec<f32> = kernel_static.to_vec();

        for &dil in &[0usize, 1, 2, 3] {
            let mut out_static = vec![0.0f32; v_tilde.len()];
            let mut out_dyn = vec![0.0f32; v_tilde.len()];
            conv_causal_into(&v_tilde, &mut out_static, kernel_static, dil);
            conv_causal_dyn_into(&v_tilde, &mut out_dyn, &kernel_dyn, dil);
            assert_eq!(
                out_static, out_dyn,
                "dyn must match static at K=4 (dilation={})",
                dil
            );
        }
    }

    #[test]
    fn dyn_identity_k8_is_passthrough() {
        // Identity kernel of length 8: out == v_tilde for any input.
        let v_tilde = [1.0f32, 2.0, 3.0, 4.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0];
        let mut out = [0.0f32; 10];
        let kernel = identity_kernel_dyn(8);
        conv_causal_dyn_into(&v_tilde, &mut out, &kernel, 1);
        assert_eq!(out, v_tilde, "K=8 identity → out == v_tilde");

        // Identity holds for any dilation (current tap always in range).
        let mut out2 = [0.0f32; 10];
        conv_causal_dyn_into(&v_tilde, &mut out2, &kernel, 3);
        assert_eq!(out2, v_tilde, "K=8 identity holds for any dilation");
    }

    #[test]
    fn dyn_k1_is_scalar_multiply() {
        // Kernel of length 1: out[t] = kernel[0] * v[t]. Pure scalar multiply.
        let v_tilde = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut out = [0.0f32; 5];
        let kernel = [0.5f32];
        conv_causal_dyn_into(&v_tilde, &mut out, &kernel, 1);
        let expected = [0.5f32, 1.0, 1.5, 2.0, 2.5];
        for i in 0..5 {
            assert!((out[i] - expected[i]).abs() < 1e-6, "out[{i}] = {}", out[i]);
        }
    }

    #[test]
    fn dyn_k8_known_convolution() {
        // 8-tap averaging kernel over a ramp input.
        // kernel = [0.125; 8], v = [1, 2, 3, ...], dilation = 1.
        // At t < 7: partial window, out[t] = 0.125 * sum(v[0..=t]).
        // At t >= 7: full window, out[t] = 0.125 * sum(v[t-7..=t]).
        let v_tilde: Vec<f32> = (1..=12).map(|i| i as f32).collect();
        let mut out = vec![0.0f32; v_tilde.len()];
        let kernel = vec![0.125f32; 8];
        conv_causal_dyn_into(&v_tilde, &mut out, &kernel, 1);

        // t=0: window [1] → 0.125 * 1 = 0.125
        assert!((out[0] - 0.125).abs() < 1e-6, "out[0] = {}", out[0]);
        // t=6: window [1..=7] → 0.125 * 28 = 3.5
        assert!((out[6] - 3.5).abs() < 1e-6, "out[6] = {}", out[6]);
        // t=7: window [1..=8] → 0.125 * 36 = 4.5
        assert!((out[7] - 4.5).abs() < 1e-6, "out[7] = {}", out[7]);
        // t=11: window [5..=12] → 0.125 * (5+6+7+8+9+10+11+12) = 0.125 * 68 = 8.5
        assert!((out[11] - 8.5).abs() < 1e-6, "out[11] = {}", out[11]);
    }

    #[test]
    fn dyn_k8_dilation_stretches_receptive_field() {
        // kernel = [1.0, 0, 0, 0, 0, 0, 0, 0] (OLDEST tap activated), dilation = 2
        //   → out[t] = v[t - (K-1)*δ] = v[t - 7*2] = v[t-14].
        // At t < 14: tap out of range → out[t] = 0.
        //
        // Note: an identity kernel [0,...,0,1.0] (CURRENT tap) would give
        // out[t] = v[t] regardless of dilation — the "receptive field stretching"
        // only shows up when a non-current tap is activated. This test activates
        // the oldest tap to make the dilation effect visible.
        let v_tilde: Vec<f32> = (0..32).map(|i| (i + 1) as f32).collect();
        let mut out = vec![0.0f32; v_tilde.len()];
        let mut kernel = vec![0.0f32; 8];
        kernel[0] = 1.0; // oldest tap
        conv_causal_dyn_into(&v_tilde, &mut out, &kernel, 2);
        // out[14] should equal v[0] = 1; out[15] = v[1] = 2; etc.
        assert!(out[0] == 0.0 && out[13] == 0.0, "t < 14 → zero");
        assert!((out[14] - 1.0).abs() < 1e-6, "out[14] = {}", out[14]);
        assert!((out[31] - 18.0).abs() < 1e-6, "out[31] = {}", out[31]);
    }

    #[test]
    fn dyn_empty_input_is_noop() {
        let v_tilde: [f32; 0] = [];
        let mut out: [f32; 0] = [];
        let kernel = [1.0f32; 8];
        conv_causal_dyn_into(&v_tilde, &mut out, &kernel, 1); // must not panic
    }

    #[test]
    fn dyn_zero_kernel_is_zero_output() {
        let v_tilde = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut out = [99.0f32; 5];
        let kernel = vec![0.0f32; 12];
        conv_causal_dyn_into(&v_tilde, &mut out, &kernel, 1);
        assert!(out.iter().all(|&v| v == 0.0), "K=12 zero kernel → all zeros");
    }

    // ── Temporal conv (conv_temporal_step_into) ─────────────────────────────

    #[test]
    fn temporal_empty_history_degenerates_to_current_tap() {
        // H=0 (no history) → out = kernel[K-1] * v_current.
        let v_current = [1.0f32, 2.0, 3.0, 4.0];
        let history: Vec<Vec<f32>> = vec![];
        let mut out = [0.0f32; 4];
        // identity kernel [0,0,0,1] → out = 1.0 * v_current
        let kernel = [0.0f32, 0.0, 0.0, 1.0];
        conv_temporal_step_into(&history, &v_current, &mut out, &kernel, 1);
        assert_eq!(out, v_current, "H=0 identity → passthrough");

        // averaging kernel [0.25; 4] → out = 0.25 * v_current
        let kernel_avg = [0.25f32; 4];
        conv_temporal_step_into(&history, &v_current, &mut out, &kernel_avg, 1);
        for i in 0..4 {
            assert!((out[i] - 0.25 * v_current[i]).abs() < 1e-6, "out[{i}] = {}", out[i]);
        }
    }

    #[test]
    fn temporal_identity_kernel_is_passthrough() {
        // Identity kernel → out == v_current regardless of history.
        let v_current = [1.0f32, 2.0, 3.0];
        let history = vec![
            vec![10.0, 20.0, 30.0],
            vec![100.0, 200.0, 300.0],
            vec![1000.0, 2000.0, 3000.0],
        ];
        let mut out = [0.0f32; 3];
        let kernel = [0.0f32, 0.0, 0.0, 1.0];
        conv_temporal_step_into(&history, &v_current, &mut out, &kernel, 1);
        assert_eq!(out, v_current, "identity kernel → out == v_current");
    }

    #[test]
    fn temporal_averaging_kernel_with_history() {
        // kernel = [0.25; 4], 3 past v's + current = 4 taps, all weighted 1/4.
        // out[j] = 0.25 * (v_hist[0][j] + v_hist[1][j] + v_hist[2][j] + v_current[j])
        let v_current = [4.0f32, 8.0];
        let history = vec![
            vec![1.0, 2.0],
            vec![2.0, 4.0],
            vec![3.0, 6.0],
        ];
        let mut out = [0.0f32; 2];
        let kernel = [0.25f32; 4];
        conv_temporal_step_into(&history, &v_current, &mut out, &kernel, 1);
        // out[0] = 0.25 * (1 + 2 + 3 + 4) = 2.5
        // out[1] = 0.25 * (2 + 4 + 6 + 8) = 5.0
        assert!((out[0] - 2.5).abs() < 1e-6, "out[0] = {}", out[0]);
        assert!((out[1] - 5.0).abs() < 1e-6, "out[1] = {}", out[1]);
    }

    #[test]
    fn temporal_partial_history_zero_pads() {
        // H=1, K=4 → only 2 taps in range (1 past + current), other 2 zero-padded.
        // kernel = [0.25; 4] → out = 0.25 * (0 + 0 + v_hist[0] + v_current)
        let v_current = [4.0f32];
        let history = vec![vec![2.0f32]];
        let mut out = [0.0f32; 1];
        let kernel = [0.25f32; 4];
        conv_temporal_step_into(&history, &v_current, &mut out, &kernel, 1);
        // out[0] = 0.25 * (2 + 4) = 1.5
        assert!((out[0] - 1.5).abs() < 1e-6, "out[0] = {}", out[0]);
    }

    #[test]
    fn temporal_dilation_stretches_receptive_field() {
        // kernel = [0,0,0,1.0] (current tap), dilation = 2 → still passthrough
        // (current tap is always timesteps_ago=0 regardless of dilation).
        let v_current = [1.0f32, 2.0];
        let history = vec![vec![10.0, 20.0], vec![100.0, 200.0]];
        let mut out = [0.0f32; 2];
        let kernel = [0.0f32, 0.0, 0.0, 1.0];
        conv_temporal_step_into(&history, &v_current, &mut out, &kernel, 2);
        assert_eq!(out, v_current, "dilation doesn't affect current tap");

        // Now activate OLDEST tap (kernel[0] = 1.0), dilation = 2, K = 4.
        // timesteps_ago = (K-1-0) * 2 = 6. Need H >= 6. We have H = 2. → zero.
        let kernel_oldest = [1.0f32, 0.0, 0.0, 0.0];
        conv_temporal_step_into(&history, &v_current, &mut out, &kernel_oldest, 2);
        assert!(out.iter().all(|&v| v == 0.0), "oldest tap OOR with dilation 2");
    }

    #[test]
    fn temporal_matches_flat_conv_at_full_history() {
        // Sanity: temporal conv with full history should match flat conv_causal_dyn_into
        // at the LAST position. Build a flat signal [v0, v1, v2, v3] of length 4
        // per channel, convolve flat, then compare temporal(last) = flat[3].
        let v0 = [1.0f32, 2.0];
        let v1 = [3.0f32, 4.0];
        let v2 = [5.0f32, 6.0];
        let v3 = [7.0f32, 8.0]; // current

        // Flat: interleave channels. conv_causal_dyn_into treats the slice as 1D,
        // so we run it per-channel by slicing strided.
        let flat_signal_ch0 = vec![v0[0], v1[0], v2[0], v3[0]];
        let mut flat_out_ch0 = vec![0.0f32; 4];
        let kernel = [0.25f32; 4];
        conv_causal_dyn_into(&flat_signal_ch0, &mut flat_out_ch0, &kernel, 1);
        // flat_out_ch0[3] = 0.25 * (1 + 3 + 5 + 7) = 4.0

        let flat_signal_ch1 = vec![v0[1], v1[1], v2[1], v3[1]];
        let mut flat_out_ch1 = vec![0.0f32; 4];
        conv_causal_dyn_into(&flat_signal_ch1, &mut flat_out_ch1, &kernel, 1);

        // Temporal: history = [v0, v1, v2], current = v3.
        let history = vec![v0.to_vec(), v1.to_vec(), v2.to_vec()];
        let mut temporal_out = [0.0f32; 2];
        conv_temporal_step_into(&history, &v3, &mut temporal_out, &kernel, 1);

        // Compare: temporal_out should equal flat_out at position 3 (last).
        assert!((temporal_out[0] - flat_out_ch0[3]).abs() < 1e-6);
        assert!((temporal_out[1] - flat_out_ch1[3]).abs() < 1e-6);
    }

    #[test]
    fn temporal_backward_truncated_identity_kernel() {
        // Identity kernel: backward should flow full grad to v_current.
        let grad_out = [1.0f32, 2.0, 3.0];
        let mut grad_v = [0.0f32; 3];
        let kernel = [0.0f32, 0.0, 0.0, 1.0];
        conv_temporal_step_backward_truncated_into(&grad_out, &mut grad_v, &kernel);
        assert_eq!(grad_v, grad_out, "identity backward → full grad flow");
    }

    #[test]
    fn temporal_backward_truncated_averaging_kernel() {
        // Averaging kernel [0.25; 4]: backward flows kernel[K-1] = 0.25.
        let grad_out = [4.0f32, 8.0];
        let mut grad_v = [0.0f32; 2];
        let kernel = [0.25f32; 4];
        conv_temporal_step_backward_truncated_into(&grad_out, &mut grad_v, &kernel);
        assert!((grad_v[0] - 1.0).abs() < 1e-6, "grad_v[0] = {}", grad_v[0]);
        assert!((grad_v[1] - 2.0).abs() < 1e-6, "grad_v[1] = {}", grad_v[1]);
    }

    #[test]
    fn temporal_backward_truncated_accumulates() {
        // grad_v accumulates (+=), so multiple calls sum.
        let kernel = [0.5f32]; // K=1, current_weight = 0.5
        let mut grad_v = [1.0f32, 1.0];
        conv_temporal_step_backward_truncated_into(&[2.0, 4.0], &mut grad_v, &kernel);
        // grad_v[0] = 1.0 + 0.5*2.0 = 2.0
        // grad_v[1] = 1.0 + 0.5*4.0 = 3.0
        assert!((grad_v[0] - 2.0).abs() < 1e-6);
        assert!((grad_v[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn temporal_backward_matches_numeric_gradient() {
        // Finite-difference check: truncated backward should match numeric grad
        // w.r.t. v_current ONLY (history treated as constant).
        //
        // Note on tolerance: f32 finite-difference suffers catastrophic
        // cancellation when subtracting close squared values (L(+) - L(-) is
        // a small difference of ~0.29-sized numbers). Using eps = 1e-2 keeps
        // truncation error O(eps^2) = 1e-4 while making the difference large
        // enough to preserve f32 precision. Relative tolerance 1% is the
        // standard bar for f32 central differences on small magnitudes.
        let v_current = [1.0f32, 2.0, 3.0];
        let history = vec![
            vec![0.5, 1.0, 1.5],
            vec![0.25, 0.5, 0.75],
            vec![0.125, 0.25, 0.375],
        ];
        let kernel = [0.1f32, 0.2, 0.3, 0.4]; // arbitrary
        let mut out = [0.0f32; 3];
        conv_temporal_step_into(&history, &v_current, &mut out, &kernel, 1);

        // Arbitrary downstream loss: L = sum(out^2).
        // dL/dout[j] = 2 * out[j].
        let grad_out: Vec<f32> = out.iter().map(|&o| 2.0 * o).collect();

        // Analytic grad (truncated).
        let mut grad_v_analytic = [0.0f32; 3];
        conv_temporal_step_backward_truncated_into(&grad_out, &mut grad_v_analytic, &kernel);

        // Numeric grad: perturb each v_current[j] by eps, recompute loss, diff.
        let eps = 1e-2f32;
        for j in 0..3 {
            let mut v_plus = v_current;
            v_plus[j] += eps;
            let mut out_plus = [0.0f32; 3];
            conv_temporal_step_into(&history, &v_plus, &mut out_plus, &kernel, 1);
            let loss_plus: f32 = out_plus.iter().map(|&o| o * o).sum();

            let mut v_minus = v_current;
            v_minus[j] -= eps;
            let mut out_minus = [0.0f32; 3];
            conv_temporal_step_into(&history, &v_minus, &mut out_minus, &kernel, 1);
            let loss_minus: f32 = out_minus.iter().map(|&o| o * o).sum();

            let numeric = (loss_plus - loss_minus) / (2.0 * eps);
            let rel_err = (numeric - grad_v_analytic[j]).abs() / grad_v_analytic[j].abs().max(1e-6);
            assert!(
                rel_err < 0.02,
                "j={}: numeric={numeric}, analytic={}, rel_err={rel_err}",
                j,
                grad_v_analytic[j],
            );
        }
    }
}
