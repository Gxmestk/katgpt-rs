//! KDA (Kimi Delta Attention) forward layer.
//!
//! Implements the per-token decode forward for Kimi Linear's KDA linear attention
//! variant (arxiv 2510.26692). Used in 6 of 8 layers of Kimi-K3-0.40B (layers
//! 1, 2, 3, 5, 6, 7); layers 4 + 8 use MLA (Phase 2).
//!
//! # The mechanism (single-token decode)
//!
//! ```text
//! 1. Projections:
//!    z_qk = W^{q/k} · h       (SHARED q/k projection, pre-conv)
//!    z_v  = W^v     · h       (separate v projection, pre-conv)
//!    log_α_flat = W^↑_α · (W^↓_α · h)   (low-rank α projection, log-space)
//!    β = sigmoid(W^β · h)     (scalar per head)
//!
//! 2. ShortConv + activation:
//!    z_qk_conv = ShortConv1D(z_qk)    (depthwise causal conv, k=4)
//!    z_v_conv  = ShortConv1D(z_v)
//!    qk_swish = Swish(z_qk_conv)      (= z * sigmoid(z))
//!    v_swish  = Swish(z_v_conv)
//!
//! 3. Per-head L2Norm (q and k, NOT v):
//!    q_h = L2Norm(qk_swish[h*dk..(h+1)*dk])
//!    k_h = L2Norm(qk_swish[h*dk..(h+1)*dk])   (same slice — shared projection)
//!    v_h = v_swish[h*dk..(h+1)*dk]             (no norm)
//!
//! 4. Per-head α + β → recurrent step (KDA eq 1):
//!    α_h = exp(log_α_flat[h*dk..(h+1)*dk]).max(eps)    (per-channel decay)
//!    β_h = β[h]                                          (already sigmoid'd)
//!    erase_b_h = β_h · ones[dk]                          (β broadcast)
//!
//!    Call gdn2_recurrent_step(k_h, v_h, q_h, S_h, α_h, erase_b_h, β_h,
//!          ..., gate_config = Gdn2GateConfig::Kda)
//!
//! 5. Head-wise RMSNorm on the per-head output (o_h ∈ R^{dk}).
//!
//! 6. Output gate + final projection:
//!    gate = sigmoid(W^↑_g · (W^↓_g · h))    (low-rank output gate)
//!    o_concat = concat(o_0_norm, ..., o_{H-1}_norm)
//!    output = W_o · (gate ⊙ o_concat)
//! ```
//!
//! # Substrate reuse (Research 329 §4)
//!
//! The recurrent kernel itself is **already shipped** as `gdn2_recurrent_step`
//! with `Gdn2GateConfig::Kda` + `b = β·ones` + `alpha = per-channel α` +
//! `w_val = β`. Research 329 §4.2 proves this computes KDA eq 1 bit-identically.
//! This module wires the KDA-specific projections + activations around that kernel.
//!
//! # Open uncertainties (Research 329 §10)
//!
//! - **The exact α activation f(·).** Paper says "decay function similar to GDN/Mamba"
//!   but does not state analytically. Pseudocode suggests log-space parameterization
//!   (`α = exp(log_α)`). This module accepts raw `log_α` from the projection and
//!   applies `exp()`, clamped at `alpha_eps` to avoid denormals. The G1 test
//!   uses the same convention; Phase 6 real-model GOAT will catch any mismatch.
//! - **q/k shared projection.** The paper parameterizes q/k with a SHARED `W^{q/k}`.
//!   After ShortConv + Swish + L2Norm, q_h and k_h are the same vector. This
//!   module computes them once and reuses the slice for both readout and update.
//! - **Head-wise RMSNorm γ shape.** `[dk]` (shared across heads). Phase 6 verifies
//!   against the actual safetensors.

use crate::gdn2::kernel::{gdn2_recurrent_step, l2_normalize};
use crate::gdn2::short_conv::ShortConv1D;
use crate::gdn2::types::{Gdn2GateConfig, Gdn2HeadState};
use katgpt_core::simd::{simd_dot_f32, simd_matmul_rows};

// ─── Config ─────────────────────────────────────────────────────────────────

/// KDA configuration parameters.
///
/// Mirrors the Kimi-K3-0.40B `kda_*` config fields. See Research 329 §5 for the
/// 0.40B-specific values.
#[derive(Clone, Debug)]
pub struct KdaConfig {
    /// Per-head key/query/value dim (`d_k = d_v`). Kimi-K3-0.40B: 128.
    pub head_dim: usize,
    /// Number of attention heads (`n_h`). Kimi-K3-0.40B: 8.
    pub n_heads: usize,
    /// Hidden dim (`d`). Kimi-K3-0.40B: 1024.
    pub hidden_size: usize,
    /// ShortConv kernel size. Kimi-K3-0.40B: 4.
    pub conv_kernel_size: usize,
    /// Low-rank projection rank for α (per-channel decay). Kimi-K3-0.40B: 128 (= d_k).
    pub alpha_rank: usize,
    /// Low-rank projection rank for the output gate. Kimi-K3-0.40B: 128 (= d_k).
    pub gate_rank: usize,
    /// Numerical floor for α (avoid denormals from `exp(very_negative)`). Default 1e-5.
    pub alpha_eps: f32,
    /// RMSNorm epsilon. Default 1e-6.
    pub rms_eps: f32,
}

impl KdaConfig {
    /// Kimi-K3-0.40B KDA configuration (text path, `kimi_linear` model type).
    pub fn kimi_k3_0_40b() -> Self {
        Self {
            head_dim: 128,
            n_heads: 8,
            hidden_size: 1024,
            conv_kernel_size: 4,
            alpha_rank: 128,
            gate_rank: 128,
            alpha_eps: 1e-5,
            rms_eps: 1e-6,
        }
    }

    /// Total q/k projection output width: `head_dim * n_heads`.
    #[inline]
    pub fn qk_proj_dim(&self) -> usize {
        self.head_dim * self.n_heads
    }
}

// ─── Weights ────────────────────────────────────────────────────────────────

/// KDA layer weight matrices (row-major `Vec<f32>`).
///
/// Naming follows the Kimi Linear paper conventions. See Research 329 §6 for the
/// expected safetensors tensor-name mapping (Phase 5 loader responsibility).
pub struct KdaWeights {
    /// Shared q/k projection `W^{q/k}`. Shape `[head_dim * n_heads, hidden_size]`.
    /// Applied before the shared q/k ShortConv.
    pub qk_proj: Vec<f32>,
    /// Value projection `W^v`. Shape `[head_dim * n_heads, hidden_size]`.
    pub v_proj: Vec<f32>,
    /// β projection `W^β` (scalar per head). Shape `[n_heads, hidden_size]`.
    pub beta_proj: Vec<f32>,
    /// α low-rank down-projection `W^↓_α`. Shape `[alpha_rank, hidden_size]`.
    pub alpha_down: Vec<f32>,
    /// α low-rank up-projection `W^↑_α`. Shape `[head_dim * n_heads, alpha_rank]`.
    pub alpha_up: Vec<f32>,
    /// ShortConv depthwise filter for q/k. Shape `[head_dim * n_heads, conv_kernel_size]`.
    pub qk_conv_weight: Vec<f32>,
    /// ShortConv depthwise filter for v. Shape `[head_dim * n_heads, conv_kernel_size]`.
    pub v_conv_weight: Vec<f32>,
    /// Output gate low-rank down-projection `W^↓_g`. Shape `[gate_rank, hidden_size]`.
    pub gate_down: Vec<f32>,
    /// Output gate low-rank up-projection `W^↑_g`. Shape `[head_dim * n_heads, gate_rank]`.
    pub gate_up: Vec<f32>,
    /// Head-wise RMSNorm weight γ (shared across heads). Shape `[head_dim]`.
    pub head_norm_weight: Vec<f32>,
    /// Output projection `W_o`. Shape `[hidden_size, head_dim * n_heads]`.
    pub o_proj: Vec<f32>,
}

impl KdaWeights {
    /// Construct random weights from a seeded RNG (for G1 testing).
    ///
    /// Uses a simple LCG to avoid pulling a new RNG crate dep. Weights are drawn
    /// from `[-1/sqrt(in_dim), 1/sqrt(in_dim)]` (Xavier-ish initialization for
    /// stable forward passes on random weights).
    pub fn random(config: &KdaConfig, seed: u64) -> Self {
        let mut rng = SimpleRng::new(seed);
        let d = config.hidden_size;
        let dk = config.head_dim;
        let n_h = config.n_heads;
        let qk_dim = dk * n_h;
        let r_a = config.alpha_rank;
        let r_g = config.gate_rank;
        let ks = config.conv_kernel_size;

        Self {
            qk_proj: random_matrix(&mut rng, qk_dim, d),
            v_proj: random_matrix(&mut rng, qk_dim, d),
            beta_proj: random_matrix(&mut rng, n_h, d),
            alpha_down: random_matrix(&mut rng, r_a, d),
            alpha_up: random_matrix(&mut rng, qk_dim, r_a),
            qk_conv_weight: random_matrix(&mut rng, qk_dim, ks),
            v_conv_weight: random_matrix(&mut rng, qk_dim, ks),
            gate_down: random_matrix(&mut rng, r_g, d),
            gate_up: random_matrix(&mut rng, qk_dim, r_g),
            head_norm_weight: random_matrix(&mut rng, dk, 1),
            o_proj: random_matrix(&mut rng, d, qk_dim),
        }
    }
}

/// Tiny deterministic LCG RNG (mirrors the MLA pattern in `mla.rs`).
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }

    #[inline]
    fn next_u32(&mut self) -> u32 {
        // xorshift64* — fast + deterministic.
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        ((self.state.wrapping_mul(0x2545_F491_4F6C_DD1D)) >> 32) as u32
    }

    /// Uniform in `[-scale, scale]`.
    #[inline]
    fn next_f32(&mut self, scale: f32) -> f32 {
        let u = self.next_u32();
        // Map u to [-1, 1] then scale.
        let signed = (u as f32 / u32::MAX as f32) * 2.0 - 1.0;
        signed * scale
    }
}

/// Allocate a `[rows, cols]` row-major matrix with Xavier-ish init.
fn random_matrix(rng: &mut SimpleRng, rows: usize, cols: usize) -> Vec<f32> {
    if cols == 0 {
        return Vec::new();
    }
    let scale = 1.0 / (cols as f32).sqrt();
    (0..rows * cols).map(|_| rng.next_f32(scale)).collect()
}

// ─── State cache ────────────────────────────────────────────────────────────

/// KDA recurrent state + conv ring buffers for a single layer.
///
/// The recurrent state matrix S ∈ R^{d_k × d_v} per head lives in `Gdn2HeadState`
/// (reused from the gdn2 substrate). The ShortConv ring buffers are KDA-specific
/// (the gdn2 forward doesn't apply a short conv).
#[derive(Clone)]
pub struct KdaLayerCache {
    /// Per-head recurrent state matrices (d_k × d_v each). One per head (NOT per
    /// KV group — KDA does not use GQA in the Kimi-K3-0.40B config).
    pub heads: Vec<Gdn2HeadState>,
    /// ShortConv for the shared q/k projection output.
    pub qk_conv: ShortConv1D,
    /// ShortConv for the v projection output.
    pub v_conv: ShortConv1D,
}

impl KdaLayerCache {
    /// Allocate zeroed cache for one KDA layer.
    pub fn new(config: &KdaConfig) -> Self {
        let dk = config.head_dim;
        let dv = config.head_dim;
        let n_h = config.n_heads;
        let qk_dim = dk * n_h;
        Self {
            heads: (0..n_h).map(|_| Gdn2HeadState::new(dk, dv)).collect(),
            qk_conv: ShortConv1D::new(qk_dim, config.conv_kernel_size),
            v_conv: ShortConv1D::new(qk_dim, config.conv_kernel_size),
        }
    }

    /// Reset state matrices + conv ring buffers to zeros (reuse allocations).
    pub fn reset(&mut self) {
        for h in &mut self.heads {
            h.reset();
        }
        self.qk_conv.reset();
        self.v_conv.reset();
    }
}

/// Multi-layer KDA cache (for models with multiple KDA layers — Kimi-K3-0.40B has 6).
#[derive(Clone)]
pub struct KdaCache {
    /// Per-layer state + conv ring buffers.
    pub layers: Vec<KdaLayerCache>,
}

impl KdaCache {
    /// Allocate zeroed cache for `n_layers` KDA layers.
    pub fn new(config: &KdaConfig, n_layers: usize) -> Self {
        Self {
            layers: (0..n_layers).map(|_| KdaLayerCache::new(config)).collect(),
        }
    }

    /// Reset all layers (reuse allocations).
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }
}

// ─── Forward scratch ────────────────────────────────────────────────────────

/// Pre-allocated scratch buffers for `kda_forward_token` (zero-alloc hot path).
///
/// All buffers are sized from the config at construction time; the forward path
/// only mutates them in place.
pub struct KdaForwardScratch {
    /// q/k projection output (pre-conv). Shape `[head_dim * n_heads]`.
    pub z_qk: Vec<f32>,
    /// v projection output (pre-conv). Shape `[head_dim * n_heads]`.
    pub z_v: Vec<f32>,
    /// q/k post-conv. Shape `[head_dim * n_heads]`.
    pub z_qk_conv: Vec<f32>,
    /// v post-conv. Shape `[head_dim * n_heads]`.
    pub z_v_conv: Vec<f32>,
    /// α low-rank down-projection intermediate. Shape `[alpha_rank]`.
    pub alpha_hidden: Vec<f32>,
    /// α flat (log-space) after up-projection. Shape `[head_dim * n_heads]`.
    pub log_alpha_flat: Vec<f32>,
    /// β per head (post-sigmoid). Shape `[n_heads]`.
    pub beta: Vec<f32>,
    /// β projection pre-sigmoid. Shape `[n_heads]`.
    pub beta_pre: Vec<f32>,
    /// Per-head β-broadcast erase buffer (= β · ones). Shape `[head_dim]`.
    pub erase_b_h: Vec<f32>,
    /// Per-head α (post-exp, post-clamp). Shape `[head_dim]`.
    pub alpha_h: Vec<f32>,
    /// Per-head output buffer (readout result, pre-norm). Shape `[head_dim]`.
    pub o_h: Vec<f32>,
    /// Per-head RMSNormed output. Shape `[head_dim]`.
    pub o_h_norm: Vec<f32>,
    /// Concatenated per-head outputs (post-norm). Shape `[head_dim * n_heads]`.
    pub o_concat: Vec<f32>,
    /// Output gate low-rank down-projection intermediate. Shape `[gate_rank]`.
    pub gate_hidden: Vec<f32>,
    /// Output gate (post-sigmoid). Shape `[head_dim * n_heads]`.
    pub gate: Vec<f32>,
    /// Output gate pre-sigmoid. Shape `[head_dim * n_heads]`.
    pub gate_pre: Vec<f32>,
    /// Final output (post o_proj). Shape `[hidden_size]`.
    pub output: Vec<f32>,
    // ── gdn2_recurrent_step inner scratch (passed per call) ──
    /// gdn2 temp buffer. Shape `[head_dim]`.
    pub gdn2_temp: Vec<f32>,
    /// gdn2 delta buffer. Shape `[head_dim]`.
    pub gdn2_delta: Vec<f32>,
    /// gdn2 write_w_channel buffer (unused for Kda mode but required by signature).
    /// Shape `[head_dim]`, filled with 1.0.
    pub gdn2_write_w: Vec<f32>,
}

impl KdaForwardScratch {
    /// Allocate scratch for the given config.
    pub fn new(config: &KdaConfig) -> Self {
        let dk = config.head_dim;
        let n_h = config.n_heads;
        let d = config.hidden_size;
        let qk_dim = dk * n_h;
        Self {
            z_qk: vec![0.0; qk_dim],
            z_v: vec![0.0; qk_dim],
            z_qk_conv: vec![0.0; qk_dim],
            z_v_conv: vec![0.0; qk_dim],
            alpha_hidden: vec![0.0; config.alpha_rank],
            log_alpha_flat: vec![0.0; qk_dim],
            beta: vec![0.0; n_h],
            beta_pre: vec![0.0; n_h],
            erase_b_h: vec![0.0; dk],
            alpha_h: vec![0.0; dk],
            o_h: vec![0.0; dk],
            o_h_norm: vec![0.0; dk],
            o_concat: vec![0.0; qk_dim],
            gate_hidden: vec![0.0; config.gate_rank],
            gate: vec![0.0; qk_dim],
            gate_pre: vec![0.0; qk_dim],
            output: vec![0.0; d],
            gdn2_temp: vec![0.0; dk],
            gdn2_delta: vec![0.0; dk],
            gdn2_write_w: vec![1.0; dk],
        }
    }
}

// ─── Forward kernel ─────────────────────────────────────────────────────────

/// Per-token KDA forward pass.
///
/// Updates the layer's recurrent state in-place and returns a slice into
/// `scratch.output` of length `config.hidden_size`.
///
/// # Arguments
/// * `config` — KDA configuration
/// * `weights` — KDA weight matrices
/// * `cache` — per-layer recurrent state + conv ring buffers
/// * `scratch` — pre-allocated scratch (reused across calls)
/// * `h` — input hidden state `[hidden_size]`
///
/// # Returns
/// A mutable slice into `scratch.output[..hidden_size]`.
pub fn kda_forward_token<'s>(
    config: &KdaConfig,
    weights: &KdaWeights,
    cache: &mut KdaLayerCache,
    scratch: &'s mut KdaForwardScratch,
    h: &[f32],
) -> &'s mut [f32] {
    let d = config.hidden_size;
    let dk = config.head_dim;
    let n_h = config.n_heads;
    let qk_dim = dk * n_h;
    debug_assert_eq!(h.len(), d, "hidden state dim mismatch");

    // ── Step 1: Projections ─────────────────────────────────────────────────
    // z_qk = W^{q/k} · h    [qk_dim]
    simd_matmul_rows(&mut scratch.z_qk, &weights.qk_proj, h, qk_dim, d);
    // z_v = W^v · h         [qk_dim]
    simd_matmul_rows(&mut scratch.z_v, &weights.v_proj, h, qk_dim, d);

    // α low-rank: alpha_hidden = W^↓_α · h   [alpha_rank]
    simd_matmul_rows(
        &mut scratch.alpha_hidden,
        &weights.alpha_down,
        h,
        config.alpha_rank,
        d,
    );
    // log_alpha_flat = W^↑_α · alpha_hidden  [qk_dim]
    simd_matmul_rows(
        &mut scratch.log_alpha_flat,
        &weights.alpha_up,
        &scratch.alpha_hidden,
        qk_dim,
        config.alpha_rank,
    );

    // β pre-sigmoid: beta_pre = W^β · h       [n_heads]
    simd_matmul_rows(&mut scratch.beta_pre, &weights.beta_proj, h, n_h, d);

    // ── Step 2: ShortConv (depthwise causal) ────────────────────────────────
    // Override the identity init with the real weights before the forward step.
    // This is a memcpy (cheaper than re-allocating); in production, weights are
    // loaded once at boot and never change.
    cache.qk_conv.weight.copy_from_slice(&weights.qk_conv_weight);
    cache.v_conv.weight.copy_from_slice(&weights.v_conv_weight);
    cache.qk_conv.forward(&scratch.z_qk, &mut scratch.z_qk_conv);
    cache.v_conv.forward(&scratch.z_v, &mut scratch.z_v_conv);

    // ── Step 3: Swish activation (z * sigmoid(z)) ───────────────────────────
    swish_inplace(&mut scratch.z_qk_conv);
    swish_inplace(&mut scratch.z_v_conv);

    // β = sigmoid(beta_pre)
    for i in 0..n_h {
        scratch.beta[i] = sigmoid(scratch.beta_pre[i]);
    }

    // ── Step 4: Per-head recurrent step (KDA eq 1 via gdn2 kernel) ──────────
    // q and k share the projection + conv + swish output; both get L2Norm'd.
    // We L2Norm the shared slice ONCE and reuse it as both q and k (same vector).
    //
    // Per-head loop:
    //   - L2Norm the head's qk slice (this becomes q_h = k_h)
    //   - α_h = exp(log_alpha_flat[h*dk..(h+1)*dk]).max(eps)
    //   - β_h = beta[h]
    //   - erase_b_h = β_h · ones[dk]
    //   - v_h = z_v_conv[h*dk..(h+1)*dk]   (NO L2Norm)
    //   - Call gdn2_recurrent_step(k_h, v_h, q_h, S_h, alpha_h, erase_b_h, β_h,
    //         write_w_channel, out=o_h, temp, delta, dk, dk, Kda)
    for head in 0..n_h {
        let off = head * dk;
        // q_h = k_h = L2Norm(z_qk_conv[off..off+dk]). Same slice, normed once.
        let qk_h = &mut scratch.z_qk_conv[off..off + dk];
        l2_normalize(qk_h);

        // v_h = z_v_conv[h slice] (borrow only — no norm).
        // We need a separate copy because the gdn2 kernel takes `k` and `v` as
        // separate slices and we can't alias scratch.z_qk_conv (k) with anything
        // else here. But q_h = k_h literally, so we pass the same slice twice.
        let v_h = &scratch.z_v_conv[off..off + dk];

        // α_h = exp(log_α).max(eps)
        for i in 0..dk {
            let la = scratch.log_alpha_flat[off + i];
            let a = la.exp();
            scratch.alpha_h[i] = if a < config.alpha_eps {
                config.alpha_eps
            } else {
                a
            };
        }

        // β_h + erase_b_h broadcast
        let beta_h = scratch.beta[head];
        for i in 0..dk {
            scratch.erase_b_h[i] = beta_h;
        }

        // Call the gdn2 kernel with Kda gate config.
        // k = qk_h (same as q — they share the projection), v = v_h, q = qk_h.
        // write_w_channel is unused for Kda mode (kernel uses w_val = β_h).
        let s = &mut cache.heads[head].s;
        gdn2_recurrent_step(
            qk_h,                       // k
            v_h,                        // v
            qk_h,                       // q (same as k — shared projection)
            s,                          // state matrix (dk × dv), updated in-place
            &scratch.alpha_h,           // per-channel decay
            &scratch.erase_b_h,         // β-broadcast erase gate
            beta_h,                     // w_val = β (scalar write weight for Kda)
            &scratch.gdn2_write_w,      // write_w_channel (unused for Kda)
            &mut scratch.o_h,           // output [dv]
            &mut scratch.gdn2_temp,     // temp buffer [dv]
            &mut scratch.gdn2_delta,    // delta buffer [dv]
            dk,
            dk,
            Gdn2GateConfig::Kda,
        );

        // Head-wise RMSNorm on o_h → o_h_norm
        rmsnorm_into(
            &scratch.o_h,
            &weights.head_norm_weight,
            config.rms_eps,
            &mut scratch.o_h_norm,
        );

        // Copy o_h_norm into o_concat[head slice]
        let cat_off = head * dk;
        scratch.o_concat[cat_off..cat_off + dk].copy_from_slice(&scratch.o_h_norm);
    }

    // ── Step 5: Output gate (low-rank) ──────────────────────────────────────
    // gate_hidden = W^↓_g · h   [gate_rank]
    simd_matmul_rows(
        &mut scratch.gate_hidden,
        &weights.gate_down,
        h,
        config.gate_rank,
        d,
    );
    // gate_pre = W^↑_g · gate_hidden  [qk_dim]
    simd_matmul_rows(
        &mut scratch.gate_pre,
        &weights.gate_up,
        &scratch.gate_hidden,
        qk_dim,
        config.gate_rank,
    );
    // gate = sigmoid(gate_pre)
    for i in 0..qk_dim {
        scratch.gate[i] = sigmoid(scratch.gate_pre[i]);
    }

    // o_concat *= gate (elementwise)
    for i in 0..qk_dim {
        scratch.o_concat[i] *= scratch.gate[i];
    }

    // ── Step 6: Output projection ───────────────────────────────────────────
    // output = W_o · o_concat   [d]
    simd_matmul_rows(&mut scratch.output, &weights.o_proj, &scratch.o_concat, d, qk_dim);

    &mut scratch.output[..d]
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Swish activation: `x[i] = x[i] * sigmoid(x[i])`. In-place.
#[inline]
fn swish_inplace(v: &mut [f32]) {
    for x in v.iter_mut() {
        *x = *x * sigmoid(*x);
    }
}

/// Sigmoid via `katgpt_core::simd::fast_sigmoid`.
#[inline]
fn sigmoid(x: f32) -> f32 {
    katgpt_core::simd::fast_sigmoid(x)
}

/// RMSNorm into a separate output buffer: `out = γ * (x / sqrt(mean(x²) + eps))`.
///
/// Avoids the katgpt-types rmsnorm which is in-place; we need a separate output
/// here because the gdn2 kernel wrote into `o_h` and we want to preserve it.
fn rmsnorm_into(x: &[f32], gamma: &[f32], eps: f32, out: &mut [f32]) {
    debug_assert_eq!(x.len(), out.len());
    debug_assert_eq!(x.len(), gamma.len());
    // mean square
    let sum_sq: f32 = simd_dot_f32(x, x, x.len());
    let mean_sq = sum_sq / x.len() as f32;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * inv_rms * gamma[i];
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> KdaConfig {
        KdaConfig {
            head_dim: 8,
            n_heads: 2,
            hidden_size: 16,
            conv_kernel_size: 4,
            alpha_rank: 8,
            gate_rank: 8,
            alpha_eps: 1e-5,
            rms_eps: 1e-6,
        }
    }

    #[test]
    fn smoke_kimi_k3_0_40b_dims_finite() {
        // Verify the full 0.40B config dimensions work end-to-end.
        let config = KdaConfig::kimi_k3_0_40b();
        let weights = KdaWeights::random(&config, 99);
        let mut cache = KdaLayerCache::new(&config);
        let mut scratch = KdaForwardScratch::new(&config);
        let h = vec![0.1f32; config.hidden_size];
        let out = kda_forward_token(&config, &weights, &mut cache, &mut scratch, &h);
        assert_eq!(out.len(), 1024);
        for &v in out.iter() {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }

    #[test]
    fn smoke_small_config_finite_3_tokens() {
        let config = small_config();
        let weights = KdaWeights::random(&config, 42);
        let mut cache = KdaLayerCache::new(&config);
        let mut scratch = KdaForwardScratch::new(&config);
        for t in 0..3 {
            let h: Vec<f32> = (0..config.hidden_size)
                .map(|i| ((i + t * 7) as f32).sin() * 0.1)
                .collect();
            let out = kda_forward_token(&config, &weights, &mut cache, &mut scratch, &h);
            assert_eq!(out.len(), config.hidden_size);
            for &v in out.iter() {
                assert!(v.is_finite(), "non-finite output at token {t}: {v}");
            }
        }
    }

    #[test]
    fn cache_reset_zeros_state() {
        let config = small_config();
        let mut cache = KdaLayerCache::new(&config);
        // Mutate state
        cache.heads[0].s[0] = 5.0;
        cache.qk_conv.buf[0] = 1.0;
        cache.reset();
        assert!(cache.heads[0].s.iter().all(|&x| x == 0.0));
        assert!(cache.qk_conv.buf.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn determinism_same_inputs_same_output() {
        // Run forward twice with same inputs + fresh cache → outputs must be bit-identical.
        let config = small_config();
        let weights = KdaWeights::random(&config, 7);

        let mut c1 = KdaLayerCache::new(&config);
        let mut s1 = KdaForwardScratch::new(&config);
        let h = vec![0.2f32; config.hidden_size];
        let o1 = kda_forward_token(&config, &weights, &mut c1, &mut s1, &h).to_vec();

        let mut c2 = KdaLayerCache::new(&config);
        let mut s2 = KdaForwardScratch::new(&config);
        let o2 = kda_forward_token(&config, &weights, &mut c2, &mut s2, &h).to_vec();

        assert_eq!(o1, o2, "same inputs + fresh state → bit-identical output");
    }

    #[test]
    fn state_grows_across_tokens() {
        // After 2 tokens, the state matrix should have non-zero magnitude
        // (the second token read + wrote into it).
        let config = small_config();
        let weights = KdaWeights::random(&config, 11);
        let mut cache = KdaLayerCache::new(&config);
        let mut scratch = KdaForwardScratch::new(&config);

        let h1 = vec![0.3f32; config.hidden_size];
        kda_forward_token(&config, &weights, &mut cache, &mut scratch, &h1);

        let s1_norm: f32 = cache.heads[0].s.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            s1_norm > 0.0,
            "state should be non-zero after first token, got norm {s1_norm}"
        );

        let h2 = vec![0.5f32; config.hidden_size];
        kda_forward_token(&config, &weights, &mut cache, &mut scratch, &h2);

        let s2_norm: f32 = cache.heads[0].s.iter().map(|x| x * x).sum::<f32>().sqrt();
        // State should evolve (not identical to after token 1).
        assert!(
            (s1_norm - s2_norm).abs() > 1e-6 || s1_norm != s2_norm,
            "state should evolve across tokens (s1={s1_norm}, s2={s2_norm})"
        );
    }
}
