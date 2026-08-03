//! Kimi-K3 full-model analytic backward (Plan 318 Phase C C6).
//!
//! Composes the three per-primitive backward modules (MLA + MoE + KDA from
//! C4/C5) with the model-level composition backward (attn-res + RMSNorm +
//! dense SiTU FFN + LM head + embedding) into a full-model gradient pass.
//!
//! # Architecture
//!
//! The Kimi-K3 model uses a non-standard residual structure:
//! - A running `prefix_sum` accumulates attention + FFN outputs across layers
//! - At block boundaries (`layer_idx % block_size == 0`), the current
//!   `prefix_sum` is pushed to `block_state` and reset to zero
//! - Before each sub-layer (attention + FFN), `prefix_sum` is mixed with past
//!   block residuals via `apply_attn_res` (softmax-weighted averaging)
//!
//! The backward distributes gradients through this prefix_sum + block_state +
//! attn-res pattern.
//!
//! # Cross-token gradient flow
//!
//! MLA handles cross-token KV-cache gradients internally (each token attends to
//! ALL cached tokens via shared W_UK/W_UV). KDA uses BPTT across the conv ring +
//! SSM state. MoE/Dense FFN are stateless. The model-level backward processes
//! layers in reverse; for each layer, it collects dL/d(attention_output) for
//! all tokens, then runs the primitive backward (MLA cross-token or KDA BPTT).
//!
//! # Scope
//!
//! CPU reference for the GPU training loop (C10). Gated behind `kimi_k3_backward`.

use katgpt_attn::gdn2::kda_backward::{
    KdaGradients, KdaSavedActivations, kda_backward_sequence, kda_forward_token_with_saved,
};
use katgpt_attn::mla_backward::{
    MlaGradients, MlaSavedActivations, mla_backward_token, mla_forward_token_with_saved,
    rmsnorm_backward as mla_rmsnorm_backward,
};
use katgpt_core::simd::{
    simd_dot_f32, simd_matmul_rows, simd_outer_product_acc, simd_sum_sq,
    simd_transpose_matvec_into,
};
use katgpt_core::types::math::{rmsnorm_with_gamma_eps, situ};
use katgpt_kv::shard_kv::rope::RopeFreqs;
use katgpt_transformer::attn_res::{
    AttnResBlockState, AttnResConfig, AttnResScratch, AttnResWeights, apply_attn_res,
};
use katgpt_transformer::moe::SwiGluExpertWeights;
use katgpt_transformer::moe_backward::{
    MoeGradients, MoeSavedActivations, moe_backward_token, moe_forward_token_with_saved,
};

use super::decoder_layer::{
    KimiAttentionConfig, KimiAttentionScratch, KimiAttentionState, KimiAttentionWeights,
    KimiDecoderLayerConfig, KimiDecoderLayerWeights, KimiFfnConfig, KimiFfnScratch, KimiFfnWeights,
};
use super::loader::KimiK3ModelWeights;
use super::model::{KimiK3ModelConfig, KimiK3Runtime};

// ─── Dense SiTU FFN saved activations (layer 0 only) ────────────────────────

#[derive(Clone)]
pub struct DenseFfnSavedActivations {
    pub h: Vec<f32>,
    pub gate_inter: Vec<f32>,
    pub up_inter: Vec<f32>,
    pub act_out: Vec<f32>,
}

/// Dense FFN gradients (gate_proj, up_proj, down_proj).
pub struct DenseFfnGradients {
    pub gate_proj: Vec<f32>,
    pub up_proj: Vec<f32>,
    pub down_proj: Vec<f32>,
}

// ─── Per-layer saved activations ───────────────────────────────────────────

/// Saved activations for one decoder layer at one token position.
#[derive(Clone)]
pub struct LayerSavedActivations {
    /// `prefix_sum` at layer entry (input from previous layer / embedding). `[d]`.
    pub prefix_sum_in: Vec<f32>,
    /// Whether `apply_attn_res` was called for the self block.
    pub has_self_attn_res: bool,
    /// The mixed hidden (input to input_layernorm). `[d]`.
    pub mixed_self: Vec<f32>,
    /// RMSNorm inv_rms for input_layernorm.
    pub self_inv_rms: f32,
    /// Whether this layer is a block boundary.
    pub is_boundary: bool,
    /// Block-state entries snapshot BEFORE the self-attn-res mixing. `[num_entries][d]`.
    pub block_state_self: Vec<Vec<f32>>,
    /// Attention output. `[d]`.
    pub attn_out: Vec<f32>,
    /// `prefix_sum` after attention accumulation. `[d]`.
    pub prefix_sum_after_attn: Vec<f32>,
    /// The mixed hidden (input to post_attention_layernorm). `[d]`.
    pub mixed_mlp: Vec<f32>,
    /// RMSNorm inv_rms for post_attention_layernorm.
    pub mlp_inv_rms: f32,
    /// Block-state entries for the MLP attn-res. `[num_entries][d]`.
    pub block_state_mlp: Vec<Vec<f32>>,
    /// FFN output. `[d]`.
    pub ffn_out: Vec<f32>,
    /// MLA saved activations (if MLA layer).
    pub mla_saved: Option<MlaSavedActivations>,
    /// KDA saved activations (if KDA layer).
    pub kda_saved: Option<KdaSavedActivations>,
    /// MoE saved activations (if MoE layer).
    pub moe_saved: Option<MoeSavedActivations>,
    /// Dense FFN intermediates (if dense layer).
    pub dense_saved: Option<DenseFfnSavedActivations>,
}

/// Model-level saved activations for one token.
pub struct TokenSavedActivations {
    pub token_id: u32,
    pub layers: Vec<LayerSavedActivations>,
    /// `prefix_sum` after last layer (before output attn-res). `[d]`.
    pub prefix_sum_final: Vec<f32>,
    /// Hidden after output attn-res, before final norm (= final norm input). `[d]`.
    pub pre_final_norm: Vec<f32>,
    /// Block-state snapshot before output attn-res. `[num_entries][d]`.
    pub block_state_final: Vec<Vec<f32>>,
    pub has_output_attn_res: bool,
    /// Final normed hidden (input to LM head). `[d]`.
    pub final_hidden: Vec<f32>,
    pub final_norm_inv_rms: f32,
    /// Logits. `[vocab_size]`.
    pub logits: Vec<f32>,
    pub pos: usize,
}

impl TokenSavedActivations {
    /// Allocate an empty struct (layers vector empty; will be filled by forward).
    pub fn new() -> Self {
        Self {
            token_id: 0,
            layers: Vec::new(),
            prefix_sum_final: Vec::new(),
            pre_final_norm: Vec::new(),
            block_state_final: Vec::new(),
            has_output_attn_res: false,
            final_hidden: Vec::new(),
            final_norm_inv_rms: 0.0,
            logits: Vec::new(),
            pos: 0,
        }
    }
}

impl Default for TokenSavedActivations {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Per-layer gradients ───────────────────────────────────────────────────

pub struct LayerGradients {
    pub input_layernorm_weight: Vec<f32>,
    pub post_attention_layernorm_weight: Vec<f32>,
    pub self_attn_res_norm: Vec<f32>,
    pub self_attn_res_proj: Vec<f32>,
    pub mlp_attn_res_norm: Vec<f32>,
    pub mlp_attn_res_proj: Vec<f32>,
    pub mla_grads: Option<MlaGradients>,
    pub kda_grads: Option<KdaGradients>,
    pub moe_grads: Option<MoeGradients>,
    pub dense_grads: Option<DenseFfnGradients>,
}

impl LayerGradients {
    pub fn zeros_like(config: &KimiK3ModelConfig, layer_idx: usize, weights: &KimiDecoderLayerWeights) -> Self {
        let d = config.hidden_size;
        let is_mla = config.is_mla_layer(layer_idx);
        let is_dense = layer_idx == 0;

        let (mla_grads, kda_grads) = if is_mla {
            let KimiAttentionWeights::Mla(w) = &weights.attention else {
                panic!("MLA layer but non-MLA weights");
            };
            (Some(MlaGradients::zeros_like(w)), None)
        } else {
            let KimiAttentionWeights::Kda(w) = &weights.attention else {
                panic!("KDA layer but non-KDA weights");
            };
            (None, Some(KdaGradients::zeros_like(w)))
        };

        let (moe_grads, dense_grads) = if is_dense {
            let KimiFfnWeights::Dense(e) = &weights.ffn else {
                panic!("Dense layer but non-dense FFN");
            };
            (
                None,
                Some(DenseFfnGradients {
                    gate_proj: vec![0.0; e.gate_proj.len()],
                    up_proj: vec![0.0; e.up_proj.len()],
                    down_proj: vec![0.0; e.down_proj.len()],
                }),
            )
        } else {
            let KimiFfnWeights::Moe(w) = &weights.ffn else {
                panic!("MoE layer but non-MoE FFN");
            };
            (Some(MoeGradients::zeros_like(w)), None)
        };

        Self {
            input_layernorm_weight: vec![0.0; d],
            post_attention_layernorm_weight: vec![0.0; d],
            self_attn_res_norm: vec![0.0; d],
            self_attn_res_proj: vec![0.0; d],
            mlp_attn_res_norm: vec![0.0; d],
            mlp_attn_res_proj: vec![0.0; d],
            mla_grads,
            kda_grads,
            moe_grads,
            dense_grads,
        }
    }
}

/// Full model gradient accumulator.
pub struct KimiK3ModelGradients {
    pub embed_weight: Vec<f32>,
    pub layers: Vec<LayerGradients>,
    pub final_norm_weight: Vec<f32>,
    pub lm_head_weight: Vec<f32>,
    pub output_attn_res_norm: Vec<f32>,
    pub output_attn_res_proj: Vec<f32>,
}

impl KimiK3ModelGradients {
    pub fn zeros_like(config: &KimiK3ModelConfig, weights: &KimiK3ModelWeights) -> Self {
        let layers = (0..config.num_layers)
            .map(|i| LayerGradients::zeros_like(config, i, &weights.layers[i]))
            .collect();
        Self {
            embed_weight: vec![0.0; weights.embed_weight.len()],
            layers,
            final_norm_weight: vec![0.0; weights.final_norm_weight.len()],
            lm_head_weight: vec![0.0; weights.lm_head_weight.len()],
            output_attn_res_norm: vec![0.0; config.hidden_size],
            output_attn_res_proj: vec![0.0; config.hidden_size],
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Forward with saved activations
// ═══════════════════════════════════════════════════════════════════════════

/// Run the full Kimi-K3 forward for one token, saving all activations.
///
/// Mirrors `kimi_k3_forward_token` exactly but captures all intermediates.
/// Returns a reference to `runtime.logits`.
#[allow(clippy::too_many_arguments)]
pub fn kimi_k3_forward_token_saved(
    config: &KimiK3ModelConfig,
    weights: &KimiK3ModelWeights,
    runtime: &mut KimiK3Runtime,
    token_id: u32,
    pos: usize,
    saved: &mut TokenSavedActivations,
) {
    let d = config.hidden_size;

    runtime.block_state.clear();
    saved.layers.clear();
    saved.pos = pos;
    saved.token_id = token_id;

    // Embedding lookup
    let embed_start = (token_id as usize) * d;
    runtime.hidden.copy_from_slice(&weights.embed_weight[embed_start..embed_start + d]);

    // Decoder layers
    for (layer_idx, layer_w) in weights.layers.iter().enumerate() {
        let layer_cfg = config.layer_config(layer_idx);
        let layer_rt = &mut runtime.layers[layer_idx];

        let mut layer_saved = LayerSavedActivations {
            prefix_sum_in: runtime.hidden.clone(),
            has_self_attn_res: false,
            mixed_self: Vec::new(),
            self_inv_rms: 0.0,
            is_boundary: false,
            block_state_self: Vec::new(),
            attn_out: Vec::new(),
            prefix_sum_after_attn: Vec::new(),
            mixed_mlp: Vec::new(),
            mlp_inv_rms: 0.0,
            block_state_mlp: Vec::new(),
            ffn_out: Vec::new(),
            mla_saved: None,
            kda_saved: None,
            moe_saved: None,
            dense_saved: None,
        };

        forward_layer_saved(
            layer_idx,
            &layer_cfg,
            layer_w,
            &mut layer_rt.attn_state,
            &mut layer_rt.attn_scratch,
            &mut layer_rt.ffn_scratch,
            &mut layer_rt.attn_res_self_scratch,
            &mut layer_rt.attn_res_mlp_scratch,
            &mut runtime.block_state,
            Some(&mut runtime.rope_freqs),
            &mut runtime.hidden,
            &mut runtime.scratch_hidden,
            &mut layer_saved,
        );

        saved.layers.push(layer_saved);
    }

    // Output attn-res
    saved.prefix_sum_final = runtime.hidden.clone();
    saved.block_state_final = runtime.block_state.residuals.clone();
    saved.has_output_attn_res = !runtime.block_state.is_empty();

    if !runtime.block_state.is_empty() {
        let mixed = apply_attn_res(
            &config.attn_res_config,
            &weights.output_attn_res,
            &runtime.block_state,
            &mut runtime.output_attn_res_scratch,
            &runtime.hidden,
        );
        runtime.hidden.copy_from_slice(mixed);
    }

    // Save post-output-attn-res hidden (= final norm input)
    saved.pre_final_norm = runtime.hidden.clone();

    // Final RMSNorm
    let sum_sq = simd_sum_sq(&runtime.hidden, d);
    let inv_rms = 1.0 / ((sum_sq / d as f32 + config.rms_eps).sqrt());
    saved.final_norm_inv_rms = inv_rms;
    rmsnorm_with_gamma_eps(&mut runtime.hidden, &weights.final_norm_weight, config.rms_eps as f64);
    saved.final_hidden = runtime.hidden.clone();

    // LM head
    simd_matmul_rows(
        &mut runtime.logits,
        &weights.lm_head_weight,
        &runtime.hidden,
        config.vocab_size,
        d,
    );
    saved.logits = runtime.logits.clone();
}

#[allow(clippy::too_many_arguments)]
fn forward_layer_saved(
    layer_idx: usize,
    config: &KimiDecoderLayerConfig,
    weights: &KimiDecoderLayerWeights,
    attn_state: &mut KimiAttentionState,
    attn_scratch: &mut KimiAttentionScratch,
    ffn_scratch: &mut KimiFfnScratch,
    attn_res_self_scratch: &mut AttnResScratch,
    attn_res_mlp_scratch: &mut AttnResScratch,
    block_state: &mut AttnResBlockState,
    rope_freqs: Option<&mut RopeFreqs>,
    prefix_sum: &mut [f32],
    scratch_hidden: &mut [f32],
    saved: &mut LayerSavedActivations,
) {
    let d = config.attn_res.d();
    let eps = config.rms_eps;
    let block_size = config.attn_res.block_size;
    let is_boundary = layer_idx.is_multiple_of(block_size);
    saved.is_boundary = is_boundary;

    // Step 1: self-attn-res mixing
    if !block_state.is_empty() {
        saved.has_self_attn_res = true;
        saved.block_state_self = block_state.residuals.clone();
        let mixed = apply_attn_res(
            &config.attn_res,
            &weights.self_attn_res,
            block_state,
            attn_res_self_scratch,
            prefix_sum,
        );
        scratch_hidden.copy_from_slice(mixed);
    } else {
        scratch_hidden.copy_from_slice(prefix_sum);
    }
    saved.mixed_self = scratch_hidden.to_vec();

    // Step 2: boundary push
    if is_boundary {
        block_state.push(prefix_sum);
        for x in prefix_sum.iter_mut() {
            *x = 0.0;
        }
    }

    // Step 3: input_layernorm + attention
    let sum_sq = simd_sum_sq(scratch_hidden, d);
    let inv_rms = 1.0 / ((sum_sq / d as f32 + eps).sqrt());
    saved.self_inv_rms = inv_rms;
    rmsnorm_with_gamma_eps(scratch_hidden, &weights.input_layernorm_weight, eps as f64);

    let attn_out: Vec<f32> = match (&config.attention, &weights.attention) {
        (KimiAttentionConfig::Mla(cfg), KimiAttentionWeights::Mla(w)) => {
            let KimiAttentionState::Mla(cache) = attn_state else { panic!("MLA state mismatch") };
            let KimiAttentionScratch::Mla(scratch) = attn_scratch else { panic!("MLA scratch mismatch") };
            let Some(rf) = rope_freqs else { panic!("MLA needs rope") };
            let (out, s) = mla_forward_token_with_saved(cfg, w, cache, scratch, rf, scratch_hidden);
            saved.mla_saved = Some(s);
            out
        }
        (KimiAttentionConfig::Kda(cfg), KimiAttentionWeights::Kda(w)) => {
            let KimiAttentionState::Kda(cache) = attn_state else { panic!("KDA state mismatch") };
            let KimiAttentionScratch::Kda(scratch) = attn_scratch else { panic!("KDA scratch mismatch") };
            let (out, s) = kda_forward_token_with_saved(cfg, w, cache, scratch, scratch_hidden);
            saved.kda_saved = Some(s);
            out
        }
        _ => panic!("attention mismatch"),
    };
    saved.attn_out = attn_out.clone();

    // Step 4: prefix_sum += attn_out
    for i in 0..d {
        prefix_sum[i] += attn_out[i];
    }
    saved.prefix_sum_after_attn = prefix_sum.to_vec();

    // Step 5: MLP attn-res
    saved.block_state_mlp = block_state.residuals.clone();
    let mixed = apply_attn_res(
        &config.attn_res,
        &weights.mlp_attn_res,
        block_state,
        attn_res_mlp_scratch,
        prefix_sum,
    );
    scratch_hidden.copy_from_slice(mixed);
    saved.mixed_mlp = scratch_hidden.to_vec();

    // Step 6: post_attention_layernorm + FFN
    let sum_sq = simd_sum_sq(scratch_hidden, d);
    let inv_rms = 1.0 / ((sum_sq / d as f32 + eps).sqrt());
    saved.mlp_inv_rms = inv_rms;
    rmsnorm_with_gamma_eps(scratch_hidden, &weights.post_attention_layernorm_weight, eps as f64);

    let ffn_out: Vec<f32> = match (&config.ffn, &weights.ffn) {
        (KimiFfnConfig::Dense { situ_beta, situ_linear_beta, .. }, KimiFfnWeights::Dense(expert)) => {
            let (out, s) = dense_situ_ffn_forward_saved(
                expert, scratch_hidden, ffn_scratch, *situ_beta, *situ_linear_beta,
            );
            saved.dense_saved = Some(s);
            out
        }
        (KimiFfnConfig::Moe(cfg), KimiFfnWeights::Moe(w)) => {
            let (out, s) = moe_forward_token_with_saved(w, cfg, scratch_hidden, &mut ffn_scratch.moe);
            ffn_scratch.dense_out[..d].copy_from_slice(&out[..d]);
            saved.moe_saved = Some(s);
            out
        }
        _ => panic!("FFN mismatch"),
    };
    saved.ffn_out = ffn_out.clone();

    // Step 7: prefix_sum += ffn_out
    for i in 0..d {
        prefix_sum[i] += ffn_out[i];
    }
}

fn dense_situ_ffn_forward_saved(
    expert: &SwiGluExpertWeights,
    hidden: &[f32],
    scratch: &mut KimiFfnScratch,
    beta: f32,
    linear_beta: Option<f32>,
) -> (Vec<f32>, DenseFfnSavedActivations) {
    let d_in = scratch.dense_out.len();
    let d_ffn = expert.gate_proj.len() / d_in;

    simd_matmul_rows(&mut scratch.dense_gate, &expert.gate_proj, hidden, d_ffn, d_in);
    simd_matmul_rows(&mut scratch.dense_up, &expert.up_proj, hidden, d_ffn, d_in);
    situ(&mut scratch.dense_act, &scratch.dense_gate, &scratch.dense_up, beta, linear_beta);
    simd_matmul_rows(&mut scratch.dense_out, &expert.down_proj, &scratch.dense_act, d_in, d_ffn);

    let saved = DenseFfnSavedActivations {
        h: hidden.to_vec(),
        gate_inter: scratch.dense_gate.clone(),
        up_inter: scratch.dense_up.clone(),
        act_out: scratch.dense_act.clone(),
    };
    (scratch.dense_out.clone(), saved)
}

// ═══════════════════════════════════════════════════════════════════════════
// Backward
// ═══════════════════════════════════════════════════════════════════════════

/// Full-model backward for a sequence of tokens.
///
/// Computes gradients w.r.t. all model parameters by composing the three
/// per-primitive backward modules (MLA + MoE + KDA) with the model-level
/// composition backward (attn-res + RMSNorm + dense FFN + LM head + embedding).
///
/// # Gradient flow
///
/// Forward per token: `embed → [layers] → output_attn_res → final_norm → lm_head → logits`
///
/// Each layer forward (prefix_sum pattern):
/// ```text
/// mixed_self = attn_res(ps_in, blocks_self) or ps_in
/// if boundary: blocks.push(ps_in); ps_in = 0
/// attn_out = attention(rmsnorm(mixed_self))
/// ps_after_attn = ps_in + attn_out
/// mixed_mlp = attn_res(ps_after_attn, blocks_mlp)
/// ffn_out = ffn(rmsnorm(mixed_mlp))
/// ps_out = ps_after_attn + ffn_out
/// ```
///
/// # Arguments
/// - `config` — model config
/// - `weights` — model weights
/// - `runtime` — runtime state (for MLA KV cache access)
/// - `saved_tokens` — saved activations from forward (L tokens)
/// - `d_logits` — per-token gradient w.r.t. logits `[L][vocab_size]`
/// - `grads` — gradient accumulator (must be zeroed)
pub fn kimi_k3_backward_sequence(
    config: &KimiK3ModelConfig,
    weights: &KimiK3ModelWeights,
    runtime: &KimiK3Runtime,
    saved_tokens: &[TokenSavedActivations],
    d_logits: &[Vec<f32>],
    grads: &mut KimiK3ModelGradients,
) {
    let d = config.hidden_size;
    let v = config.vocab_size;
    let l = saved_tokens.len();
    debug_assert_eq!(d_logits.len(), l);
    if l == 0 {
        return;
    }

    // ── Step 1: Per-token output backward (LM head + final norm + output attn-res) ──
    // Produces d_prefix_final[t] = dL/d(prefix_sum at model exit) for each token.
    let mut d_prefix: Vec<Vec<f32>> = (0..l).map(|_| vec![0.0f32; d]).collect();
    // Block-state gradient accumulators (per token). These track gradient w.r.t.
    // block entries pushed at boundary layers. Entries get popped in reverse
    // layer processing when the originating boundary layer is reached.
    let mut block_grads: Vec<Vec<Vec<f32>>> = Vec::with_capacity(l);

    for t in 0..l {
        let saved = &saved_tokens[t];

        // LM head backward: dL/d(final_hidden) = lm_head^T · dL/d(logits)
        let mut d_fh = vec![0.0f32; d];
        simd_transpose_matvec_into(&mut d_fh, &weights.lm_head_weight, &d_logits[t], v, d);
        simd_outer_product_acc(&mut grads.lm_head_weight, &d_logits[t], &saved.final_hidden, v, d);

        // Final RMSNorm backward → dL/d(pre_final_norm = post-output-attn-res hidden)
        let d_pre_fn = mla_rmsnorm_backward(
            &d_fh,
            &saved.pre_final_norm,
            &weights.final_norm_weight,
            saved.final_norm_inv_rms,
            &mut grads.final_norm_weight,
            config.rms_eps,
        );

        // Output attn-res backward
        let num_blocks = saved.block_state_final.len();
        let mut bg: Vec<Vec<f32>> = (0..num_blocks).map(|_| vec![0.0f32; d]).collect();

        if saved.has_output_attn_res {
            attn_res_backward(
                &config.attn_res_config,
                &weights.output_attn_res,
                &saved.block_state_final,
                &saved.prefix_sum_final,
                &d_pre_fn,
                &mut bg,
                &mut d_prefix[t],
                &mut grads.output_attn_res_norm,
                &mut grads.output_attn_res_proj,
            );
        } else {
            d_prefix[t].copy_from_slice(&d_pre_fn);
        }
        block_grads.push(bg);
    }

    // ── Step 2: Layer-by-layer backward (reverse) ──
    for layer_idx in (0..config.num_layers).rev() {
        let layer_cfg = config.layer_config(layer_idx);
        let layer_w = &weights.layers[layer_idx];
        let layer_grads = &mut grads.layers[layer_idx];
        let is_mla = config.is_mla_layer(layer_idx);
        let is_boundary = layer_idx.is_multiple_of(config.attn_res_config.block_size);

        // ── Step 2a: MLP/FFN block backward (per token) ──
        // Produces d_attn_out[t] = dL/d(attention_output) for each token.
        let mut d_attn_out: Vec<Vec<f32>> = (0..l).map(|_| vec![0.0f32; d]).collect();
        // Also save d_ps_after_attn for the self-attn residual backward.
        let mut d_ps_after_attn_all: Vec<Vec<f32>> = (0..l).map(|_| vec![0.0f32; d]).collect();

        for t in 0..l {
            let saved = &saved_tokens[t].layers[layer_idx];

            // FFN residual: ps_out = ps_after_attn + ffn_out
            // dL/d(ffn_out) = d_prefix[t]; dL/d(ps_after_attn)_from_ffn = d_prefix[t]
            let d_ffn_out = d_prefix[t].clone();

            // FFN backward → dL/d(normed_mlp) + FFN weight grads
            let d_normed_mlp = ffn_backward(&layer_cfg.ffn, &layer_w.ffn, saved, &d_ffn_out, layer_grads);

            // Post-attn RMSNorm backward → dL/d(mixed_mlp)
            let d_mixed_mlp = mla_rmsnorm_backward(
                &d_normed_mlp,
                &saved.mixed_mlp,
                &layer_w.post_attention_layernorm_weight,
                saved.mlp_inv_rms,
                &mut layer_grads.post_attention_layernorm_weight,
                config.rms_eps,
            );

            // MLP attn-res backward → distributes to ps_after_attn + block grads
            let num_mlp_blocks = saved.block_state_mlp.len();
            while block_grads[t].len() < num_mlp_blocks {
                block_grads[t].push(vec![0.0f32; d]);
            }
            let mut d_mlp_blocks: Vec<Vec<f32>> =
                (0..num_mlp_blocks).map(|_| vec![0.0f32; d]).collect();
            // d_prefix[t] is the FFN residual gradient; now add attn-res contribution
            let mut d_ps_after = d_prefix[t].clone(); // dL/d(ps_after_attn)_from_ffn

            attn_res_backward(
                &config.attn_res_config,
                &layer_w.mlp_attn_res,
                &saved.block_state_mlp,
                &saved.prefix_sum_after_attn,
                &d_mixed_mlp,
                &mut d_mlp_blocks,
                &mut d_ps_after,
                &mut layer_grads.mlp_attn_res_norm,
                &mut layer_grads.mlp_attn_res_proj,
            );

            // Accumulate block grads
            for i in 0..num_mlp_blocks {
                for j in 0..d {
                    block_grads[t][i][j] += d_mlp_blocks[i][j];
                }
            }

            // d_ps_after is now dL/d(ps_after_attn) = FFN residual + attn-res contribution
            d_ps_after_attn_all[t] = d_ps_after.clone();
            // Self-attn residual: dL/d(attn_out) = d_ps_after_attn
            d_attn_out[t] = d_ps_after;
        }

        // ── Step 2b: Attention backward (MLA cross-token or KDA BPTT) ──
        // Produces d_normed_self[t] = dL/d(normed_self) for each token.
        let mut d_normed_self: Vec<Vec<f32>> = (0..l).map(|_| vec![0.0f32; d]).collect();

        if is_mla {
            let KimiAttentionWeights::Mla(mla_w) = &layer_w.attention else {
                panic!("MLA layer but non-MLA weights");
            };
            let KimiAttentionState::Mla(cache) = &runtime.layers[layer_idx].attn_state else {
                panic!("MLA layer but non-MLA cache");
            };

            // Collect all MLA saved activations for this layer.
            // mla_backward_token needs &[MlaSavedActivations] (owned refs).
            let all_saved: Vec<MlaSavedActivations> = (0..l)
                .map(|t| saved_tokens[t].layers[layer_idx].mla_saved.clone().unwrap())
                .collect();

            // all_dh[t] accumulates dL/d(input hidden to MLA) = dL/d(normed_self).
            let mut all_dh: Vec<Vec<f32>> = (0..l).map(|_| vec![0.0f32; d]).collect();
            let mla_grads = layer_grads.mla_grads.as_mut().unwrap();
            // Create a fresh RopeFreqs for the backward (MLA backward mutates it
            // for RoPE inversion via apply(negate=true)).
            let mut rf = RopeFreqs::new_with_theta(
                config.mla_config.qk_rope_head_dim,
                config.mla_config.rope_theta,
            );

            for t in 0..l {
                mla_backward_token(
                    &config.mla_config,
                    mla_w,
                    cache,
                    &all_saved[t],
                    &all_saved,
                    &mut rf,
                    &d_attn_out[t],
                    &mut all_dh,
                    mla_grads,
                );
            }
            d_normed_self = all_dh;
        } else {
            // KDA BPTT across the sequence
            let KimiAttentionWeights::Kda(kda_w) = &layer_w.attention else {
                panic!("KDA layer but non-KDA weights");
            };
            let all_saved_kda: Vec<KdaSavedActivations> = (0..l)
                .map(|t| saved_tokens[t].layers[layer_idx].kda_saved.clone().unwrap())
                .collect();
            let kda_grads = layer_grads.kda_grads.as_mut().unwrap();

            kda_backward_sequence(
                &config.kda_config,
                kda_w,
                &all_saved_kda,
                &d_attn_out,
                &mut d_normed_self,
                kda_grads,
            );
        }

        // ── Step 2c: Self-attn block backward (per token) → d_prefix[t] for previous layer ──
        for t in 0..l {
            let saved = &saved_tokens[t].layers[layer_idx];

            // Input RMSNorm backward → dL/d(mixed_self)
            let d_mixed_self = mla_rmsnorm_backward(
                &d_normed_self[t],
                &saved.mixed_self,
                &layer_w.input_layernorm_weight,
                saved.self_inv_rms,
                &mut layer_grads.input_layernorm_weight,
                config.rms_eps,
            );

            // Pop block entry if boundary (gradient for the pushed ps_in).
            // The push happens AFTER self-attn-res but BEFORE attention.
            // So we pop AFTER computing d_mixed_self but BEFORE self-attn-res backward.
            let mut d_ps_in = if is_boundary {
                block_grads[t].pop().unwrap_or_else(|| vec![0.0f32; d])
            } else {
                vec![0.0f32; d]
            };

            // Self-attn residual: for non-boundary, ps_after_attn = ps_in + attn_out
            // → dL/d(ps_in)_from_residual = d_ps_after_attn
            // For boundary, ps_in was reset to 0 → no residual gradient.
            if !is_boundary {
                for i in 0..d {
                    d_ps_in[i] += d_ps_after_attn_all[t][i];
                }
            }

            // Self attn-res backward (if applied) or copy gradient.
            if saved.has_self_attn_res {
                let num_self_blocks = saved.block_state_self.len();
                // block_grads[t] should have exactly num_self_blocks entries now
                // (after the boundary pop above).
                let mut d_self_blocks: Vec<Vec<f32>> =
                    (0..num_self_blocks).map(|_| vec![0.0f32; d]).collect();
                let mut d_ps_from_attnres = vec![0.0f32; d];

                attn_res_backward(
                    &config.attn_res_config,
                    &layer_w.self_attn_res,
                    &saved.block_state_self,
                    &saved.prefix_sum_in,
                    &d_mixed_self,
                    &mut d_self_blocks,
                    &mut d_ps_from_attnres,
                    &mut layer_grads.self_attn_res_norm,
                    &mut layer_grads.self_attn_res_proj,
                );

                // Accumulate block grads
                for i in 0..num_self_blocks {
                    for j in 0..d {
                        block_grads[t][i][j] += d_self_blocks[i][j];
                    }
                }
                // Add attn-res gradient for ps_in
                for i in 0..d {
                    d_ps_in[i] += d_ps_from_attnres[i];
                }
            } else {
                // mixed_self = ps_in (copy)
                for i in 0..d {
                    d_ps_in[i] += d_mixed_self[i];
                }
            }

            // d_ps_in becomes d_prefix for the previous layer
            d_prefix[t] = d_ps_in;
        }
    }

    // ── Step 3: Embedding backward ──
    // d_prefix[t] = dL/d(embedding for token t)
    for t in 0..l {
        let token_id = saved_tokens[t].token_id as usize;
        let base = token_id * d;
        for (j, slot) in (0..d).zip(&mut grads.embed_weight[base..base + d]) {
            *slot += d_prefix[t][j];
        }
    }
}

/// FFN backward (MoE or Dense). Returns dL/d(normed_mlp) and accumulates weight grads.
fn ffn_backward(
    ffn_config: &KimiFfnConfig,
    ffn_weights: &KimiFfnWeights,
    saved: &LayerSavedActivations,
    d_output: &[f32],
    layer_grads: &mut LayerGradients,
) -> Vec<f32> {
    match (ffn_config, ffn_weights) {
        (KimiFfnConfig::Dense { situ_beta, situ_linear_beta, .. }, KimiFfnWeights::Dense(expert)) => {
            let dense_saved = saved.dense_saved.as_ref().unwrap();
            let dense_grads = layer_grads.dense_grads.as_mut().unwrap();
            dense_situ_ffn_backward(expert, dense_saved, d_output, dense_grads, *situ_beta, *situ_linear_beta)
        }
        (KimiFfnConfig::Moe(moe_cfg), KimiFfnWeights::Moe(moe_w)) => {
            let moe_saved = saved.moe_saved.as_ref().unwrap();
            let moe_grads = layer_grads.moe_grads.as_mut().unwrap();
            let mut dh = vec![0.0f32; d_output.len()];
            moe_backward_token(moe_cfg, moe_w, moe_saved, d_output, &mut dh, moe_grads);
            dh
        }
        _ => panic!("FFN config/weights mismatch"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Attn-res backward
// ═══════════════════════════════════════════════════════════════════════════

/// Backward through `apply_attn_res`.
///
/// Forward:
/// ```text
/// score_weight = norm_weight ⊙ proj_weight
/// for each entry v[i] (block residuals + prefix_sum):
///     score[i] = inv_rms(v[i]) * dot(v[i], score_weight)
/// probs = softmax(scores)
/// out = Σ_i probs[i] * v[i]
/// ```
///
/// # Arguments
/// - `d_out` — gradient w.r.t. the attn-res output `[d]`
/// - `block_values` — the block residual entries `[num_blocks][d]`
/// - `prefix_sum` — the prefix_sum entry `[d]`
/// - `d_block_values` — accumulator for block residual gradients (mutated)
/// - `d_prefix_sum` — accumulator for prefix_sum gradient (mutated, += )
/// - `d_norm_weight`, `d_proj_weight` — weight gradient accumulators
#[allow(clippy::too_many_arguments)]
fn attn_res_backward(
    config: &AttnResConfig,
    weights: &AttnResWeights,
    block_values: &[Vec<f32>],
    prefix_sum: &[f32],
    d_out: &[f32],
    d_block_values: &mut [Vec<f32>],
    d_prefix_sum: &mut [f32],
    d_norm_weight: &mut [f32],
    d_proj_weight: &mut [f32],
) {
    let d = config.d();
    let eps = config.rms_eps;
    let num_entries = block_values.len() + 1; // blocks + prefix_sum

    // Reconstruct score_weight = norm_weight ⊙ proj_weight
    let score_weight: Vec<f32> = (0..d)
        .map(|i| weights.norm_weight[i] * weights.proj_weight[i])
        .collect();

    // Reconstruct scores + inv_rms for each entry
    let mut scores = vec![0.0f32; num_entries];
    let mut inv_rms_values = vec![0.0f32; num_entries];

    // Block entries
    for (i, residual) in block_values.iter().enumerate() {
        let sum_sq = simd_sum_sq(residual, d);
        let inv_rms = 1.0 / ((sum_sq / d as f32 + eps).sqrt());
        inv_rms_values[i] = inv_rms;
        let raw_dot = simd_dot_f32(residual, &score_weight, d);
        scores[i] = inv_rms * raw_dot;
    }
    // Prefix sum entry
    {
        let sum_sq = simd_sum_sq(prefix_sum, d);
        let inv_rms = 1.0 / ((sum_sq / d as f32 + eps).sqrt());
        inv_rms_values[num_entries - 1] = inv_rms;
        let raw_dot = simd_dot_f32(prefix_sum, &score_weight, d);
        scores[num_entries - 1] = inv_rms * raw_dot;
    }

    // Softmax reconstruction (from scores)
    let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut exp_scores = vec![0.0f32; num_entries];
    let mut sum_exp = 0.0f32;
    for i in 0..num_entries {
        exp_scores[i] = (scores[i] - max_score).exp();
        sum_exp += exp_scores[i];
    }
    let probs: Vec<f32> = exp_scores.iter().map(|&e| e / sum_exp).collect();

    // ── Backward through weighted sum: out = Σ_i probs[i] * v[i] ──
    // dL/dv[i] += probs[i] * dL/dout
    // dL/dprob[i] = dot(v[i], dL/dout)
    let mut d_probs = vec![0.0f32; num_entries];
    for (i, residual) in block_values.iter().enumerate() {
        for j in 0..d {
            d_block_values[i][j] += probs[i] * d_out[j];
        }
        d_probs[i] = simd_dot_f32(residual, d_out, d);
    }
    // Prefix sum entry
    {
        for j in 0..d {
            d_prefix_sum[j] += probs[num_entries - 1] * d_out[j];
        }
        d_probs[num_entries - 1] = simd_dot_f32(prefix_sum, d_out, d);
    }

    // ── Softmax backward: dL/dscore[i] = prob[i] * (d_prob[i] - Σ_j prob[j]*d_prob[j]) ──
    let dot_prob_dprob: f32 = probs.iter().zip(d_probs.iter()).map(|(&p, &dp)| p * dp).sum();
    let d_scores: Vec<f32> = probs
        .iter()
        .zip(d_probs.iter())
        .map(|(&p, &dp)| p * (dp - dot_prob_dprob))
        .collect();

    // ── Score backward: score[i] = inv_rms(v[i]) * dot(v[i], score_weight) ──
    // Let r_i = inv_rms(v[i]), s_i = dot(v[i], score_weight)
    // score[i] = r_i * s_i
    // dL/d(r_i) = dL/dscore[i] * s_i
    // dL/d(s_i) = dL/dscore[i] * r_i
    // dL/d(score_weight) += Σ_i dL/d(s_i) * r_i * v[i]
    // dL/d(v[i]) += dL/d(s_i) * r_i * score_weight   (from dot product)
    //            += dL/d(r_i) * d(r_i)/d(v[i])       (from inv_rms)

    let mut d_score_weight = vec![0.0f32; d];
    for i in 0..num_entries {
        let r_i = inv_rms_values[i];
        let s_i = scores[i] / r_i; // = dot(v[i], score_weight)
        let d_score_i = d_scores[i];
        let d_r_i = d_score_i * s_i;
        let d_s_i = d_score_i * r_i;

        // dL/d(score_weight) += d_s_i * v[i] (from dot(v[i], score_weight))
        // The v[i] is block_values[i] or prefix_sum
        let v_i: &[f32] = if i < block_values.len() {
            &block_values[i]
        } else {
            prefix_sum
        };

        for j in 0..d {
            d_score_weight[j] += d_s_i * v_i[j];
        }

        // dL/d(v[i]) from the dot product: d_s_i * score_weight[j]
        // (s_i = dot(v[i], score_weight), so d(s_i)/d(v[i][j]) = score_weight[j])
        for j in 0..d {
            let dot_grad = d_s_i * score_weight[j];

            // dL/d(v[i]) from inv_rms: d_r_i * d(r_i)/d(v[i][j])
            // r_i = 1/sqrt(mean(v²) + eps) = (mean(v²) + eps)^(-1/2)
            // dr/dv[j] = -1/2 * (mean(v²)+eps)^(-3/2) * (2*v[j]/n)
            //          = -v[j] * r_i³ / n
            let inv_rms_grad = d_r_i * (-v_i[j] * r_i * r_i * r_i / d as f32);

            let total_v_grad = dot_grad + inv_rms_grad;
            if i < block_values.len() {
                d_block_values[i][j] += total_v_grad;
            } else {
                d_prefix_sum[j] += total_v_grad;
            }
        }
    }

    // ── score_weight = norm_weight ⊙ proj_weight ──
    // dL/d(norm_weight[j]) = dL/d(score_weight[j]) * proj_weight[j]
    // dL/d(proj_weight[j]) = dL/d(score_weight[j]) * norm_weight[j]
    for j in 0..d {
        d_norm_weight[j] += d_score_weight[j] * weights.proj_weight[j];
        d_proj_weight[j] += d_score_weight[j] * weights.norm_weight[j];
    }
}

// ─── Dense SiTU FFN backward ────────────────────────────────────────────────

/// Backward through dense SiTU FFN: `down_proj(SiTU(gate_proj(h), up_proj(h)))`.
///
/// Returns dL/d(h) and accumulates weight gradients.
fn dense_situ_ffn_backward(
    expert: &SwiGluExpertWeights,
    saved: &DenseFfnSavedActivations,
    d_output: &[f32],
    grads: &mut DenseFfnGradients,
    beta: f32,
    linear_beta: Option<f32>,
) -> Vec<f32> {
    let d_in = d_output.len();
    let d_ffn = expert.gate_proj.len() / d_in;

    // out = down_proj · act  →  dL/d(act) = down_proj^T · dL/d(out)
    let mut d_act = vec![0.0f32; d_ffn];
    simd_transpose_matvec_into(&mut d_act, &expert.down_proj, d_output, d_in, d_ffn);
    // dL/d(down_proj) += outer(d_output, act)
    simd_outer_product_acc(&mut grads.down_proj, d_output, &saved.act_out, d_in, d_ffn);

    // SiTU backward: act = SiTU(gate, up)
    let mut d_gate = vec![0.0f32; d_ffn];
    let mut d_up = vec![0.0f32; d_ffn];
    situ_backward(
        &d_act,
        &saved.gate_inter,
        &saved.up_inter,
        &mut d_gate,
        &mut d_up,
        beta,
        linear_beta,
    );

    // gate = gate_proj · h  →  dL/d(h) += gate_proj^T · dL/d(gate)
    let mut d_h = vec![0.0f32; d_in];
    simd_transpose_matvec_into(&mut d_h, &expert.gate_proj, &d_gate, d_ffn, d_in);
    simd_outer_product_acc(&mut grads.gate_proj, &d_gate, &saved.h, d_ffn, d_in);

    // up = up_proj · h  →  dL/d(h) += up_proj^T · dL/d(up)
    let mut d_h_up = vec![0.0f32; d_in];
    simd_transpose_matvec_into(&mut d_h_up, &expert.up_proj, &d_up, d_ffn, d_in);
    simd_outer_product_acc(&mut grads.up_proj, &d_up, &saved.h, d_ffn, d_in);

    for i in 0..d_in {
        d_h[i] += d_h_up[i];
    }

    d_h
}

/// SiTU activation backward.
fn situ_backward(
    d_act: &[f32],
    gate_inter: &[f32],
    up_inter: &[f32],
    d_gate: &mut [f32],
    d_up: &mut [f32],
    beta: f32,
    linear_beta: Option<f32>,
) {
    let inv_beta = 1.0 / beta;
    let n = d_act.len();

    if let Some(lb) = linear_beta {
        let inv_lb = 1.0 / lb;
        for i in 0..n {
            let g = gate_inter[i];
            let u = up_inter[i];
            let da = d_act[i];
            let gs = 1.0 / (1.0 + (-g).exp()); // sigmoid(g)
            let gt = (g * inv_beta).tanh();    // tanh(g/beta)
            let ut = lb * (u * inv_lb).tanh(); // lb * tanh(u/lb)

            let d_act_dg = (1.0 - gt * gt) * gs * ut + beta * gt * gs * (1.0 - gs) * ut;
            let tanh_u = (u * inv_lb).tanh();
            let d_act_du = beta * gt * gs * (1.0 - tanh_u * tanh_u);

            d_gate[i] = da * d_act_dg;
            d_up[i] = da * d_act_du;
        }
    } else {
        for i in 0..n {
            let g = gate_inter[i];
            let u = up_inter[i];
            let da = d_act[i];
            let gs = 1.0 / (1.0 + (-g).exp());
            let gt = (g * inv_beta).tanh();

            let d_act_dg = (1.0 - gt * gt) * gs * u + beta * gt * gs * (1.0 - gs) * u;
            let d_act_du = beta * gt * gs;

            d_gate[i] = da * d_act_dg;
            d_up[i] = da * d_act_du;
        }
    }
}
