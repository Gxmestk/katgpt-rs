//! Safetensors loader for Kimi-K3-0.40B.
//!
//! This module loads `model.safetensors` and maps tensor names to the
//! weight structs (MLA, KDA, MoE, attn-res).
//!
//! # Tensor name mapping (VERIFIED against actual safetensors header)
//!
//! Verified 2026-08-01 via HTTP Range request on the real file header
//! (Research 331). All tensor names below match the actual file. See
//! `_ref_kimi_k3_tensors.json` in riir-ai `.research/` for the full dump.
//!
//! ## Top-level prefix: `language_model.`
//!
//! All language-model tensors are prefixed with `language_model.` (the model
//! is a `KimiK3ForConditionalGeneration` with a vision tower + mm_projector
//! that share the safetensors file). The vision tower (`vision_tower.*`) and
//! mm_projector (`mm_projector.*`) tensors are out of scope (text-only path).
//!
//! ## Layer topology (VERIFIED)
//!
//! | Layer | Attention | FFN   |
//! |-------|-----------|-------|
//! | 0     | KDA       | Dense |
//! | 1-2   | KDA       | MoE   |
//! | 3     | MLA       | MoE   |
//! | 4-6   | KDA       | MoE   |
//! | 7     | MLA       | MoE   |
//!
//! Config says `full_attn_layers: [4, 8]` (1-indexed) → MLA at 0-indexed 3,7.
//! KDA layers: `kda_layers: [1, 2, 3, 5, 6, 7]` (1-indexed) → 0-indexed
//! [0, 1, 2, 4, 5, 6]. Layer 0 is the only dense layer (`first_k_dense_replace: 1`).
//!
//! ## Model-level tensors
//!
//! | Tensor name | Field | Shape |
//! |-------------|-------|-------|
//! | `language_model.model.embed_tokens.weight` | `embed_weight` | `[163840, 1024]` |
//! | `language_model.model.norm.weight` | `final_norm_weight` | `[1024]` |
//! | `language_model.lm_head.weight` | `lm_head_weight` | `[163840, 1024]` |
//! | `language_model.model.output_attn_res_norm.weight` | `output_attn_res.norm` | `[1024]` |
//! | `language_model.model.output_attn_res_proj.weight` | `output_attn_res.proj` | `[1, 1024]` |
//!
//! ## Per-layer tensors (layer index `N`)
//!
//! Common (all layers):
//! - `language_model.model.layers.N.input_layernorm.weight` → `input_layernorm_weight`
//! - `language_model.model.layers.N.post_attention_layernorm.weight` → `post_attention_layernorm_weight`
//! - `language_model.model.layers.N.self_attention_res_norm.weight` → `self_attn_res.norm`
//! - `language_model.model.layers.N.self_attention_res_proj.weight` → `self_attn_res.proj`
//! - `language_model.model.layers.N.mlp_res_norm.weight` → `mlp_attn_res.norm`
//! - `language_model.model.layers.N.mlp_res_proj.weight` → `mlp_attn_res.proj`
//!
//! MLA layers (3, 7) — `language_model.model.layers.N.self_attn.*`:
//! - `.kv_a_proj_with_mqa.weight` → fused `w_dkv` + `w_kr` (split at `d_c=128`)
//! - `.kv_b_proj.weight` → fused `w_uk` + `w_uv` (split at `d_h·n_h=512`)
//! - `.q_a_proj.weight` → `w_dq`
//! - `.q_a_layernorm.weight` → `q_a_norm_weight`
//! - `.kv_a_layernorm.weight` → `kv_a_norm_weight`
//! - `.q_b_proj.weight` → fused `w_uq` + `w_qr` (split at `d_h·n_h=512`)
//! - `.o_proj.weight` → `w_o`  (shape `[1024, 512]`)
//! - `.g_proj.weight` → `w_g`  (shape `[512, 1024]` — gate BEFORE o_proj)
//!
//! KDA layers (0,1,2,4,5,6) — `language_model.model.layers.N.self_attn.*`:
//! - `.q_proj.weight` → `q_proj`
//! - `.k_proj.weight` → `k_proj`
//! - `.v_proj.weight` → `v_proj`
//! - `.q_conv1d.weight` → `q_conv_weight`
//! - `.k_conv1d.weight` → `k_conv_weight`
//! - `.v_conv1d.weight` → `v_conv_weight`
//! - `.A_log` → `a_log`
//! - `.f_a_proj.weight` → `f_a_proj`
//! - `.f_b_proj.weight` → `f_b_proj`
//! - `.dt_bias` → `dt_bias`
//! - `.b_proj.weight` → `beta_proj`
//! - `.g_proj.weight` → `g_proj` (output gate, shape `[256, 1024]`)
//! - `.o_norm.weight` → `o_norm_weight`
//! - `.o_proj.weight` → `o_proj` (shape `[1024, 256]`)
//!
//! Dense MLP (layer 0) — `language_model.model.layers.N.mlp.*`:
//! - `.gate_proj.weight` → `gate_proj`
//! - `.up_proj.weight` → `up_proj`
//! - `.down_proj.weight` → `down_proj`
//!
//! MoE (layers 1-7) — `language_model.model.layers.N.block_sparse_moe.*`:
//! - `.gate.weight` → `router_weight`
//! - `.gate.e_score_correction_bias` → `e_score_correction_bias`
//! - `.experts.N.w1.weight` → `experts[N].gate_proj`
//! - `.experts.N.w2.weight` → `experts[N].down_proj`
//! - `.experts.N.w3.weight` → `experts[N].up_proj`
//! - `.shared_experts.gate_proj.weight` → `shared_experts[0].gate_proj`
//! - `.shared_experts.up_proj.weight` → `shared_experts[0].up_proj`
//! - `.shared_experts.down_proj.weight` → `shared_experts[0].down_proj`
//! - `.routed_expert_down_proj.weight` → `routed_expert_down_proj`
//! - `.routed_expert_up_proj.weight` → `routed_expert_up_proj`
//! - `.routed_expert_norm.weight` → `routed_expert_norm_weight`

use katgpt_attn::gdn2::kda_forward::KdaWeights;
use katgpt_attn::mla::MlaWeights;
use katgpt_transformer::attn_res::AttnResWeights;
use katgpt_transformer::moe::{MoeWeights, SwiGluExpertWeights};

use super::decoder_layer::{
    KimiAttentionWeights, KimiDecoderLayerWeights, KimiFfnWeights,
};

/// Errors that can occur during model loading.
#[derive(Debug)]
pub enum LoadError {
    /// I/O error reading the safetensors file.
    Io(std::io::Error),
    /// Safetensors parsing error.
    Safetensors(String),
    /// A required tensor is missing from the file.
    MissingTensor(String),
    /// A tensor has an unexpected shape.
    ShapeMismatch { tensor: String, expected: String, actual: String },
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "loader I/O error: {e}"),
            Self::Safetensors(e) => write!(f, "safetensors error: {e}"),
            Self::MissingTensor(name) => write!(f, "missing tensor: {name}"),
            Self::ShapeMismatch { tensor, expected, actual } => {
                write!(f, "shape mismatch for '{tensor}': expected {expected}, got {actual}")
            }
        }
    }
}

impl std::error::Error for LoadError {}

/// Kimi-K3 model weights loaded from safetensors.
///
/// Contains:
/// - Token embedding weight
/// - Per-layer decoder weights (8 layers)
/// - Final norm weight
/// - LM head weight (untied — `tie_word_embeddings: false`)
/// - Output-level attn-res weights
#[derive(Clone)]
pub struct KimiK3ModelWeights {
    /// Token embedding `[vocab_size, hidden_size]`.
    pub embed_weight: Vec<f32>,
    /// Per-layer decoder weights (8 layers).
    pub layers: Vec<KimiDecoderLayerWeights>,
    /// Final RMSNorm gamma `[hidden_size]`.
    pub final_norm_weight: Vec<f32>,
    /// LM head `[vocab_size, hidden_size]` (untied from embedding).
    pub lm_head_weight: Vec<f32>,
    /// Output-level attention residual weights.
    pub output_attn_res: AttnResWeights,
}

/// Load a tensor from the safetensors view as f32.
///
/// Safetensors stores tensors in their original dtype (f32 for Kimi-K3-0.40B).
/// This function extracts the raw bytes and converts to `Vec<f32>`.
fn extract_f32_tensor(
    st: &safetensors::SafeTensors,
    name: &str,
) -> Result<Vec<f32>, LoadError> {
    let view = st
        .tensor(name)
        .map_err(|e| LoadError::Safetensors(format!("tensor '{name}': {e}")))?;

    // Kimi-K3-0.40B weights are stored as f32.
    let data = view.data();
    let f32_data: Vec<f32> = bytemuck::cast_slice::<u8, f32>(data).to_vec();

    Ok(f32_data)
}

/// Extract a tensor or return an error with a clear message.
macro_rules! get_tensor {
    ($st:expr, $name:expr) => {{
        match extract_f32_tensor(&$st, $name) {
            Ok(v) => v,
            Err(LoadError::Safetensors(_)) => {
                return Err(LoadError::MissingTensor($name.to_string()));
            }
            Err(e) => return Err(e),
        }
    }};
}

/// Load MLA weights for one layer.
///
/// Handles the fused projections:
/// - `kv_a_proj_with_mqa` → split into `w_dkv` (first `d_c` rows) + `w_kr` (remaining `d_r` rows)
/// - `kv_b_proj` → split into `w_uk` (first `d_h·n_h` rows) + `w_uv` (remaining `v_h·n_h` rows)
/// - `q_b_proj` → split into `w_uq` (first `d_h·n_h` rows) + `w_qr` (remaining `d_r·n_h` rows)
#[allow(clippy::too_many_arguments)]
fn load_mla_layer(
    st: &safetensors::SafeTensors,
    layer_idx: usize,
    d: usize,
    d_c: usize,
    d_qc: usize,
    d_h: usize,
    d_r: usize,
    v_h: usize,
    n_h: usize,
) -> Result<MlaWeights, LoadError> {
    let prefix = format!("language_model.model.layers.{layer_idx}.self_attn");

    // Fused kv_a_proj_with_mqa: [d_c + d_r, d] → split into w_dkv [d_c, d] + w_kr [d_r, d]
    // Verified shape: [160, 1024] = [(128+32), 1024] ✓
    let kv_a = get_tensor!(st, &format!("{prefix}.kv_a_proj_with_mqa.weight"));
    let kv_a_rows = d_c + d_r;
    let row_len = d;
    let w_dkv: Vec<f32> = kv_a[..d_c * row_len].to_vec();
    let w_kr: Vec<f32> = kv_a[d_c * row_len..kv_a_rows * row_len].to_vec();

    // Fused kv_b_proj: [n_h*(d_h + v_h), d_c] → de-interleave into w_uk + w_uv.
    //
    // PyTorch nn.Linear(d_c, n_h*(d_h+v_h)) stores the weight as
    // [n_h*(d_h+v_h), d_c]. The output features are laid out per-head:
    //   row 0..d_h-1:           head 0 content key rows
    //   row d_h..d_h+v_h-1:     head 0 value rows
    //   row d_h+v_h..2d_h+v_h-1: head 1 content key rows
    //   ...
    //
    // The Rust MLA forward expects w_uk [d_h*n_h, d_c] and w_uv [v_h*n_h, d_c]
    // as BLOCK layouts (all keys first, all values second). So we must
    // de-interleave the per-head [key, value] blocks.
    let kv_b = get_tensor!(st, &format!("{prefix}.kv_b_proj.weight"));
    let kv_b_row_len = d_c; // in_features
    let kv_b_per_head_rows = d_h + v_h;
    debug_assert_eq!(kv_b.len(), n_h * kv_b_per_head_rows * kv_b_row_len, "kv_b_proj size mismatch");
    let mut w_uk = vec![0.0f32; d_h * n_h * d_c];
    let mut w_uv = vec![0.0f32; v_h * n_h * d_c];
    for head in 0..n_h {
        let src_k_start = head * kv_b_per_head_rows * kv_b_row_len;
        let src_v_start = src_k_start + d_h * kv_b_row_len;
        let dst_k_start = head * d_h * d_c;
        let dst_v_start = head * v_h * d_c;
        w_uk[dst_k_start..dst_k_start + d_h * d_c]
            .copy_from_slice(&kv_b[src_k_start..src_k_start + d_h * d_c]);
        w_uv[dst_v_start..dst_v_start + v_h * d_c]
            .copy_from_slice(&kv_b[src_v_start..src_v_start + v_h * d_c]);
    }

    // q_a_proj: [d_qc, d] — verified shape [256, 1024] ✓
    let w_dq = get_tensor!(st, &format!("{prefix}.q_a_proj.weight"));

    // q_a_layernorm: [d_qc] — verified shape [256] ✓
    let q_a_norm_weight = get_tensor!(st, &format!("{prefix}.q_a_layernorm.weight"));

    // kv_a_layernorm: [d_c] — verified shape [128] ✓
    let kv_a_norm_weight = get_tensor!(st, &format!("{prefix}.kv_a_layernorm.weight"));

    // Fused q_b_proj: [n_h*(d_h + d_r), d_qc] → de-interleave into w_uq + w_qr.
    //
    // Same per-head interleaving as kv_b_proj:
    //   row 0..d_h-1:        head 0 content query rows
    //   row d_h..d_h+d_r-1:  head 0 rope query rows
    //   ...
    let q_b = get_tensor!(st, &format!("{prefix}.q_b_proj.weight"));
    let q_b_row_len = d_qc; // in_features
    let q_b_per_head_rows = d_h + d_r;
    debug_assert_eq!(q_b.len(), n_h * q_b_per_head_rows * q_b_row_len, "q_b_proj size mismatch");
    let mut w_uq = vec![0.0f32; d_h * n_h * d_qc];
    let mut w_qr = vec![0.0f32; d_r * n_h * d_qc];
    for head in 0..n_h {
        let src_q_start = head * q_b_per_head_rows * q_b_row_len;
        let src_r_start = src_q_start + d_h * q_b_row_len;
        let dst_q_start = head * d_h * d_qc;
        let dst_r_start = head * d_r * d_qc;
        w_uq[dst_q_start..dst_q_start + d_h * d_qc]
            .copy_from_slice(&q_b[src_q_start..src_q_start + d_h * d_qc]);
        w_qr[dst_r_start..dst_r_start + d_r * d_qc]
            .copy_from_slice(&q_b[src_r_start..src_r_start + d_r * d_qc]);
    }

    // o_proj: [d, v_h*n_h] — verified shape [1024, 512] ✓
    let w_o = get_tensor!(st, &format!("{prefix}.o_proj.weight"));

    // g_proj (output gate): [v_h*n_h, d] — verified shape [512, 1024] ✓
    // Applied BEFORE o_proj in the actual model (gate on attn_output, not on final output).
    let w_g = get_tensor!(st, &format!("{prefix}.g_proj.weight"));

    Ok(MlaWeights {
        w_dkv,
        w_dq,
        w_uq,
        w_qr,
        w_uk,
        w_uv,
        w_kr,
        w_o,
        q_a_norm_weight,
        kv_a_norm_weight,
        w_g: Some(w_g),
    })
}

/// Load KDA weights for one layer.
fn load_kda_layer(
    st: &safetensors::SafeTensors,
    layer_idx: usize,
) -> Result<KdaWeights, LoadError> {
    let prefix = format!("language_model.model.layers.{layer_idx}.self_attn");

    // ShortConv1D weights need per-channel tap reversal.
    //
    // safetensors stores them as [D, 1, W] (PyTorch Conv1d format) where
    // index W-1 is the NEWEST (current) sample tap and index 0 is the OLDEST.
    // Rust's ShortConv1D expects weight[c*ks + 0] = newest, weight[c*ks + ks-1] = oldest.
    //
    // Without reversal, taps 0 and W-1 (and 1 and W-2, etc.) are swapped,
    // causing the KDA attention output to diverge from the reference model.
    let conv_kernel_size = 4usize; // from config: short_conv_kernel_size = 4

    Ok(KdaWeights {
        q_proj: get_tensor!(st, &format!("{prefix}.q_proj.weight")),
        k_proj: get_tensor!(st, &format!("{prefix}.k_proj.weight")),
        v_proj: get_tensor!(st, &format!("{prefix}.v_proj.weight")),
        q_conv_weight: reverse_conv_taps(
            get_tensor!(st, &format!("{prefix}.q_conv1d.weight")),
            conv_kernel_size,
        ),
        k_conv_weight: reverse_conv_taps(
            get_tensor!(st, &format!("{prefix}.k_conv1d.weight")),
            conv_kernel_size,
        ),
        v_conv_weight: reverse_conv_taps(
            get_tensor!(st, &format!("{prefix}.v_conv1d.weight")),
            conv_kernel_size,
        ),
        a_log: get_tensor!(st, &format!("{prefix}.A_log")),
        f_a_proj: get_tensor!(st, &format!("{prefix}.f_a_proj.weight")),
        f_b_proj: get_tensor!(st, &format!("{prefix}.f_b_proj.weight")),
        dt_bias: get_tensor!(st, &format!("{prefix}.dt_bias")),
        beta_proj: get_tensor!(st, &format!("{prefix}.b_proj.weight")),
        g_proj: get_tensor!(st, &format!("{prefix}.g_proj.weight")),
        o_norm_weight: get_tensor!(st, &format!("{prefix}.o_norm.weight")),
        o_proj: get_tensor!(st, &format!("{prefix}.o_proj.weight")),
    })
}

/// Reverse per-channel taps in a flattened Conv1d weight tensor.
///
/// Input: `[n_channels * kernel_size]` stored as [D, 1, W] (oldest-to-newest per channel).
/// Output: `[n_channels * kernel_size]` stored as newest-to-oldest per channel.
///
/// This converts PyTorch Conv1d weight layout to Rust ShortConv1D's expected layout.
fn reverse_conv_taps(mut weights: Vec<f32>, kernel_size: usize) -> Vec<f32> {
    let n_channels = weights.len() / kernel_size;
    for c in 0..n_channels {
        let start = c * kernel_size;
        weights[start..start + kernel_size].reverse();
    }
    weights
}

/// Load dense MLP weights for layer 0.
fn load_dense_mlp(
    st: &safetensors::SafeTensors,
    layer_idx: usize,
) -> Result<SwiGluExpertWeights, LoadError> {
    let prefix = format!("language_model.model.layers.{layer_idx}.mlp");

    Ok(SwiGluExpertWeights {
        gate_proj: get_tensor!(st, &format!("{prefix}.gate_proj.weight")),
        down_proj: get_tensor!(st, &format!("{prefix}.down_proj.weight")),
        up_proj: get_tensor!(st, &format!("{prefix}.up_proj.weight")),
    })
}

/// Load MoE weights for layers 1-7.
fn load_moe_layer(
    st: &safetensors::SafeTensors,
    layer_idx: usize,
    num_experts: usize,
    num_shared_experts: usize,
) -> Result<MoeWeights, LoadError> {
    let prefix = format!("language_model.model.layers.{layer_idx}.block_sparse_moe");

    // Router centroid: [N_r, d] — `.gate.weight`
    let router_weight = get_tensor!(st, &format!("{prefix}.gate.weight"));
    // noaux_tc bias: [N_r] — `.gate.e_score_correction_bias`
    let e_score_correction_bias = get_tensor!(st, &format!("{prefix}.gate.e_score_correction_bias"));

    // Routed experts — `.experts.N.w1/w2/w3`
    let mut experts = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        let eprefix = format!("{prefix}.experts.{e}");
        experts.push(SwiGluExpertWeights {
            gate_proj: get_tensor!(st, &format!("{eprefix}.w1.weight")),
            down_proj: get_tensor!(st, &format!("{eprefix}.w2.weight")),
            up_proj: get_tensor!(st, &format!("{eprefix}.w3.weight")),
        });
    }

    // Shared experts — `.shared_experts.gate_proj/up_proj/down_proj`
    let mut shared_experts = Vec::with_capacity(num_shared_experts);
    for _s in 0..num_shared_experts {
        shared_experts.push(SwiGluExpertWeights {
            gate_proj: get_tensor!(st, &format!("{prefix}.shared_experts.gate_proj.weight")),
            down_proj: get_tensor!(st, &format!("{prefix}.shared_experts.down_proj.weight")),
            up_proj: get_tensor!(st, &format!("{prefix}.shared_experts.up_proj.weight")),
        });
    }

    // Latent MoE wrapper — `.routed_expert_*`
    let routed_expert_down_proj = Some(get_tensor!(st, &format!("{prefix}.routed_expert_down_proj.weight")));
    let routed_expert_up_proj = Some(get_tensor!(st, &format!("{prefix}.routed_expert_up_proj.weight")));
    let routed_expert_norm_weight = Some(get_tensor!(st, &format!("{prefix}.routed_expert_norm.weight")));

    Ok(MoeWeights {
        router_weight,
        e_score_correction_bias,
        experts,
        shared_experts,
        routed_expert_down_proj,
        routed_expert_up_proj,
        routed_expert_norm_weight,
    })
}

/// Load one decoder layer.
///
/// Determines attention type (MLA vs KDA) and FFN type (Dense vs MoE)
/// from the `is_mla` flag + layer index. The caller computes `is_mla` from
/// the model config's `mla_layer_indices`.
#[allow(clippy::too_many_arguments)]
fn load_decoder_layer(
    st: &safetensors::SafeTensors,
    layer_idx: usize,
    is_mla: bool,
    d: usize,
    num_experts: usize,
    num_shared_experts: usize,
    // MLA dims
    d_c: usize,
    d_qc: usize,
    d_h: usize,
    d_r: usize,
    v_h: usize,
    n_h: usize,
) -> Result<KimiDecoderLayerWeights, LoadError> {
    // Topology: MLA layers use full attention; others use KDA (linear/delta).
    // Dense MLP at layer 0 only (first_k_dense_replace: 1).
    let is_dense = layer_idx == 0;

    let attention = if is_mla {
        KimiAttentionWeights::Mla(load_mla_layer(st, layer_idx, d, d_c, d_qc, d_h, d_r, v_h, n_h)?)
    } else {
        KimiAttentionWeights::Kda(load_kda_layer(st, layer_idx)?)
    };

    let ffn = if is_dense {
        KimiFfnWeights::Dense(load_dense_mlp(st, layer_idx)?)
    } else {
        KimiFfnWeights::Moe(load_moe_layer(st, layer_idx, num_experts, num_shared_experts)?)
    };

    // Common layer norm + attn-res weights
    let lpfx = format!("language_model.model.layers.{layer_idx}");
    let input_layernorm_weight =
        get_tensor!(st, &format!("{lpfx}.input_layernorm.weight"));
    let post_attention_layernorm_weight =
        get_tensor!(st, &format!("{lpfx}.post_attention_layernorm.weight"));

    let self_attn_res = AttnResWeights {
        norm_weight: get_tensor!(st, &format!("{lpfx}.self_attention_res_norm.weight")),
        proj_weight: get_tensor!(st, &format!("{lpfx}.self_attention_res_proj.weight")),
    };

    let mlp_attn_res = AttnResWeights {
        norm_weight: get_tensor!(st, &format!("{lpfx}.mlp_res_norm.weight")),
        proj_weight: get_tensor!(st, &format!("{lpfx}.mlp_res_proj.weight")),
    };

    Ok(KimiDecoderLayerWeights {
        input_layernorm_weight,
        post_attention_layernorm_weight,
        attention,
        ffn,
        self_attn_res,
        mlp_attn_res,
    })
}

/// Load the full Kimi-K3-0.40B model from a safetensors file.
///
/// # Zero-copy mmap (native)
///
/// On native targets, the file is memory-mapped via `memmap2` — the 1.5 GB
/// safetensors data section is read directly from the OS page cache, with
/// no intermediate `Vec<u8>` buffer. This eliminates the largest copy in
/// the original `read_to_end` path (255 ms at ~6 GB/s on Apple Silicon).
///
/// Per-tensor `Vec<f32>` materialization still happens (the forward path
/// takes `&[f32]` slices, and MLA de-interleaving needs writable storage).
/// That copy is necessary + bounded — total weight bytes are still 1.5 GB
/// of f32, just owned by the weight structs instead of by a transient
/// file-read buffer.
///
/// On `wasm32-unknown-unknown`, `memmap2` is unavailable → falls back to
/// `read_to_end`. (The wasm32 peer doesn't load real model weights today;
/// this path exists for completeness.)
///
/// # Arguments
/// - `path` — path to `model.safetensors`
///
/// # Returns
/// A `KimiK3ModelWeights` with all 8 decoder layers + embeddings + lm_head.
///
/// # Configuration
///
/// Uses the Kimi-K3-0.40B config values from config.json (Research 331):
/// - `hidden_size: 1024`, `num_hidden_layers: 8`, `vocab_size: 163840`
/// - MLA: `kv_lora_rank=128, q_lora_rank=256, qk_nope=64, qk_rope=32, v_head=64, n_heads=8`
/// - KDA: `head_dim=32, n_heads=8`
/// - MoE: `num_experts=8, num_shared_experts=1`
/// - Dense MLP layer: 0 only (`first_k_dense_replace: 1`)
/// - MLA layers: 3, 7 (config `full_attn_layers: [4,8]` = 1-indexed)
/// - KDA layers: 0, 1, 2, 4, 5, 6
pub fn load_kimi_k3(path: &str) -> Result<KimiK3ModelWeights, LoadError> {
    // ── Native: mmap the safetensors file (zero-copy file buffer) ──────────
    // The mmap stays alive for the duration of `load_kimi_k3_from_bytes`;
    // SafeTensors borrows from it via the `&[u8]` slice. After the function
    // returns, the weight structs own their materialized `Vec<f32>` storage
    // + the mmap is dropped. The OS reclaims the page cache lazily.
    #[cfg(not(target_arch = "wasm32"))]
    {
        use memmap2::MmapOptions;
        let file = std::fs::File::open(path).map_err(LoadError::Io)?;
        // Safety: model files are write-once artifacts. Concurrent writes
        // from another process are a deployment-error condition, not a
        // runtime hazard. We map read-only.
        let mmap = unsafe { MmapOptions::new().map(&file) }.map_err(LoadError::Io)?;
        load_kimi_k3_from_bytes(&mmap)
    }

    // ── wasm32: no mmap available, fall back to read_to_end ───────────────
    #[cfg(target_arch = "wasm32")]
    {
        let file = std::fs::File::open(path).map_err(LoadError::Io)?;
        let mut reader = std::io::BufReader::new(file);
        let mut buf = Vec::new();
        std::io::Read::read_to_end(&mut reader, &mut buf).map_err(LoadError::Io)?;
        load_kimi_k3_from_bytes(&buf)
    }
}

/// Parse + materialize Kimi-K3 weights from in-memory safetensors bytes.
///
/// This is the pure deserialization step, separated from the file-reading step
/// so the same code path serves both the mmap (native) + `read_to_end` (wasm32)
/// front-ends. Tests can construct a synthetic safetensors blob + call this
/// directly without touching the filesystem.
pub fn load_kimi_k3_from_bytes(bytes: &[u8]) -> Result<KimiK3ModelWeights, LoadError> {
    // Parse safetensors from the in-memory bytes (mmap'd on native).
    let st = safetensors::SafeTensors::deserialize(bytes)
        .map_err(|e| LoadError::Safetensors(e.to_string()))?;

    // Kimi-K3-0.40B config values
    let d = 1024;
    let num_layers = 8;
    let num_experts = 8;
    let num_shared_experts = 1;
    // MLA dims
    let d_c = 128;
    let d_qc = 256;
    let d_h = 64;
    let d_r = 32;
    let v_h = 64;
    let n_h = 8;

    // Model-level tensors — all prefixed with `language_model.` (the model is
    // KimiK3ForConditionalGeneration; vision tower + mm_projector share the file).
    let embed_weight = get_tensor!(st, "language_model.model.embed_tokens.weight");
    let final_norm_weight = get_tensor!(st, "language_model.model.norm.weight");
    // tie_word_embeddings: false → lm_head is separate
    let lm_head_weight = get_tensor!(st, "language_model.lm_head.weight");

    let output_attn_res = AttnResWeights {
        norm_weight: get_tensor!(st, "language_model.model.output_attn_res_norm.weight"),
        proj_weight: get_tensor!(st, "language_model.model.output_attn_res_proj.weight"),
    };

    // Per-layer weights
    // TODO(388): parameterize by config for 4B model loading (Phase B).
    // Currently hardcoded for the 0.40B fixture.
    let mla_layer_indices = [3usize, 7];
    let mut layers = Vec::with_capacity(num_layers);
    for layer_idx in 0..num_layers {
        layers.push(load_decoder_layer(
            &st,
            layer_idx,
            mla_layer_indices.contains(&layer_idx),
            d,
            num_experts,
            num_shared_experts,
            d_c,
            d_qc,
            d_h,
            d_r,
            v_h,
            n_h,
        )?);
    }

    Ok(KimiK3ModelWeights {
        embed_weight,
        layers,
        final_norm_weight,
        lm_head_weight,
        output_attn_res,
    })
}
