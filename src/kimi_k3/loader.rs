//! Safetensors loader for Kimi-K3-0.40B.
//!
//! This module loads `model.safetensors` and maps tensor names to the
//! corrected weight structs (MLA, KDA, MoE, attn-res) per Research 330.
//!
//! # Tensor name mapping (from HF naming convention)
//!
//! The safetensors file uses HuggingFace naming conventions. The exact tensor
//! names were not verified against the actual file header (Phase 6 concern).
//! The mapping below follows the DeepSeek-V2 + Kimi-K3 convention documented
//! in Research 327 §5 + Research 330 §2-5.
//!
//! ## Model-level tensors
//!
//! | Tensor name | Field | Shape |
//! |-------------|-------|-------|
//! | `model.embed_tokens.weight` | `embed_weight` | `[vocab_size, hidden_size]` |
//! | `model.norm.weight` | `final_norm_weight` | `[hidden_size]` |
//! | `lm_head.weight` | `lm_head_weight` | `[vocab_size, hidden_size]` |
//! | `model.output_attn_res_norm.weight` | `output_attn_res.norm` | `[hidden_size]` |
//! | `model.output_attn_res_proj.weight` | `output_attn_res.proj` | `[hidden_size]` |
//!
//! ## Per-layer tensors (layer index `N`)
//!
//! Common (all layers):
//! - `model.layers.N.input_layernorm.weight` → `input_layernorm_weight`
//! - `model.layers.N.post_attention_layernorm.weight` → `post_attention_layernorm_weight`
//! - `model.layers.N.self_attention_res_norm.weight` → `self_attn_res.norm`
//! - `model.layers.N.self_attention_res_proj.weight` → `self_attn_res.proj`
//! - `model.layers.N.mlp_res_norm.weight` → `mlp_attn_res.norm`
//! - `model.layers.N.mlp_res_proj.weight` → `mlp_attn_res.proj`
//!
//! MLA layers (0, 4) — `model.layers.N.attention.*`:
//! - `.kv_a_proj_with_mqa.weight` → fused `w_dkv` + `w_kr` (split at `d_c`)
//! - `.kv_b_proj.weight` → fused `w_uk` + `w_uv` (split at `d_h·n_h`)
//! - `.q_a_proj.weight` → `w_dq`
//! - `.q_a_layernorm.weight` → `q_a_norm_weight`
//! - `.kv_a_layernorm.weight` → `kv_a_norm_weight`
//! - `.q_b_proj.weight` → fused `w_uq` + `w_qr` (split at `d_h·n_h`)
//! - `.o_proj.weight` → `w_o`
//! - `.g_proj.weight` → `w_g` (output gate)
//!
//! KDA layers (1,2,3,5,6,7) — `model.layers.N.attention.*`:
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
//! - `.g_proj.weight` → `g_proj` (output gate)
//! - `.o_norm.weight` → `o_norm_weight`
//! - `.o_proj.weight` → `o_proj`
//!
//! Dense MLP (layer 0) — `model.layers.N.mlp.*`:
//! - `.w1.weight` → `gate_proj`
//! - `.w2.weight` → `down_proj`
//! - `.w3.weight` → `up_proj`
//!
//! MoE (layers 1-7) — `model.layers.N.mlp.*`:
//! - `.gate.weight` → `router_weight`
//! - `.e_score_correction_bias` → `e_score_correction_bias`
//! - `.experts.N.w1.weight` → `experts[N].gate_proj`
//! - `.experts.N.w2.weight` → `experts[N].down_proj`
//! - `.experts.N.w3.weight` → `experts[N].up_proj`
//! - `.shared_experts.w1.weight` → `shared_experts[0].gate_proj`
//! - `.shared_experts.w2.weight` → `shared_experts[0].down_proj`
//! - `.shared_experts.w3.weight` → `shared_experts[0].up_proj`
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
    _v_h: usize,
    n_h: usize,
) -> Result<MlaWeights, LoadError> {
    let prefix = format!("model.layers.{layer_idx}.attention");

    // Fused kv_a_proj_with_mqa: [d_c + d_r, d] → split into w_dkv [d_c, d] + w_kr [d_r, d]
    let kv_a = get_tensor!(st, &format!("{prefix}.kv_a_proj_with_mqa.weight"));
    let kv_a_rows = d_c + d_r;
    let row_len = d;
    let w_dkv: Vec<f32> = kv_a[..d_c * row_len].to_vec();
    let w_kr: Vec<f32> = kv_a[d_c * row_len..kv_a_rows * row_len].to_vec();

    // Fused kv_b_proj: [(d_h + v_h) * n_h... wait, actually kv_b_proj is [d_h*n_h + v_h*n_h, d_c]
    // Wait — let me reconsider. DeepSeek-V2's kv_b_proj produces both UK and UV:
    //   output shape = [(d_h*n_h) + (v_h*n_h), d_c] but it's actually uk then uv
    // No — the standard DeepSeek-V2 layout: kv_b_proj has output dim = n_h * (d_h + v_h)?
    // Actually: UK is [d_h*n_h, d_c] and UV is [v_h*n_h, d_c], and kv_b_proj is
    // [d_h*n_h + v_h*n_h, d_c] concatenated row-wise.
    // But d_h == v_h for Kimi-K3 (both 64), so total = 2 * 64 * 8 = 1024 rows.
    let kv_b = get_tensor!(st, &format!("{prefix}.kv_b_proj.weight"));
    let uk_rows = d_h * n_h;
    let w_uk: Vec<f32> = kv_b[..uk_rows * d_c].to_vec();
    let w_uv: Vec<f32> = kv_b[uk_rows * d_c..].to_vec();

    // q_a_proj: [d_qc, d]
    let w_dq = get_tensor!(st, &format!("{prefix}.q_a_proj.weight"));

    // q_a_layernorm: [d_qc]
    let q_a_norm_weight = get_tensor!(st, &format!("{prefix}.q_a_layernorm.weight"));

    // kv_a_layernorm: [d_c]
    let kv_a_norm_weight = get_tensor!(st, &format!("{prefix}.kv_a_layernorm.weight"));

    // Fused q_b_proj: [(d_h + d_r)*n_h, d_qc] → split into w_uq [d_h*n_h, d_qc] + w_qr [d_r*n_h, d_qc]
    let q_b = get_tensor!(st, &format!("{prefix}.q_b_proj.weight"));
    let uq_rows = d_h * n_h;
    let w_uq: Vec<f32> = q_b[..uq_rows * d_qc].to_vec();
    let w_qr: Vec<f32> = q_b[uq_rows * d_qc..].to_vec();

    // o_proj: [d, v_h*n_h]
    let w_o = get_tensor!(st, &format!("{prefix}.o_proj.weight"));

    // g_proj (output gate): [d, d]
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
    let prefix = format!("model.layers.{layer_idx}.attention");

    Ok(KdaWeights {
        q_proj: get_tensor!(st, &format!("{prefix}.q_proj.weight")),
        k_proj: get_tensor!(st, &format!("{prefix}.k_proj.weight")),
        v_proj: get_tensor!(st, &format!("{prefix}.v_proj.weight")),
        q_conv_weight: get_tensor!(st, &format!("{prefix}.q_conv1d.weight")),
        k_conv_weight: get_tensor!(st, &format!("{prefix}.k_conv1d.weight")),
        v_conv_weight: get_tensor!(st, &format!("{prefix}.v_conv1d.weight")),
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

/// Load dense MLP weights for layer 0.
fn load_dense_mlp(
    st: &safetensors::SafeTensors,
    layer_idx: usize,
) -> Result<SwiGluExpertWeights, LoadError> {
    let prefix = format!("model.layers.{layer_idx}.mlp");

    Ok(SwiGluExpertWeights {
        gate_proj: get_tensor!(st, &format!("{prefix}.w1.weight")),
        down_proj: get_tensor!(st, &format!("{prefix}.w2.weight")),
        up_proj: get_tensor!(st, &format!("{prefix}.w3.weight")),
    })
}

/// Load MoE weights for layers 1-7.
fn load_moe_layer(
    st: &safetensors::SafeTensors,
    layer_idx: usize,
    num_experts: usize,
    num_shared_experts: usize,
) -> Result<MoeWeights, LoadError> {
    let prefix = format!("model.layers.{layer_idx}.mlp");

    // Router centroid: [N_r, d]
    let router_weight = get_tensor!(st, &format!("{prefix}.gate.weight"));
    // noaux_tc bias: [N_r]
    let e_score_correction_bias = get_tensor!(st, &format!("{prefix}.e_score_correction_bias"));

    // Routed experts
    let mut experts = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        let eprefix = format!("{prefix}.experts.{e}");
        experts.push(SwiGluExpertWeights {
            gate_proj: get_tensor!(st, &format!("{eprefix}.w1.weight")),
            down_proj: get_tensor!(st, &format!("{eprefix}.w2.weight")),
            up_proj: get_tensor!(st, &format!("{eprefix}.w3.weight")),
        });
    }

    // Shared experts
    let mut shared_experts = Vec::with_capacity(num_shared_experts);
    for s in 0..num_shared_experts {
        let sprefix = format!("{prefix}.shared_experts.{s}");
        shared_experts.push(SwiGluExpertWeights {
            gate_proj: get_tensor!(st, &format!("{sprefix}.w1.weight")),
            down_proj: get_tensor!(st, &format!("{sprefix}.w2.weight")),
            up_proj: get_tensor!(st, &format!("{sprefix}.w3.weight")),
        });
    }

    // Latent MoE wrapper (optional but present for Kimi-K3)
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
/// Determines attention type (MLA for layers 0,4; KDA for 1,2,3,5,6,7) and
/// FFN type (Dense for layer 0; MoE for 1-7) from the layer index.
#[allow(clippy::too_many_arguments)]
fn load_decoder_layer(
    st: &safetensors::SafeTensors,
    layer_idx: usize,
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
    let is_mla = layer_idx == 0 || layer_idx == 4;
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
    let input_layernorm_weight =
        get_tensor!(st, &format!("model.layers.{layer_idx}.input_layernorm.weight"));
    let post_attention_layernorm_weight =
        get_tensor!(st, &format!("model.layers.{layer_idx}.post_attention_layernorm.weight"));

    let self_attn_res = AttnResWeights {
        norm_weight: get_tensor!(st, &format!("model.layers.{layer_idx}.self_attention_res_norm.weight")),
        proj_weight: get_tensor!(st, &format!("model.layers.{layer_idx}.self_attention_res_proj.weight")),
    };

    let mlp_attn_res = AttnResWeights {
        norm_weight: get_tensor!(st, &format!("model.layers.{layer_idx}.mlp_res_norm.weight")),
        proj_weight: get_tensor!(st, &format!("model.layers.{layer_idx}.mlp_res_proj.weight")),
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
/// # Arguments
/// - `path` — path to `model.safetensors`
///
/// # Returns
/// A `KimiK3ModelWeights` with all 8 decoder layers + embeddings + lm_head.
///
/// # Configuration
///
/// Uses the Kimi-K3-0.40B config values from Research 330 §7:
/// - `hidden_size: 1024`, `num_hidden_layers: 8`, `vocab_size: 163840`
/// - MLA: `kv_lora_rank=128, q_lora_rank=256, qk_nope=64, qk_rope=32, v_head=64, n_heads=8`
/// - KDA: `head_dim=32, n_heads=8`
/// - MoE: `num_experts=8, num_shared_experts=1`
/// - Layer 0: dense MLP; layers 1-7: MoE
/// - MLA layers: 0, 4; KDA layers: 1,2,3,5,6,7
pub fn load_kimi_k3(path: &str) -> Result<KimiK3ModelWeights, LoadError> {
    // Read the safetensors file
    let file = std::fs::File::open(path).map_err(LoadError::Io)?;
    let mut reader = std::io::BufReader::new(file);
    let mut buffer = Vec::new();
    std::io::Read::read_to_end(&mut reader, &mut buffer).map_err(LoadError::Io)?;

    // Parse safetensors
    let st = safetensors::SafeTensors::deserialize(&buffer)
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

    // Model-level tensors
    let embed_weight = get_tensor!(st, "model.embed_tokens.weight");
    let final_norm_weight = get_tensor!(st, "model.norm.weight");
    // tie_word_embeddings: false → lm_head is separate
    let lm_head_weight = get_tensor!(st, "lm_head.weight");

    let output_attn_res = AttnResWeights {
        norm_weight: get_tensor!(st, "model.output_attn_res_norm.weight"),
        proj_weight: get_tensor!(st, "model.output_attn_res_proj.weight"),
    };

    // Per-layer weights
    let mut layers = Vec::with_capacity(num_layers);
    for layer_idx in 0..num_layers {
        layers.push(load_decoder_layer(
            &st,
            layer_idx,
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
