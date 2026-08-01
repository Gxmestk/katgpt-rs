//! Kimi-K3 native support — decoder layer composition + model loader.
//!
//! This module is the Phase 5+ composition layer for Kimi-K3-0.40B
//! (Proposal 032). It lives in the katgpt-rs root crate because the decoder
//! layer needs BOTH:
//! - MLA/KDA from `katgpt-attn`
//! - MoE/attn-res from `katgpt-transformer`
//!
//! The root crate depends on both leaf crates, making it the natural
//! composition point.
//!
//! # Feature gates
//!
//! - `kimi_k3` — decoder layer composition (this module). Pulls in all four
//!   substrate primitives: `mla_attention`, `kda_linear`, `transformer_moe`,
//!   `transformer_attn_res`.
//! - `kimi_k3_loader` — adds the safetensors loader + tiktoken tokenizer
//!   (heavy deps: `safetensors`, `base64`).
//!
//! Both are opt-in (default-off). Promotion to default requires the Phase 6
//! real-model GOAT gate (logits match PyTorch reference).

pub mod decoder_layer;

#[cfg(feature = "kimi_k3_loader")]
pub mod model;

pub use decoder_layer::{
    KimiAttentionConfig, KimiAttentionState, KimiAttentionScratch, KimiAttentionWeights,
    KimiDecoderLayerConfig, KimiDecoderLayerWeights, KimiFfnConfig, KimiFfnScratch,
    KimiFfnWeights, kimi_decoder_layer_forward,
};

#[cfg(feature = "kimi_k3_loader")]
pub use model::{
    ForwardTiming, KimiK3ModelConfig, KimiK3Runtime, kimi_k3_forward_token, kimi_k3_forward_token_timed,
    kimi_k3_forward_token_traced,
};

// Safetensors loader + tiktoken tokenizer (gated by `kimi_k3_loader`).
#[cfg(feature = "kimi_k3_loader")]
pub mod loader;
#[cfg(feature = "kimi_k3_loader")]
pub mod tiktoken;
