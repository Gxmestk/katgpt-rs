//! Plan 318 C9: training-suitable weight initialization tests.
//!
//! Verifies that `KimiK3ModelWeights::random_train_init` produces:
//! - All-finite weights (no NaN/Inf).
//! - Embedding + LM head with small std (~0.02, GPT-2/LLaMA convention).
//! - MoE router with small values (prevents expert collapse).
//! - MoE `e_score_correction_bias` initialized to zero.
//! - RMSNorm gammas at 1.0.
//! - A forward pass that produces finite logits.

#![cfg(feature = "kimi_k3_backward")]

use katgpt_rs::kimi_k3::loader::KimiK3ModelWeights;
use katgpt_rs::kimi_k3::model::{KimiK3ModelConfig, KimiK3Runtime};

/// Check that all values in a slice are finite (no NaN/Inf).
fn assert_all_finite(name: &str, v: &[f32]) {
    for (i, &val) in v.iter().enumerate() {
        assert!(val.is_finite(), "{name}[{i}] is not finite: {val}");
    }
}

/// Compute the standard deviation of a slice.
fn std_dev(v: &[f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    let var = v.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32;
    var.sqrt()
}

#[test]
fn train_init_all_finite_0_40b() {
    let config = KimiK3ModelConfig::kimi_k3_0_40b();
    let weights = KimiK3ModelWeights::random_train_init(&config, 42);

    // Top-level.
    assert_all_finite("embed_weight", &weights.embed_weight);
    assert_all_finite("lm_head_weight", &weights.lm_head_weight);
    assert_all_finite("final_norm_weight", &weights.final_norm_weight);
    assert_all_finite("output_attn_res.norm", &weights.output_attn_res.norm_weight);
    assert_all_finite("output_attn_res.proj", &weights.output_attn_res.proj_weight);

    // Per-layer.
    for (li, layer) in weights.layers.iter().enumerate() {
        assert_all_finite(
            &format!("layer{li}.input_ln"),
            &layer.input_layernorm_weight,
        );
        assert_all_finite(
            &format!("layer{li}.post_attn_ln"),
            &layer.post_attention_layernorm_weight,
        );
        assert_all_finite(
            &format!("layer{li}.self_attn_res.norm"),
            &layer.self_attn_res.norm_weight,
        );
        assert_all_finite(
            &format!("layer{li}.self_attn_res.proj"),
            &layer.self_attn_res.proj_weight,
        );
        assert_all_finite(
            &format!("layer{li}.mlp_attn_res.norm"),
            &layer.mlp_attn_res.norm_weight,
        );
        assert_all_finite(
            &format!("layer{li}.mlp_attn_res.proj"),
            &layer.mlp_attn_res.proj_weight,
        );

        match &layer.attention {
            katgpt_rs::kimi_k3::decoder_layer::KimiAttentionWeights::Mla(m) => {
                assert_all_finite(&format!("layer{li}.mla.w_dkv"), &m.w_dkv);
                assert_all_finite(&format!("layer{li}.mla.w_o"), &m.w_o);
                assert_all_finite(&format!("layer{li}.mla.q_a_norm"), &m.q_a_norm_weight);
                if let Some(ref wg) = m.w_g {
                    assert_all_finite(&format!("layer{li}.mla.w_g"), wg);
                }
            }
            katgpt_rs::kimi_k3::decoder_layer::KimiAttentionWeights::Kda(k) => {
                assert_all_finite(&format!("layer{li}.kda.q_proj"), &k.q_proj);
                assert_all_finite(&format!("layer{li}.kda.a_log"), &k.a_log);
                assert_all_finite(&format!("layer{li}.kda.dt_bias"), &k.dt_bias);
            }
        }

        match &layer.ffn {
            katgpt_rs::kimi_k3::decoder_layer::KimiFfnWeights::Dense(e) => {
                assert_all_finite(&format!("layer{li}.dense.gate"), &e.gate_proj);
            }
            katgpt_rs::kimi_k3::decoder_layer::KimiFfnWeights::Moe(m) => {
                assert_all_finite(&format!("layer{li}.moe.router"), &m.router_weight);
                assert_all_finite(&format!("layer{li}.moe.bias"), &m.e_score_correction_bias);
                for (ei, e) in m.experts.iter().enumerate() {
                    assert_all_finite(&format!("layer{li}.moe.expert{ei}.gate"), &e.gate_proj);
                }
            }
        }
    }
}

#[test]
fn train_init_embedding_small_std() {
    // GPT-2/LLaMA convention: N(0, 0.02²) → std ≈ 0.02.
    let config = KimiK3ModelConfig::kimi_k3_0_40b();
    let weights = KimiK3ModelWeights::random_train_init(&config, 42);
    let embed_std = std_dev(&weights.embed_weight);
    let lm_head_std = std_dev(&weights.lm_head_weight);
    eprintln!("embed std = {embed_std:.4}, lm_head std = {lm_head_std:.4}");
    // Allow some tolerance for finite-sample variance.
    assert!(
        embed_std > 0.01 && embed_std < 0.04,
        "embed std out of range: {embed_std:.4} (expected ~0.02)"
    );
    assert!(
        lm_head_std > 0.01 && lm_head_std < 0.04,
        "lm_head std out of range: {lm_head_std:.4} (expected ~0.02)"
    );
}

#[test]
fn train_init_norm_gammas_are_one() {
    let config = KimiK3ModelConfig::kimi_k3_0_40b();
    let weights = KimiK3ModelWeights::random_train_init(&config, 42);
    // All RMSNorm gammas should be exactly 1.0.
    for &g in &weights.final_norm_weight {
        assert_eq!(g, 1.0, "final_norm gamma != 1.0: {g}");
    }
    for &g in &weights.output_attn_res.norm_weight {
        assert_eq!(g, 1.0, "output_attn_res norm gamma != 1.0: {g}");
    }
    for layer in &weights.layers {
        for &g in &layer.input_layernorm_weight {
            assert_eq!(g, 1.0);
        }
        for &g in &layer.post_attention_layernorm_weight {
            assert_eq!(g, 1.0);
        }
        for &g in &layer.self_attn_res.norm_weight {
            assert_eq!(g, 1.0);
        }
        for &g in &layer.mlp_attn_res.norm_weight {
            assert_eq!(g, 1.0);
        }
    }
}

#[test]
fn train_init_moe_router_small() {
    // MoE router centroids should have small values (std ~0.02).
    let config = KimiK3ModelConfig::kimi_k3_0_40b();
    let weights = KimiK3ModelWeights::random_train_init(&config, 42);

    // Find the first MoE layer.
    for layer in &weights.layers {
        if let katgpt_rs::kimi_k3::decoder_layer::KimiFfnWeights::Moe(m) = &layer.ffn {
            let router_std = std_dev(&m.router_weight);
            eprintln!("MoE router std = {router_std:.4}");
            assert!(
                router_std < 0.05,
                "MoE router std too large: {router_std:.4} (expected ~0.02)"
            );
            // e_score_correction_bias should be zero.
            for &b in &m.e_score_correction_bias {
                assert_eq!(b, 0.0, "e_score_correction_bias != 0: {b}");
            }
            break;
        }
    }
}

#[test]
fn train_init_forward_pass_finite_logits() {
    // The init must produce a forward pass with finite logits (no NaN/Inf).
    let config = KimiK3ModelConfig::kimi_k3_0_40b();
    let weights = KimiK3ModelWeights::random_train_init(&config, 42);
    let mut runtime = KimiK3Runtime::new(&config, 16);

    // Forward one token.
    let tokens = vec![1u32, 5, 10, 15];
    let _ = tokens; // just for documentation
    runtime.reset();
    let logits =
        katgpt_rs::kimi_k3::model::kimi_k3_forward_token(&config, &weights, &mut runtime, 1u32);

    // Check all logits are finite.
    let v = config.vocab_size;
    assert_eq!(logits.len(), v, "logits length mismatch");
    for (i, &l) in logits.iter().enumerate() {
        assert!(l.is_finite(), "logit[{i}] is not finite: {l}");
    }

    eprintln!(
        "✓ Train-init forward pass: {} finite logits, max|logit| = {:.4}",
        v,
        logits.iter().fold(0.0f32, |a, &b| a.max(b.abs()))
    );
}
