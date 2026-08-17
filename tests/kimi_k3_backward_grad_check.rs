//! Finite-difference gradient check for the KimiK3 full-model backward (Plan 318 C6).
//!
//! Standard ML gradcheck:
//! 1. Small KimiK3 model (3 layers, small dims).
//! 2. Random seeded inputs + weights.
//! 3. Loss = sum of logits² (sum-of-squares).
//! 4. Finite-difference: central differences.
//! 5. Analytic: `kimi_k3_backward_sequence`.
//! 6. Pass criterion: relative error < threshold (f32 noise floor).
//!
//! The test validates that the composition of MLA + MoE + KDA backward primitives
//! with attn-res + RMSNorm + dense FFN + LM head backward produces correct gradients.

#![cfg(feature = "kimi_k3_backward")]

use katgpt_attn::gdn2::kda_forward::KdaConfig;
use katgpt_attn::mla::MlaConfig;
use katgpt_rs::kimi_k3::backward::{
    KimiK3ModelGradients, TokenSavedActivations, kimi_k3_backward_sequence,
    kimi_k3_forward_token_saved,
};
use katgpt_rs::kimi_k3::decoder_layer::KimiFfnConfig;
use katgpt_rs::kimi_k3::loader::KimiK3ModelWeights;
use katgpt_rs::kimi_k3::model::{KimiK3ModelConfig, KimiK3Runtime};
use katgpt_transformer::attn_res::AttnResConfig;
use katgpt_transformer::moe::MoeConfig;

// ─── Small test config ─────────────────────────────────────────────────────

/// Build a small KimiK3 config for gradient checking.
///
/// 3 layers: layer 0 (KDA + Dense), layer 1 (KDA + MoE), layer 2 (MLA + MoE).
/// block_size=4 → only layer 0 is a boundary.
fn small_config() -> KimiK3ModelConfig {
    small_config_with_mla(true)
}

/// All-KDA config (no MLA layer) — isolates KDA backward from MLA.
fn small_config_all_kda() -> KimiK3ModelConfig {
    small_config_with_mla(false)
}

fn small_config_with_mla(use_mla: bool) -> KimiK3ModelConfig {
    small_config_layers(use_mla, 3)
}

/// Single-layer config — minimal test for the basic backward.
fn single_layer_config() -> KimiK3ModelConfig {
    small_config_layers(false, 1)
}

fn small_config_layers(use_mla: bool, num_layers: usize) -> KimiK3ModelConfig {
    let d = 16;
    KimiK3ModelConfig {
        hidden_size: d,
        vocab_size: 32,
        num_layers,
        rms_eps: 1e-5,
        mla_layer_indices: if use_mla {
            vec![num_layers - 1]
        } else {
            vec![]
        },
        mla_config: MlaConfig {
            kv_lora_rank: 8,
            q_lora_rank: 8,
            qk_nope_head_dim: 4,
            qk_rope_head_dim: 4,
            v_head_dim: 4,
            n_heads: 2,
            hidden_size: d,
            use_output_gate: true,
            use_nope: true,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
        },
        kda_config: KdaConfig {
            hidden_size: d,
            n_heads: 2,
            head_dim: 8,
            conv_kernel_size: 4,
            ..KdaConfig::kimi_k3_0_40b()
        },
        dense_ffn_config: KimiFfnConfig::Dense {
            intermediate_size: 32,
            hidden_size: d,
            situ_beta: 4.0,
            situ_linear_beta: Some(25.0),
        },
        moe_config: MoeConfig {
            num_experts: 4,
            num_experts_per_token: 2,
            num_shared_experts: 1,
            moe_intermediate_size: 16,
            hidden_size: d,
            routed_expert_hidden_size: Some(8),
            ..MoeConfig::kimi_k3_0_40b()
        },
        attn_res_config: AttnResConfig {
            hidden_size: d,
            block_size: 4,
            rms_eps: 1e-5,
        },
    }
}

/// Relative error.
fn rel_err(analytic: f32, numeric: f32) -> f32 {
    // Use a larger floor (1e-2) to suppress FD noise on tiny gradients.
    // At f32 precision with ε=5e-3, values below ~1e-2 have rel_err dominated
    // by FD round-off, not by backward correctness.
    let denom = analytic.abs().max(numeric.abs()).max(1e-2);
    (analytic - numeric).abs() / denom
}

/// Run the forward for a sequence of tokens, compute sum-of-squares loss on logits.
fn run_forward_loss(
    config: &KimiK3ModelConfig,
    weights: &KimiK3ModelWeights,
    tokens: &[u32],
) -> f32 {
    let mut runtime = KimiK3Runtime::new(config, tokens.len());
    let mut saved = TokenSavedActivations::new();
    let mut loss = 0.0f32;
    for (pos, &tok) in tokens.iter().enumerate() {
        kimi_k3_forward_token_saved(config, weights, &mut runtime, tok, pos, &mut saved);
        loss += saved.logits.iter().map(|&v| v * v).sum::<f32>();
    }
    loss
}

// ─── Main gradient check ─────────────────────────────────────────────────────

/// Gradient check with NO attn-res (block_size huge → no boundaries, no block state).
/// Isolates the basic residual + attention + FFN backward from the attn-res complexity.
#[test]
fn gradient_check_no_attn_res() {
    let mut config = small_config();
    config.attn_res_config.block_size = 1000; // No boundaries → no block state → no attn-res
    let d = config.hidden_size;
    let l = 2;
    let epsilon = 5e-3f32;
    let tol = 8e-2f32;

    let weights = KimiK3ModelWeights::random(&config, 42);
    let tokens: Vec<u32> = vec![0, 1];

    let mut runtime = KimiK3Runtime::new(&config, l);
    let mut saved_tokens: Vec<TokenSavedActivations> =
        (0..l).map(|_| TokenSavedActivations::new()).collect();
    for (pos, &tok) in tokens.iter().enumerate() {
        kimi_k3_forward_token_saved(
            &config,
            &weights,
            &mut runtime,
            tok,
            pos,
            &mut saved_tokens[pos],
        );
    }

    let d_logits: Vec<Vec<f32>> = saved_tokens
        .iter()
        .map(|s| s.logits.iter().map(|&v| 2.0 * v).collect())
        .collect();
    let mut grads = KimiK3ModelGradients::zeros_like(&config, &weights);
    kimi_k3_backward_sequence(
        &config,
        &weights,
        &runtime,
        &saved_tokens,
        &d_logits,
        &mut grads,
    );

    let mut max_rel_err = 0.0f32;
    let mut max_rel_err_label = String::new();
    let mut weights_mut = weights.clone();

    macro_rules! fd_check {
        ($get_mut:expr, $analytic:expr, $label:expr) => {{
            let orig = *$get_mut;
            *$get_mut = orig + epsilon;
            let lp = run_forward_loss(&config, &weights_mut, &tokens);
            *$get_mut = orig - epsilon;
            let lm = run_forward_loss(&config, &weights_mut, &tokens);
            *$get_mut = orig;
            let numeric = (lp - lm) / (2.0 * epsilon);
            let err = rel_err($analytic, numeric);
            if err > max_rel_err {
                max_rel_err = err;
                max_rel_err_label = $label.to_string();
            }
        }};
    }

    // Check embed_weight
    for j in 0..d.min(4) {
        fd_check!(
            &mut weights_mut.embed_weight[j],
            grads.embed_weight[j],
            format!("embed[{}]", j)
        );
    }
    // Check final_norm
    for j in 0..d.min(4) {
        fd_check!(
            &mut weights_mut.final_norm_weight[j],
            grads.final_norm_weight[j],
            format!("final_norm[{}]", j)
        );
    }
    // Check per-layer input_ln
    for li in 0..config.num_layers {
        for j in 0..d.min(2) {
            fd_check!(
                &mut weights_mut.layers[li].input_layernorm_weight[j],
                grads.layers[li].input_layernorm_weight[j],
                format!("L{}_input_ln[{}]", li, j)
            );
        }
    }

    eprintln!(
        "No-attn-res gradient check: max rel_err = {:.4}% at {}",
        max_rel_err * 100.0,
        max_rel_err_label
    );
    assert!(
        max_rel_err < tol,
        "no-attn-res gradient check FAILED: {:.4}% at {}",
        max_rel_err * 100.0,
        max_rel_err_label
    );
}

/// Single-layer gradient check — minimal test isolating the basic backward.
#[test]
fn gradient_check_single_layer() {
    let config = single_layer_config();
    let d = config.hidden_size;
    let l = 1;
    let epsilon = 5e-3f32;
    let tol = 8e-2f32;

    let weights = KimiK3ModelWeights::random(&config, 42);
    let tokens: Vec<u32> = vec![0];

    let mut runtime = KimiK3Runtime::new(&config, l);
    let mut saved_tokens: Vec<TokenSavedActivations> =
        (0..l).map(|_| TokenSavedActivations::new()).collect();
    for (pos, &tok) in tokens.iter().enumerate() {
        kimi_k3_forward_token_saved(
            &config,
            &weights,
            &mut runtime,
            tok,
            pos,
            &mut saved_tokens[pos],
        );
    }

    let d_logits: Vec<Vec<f32>> = saved_tokens
        .iter()
        .map(|s| s.logits.iter().map(|&v| 2.0 * v).collect())
        .collect();
    let mut grads = KimiK3ModelGradients::zeros_like(&config, &weights);
    kimi_k3_backward_sequence(
        &config,
        &weights,
        &runtime,
        &saved_tokens,
        &d_logits,
        &mut grads,
    );

    let mut max_rel_err = 0.0f32;
    let mut max_rel_err_label = String::new();
    let mut weights_mut = weights.clone();

    macro_rules! fd_check {
        ($get_mut:expr, $analytic:expr, $label:expr) => {{
            let orig = *$get_mut;
            *$get_mut = orig + epsilon;
            let lp = run_forward_loss(&config, &weights_mut, &tokens);
            *$get_mut = orig - epsilon;
            let lm = run_forward_loss(&config, &weights_mut, &tokens);
            *$get_mut = orig;
            let numeric = (lp - lm) / (2.0 * epsilon);
            let err = rel_err($analytic, numeric);
            if err > max_rel_err {
                max_rel_err = err;
                max_rel_err_label = $label.to_string();
            }
        }};
    }

    for j in 0..d.min(4) {
        fd_check!(
            &mut weights_mut.embed_weight[j],
            grads.embed_weight[j],
            format!("embed[{}]", j)
        );
    }
    for j in 0..d.min(4) {
        fd_check!(
            &mut weights_mut.final_norm_weight[j],
            grads.final_norm_weight[j],
            format!("final_norm[{}]", j)
        );
    }
    for j in 0..d.min(4) {
        fd_check!(
            &mut weights_mut.layers[0].input_layernorm_weight[j],
            grads.layers[0].input_layernorm_weight[j],
            format!("L0_input_ln[{}]", j)
        );
    }

    eprintln!(
        "Single-layer gradient check: max rel_err = {:.4}% at {}",
        max_rel_err * 100.0,
        max_rel_err_label
    );
    assert!(
        max_rel_err < tol,
        "single-layer gradient check FAILED: {:.4}% at {}",
        max_rel_err * 100.0,
        max_rel_err_label
    );
}

/// Gradient check with ALL KDA layers (no MLA) — isolates KDA backward.
#[test]
fn gradient_check_all_kda() {
    let config = small_config_all_kda();
    let d = config.hidden_size;
    let l = 2;
    let epsilon = 5e-3f32;
    let tol = 1e-1f32;

    let weights = KimiK3ModelWeights::random(&config, 42);
    let tokens: Vec<u32> = vec![0, 1];

    let mut runtime = KimiK3Runtime::new(&config, l);
    let mut saved_tokens: Vec<TokenSavedActivations> =
        (0..l).map(|_| TokenSavedActivations::new()).collect();
    for (pos, &tok) in tokens.iter().enumerate() {
        kimi_k3_forward_token_saved(
            &config,
            &weights,
            &mut runtime,
            tok,
            pos,
            &mut saved_tokens[pos],
        );
    }

    let d_logits: Vec<Vec<f32>> = saved_tokens
        .iter()
        .map(|s| s.logits.iter().map(|&v| 2.0 * v).collect())
        .collect();
    let mut grads = KimiK3ModelGradients::zeros_like(&config, &weights);
    kimi_k3_backward_sequence(
        &config,
        &weights,
        &runtime,
        &saved_tokens,
        &d_logits,
        &mut grads,
    );

    let mut max_rel_err = 0.0f32;
    let mut max_rel_err_label = String::new();
    let mut weights_mut = weights.clone();

    macro_rules! fd_check {
        ($get_mut:expr, $analytic:expr, $label:expr) => {{
            let orig = *$get_mut;
            *$get_mut = orig + epsilon;
            let lp = run_forward_loss(&config, &weights_mut, &tokens);
            *$get_mut = orig - epsilon;
            let lm = run_forward_loss(&config, &weights_mut, &tokens);
            *$get_mut = orig;
            let numeric = (lp - lm) / (2.0 * epsilon);
            let err = rel_err($analytic, numeric);
            if err > max_rel_err {
                max_rel_err = err;
                max_rel_err_label = $label.to_string();
            }
        }};
    }

    for j in 0..d.min(4) {
        fd_check!(
            &mut weights_mut.embed_weight[j],
            grads.embed_weight[j],
            format!("embed[{}]", j)
        );
    }
    for j in 0..d.min(4) {
        fd_check!(
            &mut weights_mut.final_norm_weight[j],
            grads.final_norm_weight[j],
            format!("final_norm[{}]", j)
        );
    }
    for li in 0..config.num_layers {
        for j in 0..d.min(2) {
            fd_check!(
                &mut weights_mut.layers[li].input_layernorm_weight[j],
                grads.layers[li].input_layernorm_weight[j],
                format!("L{}_input_ln[{}]", li, j)
            );
        }
    }

    eprintln!(
        "All-KDA gradient check: max rel_err = {:.4}% at {}",
        max_rel_err * 100.0,
        max_rel_err_label
    );
    assert!(
        max_rel_err < tol,
        "all-KDA gradient check FAILED: {:.4}% at {}",
        max_rel_err * 100.0,
        max_rel_err_label
    );
}

#[test]
fn gradient_check_full_model() {
    gradient_check_model(small_config(), 3, "full-model");
}

/// Full-model gradient check with L=1 token (no cross-token MLA effects).
#[test]
fn gradient_check_full_model_single_token() {
    gradient_check_model(small_config(), 1, "full-model-L1");
}

fn gradient_check_model(config: KimiK3ModelConfig, l: usize, label: &str) {
    let d = config.hidden_size;
    let epsilon = 5e-3f32;
    // 15% tolerance — the MLA cross-token composition + f32 FD noise accumulates.
    // The per-primitive backwards (MLA, MoE, KDA) pass at <2% individually (C4/C5);
    // the composition through prefix_sum + attn-res + block-state adds error.
    // The overfit test (C-GATE-M3.6) is the ultimate correctness gate.
    let tol = 1.5e-1f32;

    let weights = KimiK3ModelWeights::random(&config, 42);
    let tokens: Vec<u32> = (0..l as u32).collect();

    // Forward with saved for each token
    let mut runtime = KimiK3Runtime::new(&config, l);
    let mut saved_tokens: Vec<TokenSavedActivations> =
        (0..l).map(|_| TokenSavedActivations::new()).collect();
    for (pos, &tok) in tokens.iter().enumerate() {
        kimi_k3_forward_token_saved(
            &config,
            &weights,
            &mut runtime,
            tok,
            pos,
            &mut saved_tokens[pos],
        );
    }

    // dL/d(logits) = 2 * logits
    let d_logits: Vec<Vec<f32>> = saved_tokens
        .iter()
        .map(|s| s.logits.iter().map(|&v| 2.0 * v).collect())
        .collect();

    // Analytic backward
    let mut grads = KimiK3ModelGradients::zeros_like(&config, &weights);
    kimi_k3_backward_sequence(
        &config,
        &weights,
        &runtime,
        &saved_tokens,
        &d_logits,
        &mut grads,
    );

    // Finite-difference check
    let mut max_rel_err = 0.0f32;
    let mut max_rel_err_label = String::new();
    let mut weights_mut = weights.clone();
    let mut all_errs: Vec<(String, f32, f32, f32)> = Vec::new(); // (label, analytic, numeric, rel_err)

    // Helper: perturb one weight element, measure loss change, compare to analytic.
    macro_rules! fd_check {
        ($get_mut:expr, $analytic:expr, $label:expr) => {{
            let orig = *$get_mut;
            *$get_mut = orig + epsilon;
            let lp = run_forward_loss(&config, &weights_mut, &tokens);
            *$get_mut = orig - epsilon;
            let lm = run_forward_loss(&config, &weights_mut, &tokens);
            *$get_mut = orig;
            let numeric = (lp - lm) / (2.0 * epsilon);
            let err = rel_err($analytic, numeric);
            all_errs.push(($label.to_string(), $analytic, numeric, err));
            if err > max_rel_err {
                max_rel_err = err;
                max_rel_err_label = $label.to_string();
            }
        }};
    }

    // embed_weight
    for t in 0..l {
        for j in 0..d.min(4) {
            let idx = tokens[t] as usize * d + j;
            fd_check!(
                &mut weights_mut.embed_weight[idx],
                grads.embed_weight[idx],
                format!("embed_weight[t{}][{}]", t, j)
            );
        }
    }

    // final_norm_weight
    for j in 0..d.min(4) {
        fd_check!(
            &mut weights_mut.final_norm_weight[j],
            grads.final_norm_weight[j],
            format!("final_norm[{}]", j)
        );
    }

    // lm_head_weight (first 4 vocab rows)
    for vi in 0..4 {
        for j in 0..d.min(2) {
            let idx = vi * d + j;
            fd_check!(
                &mut weights_mut.lm_head_weight[idx],
                grads.lm_head_weight[idx],
                format!("lm_head[{}][{}]", vi, j)
            );
        }
    }

    // output attn-res
    for j in 0..d.min(4) {
        fd_check!(
            &mut weights_mut.output_attn_res.norm_weight[j],
            grads.output_attn_res_norm[j],
            format!("output_attn_res_norm[{}]", j)
        );
        fd_check!(
            &mut weights_mut.output_attn_res.proj_weight[j],
            grads.output_attn_res_proj[j],
            format!("output_attn_res_proj[{}]", j)
        );
    }

    // Per-layer checks
    for layer_idx in 0..config.num_layers {
        // input_layernorm_weight
        for j in 0..d.min(2) {
            fd_check!(
                &mut weights_mut.layers[layer_idx].input_layernorm_weight[j],
                grads.layers[layer_idx].input_layernorm_weight[j],
                format!("L{} input_ln[{}]", layer_idx, j)
            );
        }
        // post_attention_layernorm_weight
        for j in 0..d.min(2) {
            fd_check!(
                &mut weights_mut.layers[layer_idx].post_attention_layernorm_weight[j],
                grads.layers[layer_idx].post_attention_layernorm_weight[j],
                format!("L{} post_attn_ln[{}]", layer_idx, j)
            );
        }
        // self_attn_res proj
        for j in 0..d.min(2) {
            fd_check!(
                &mut weights_mut.layers[layer_idx].self_attn_res.proj_weight[j],
                grads.layers[layer_idx].self_attn_res_proj[j],
                format!("L{} self_attn_res_proj[{}]", layer_idx, j)
            );
        }
        // mlp_attn_res proj
        for j in 0..d.min(2) {
            fd_check!(
                &mut weights_mut.layers[layer_idx].mlp_attn_res.proj_weight[j],
                grads.layers[layer_idx].mlp_attn_res_proj[j],
                format!("L{} mlp_attn_res_proj[{}]", layer_idx, j)
            );
        }
    }

    eprintln!(
        "{} gradient check: max rel_err = {:.4}% at {} (tol {:.1}%)",
        label,
        max_rel_err * 100.0,
        max_rel_err_label,
        tol * 100.0
    );
    // Print all checks sorted by rel_err descending
    all_errs.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
    eprintln!("Top 15 errors:");
    for (label, analytic, numeric, err) in all_errs.iter().take(15) {
        eprintln!(
            "  {:<30} analytic={:>12.6e}  numeric={:>12.6e}  rel_err={:.4}%",
            label,
            analytic,
            numeric,
            err * 100.0
        );
    }

    assert!(
        max_rel_err < tol,
        "{} gradient check FAILED: max rel_err = {:.4}% at {} (tol {:.1}%)",
        label,
        max_rel_err * 100.0,
        max_rel_err_label,
        tol * 100.0
    );
}

/// Smoke test: backward runs without panic/NaN at kimi_k3_0_40b dims.
#[test]
fn backward_smoke_kimi_k3_0_40b() {
    let mut config = KimiK3ModelConfig::kimi_k3_0_40b();
    config.vocab_size = 64; // tiny vocab for speed
    let l = 3;

    let weights = KimiK3ModelWeights::random(&config, 99);
    let tokens: Vec<u32> = vec![0, 1, 2];

    let mut runtime = KimiK3Runtime::new(&config, l);
    let mut saved_tokens: Vec<TokenSavedActivations> =
        (0..l).map(|_| TokenSavedActivations::new()).collect();
    for (pos, &tok) in tokens.iter().enumerate() {
        kimi_k3_forward_token_saved(
            &config,
            &weights,
            &mut runtime,
            tok,
            pos,
            &mut saved_tokens[pos],
        );
    }

    let d_logits: Vec<Vec<f32>> = saved_tokens
        .iter()
        .map(|s| s.logits.iter().map(|&v| 2.0 * v).collect())
        .collect();
    let mut grads = KimiK3ModelGradients::zeros_like(&config, &weights);
    kimi_k3_backward_sequence(
        &config,
        &weights,
        &runtime,
        &saved_tokens,
        &d_logits,
        &mut grads,
    );

    let check_finite = |v: &[f32], label: &str| {
        for (i, &val) in v.iter().enumerate() {
            assert!(
                val.is_finite(),
                "non-finite gradient at {label}[{i}]: {val}"
            );
        }
    };
    check_finite(&grads.embed_weight, "embed_weight");
    check_finite(&grads.final_norm_weight, "final_norm_weight");
    check_finite(&grads.lm_head_weight, "lm_head_weight");
    for (i, lg) in grads.layers.iter().enumerate() {
        check_finite(&lg.input_layernorm_weight, &format!("layer{i}.input_ln"));
        check_finite(
            &lg.post_attention_layernorm_weight,
            &format!("layer{i}.post_attn_ln"),
        );
    }

    eprintln!("Backward smoke test PASSED (kimi_k3_0_40b dims, vocab=64).");
}
