//! Equivalence test: checkpointed backward must produce bit-identical gradients
//! to the non-checkpointed backward (Plan 318 Phase C C7).
//!
//! Strategy: run both backwards on the same forward, compare every gradient
//! entry. The checkpointed backward recomputes activations, so the gradients
//! should be numerically identical (both use the same analytic formulas).
//!
//! A small tolerance accounts for f32 reordering (the checkpointed version
//! clones + re-derives some values, which can change summation order).

#![cfg(feature = "kimi_k3_backward")]

use katgpt_attn::gdn2::kda_forward::KdaConfig;
use katgpt_attn::mla::MlaConfig;
use katgpt_rs::kimi_k3::backward::{
    KimiK3ModelGradients, TokenSavedActivations, kimi_k3_backward_sequence,
    kimi_k3_forward_token_saved,
};
use katgpt_rs::kimi_k3::checkpoint::{
    SequenceCheckpoint, TokenCheckpoint, kimi_k3_backward_sequence_ckpt, kimi_k3_forward_token_ckpt,
};
use katgpt_rs::kimi_k3::decoder_layer::KimiFfnConfig;
use katgpt_rs::kimi_k3::loader::KimiK3ModelWeights;
use katgpt_rs::kimi_k3::model::{KimiK3ModelConfig, KimiK3Runtime};
use katgpt_transformer::attn_res::AttnResConfig;
use katgpt_transformer::moe::MoeConfig;

/// Build a small 3-layer config with MLA at layer 2 + KDA at layers 0,1.
fn small_config() -> KimiK3ModelConfig {
    let d = 16;
    KimiK3ModelConfig {
        hidden_size: d,
        vocab_size: 32,
        num_layers: 3,
        rms_eps: 1e-5,
        mla_layer_indices: vec![2],
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

/// All-KDA config (no MLA layer).
fn small_config_all_kda() -> KimiK3ModelConfig {
    let mut c = small_config();
    c.mla_layer_indices = vec![];
    c
}

/// Max abs diff between two gradient vectors across the whole model.
fn max_abs_diff(a: &KimiK3ModelGradients, b: &KimiK3ModelGradients) -> f32 {
    let mut max_diff = 0.0f32;
    macro_rules! cmp {
        ($av:expr, $bv:expr) => {
            for (x, y) in $av.iter().zip($bv.iter()) {
                max_diff = max_diff.max((x - y).abs());
            }
        };
    }
    cmp!(&a.embed_weight, &b.embed_weight);
    cmp!(&a.final_norm_weight, &b.final_norm_weight);
    cmp!(&a.lm_head_weight, &b.lm_head_weight);
    cmp!(&a.output_attn_res_norm, &b.output_attn_res_norm);
    cmp!(&a.output_attn_res_proj, &b.output_attn_res_proj);
    for (la, lb) in a.layers.iter().zip(b.layers.iter()) {
        cmp!(&la.input_layernorm_weight, &lb.input_layernorm_weight);
        cmp!(
            &la.post_attention_layernorm_weight,
            &lb.post_attention_layernorm_weight
        );
        cmp!(&la.self_attn_res_norm, &lb.self_attn_res_norm);
        cmp!(&la.self_attn_res_proj, &lb.self_attn_res_proj);
        cmp!(&la.mlp_attn_res_norm, &lb.mlp_attn_res_norm);
        cmp!(&la.mlp_attn_res_proj, &lb.mlp_attn_res_proj);
        if let (Some(ag), Some(bg)) = (&la.mla_grads, &lb.mla_grads) {
            cmp!(&ag.w_dkv, &bg.w_dkv);
            cmp!(&ag.w_dq, &bg.w_dq);
            cmp!(&ag.w_uq, &bg.w_uq);
            cmp!(&ag.w_qr, &bg.w_qr);
            cmp!(&ag.w_uk, &bg.w_uk);
            cmp!(&ag.w_uv, &bg.w_uv);
            cmp!(&ag.w_kr, &bg.w_kr);
            cmp!(&ag.w_o, &bg.w_o);
            cmp!(&ag.q_a_norm_weight, &bg.q_a_norm_weight);
            cmp!(&ag.kv_a_norm_weight, &bg.kv_a_norm_weight);
            if let (Some(aw), Some(bw)) = (&ag.w_g, &bg.w_g) {
                cmp!(aw, bw);
            }
        }
        if let (Some(ag), Some(bg)) = (&la.kda_grads, &lb.kda_grads) {
            cmp!(&ag.q_proj, &bg.q_proj);
            cmp!(&ag.k_proj, &bg.k_proj);
            cmp!(&ag.v_proj, &bg.v_proj);
            cmp!(&ag.q_conv_weight, &bg.q_conv_weight);
            cmp!(&ag.k_conv_weight, &bg.k_conv_weight);
            cmp!(&ag.v_conv_weight, &bg.v_conv_weight);
            cmp!(&ag.a_log, &bg.a_log);
            cmp!(&ag.f_a_proj, &bg.f_a_proj);
            cmp!(&ag.f_b_proj, &bg.f_b_proj);
            cmp!(&ag.dt_bias, &bg.dt_bias);
            cmp!(&ag.beta_proj, &bg.beta_proj);
            cmp!(&ag.g_proj, &bg.g_proj);
            cmp!(&ag.o_proj, &bg.o_proj);
        }
        if let (Some(ag), Some(bg)) = (&la.moe_grads, &lb.moe_grads) {
            cmp!(&ag.router_weight, &bg.router_weight);
            cmp!(&ag.e_score_correction_bias, &bg.e_score_correction_bias);
            for (ae, be) in ag.experts.iter().zip(bg.experts.iter()) {
                cmp!(&ae.gate_proj, &be.gate_proj);
                cmp!(&ae.up_proj, &be.up_proj);
                cmp!(&ae.down_proj, &be.down_proj);
            }
            for (ae, be) in ag.shared_experts.iter().zip(bg.shared_experts.iter()) {
                cmp!(&ae.gate_proj, &be.gate_proj);
                cmp!(&ae.up_proj, &be.up_proj);
                cmp!(&ae.down_proj, &be.down_proj);
            }
            if let (Some(ad), Some(bd)) = (&ag.routed_expert_down_proj, &bg.routed_expert_down_proj)
            {
                cmp!(ad, bd);
            }
            if let (Some(au), Some(bu)) = (&ag.routed_expert_up_proj, &bg.routed_expert_up_proj) {
                cmp!(au, bu);
            }
        }
        if let (Some(ag), Some(bg)) = (&la.dense_grads, &lb.dense_grads) {
            cmp!(&ag.gate_proj, &bg.gate_proj);
            cmp!(&ag.up_proj, &bg.up_proj);
            cmp!(&ag.down_proj, &bg.down_proj);
        }
    }
    max_diff
}

/// Standard d_logits for a sum-of-squares loss on logits: dL/d(logits) = 2*logits.
fn make_d_logits(saved_tokens: &[TokenSavedActivations]) -> Vec<Vec<f32>> {
    saved_tokens
        .iter()
        .map(|s| s.logits.iter().map(|&v| 2.0 * v).collect())
        .collect()
}

/// Run the equivalence check for a given config + token sequence.
fn run_equivalence(config: &KimiK3ModelConfig, tokens: &[u32]) -> f32 {
    let l = tokens.len();
    let weights = KimiK3ModelWeights::random(config, 42);

    // ── Non-checkpointed path ──
    let mut rt_full = KimiK3Runtime::new(config, l);
    let mut saved_tokens: Vec<TokenSavedActivations> =
        (0..l).map(|_| TokenSavedActivations::new()).collect();
    rt_full.reset();
    for (pos, &tok) in tokens.iter().enumerate() {
        kimi_k3_forward_token_saved(
            config,
            &weights,
            &mut rt_full,
            tok,
            pos,
            &mut saved_tokens[pos],
        );
    }
    let d_logits = make_d_logits(&saved_tokens);
    let mut grads_full = KimiK3ModelGradients::zeros_like(config, &weights);
    kimi_k3_backward_sequence(
        config,
        &weights,
        &rt_full,
        &saved_tokens,
        &d_logits,
        &mut grads_full,
    );

    // ── Checkpointed path ──
    let mut rt_ckpt = KimiK3Runtime::new(config, l);
    let mut ckpt_tokens: Vec<TokenCheckpoint> = (0..l)
        .map(|_| TokenCheckpoint {
            token_id: 0,
            pos: 0,
            layers: Vec::new(),
            final_stage: katgpt_rs::kimi_k3::checkpoint::FinalStageSaved {
                prefix_sum_final: Vec::new(),
                block_state_final: Vec::new(),
                has_output_attn_res: false,
                pre_final_norm: Vec::new(),
                final_hidden: Vec::new(),
                final_norm_inv_rms: 0.0,
                logits: Vec::new(),
            },
        })
        .collect();
    rt_ckpt.reset();
    for (pos, &tok) in tokens.iter().enumerate() {
        kimi_k3_forward_token_ckpt(
            config,
            &weights,
            &mut rt_ckpt,
            tok,
            pos,
            &mut ckpt_tokens[pos],
        );
    }
    let ckpt = SequenceCheckpoint {
        tokens: ckpt_tokens,
    };
    let mut grads_ckpt = KimiK3ModelGradients::zeros_like(config, &weights);
    kimi_k3_backward_sequence_ckpt(
        config,
        &weights,
        &mut rt_ckpt,
        &ckpt,
        &d_logits,
        &mut grads_ckpt,
    );

    max_abs_diff(&grads_full, &grads_ckpt)
}

#[test]
fn checkpoint_equivalence_all_kda_3_layers_2_tokens() {
    let config = small_config_all_kda();
    let tokens = vec![0u32, 1];
    let max_diff = run_equivalence(&config, &tokens);
    eprintln!("all-KDA 3-layer 2-token: max abs diff = {max_diff:.2e}");
    // Both backwards use the same analytic formulas on the same forward values;
    // differences come only from f32 reordering in cloned snapshots. Should be
    // near zero (typically < 1e-5).
    assert!(
        max_diff < 1e-3,
        "checkpoint vs non-checkpoint grad diff too large: {max_diff:.2e}"
    );
}

#[test]
fn checkpoint_equivalence_with_mla_3_tokens() {
    let config = small_config();
    let tokens = vec![0u32, 1, 2];
    let max_diff = run_equivalence(&config, &tokens);
    eprintln!("MLA+KDA 3-layer 3-token: max abs diff = {max_diff:.2e}");
    assert!(
        max_diff < 1e-3,
        "checkpoint vs non-checkpoint grad diff too large: {max_diff:.2e}"
    );
}

#[test]
fn checkpoint_equivalence_with_mla_block_boundary() {
    // block_size=2 → layers 0 and 2 are boundaries. Forces block-state push/pop
    // + attn-res mixing on multiple layers.
    let mut config = small_config();
    config.attn_res_config.block_size = 2;
    let tokens = vec![0u32, 1, 2];
    let max_diff = run_equivalence(&config, &tokens);
    eprintln!("MLA+KDA block_size=2 3-token: max abs diff = {max_diff:.2e}");
    assert!(
        max_diff < 1e-3,
        "checkpoint vs non-checkpoint grad diff too large: {max_diff:.2e}"
    );
}
