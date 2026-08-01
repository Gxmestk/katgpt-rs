//! G1 spec-match tests for the MoE forward (Proposal 032 Phase 3).
//!
//! Strategy (mirrors Research 327 MLA G1 + Research 328 §7.6):
//!
//! 1. Write an INDEPENDENT f64 reference implementation of the MoE forward
//!    (equations 12–16 from DeepSeek-V3 §3.3, distilled in Research 328).
//!    This reference does NOT reuse the f32 code — it's written from scratch
//!    from the paper equations.
//! 2. Run both impls on the same random inputs.
//! 3. Assert the f32 impl matches the f64 reference within tolerance.
//!
//! The CRITICAL test is `g1_bias_does_not_leak_into_renormalization` — it
//! implements a "buggy" reference variant that uses `biased[topk_idx]` in the
//! renormalization, runs it, asserts its output DIFFERS from the correct
//! reference, then asserts the f32 impl matches the CORRECT reference. This
//! catches the §2.2 misreading bit-identically.

use katgpt_transformer::moe::{MoeConfig, MoeForwardScratch, MoeWeights, moe_forward_token};

// ─── Independent f64 reference implementation ───────────────────────────────

/// f64 reference of a single SwiGLU expert FFN forward.
fn ref_swiglu_expert_f64(
    gate_proj: &[f64],
    up_proj: &[f64],
    down_proj: &[f64],
    hidden_in: &[f64],
    d: usize,
    d_ffn: usize,
    out: &mut [f64],
) {
    let mut intermediate = vec![0.0f64; d_ffn];
    let mut up = vec![0.0f64; d_ffn];
    // gate · h
    for o in 0..d_ffn {
        let mut acc = 0.0;
        for i in 0..d {
            acc += gate_proj[o * d + i] * hidden_in[i];
        }
        intermediate[o] = acc;
    }
    // up · h
    for o in 0..d_ffn {
        let mut acc = 0.0;
        for i in 0..d {
            acc += up_proj[o * d + i] * hidden_in[i];
        }
        up[o] = acc;
    }
    // swiglu: SiLU(gate) ⊙ up = (gate / (1 + exp(-gate))) * up
    for o in 0..d_ffn {
        let g = intermediate[o];
        let silu = g / (1.0 + (-g).exp());
        intermediate[o] = silu * up[o];
    }
    // down · intermediate
    for o in 0..d {
        let mut acc = 0.0;
        for i in 0..d_ffn {
            acc += down_proj[o * d_ffn + i] * intermediate[i];
        }
        out[o] = acc;
    }
}

/// f64 reference sigmoid (plain exp, no approximations).
#[inline]
fn ref_sigmoid_f64(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// f64 reference of the FULL MoE forward (equations 12–16 from Research 328).
///
/// `buggy_bias_in_renorm`: when true, uses `biased[topk_idx]` in the
/// renormalization (the §2.2 misreading). When false, uses RAW sigmoid scores
/// (the correct behavior). Used by `g1_bias_does_not_leak_into_renormalization`.
fn ref_moe_forward_f64(
    weights: &MoeWeights,
    config: &MoeConfig,
    hidden_in: &[f32],
    out: &mut [f64],
    buggy_bias_in_renorm: bool,
) {
    let n_r = config.n_routed();
    let k_r = config.k_routed();
    let d = config.d();
    let d_ffn = config.d_ffn();

    let hidden_f64: Vec<f64> = hidden_in.iter().map(|&v| v as f64).collect();

    // 1. Router logits + sigmoid scores
    let mut logits = vec![0.0f64; n_r];
    let mut scores = vec![0.0f64; n_r];
    for e in 0..n_r {
        let row = &weights.router_weight[e * d..e * d + d];
        let mut acc = 0.0;
        for (rw, h) in row.iter().zip(hidden_f64.iter()).take(d) {
            acc += *rw as f64 * h;
        }
        logits[e] = acc;
        scores[e] = ref_sigmoid_f64(logits[e]);
    }

    // 2. noaux_tc biased scores + top-K selection
    let mut biased: Vec<(usize, f64)> = (0..n_r)
        .map(|e| (e, scores[e] + weights.e_score_correction_bias[e] as f64))
        .collect();
    // Sort descending by biased score, take top-K.
    biased.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let topk_idx: Vec<usize> = biased.iter().take(k_r).map(|(i, _)| *i).collect();

    // 3. Renormalize
    let topk_s: Vec<f64> = if buggy_bias_in_renorm {
        // BUGGY: use biased score in renormalization
        topk_idx
            .iter()
            .map(|&i| scores[i] + weights.e_score_correction_bias[i] as f64)
            .collect()
    } else {
        // CORRECT: use raw sigmoid score
        topk_idx.iter().map(|&i| scores[i]).collect()
    };
    let sum: f64 = topk_s.iter().sum();
    let g: Vec<f64> = if config.renormalize {
        topk_s.iter().map(|&s| s / sum).collect()
    } else {
        topk_s
    };

    // 4. Shared expert (always on) → base of output
    let shared = &weights.shared_experts[0];
    ref_swiglu_expert_f64(
        &cast_f32_slice(&shared.gate_proj),
        &cast_f32_slice(&shared.up_proj),
        &cast_f32_slice(&shared.down_proj),
        &hidden_f64,
        d,
        d_ffn,
        out,
    );
    // If N_s > 1, accumulate remaining shared experts.
    let mut expert_out = vec![0.0f64; d];
    for s in 1..weights.shared_experts.len() {
        let shared = &weights.shared_experts[s];
        ref_swiglu_expert_f64(
            &cast_f32_slice(&shared.gate_proj),
            &cast_f32_slice(&shared.up_proj),
            &cast_f32_slice(&shared.down_proj),
            &hidden_f64,
            d,
            d_ffn,
            &mut expert_out,
        );
        for i in 0..d {
            out[i] += expert_out[i];
        }
    }

    // 5. Routed experts (weighted by g)
    for k in 0..k_r {
        let idx = topk_idx[k];
        let w = g[k];
        let expert = &weights.experts[idx];
        ref_swiglu_expert_f64(
            &cast_f32_slice(&expert.gate_proj),
            &cast_f32_slice(&expert.up_proj),
            &cast_f32_slice(&expert.down_proj),
            &hidden_f64,
            d,
            d_ffn,
            &mut expert_out,
        );
        for i in 0..d {
            out[i] += w * expert_out[i];
        }
    }
}

/// Cast `&[f32]` to `Vec<f64>` for the reference impl.
fn cast_f32_slice(s: &[f32]) -> Vec<f64> {
    s.iter().map(|&v| v as f64).collect()
}

/// Max abs diff between two slices.
fn max_abs_diff(a: &[f32], b: &[f64]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x - *y as f32).abs())
        .fold(0.0f32, f32::max)
}

// ─── G1 tests ───────────────────────────────────────────────────────────────

/// Tiny config: 4 experts, 1 shared, K=2, d=8, d_ffn=16.
fn tiny_config() -> MoeConfig {
    MoeConfig {
        num_experts: 4,
        num_shared_experts: 1,
        num_experts_per_token: 2,
        moe_intermediate_size: 16,
        hidden_size: 8,
        use_sigmoid_router: true,
        renormalize: true,
    }
}

#[test]
fn g1_zero_bias_matches_reference() {
    // Bias = 0 → pure sigmoid router, no noaux_tc influence on selection.
    let config = tiny_config();
    let mut weights = MoeWeights::random(&config, 42);
    // Zero out the bias.
    for b in &mut weights.e_score_correction_bias {
        *b = 0.0;
    }
    let mut scratch = MoeForwardScratch::new(&config);
    let hidden_in: Vec<f32> = (0..config.d()).map(|i| (i as f32) * 0.1 - 0.4).collect();
    let mut f32_out = vec![0.0; config.d()];
    moe_forward_token(&weights, &config, &hidden_in, &mut f32_out, &mut scratch);

    let mut ref_out = vec![0.0f64; config.d()];
    ref_moe_forward_f64(&weights, &config, &hidden_in, &mut ref_out, false);

    let diff = max_abs_diff(&f32_out, &ref_out);
    assert!(
        diff < 1e-4,
        "zero-bias f32 vs f64 max diff = {} (tol 1e-4)",
        diff
    );
}

#[test]
fn g1_nonzero_bias_matches_reference() {
    // Bias ≠ 0 → noaux_tc influences selection. f32 must still match f64.
    let config = tiny_config();
    let weights = MoeWeights::random(&config, 99);
    let mut scratch = MoeForwardScratch::new(&config);
    let hidden_in: Vec<f32> = (0..config.d()).map(|i| (i as f32) * 0.15 - 0.6).collect();
    let mut f32_out = vec![0.0; config.d()];
    moe_forward_token(&weights, &config, &hidden_in, &mut f32_out, &mut scratch);

    let mut ref_out = vec![0.0f64; config.d()];
    ref_moe_forward_f64(&weights, &config, &hidden_in, &mut ref_out, false);

    let diff = max_abs_diff(&f32_out, &ref_out);
    assert!(
        diff < 1e-4,
        "nonzero-bias f32 vs f64 max diff = {} (tol 1e-4)",
        diff
    );
}

#[test]
fn g1_bias_changes_selection() {
    // Prove the bias actually changes WHICH experts are picked.
    let config = tiny_config();
    let mut weights_zero_bias = MoeWeights::random(&config, 7);
    for b in &mut weights_zero_bias.e_score_correction_bias {
        *b = 0.0;
    }
    let mut weights_with_bias = weights_zero_bias.clone();
    // Set a strong bias that flips selection: boost expert 0 + 2, suppress 1 + 3.
    weights_with_bias.e_score_correction_bias = vec![0.9, -0.9, 0.9, -0.9];

    let hidden_in: Vec<f32> = vec![0.3; config.d()];

    let mut scratch = MoeForwardScratch::new(&config);
    let mut out = vec![0.0; config.d()];
    moe_forward_token(&weights_zero_bias, &config, &hidden_in, &mut out, &mut scratch);
    let idx_zero = scratch.topk_indices.clone();

    moe_forward_token(&weights_with_bias, &config, &hidden_in, &mut out, &mut scratch);
    let idx_biased = scratch.topk_indices.clone();

    // With the strong bias, experts {0, 2} should be selected (vs whatever the
    // raw sigmoid router picked under zero bias). We don't assert the exact
    // set under zero bias (depends on RNG), but we DO assert the sets differ.
    assert_ne!(
        idx_zero, idx_biased,
        "bias must change top-K selection — otherwise noaux_tc is a no-op"
    );

    // And under bias, experts 0 + 2 should win (they got +0.9).
    let mut sorted = idx_biased.clone();
    sorted.sort();
    assert_eq!(
        sorted, vec![0, 2],
        "with bias [+0.9, -0.9, +0.9, -0.9], experts {{0, 2}} must be selected, got {:?}",
        idx_biased
    );
}

#[test]
fn g1_bias_does_not_leak_into_renormalization() {
    // THE LOAD-BEARING TEST (Research 328 §7.6).
    //
    // The bias must participate in top-K SELECTION but NOT in renormalization.
    // We implement a "buggy" reference that uses `biased[topk_idx]` in the
    // renorm, run it, assert its output DIFFERS from the correct reference,
    // then assert the f32 impl matches the CORRECT reference.
    //
    // This catches the §2.2 misreading bit-identically: if the f32 impl used
    // biased scores in renorm, it would match the buggy reference, not the
    // correct one.
    let config = tiny_config();
    let weights = MoeWeights::random(&config, 314);
    let hidden_in: Vec<f32> = (0..config.d()).map(|i| (i as f32) * 0.2 - 0.7).collect();

    // f32 impl
    let mut scratch = MoeForwardScratch::new(&config);
    let mut f32_out = vec![0.0; config.d()];
    moe_forward_token(&weights, &config, &hidden_in, &mut f32_out, &mut scratch);

    // Correct f64 reference (raw-score renorm)
    let mut ref_correct = vec![0.0f64; config.d()];
    ref_moe_forward_f64(&weights, &config, &hidden_in, &mut ref_correct, false);

    // Buggy f64 reference (biased-score renorm)
    let mut ref_buggy = vec![0.0f64; config.d()];
    ref_moe_forward_f64(&weights, &config, &hidden_in, &mut ref_buggy, true);

    // (a) The two references must DIFFER — otherwise the test can't discriminate.
    let ref_diff: f64 = ref_correct
        .iter()
        .zip(ref_buggy.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        ref_diff > 1e-5,
        "correct vs buggy reference must differ when bias ≠ 0 (got max diff {})",
        ref_diff
    );

    // (b) The f32 impl must match the CORRECT reference, not the buggy one.
    let diff_correct = max_abs_diff(&f32_out, &ref_correct);
    let diff_buggy = max_abs_diff(&f32_out, &ref_buggy);
    assert!(
        diff_correct < 1e-4,
        "f32 must match CORRECT reference (raw-score renorm); diff = {} (tol 1e-4)",
        diff_correct
    );
    assert!(
        diff_buggy > 1e-5,
        "f32 must NOT match buggy reference (biased-score renorm); diff = {} (should be > 1e-5)",
        diff_buggy
    );
}

#[test]
fn g1_shared_expert_always_on() {
    // Disable all routed experts (zero their weights); the output must still
    // equal the shared-expert forward (proving the shared expert is ungated).
    let config = tiny_config();
    let mut weights = MoeWeights::random(&config, 55);
    for expert in &mut weights.experts {
        for v in &mut expert.gate_proj {
            *v = 0.0;
        }
        for v in &mut expert.up_proj {
            *v = 0.0;
        }
        for v in &mut expert.down_proj {
            *v = 0.0;
        }
    }
    let hidden_in: Vec<f32> = (0..config.d()).map(|i| (i as f32) * 0.1 - 0.3).collect();

    let mut scratch = MoeForwardScratch::new(&config);
    let mut f32_out = vec![0.0; config.d()];
    moe_forward_token(&weights, &config, &hidden_in, &mut f32_out, &mut scratch);

    let mut ref_out = vec![0.0f64; config.d()];
    ref_moe_forward_f64(&weights, &config, &hidden_in, &mut ref_out, false);

    let diff = max_abs_diff(&f32_out, &ref_out);
    assert!(
        diff < 1e-4,
        "shared-expert-only f32 vs f64 max diff = {} (tol 1e-4)",
        diff
    );

    // And the output must be non-zero (the shared expert actually fires).
    let magnitude: f32 = f32_out.iter().map(|v| v.abs()).sum();
    assert!(
        magnitude > 1e-3,
        "shared expert output must be non-zero, got sum |v| = {}",
        magnitude
    );
}

#[test]
fn g1_kimi_k3_0_40b_dims_match_reference() {
    // Full Kimi-K3-0.40B dims: 8 experts, 1 shared, K=2, d=1024, d_ffn=1024.
    // Tolerance 1e-3 (larger dims accumulate more f32 error).
    let config = MoeConfig::kimi_k3_0_40b();
    let weights = MoeWeights::random(&config, 2718);
    let mut scratch = MoeForwardScratch::new(&config);
    let hidden_in = vec![0.05; config.d()];
    let mut f32_out = vec![0.0; config.d()];
    moe_forward_token(&weights, &config, &hidden_in, &mut f32_out, &mut scratch);

    let mut ref_out = vec![0.0f64; config.d()];
    ref_moe_forward_f64(&weights, &config, &hidden_in, &mut ref_out, false);

    let diff = max_abs_diff(&f32_out, &ref_out);
    assert!(
        diff < 1e-3,
        "kimi_k3_0_40b dims f32 vs f64 max diff = {} (tol 1e-3)",
        diff
    );
}

#[test]
fn g1_renormalization_disabled_matches_reference() {
    // When moe_renormalize=false, the f32 impl must use raw sigmoid scores
    // as weights (not renormalized). Verify against the reference.
    let mut config = tiny_config();
    config.renormalize = false;
    let weights = MoeWeights::random(&config, 161);
    let mut scratch = MoeForwardScratch::new(&config);
    let hidden_in: Vec<f32> = (0..config.d()).map(|i| (i as f32) * 0.12).collect();
    let mut f32_out = vec![0.0; config.d()];
    moe_forward_token(&weights, &config, &hidden_in, &mut f32_out, &mut scratch);

    let mut ref_out = vec![0.0f64; config.d()];
    ref_moe_forward_f64(&weights, &config, &hidden_in, &mut ref_out, false);

    let diff = max_abs_diff(&f32_out, &ref_out);
    assert!(
        diff < 1e-4,
        "renormalize=false f32 vs f64 max diff = {} (tol 1e-4)",
        diff
    );
}
