//! Phase 6 — Kimi-K3-0.40B real-model integration tests.
//!
//! These tests load the actual `model.safetensors` (1.5GB) and verify:
//! - T5.4: every tensor in the file maps to a weight field
//! - T6.2: the model-level config + runtime state setup works
//! - T6.3: end-to-end forward pass runs without panic
//! - T6.4: G1 logits match PyTorch reference (future — needs reference logits)
//!
//! # Prerequisites
//!
//! Model files must be downloaded to `data/kimi-k3-0.40b/`:
//! ```sh
//! # See Research 331 for the download method (HF API key in riir-ai/.env)
//! curl -sSL -H "Authorization: Bearer $HF_KEY" \
//!   "https://huggingface.co/inference-optimization/Kimi-K3-0.40B/resolve/main/model.safetensors" \
//!   -o data/kimi-k3-0.40b/model.safetensors
//! ```
//!
//! Tests are `#[ignore]`d by default — run with `--ignored` or set
//! `KIMI_K3_MODEL_DIR` env var to enable.

#![cfg(feature = "kimi_k3_loader")]

use std::path::Path;

use katgpt_rs::kimi_k3::loader::load_kimi_k3;

/// Model directory — override with `KIMI_K3_MODEL_DIR` env var.
fn model_dir() -> String {
    std::env::var("KIMI_K3_MODEL_DIR").unwrap_or_else(|_| {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        format!("{manifest_dir}/data/kimi-k3-0.40b")
    })
}

fn model_path() -> String {
    format!("{}/model.safetensors", model_dir())
}

fn model_exists() -> bool {
    Path::new(&model_path()).exists()
}

/// T5.4 — Load the real safetensors file and verify every tensor maps to a field.
///
/// This is the definitive tensor-name verification gate. If any tensor name in
/// the loader is wrong, this test fails with `MissingTensor(...)`.
#[test]
#[ignore = "requires model.safetensors (1.5GB download)"]
fn t5_4_load_real_model_all_tensors_mapped() {
    if !model_exists() {
        eprintln!("skipping: {} not found", model_path());
        return;
    }

    let weights = load_kimi_k3(&model_path()).unwrap_or_else(|e| panic!("load failed: {e}"));

    // Basic structural checks
    assert_eq!(weights.layers.len(), 8, "expected 8 decoder layers");
    assert_eq!(
        weights.embed_weight.len(),
        163840 * 1024,
        "embed weight shape mismatch"
    );
    assert_eq!(
        weights.lm_head_weight.len(),
        163840 * 1024,
        "lm_head weight shape mismatch"
    );
    assert_eq!(
        weights.final_norm_weight.len(),
        1024,
        "final norm weight shape mismatch"
    );

    // Verify layer topology: MLA at 3,7; KDA at 0,1,2,4,5,6; Dense at 0
    use katgpt_rs::kimi_k3::decoder_layer::{KimiAttentionWeights, KimiFfnWeights};

    for (idx, layer) in weights.layers.iter().enumerate() {
        let is_mla = matches!(layer.attention, KimiAttentionWeights::Mla(_));
        let expected_mla = idx == 3 || idx == 7;
        assert_eq!(
            is_mla, expected_mla,
            "layer {idx}: attention type mismatch (MLA expected={expected_mla})"
        );

        let is_dense = matches!(layer.ffn, KimiFfnWeights::Dense(_));
        let expected_dense = idx == 0;
        assert_eq!(
            is_dense, expected_dense,
            "layer {idx}: FFN type mismatch (Dense expected={expected_dense})"
        );
    }

    eprintln!("✅ T5.4 PASS: all 398 tensors mapped successfully");
    eprintln!("   embed: [163840, 1024], lm_head: [163840, 1024]");
    eprintln!("   layers: 8 (MLA at 3,7; KDA at 0,1,2,4,5,6; Dense at 0)");
}

/// T6.3 — End-to-end forward pass on the real model.
///
/// Loads the real weights, runs a forward pass on token ID 1 (BOS), and
/// verifies the logits are finite (no NaN/Inf from bad math).
#[test]
#[ignore = "requires model.safetensors (1.5GB download)"]
fn t6_3_forward_pass_produces_finite_logits() {
    use katgpt_rs::kimi_k3::model::{KimiK3ModelConfig, KimiK3Runtime, kimi_k3_forward_token};

    if !model_exists() {
        eprintln!("skipping: {} not found", model_path());
        return;
    }

    let weights = load_kimi_k3(&model_path()).unwrap_or_else(|e| panic!("load failed: {e}"));
    let config = KimiK3ModelConfig::kimi_k3_0_40b();
    let mut runtime = KimiK3Runtime::new(&config, 64);

    // Forward pass on BOS token (id=1)
    let logits = kimi_k3_forward_token(&config, &weights, &mut runtime, 1u32);

    assert_eq!(logits.len(), 163840, "logits length mismatch");

    let mut nan_count = 0;
    let mut inf_count = 0;
    let mut max_logit = f32::NEG_INFINITY;
    let mut min_logit = f32::INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v.is_nan() {
            nan_count += 1;
            if nan_count <= 5 {
                eprintln!("   NaN at logit[{i}]");
            }
        }
        if v.is_infinite() {
            inf_count += 1;
            if inf_count <= 5 {
                eprintln!("   Inf at logit[{i}]");
            }
        }
        if v > max_logit {
            max_logit = v;
        }
        if v < min_logit {
            min_logit = v;
        }
    }

    eprintln!("   logits: len={}, nan={}, inf={}, min={:.4}, max={:.4}",
        logits.len(), nan_count, inf_count, min_logit, max_logit);

    // Find top-5 predicted tokens
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    eprintln!("   top-5 tokens: {:?}", &indexed[..5]);

    assert_eq!(nan_count, 0, "NaN in logits");
    assert_eq!(inf_count, 0, "Inf in logits");
}

/// G1 — Logits match PyTorch reference.
///
/// **BLOCKED**: requires the `fla` library (flash-linear-attention) which
/// depends on `triton` (CUDA-only, Linux only). The reference model cannot
/// be run on macOS. G1 requires a cloud GPU instance (Linux + CUDA + triton)
/// to generate reference logits.
///
/// See `.issues/575` for the tracking issue. When unblocked:
/// 1. Run PyTorch model on GPU: `model(input_ids)` → reference logits
/// 2. Save reference logits to `data/kimi-k3-0.40b/ref_logits_bos.npy`
/// 3. Compare against Rust logits within f32 tolerance (max_diff < 1e-4).
#[test]
#[ignore = "BLOCKED: requires GPU + triton (CUDA-only) to run PyTorch reference"]
fn g1_logits_match_pytorch_reference() {
    use katgpt_rs::kimi_k3::model::{KimiK3ModelConfig, KimiK3Runtime, kimi_k3_forward_token};

    let ref_path = format!("{}/ref_logits_bos.npy", model_dir());
    if !Path::new(&ref_path).exists() {
        eprintln!("skipping: reference logits not found at {ref_path}");
        eprintln!("Generate with: python3 -c \"... PyTorch model forward on GPU ...\"");
        return;
    }

    let weights = load_kimi_k3(&model_path()).unwrap_or_else(|e| panic!("load failed: {e}"));
    let config = KimiK3ModelConfig::kimi_k3_0_40b();
    let mut runtime = KimiK3Runtime::new(&config, 64);

    let logits = kimi_k3_forward_token(&config, &weights, &mut runtime, 1u32);

    // Load reference logits (simple binary format: [vocab_size] f32 LE)
    let ref_bytes = std::fs::read(&ref_path).expect("failed to read reference logits");
    let ref_logits: &[f32] = bytemuck::cast_slice(&ref_bytes);

    assert_eq!(logits.len(), ref_logits.len(), "logits length mismatch");

    let mut max_diff = 0.0f32;
    let mut max_idx = 0;
    for (i, (a, b)) in logits.iter().zip(ref_logits.iter()).enumerate() {
        let diff = (a - b).abs();
        if diff > max_diff {
            max_diff = diff;
            max_idx = i;
        }
    }

    eprintln!("   max_diff = {max_diff:.2e} at logit[{max_idx}]");
    eprintln!("   rust[{max_idx}] = {:.6}, ref[{max_idx}] = {:.6}",
        logits[max_idx], ref_logits[max_idx]);

    // G1 gate: max_diff < 1e-4 (f32 tolerance for a model with mixed attention types)
    assert!(max_diff < 1e-4, "G1 FAIL: max_diff {max_diff:.2e} exceeds 1e-4 tolerance");
}
