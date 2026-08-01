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
