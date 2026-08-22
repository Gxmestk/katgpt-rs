//! Plan 318 Phase A3 — Kimi-K3-4B-A2B forward pass verification on random init.
//!
//! Verifies that the `KimiK3ModelConfig::kimi_k3_4b_a2b()` config (Issue 388 /
//! Plan 318 — ~4.43B total / ~1.99B active params, 12 layers, 9 KDA + 3 MLA,
//! hidden=3072, 12 routed experts top-4, 2 shared, kv_lora_rank=512) can:
//!
//! 1. Instantiate (config + runtime allocate without panic)
//! 2. Construct a random-init weight tree matching the config
//! 3. Run a single-token forward pass producing the correct logit shape
//!    (163840 = Kimi-K3 vocab) with all finite values (no NaN / Inf)
//!
//! This is the architecture soundness gate — it proves the config composes
//! correctly through the MLA / KDA / MoE / attn-res substrate at 4B scale,
//! without needing real safetensors weights. The 4B target architecture is
//! distinct from the 4B vanilla transformer proven in the prior session
//! (Plan 318 smoke): same param count, but MLA-MoE-KDA instead of QKV-MLP.
//!
//! # Memory
//!
//! The 4B-A2B config is ~17.7 GB of `f32` weights. On M3 Max 64GB unified
//! memory this fits comfortably. On smaller machines the test will OOM —
//! gate it behind a `KIMI_K3_4B_SKIP` env var so CI on constrained runners
//! can skip it without failing.
//!
//! # Run
//!
//! ```sh
//! cargo test --release --features kimi_k3_loader --test kimi_k3_4b_a3_forward_verify -- --nocapture --ignored
//! ```
//!
//! Marked `#[ignore]` because (a) it allocates ~17.7 GB and (b) it's a
//! single-token forward at 4B (slow in debug). Run on demand.

#![cfg(feature = "kimi_k3_loader")]

use katgpt_rs::kimi_k3::loader::KimiK3ModelWeights;
use katgpt_rs::kimi_k3::model::{KimiK3ModelConfig, KimiK3Runtime, kimi_k3_forward_token};

/// Skip when the caller sets `KIMI_K3_4B_SKIP=1` (CI / constrained runners).
fn skip_requested() -> bool {
    std::env::var("KIMI_K3_4B_SKIP").ok().as_deref() == Some("1")
}

/// Count + report approximate param count for the config (informational).
///
/// This is a rough count (embed + lm_head + per-layer substrate) — the exact
/// count requires walking every substrate weight tensor. The point is to
/// sanity-check we're in the 4B ballpark, not to nail the exact figure.
fn approx_param_count(config: &KimiK3ModelConfig) -> usize {
    let d = config.hidden_size;
    let v = config.vocab_size;
    // Embed + LM head (untied)
    let mut n = 2 * v * d;
    // Final norm
    n += d;
    // Per-layer — rough: each KDA layer is ~O(d^2), each MLA layer is ~O(d^2)
    // (the lora ranks keep it bounded), each MoE is ~experts * d * d_ffn.
    // For a ballpark we use d^2 per attention + the MoE expert budget.
    for layer_idx in 0..config.num_layers {
        n += 2 * d; // two norm gammas
        n += 2 * (d * d / 4); // two attn-res (rough: ~d^2/4 each)
        let _ = layer_idx;
    }
    n
}

#[test]
#[ignore]
fn a3_4b_forward_pass_finite_logits() {
    if skip_requested() {
        eprintln!("skipping: KIMI_K3_4B_SKIP=1");
        return;
    }

    let t0 = std::time::Instant::now();

    // ── 1. Config ─────────────────────────────────────────────────────────
    let config = KimiK3ModelConfig::kimi_k3_4b_a2b();
    eprintln!("── Kimi-K3-4B-A2B Phase A3 forward verification ──");
    eprintln!(
        "  config: {} layers (MLA at {:?}), hidden={}, vocab={}",
        config.num_layers, config.mla_layer_indices, config.hidden_size, config.vocab_size,
    );
    eprintln!(
        "  MoE: {} routed (top-{}) + {} shared, moe_intermediate={}",
        config.moe_config.num_experts,
        config.moe_config.num_experts_per_token,
        config.moe_config.num_shared_experts,
        config.moe_config.moe_intermediate_size,
    );
    eprintln!(
        "  MLA: kv_lora={}, q_lora={}, n_heads={}, nope_dim={}, rope_dim={}, v_dim={}",
        config.mla_config.kv_lora_rank,
        config.mla_config.q_lora_rank,
        config.mla_config.n_heads,
        config.mla_config.qk_nope_head_dim,
        config.mla_config.qk_rope_head_dim,
        config.mla_config.v_head_dim,
    );
    eprintln!("  approx params (rough): {}", approx_param_count(&config));

    // ── 2. Random-init weights (~17.7 GB on 4B config) ───────────────────
    eprintln!("  allocating random-init weights (seed=42) ...");
    let t_weights = std::time::Instant::now();
    let weights = KimiK3ModelWeights::random(&config, 42);
    eprintln!(
        "  weights allocated in {:.2?} ({} layers)",
        t_weights.elapsed(),
        weights.layers.len(),
    );

    // ── 3. Runtime (KV caches + scratch + block state) ───────────────────
    // max_seq_len=64 keeps the MLA KV cache tiny for this single-token test.
    // The separate 256K KV cache allocation gate is in the a5 test below.
    eprintln!("  allocating runtime (max_seq_len=64) ...");
    let t_rt = std::time::Instant::now();
    let mut runtime = KimiK3Runtime::new(&config, 64);
    eprintln!("  runtime allocated in {:.2?}", t_rt.elapsed());

    // ── 4. Forward pass on BOS token (id=1) ──────────────────────────────
    eprintln!("  forward pass on token id=1 (BOS) ...");
    let t_fwd = std::time::Instant::now();
    let logits = kimi_k3_forward_token(&config, &weights, &mut runtime, 1u32);
    let fwd_us = t_fwd.elapsed().as_secs_f64() * 1e6;
    eprintln!(
        "  forward pass: {:.0} µs ({:.2} ms)",
        fwd_us,
        fwd_us / 1000.0
    );

    // ── 5. Assertions ────────────────────────────────────────────────────
    // G1 (correctness): logit shape matches vocab_size.
    assert_eq!(
        logits.len(),
        config.vocab_size,
        "logits length = {}, expected {} (vocab_size)",
        logits.len(),
        config.vocab_size,
    );
    assert_eq!(logits.len(), 163840, "Kimi-K3 vocab is 163840");

    // G1 (finiteness): no NaN / Inf in the logits.
    let mut n_finite = 0usize;
    let mut n_nan = 0usize;
    let mut n_inf = 0usize;
    let mut max_abs = 0.0f32;
    for &l in logits {
        if l.is_nan() {
            n_nan += 1;
        } else if l.is_infinite() {
            n_inf += 1;
        } else {
            n_finite += 1;
            max_abs = max_abs.max(l.abs());
        }
    }
    eprintln!("  logits: {n_finite} finite, {n_nan} NaN, {n_inf} Inf, max|logit| = {max_abs:.3}",);
    assert_eq!(n_nan, 0, "found {n_nan} NaN logits — forward diverged");
    assert_eq!(n_inf, 0, "found {n_inf} Inf logits — forward diverged");
    assert_eq!(n_finite, config.vocab_size, "all logits must be finite");

    // Sanity: argmax is a valid token id.
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &l) in logits.iter().enumerate() {
        if l > best_v {
            best_v = l;
            best = i;
        }
    }
    assert!(best < config.vocab_size, "argmax out of range");
    eprintln!("  argmax token: {best} (logit {best_v:.3})");

    eprintln!(
        "  ✅ A3 PASS: 4B-A2B forward produces {} finite logits",
        logits.len()
    );
    eprintln!("  total wall: {:.2?}", t0.elapsed());
}

/// A5: verify the 256K KV cache allocation path doesn't OOM on the 4B config.
///
/// This doesn't run a forward pass at 256K (would take minutes) — it just
/// allocates the runtime with `max_seq_len = 262144` and confirms the MLA KV
/// caches + scratch + block state all fit. The expected MLA KV cache at 256K
/// is ~1.81 GB (3 MLA layers × 576 bytes/token × 262144 tokens).
#[test]
#[ignore]
fn a5_4b_256k_kv_cache_allocates() {
    if skip_requested() {
        eprintln!("skipping: KIMI_K3_4B_SKIP=1");
        return;
    }

    let config = KimiK3ModelConfig::kimi_k3_4b_a2b();
    eprintln!("── Kimi-K3-4B-A2B Phase A5 256K KV cache allocation ──");
    eprintln!(
        "  config: {} layers (MLA at {:?}), hidden={}",
        config.num_layers, config.mla_layer_indices, config.hidden_size,
    );

    // The MLA KV cache at 256K: per MLA layer, kv_lora_rank (512) f32 per
    // token, plus the shared rope key (d_r = 64 f32). 3 MLA layers.
    let per_token_bytes = (config.mla_config.kv_lora_rank + config.mla_config.qk_rope_head_dim) * 4; // f32
    let n_mla = config.mla_layer_indices.len();
    let expected_kv_bytes = per_token_bytes * n_mla * 262144;
    eprintln!(
        "  expected MLA KV cache @ 256K: {:.2} GB ({} MLA layers × {} bytes/token)",
        expected_kv_bytes as f64 / 1e9,
        n_mla,
        per_token_bytes,
    );

    eprintln!("  allocating runtime (max_seq_len=262144) ...");
    let t = std::time::Instant::now();
    // This is the load-bearing line — if the KV cache sizing math is wrong,
    // this will OOM or panic on the allocation.
    let _runtime = KimiK3Runtime::new(&config, 262144);
    eprintln!(
        "  ✅ 256K KV cache allocated in {:.2?} (no OOM)",
        t.elapsed()
    );
}
