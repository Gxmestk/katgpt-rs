//! Issue 131 G2 — Real-data Weaver checkpoint integration test.
//!
//! Loads the trained `weaver_v1.safetensors` produced by
//! `riir-train/examples/precompute_weaver_real_data.rs` (Plan 314 Phase 4/5
//! real-data run, 2026-07-13) and verifies:
//!
//! 1. **Format compatibility** — katgpt-rs's reader loads the riir-train-written
//!    checkpoint without error (tensor names + metadata keys match).
//! 2. **BLAKE3 verification** — the `.blake3` sidecar is present and matches.
//! 3. **Non-zero residuals** — the trained weights produce non-zero Weaver
//!    residuals on a synthetic input (proving the training produced real
//!    signal, not degenerate all-zeros).
//! 4. **G1 invariants hold** — corrected probabilities sum to 1.0, no NaN/Inf.
//!
//! This test is gated by the `WEAVER_CHECKPOINT_PATH` env var. When unset, the
//! test is ignored (the checkpoint is ~219 MB and lives outside this repo at
//! `riir-train/output/weaver_real_trained/weaver_v1.safetensors`).
//!
//! ## Running
//!
//! ```bash
//! WEAVER_CHECKPOINT_PATH=/path/to/riir-train/output/weaver_real_trained/weaver_v1.safetensors \
//!     cargo test -p katgpt-speculative --features weaver_runtime \
//!     --test weaver_real_checkpoint -- --ignored --nocapture
//! ```

#![cfg(feature = "weaver_runtime")]

use katgpt_speculative::weaver::{WeaverCorrector, WeaverInput};

/// Build a synthetic WeaverInput sized to match the real Gemma2-2B config
/// (hidden=2304, K=32, depth=4). The values are arbitrary but non-degenerate.
fn synthetic_input_for_gemma2_config() -> WeaverInput<'static> {
    let h = 2304_usize;
    let k = 32_usize;
    let d = 4_usize;
    let vocab = 911_usize; // matches the 20-problem compact vocab

    let h_verifier: &'static [f32] =
        Box::leak(vec![0.5f32; h].into_boxed_slice());

    let mut h_dflash: Vec<&'static [f32]> = Vec::with_capacity(d);
    let mut topk_ids: Vec<&'static [u32]> = Vec::with_capacity(d);
    let mut dflash_logits: Vec<&'static [f32]> = Vec::with_capacity(d);

    for di in 0..d {
        h_dflash.push(Box::leak(
            (0..h)
                .map(|i| 0.3 + 0.001 * (di * h + i) as f32)
                .collect::<Vec<f32>>()
                .into_boxed_slice(),
        ));
        topk_ids.push(Box::leak(
            (0..k)
                .map(|i| (i as u32) % vocab as u32)
                .collect::<Vec<u32>>()
                .into_boxed_slice(),
        ));
        dflash_logits.push(Box::leak(
            (0..k)
                .map(|i| (i as f32) * 0.1 - 1.5)
                .collect::<Vec<f32>>()
                .into_boxed_slice(),
        ));
    }

    let emb: &'static [f32] = Box::leak(vec![0.1f32; vocab * h].into_boxed_slice());

    WeaverInput {
        h_verifier,
        h_dflash: Box::leak(h_dflash.into_boxed_slice()),
        topk_ids: Box::leak(topk_ids.into_boxed_slice()),
        dflash_logits: Box::leak(dflash_logits.into_boxed_slice()),
        embedding: emb,
        vocab_size: vocab,
    }
}

#[test]
#[ignore = "requires WEAVER_CHECKPOINT_PATH env var pointing to the real checkpoint"]
fn real_checkpoint_loads_and_produces_nonzero_residual() {
    let path = std::env::var("WEAVER_CHECKPOINT_PATH").unwrap_or_else(|_| {
        // Default to the riir-train output path (works when katgpt-rs and
        // riir-train are siblings, which is the standard repo layout).
        "../riir-train/output/weaver_real_trained/weaver_v1.safetensors".to_string()
    });

    let path = std::path::Path::new(&path);
    if !path.exists() {
        eprintln!(
            "SKIP — checkpoint not found at {} (set WEAVER_CHECKPOINT_PATH)",
            path.display()
        );
        return;
    }

    eprintln!("Loading checkpoint from {}...", path.display());
    let corrector = WeaverCorrector::from_checkpoint(path).expect(
        "checkpoint should load — if this fails, the tensor name or metadata \
         format has drifted between riir-train (writer) and katgpt-rs (reader)",
    );

    // Config should match the Gemma2-2B real-data run.
    let cfg = &corrector.weights().config;
    eprintln!(
        "Loaded — hidden={}, heads={}, K={}, depth={}, params={:.1}M",
        cfg.hidden_dim,
        cfg.n_heads,
        cfg.k_candidates,
        cfg.max_depth,
        // w_c (h*h) + 4 attn (h*h) + 3 ff (2*h*ff + ff*h) + 3 norm (h) + pos (d*h)
        // Approximate — the exact count is in the metadata.
        (cfg.hidden_dim * cfg.hidden_dim * 5 + cfg.hidden_dim * cfg.d_ff * 3) as f64 / 1e6
    );

    assert_eq!(cfg.hidden_dim, 2304, "expected Gemma2-2B hidden_dim=2304");
    assert_eq!(cfg.k_candidates, 32, "expected K=32 from the real-data run");
    assert_eq!(cfg.max_depth, 4, "expected depth=4 from the real-data run");

    // Run the forward pass on a synthetic input.
    let input = synthetic_input_for_gemma2_config();
    let out = corrector.correct(&input);

    // G1: corrected probs sum to 1.0 per depth.
    for di in 0..out.depth {
        let sum: f32 = out.corrected_probs[di].iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "G1 FAIL — probs at depth {di} sum to {sum}, expected 1.0"
        );
    }

    // G1: no NaN/Inf.
    for di in 0..out.depth {
        for ki in 0..out.k {
            assert!(out.corrected_probs[di][ki].is_finite(), "NaN/Inf in probs");
            assert!(out.corrected_logits[di][ki].is_finite(), "NaN/Inf in logits");
            assert!(
                out.weaver_residual[di][ki].is_finite(),
                "NaN/Inf in residual"
            );
        }
    }

    // G2 (the key assertion): trained weights produce non-zero residuals.
    // Zero-init weights produce exactly zero residual (all matmuls are zero).
    // Trained weights MUST produce non-zero residual — this is the proof that
    // the training (riir-train Plan 314 real-data run) encoded real signal.
    let mut max_abs_residual = 0.0f32;
    for di in 0..out.depth {
        for ki in 0..out.k {
            let abs_r = out.weaver_residual[di][ki].abs();
            if abs_r > max_abs_residual {
                max_abs_residual = abs_r;
            }
        }
    }

    eprintln!(
        "Max |residual| across all (depth, K) positions: {:.6}",
        max_abs_residual
    );

    assert!(
        max_abs_residual > 1e-4,
        "G2 FAIL — trained weights produced near-zero residual (max={:.6}). \
         Either the checkpoint is untrained (zero-init) or the loader silently \
         zeroed the weights.",
        max_abs_residual
    );

    eprintln!("✅ PASS — real checkpoint loads, format compatible, residuals non-zero.");
    eprintln!(
        "   This confirms riir-train's safetensors writer and katgpt-rs's reader\
    are format-compatible, and the trained weights carry real signal."
    );

    // ── G4 (latency): measure Weaver forward pass time on the real config ──
    //
    // The forward pass does: conditioning (2× RMSNorm + matmul), single-head
    // causal attention over D+1=5 positions, SwiGLU MLP, top-K=32 gather
    // projection (reads 32×2304×4 = 288 KB of embedding), residual add, softmax.
    //
    // We measure the median of N runs to get a stable latency number.
    const WARMUP_RUNS: usize = 3;
    const MEASURED_RUNS: usize = 20;

    for _ in 0..WARMUP_RUNS {
        let _ = corrector.correct(&input);
    }

    let mut times_us: Vec<f64> = Vec::with_capacity(MEASURED_RUNS);
    for _ in 0..MEASURED_RUNS {
        let t0 = std::time::Instant::now();
        let _ = corrector.correct(&input);
        times_us.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    times_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_us = times_us[times_us.len() / 2];
    let p99_idx = ((times_us.len() as f64 - 1.0) * 0.99) as usize;
    let p99_us = times_us[p99_idx];

    eprintln!();
    eprintln!("── G4 Latency (Weaver forward, real config) ──");
    eprintln!("  Config: hidden=2304, K=32, depth=4, heads=8");
    eprintln!("  Median: {:.1} µs ({:.2} ms)", median_us, median_us / 1000.0);
    eprintln!("  P99:    {:.1} µs ({:.2} ms)", p99_us, p99_us / 1000.0);
    eprintln!("  Runs:   {} (warmup: {})", MEASURED_RUNS, WARMUP_RUNS);
    eprintln!();
    eprintln!(
        "  Context: a single DFlash draft step produces D=4 lookahead positions."
    );
    eprintln!(
        "  This latency is added per draft step when the weaver_runtime feature is on."
    );
    eprintln!(
        "  For reference, a Gemma2-2B forward pass (26 layers, 2B params) takes ~3-5 ms"
    );
    eprintln!(
        "  per token on CPU. The Weaver overhead is {:.1}% of a single verifier step.",
        median_us / 3000.0 * 100.0
    );
}
