//! Forward-path profiling harness — isolates per-phase cost without a sampling
//! profiler.
//!
//! # Method
//!
//! The forward pass has two cost regimes per decoded token:
//!   - **O(1) per token**: embedding lookup, QKV matmuls, output projection,
//!     MLP matmuls, lm_head matmul. These are independent of how many tokens
//!     are already in the KV cache.
//!   - **O(seq) per token**: the attention head — score computation,
//!     softmax, and value accumulation each scan over all previous tokens.
//!
//! By measuring the full `forward()` call at position 0 (seq_len=1) and at
//! position N-1 (seq_len=N), the difference isolates the attention cost.
//! The position-0 cost isolates the matmul-dominated cost.
//!
//! This is a *relative* profiler — it tells you whether attention or matmuls
//! dominate, and how attention scales with seq_len. Absolute numbers depend
//! on the config and host.
//!
//! # Run
//!
//! ```bash
//! cargo test -p katgpt-forward --test prof_forward --release -- --ignored --nocapture
//! ```
//!
//! `--release` is required: debug builds measure debug-codegen overhead, not
//! the production hot path. `--ignored` because perf tests don't run in CI.

#![allow(clippy::cast_precision_loss)]

use katgpt_forward::{ForwardContext, forward};
use katgpt_transformer::{MultiLayerKVCache, TransformerWeights};
use katgpt_types::Config;

/// A medium config that stresses matmuls meaningfully (n_embd=128, 4 layers,
/// 4 heads, head_dim=32, vocab=256, mlp_hidden=512) while staying fast enough
/// to iterate on. This is NOT a built-in Config preset — it's a local
/// profiling config. Derived from `Config::micro()` with larger dims.
fn medium_config() -> Config {
    let mut c = Config::micro();
    c.vocab_size = 256;
    c.block_size = 128;
    c.n_embd = 128;
    c.n_head = 4;
    c.head_dim = 32;
    c.mlp_hidden = 512;
    c.n_layer = 4;
    c.n_kv_head = 4; // MHA (no GQA)
    c
}

/// How many iterations to run per measurement. Tuned so each sub-test takes
/// ~0.5–2 s in release mode on the medium config.
const ITERS: usize = 5_000;

fn black_box<T>(x: T) -> T {
    std::hint::black_box(x)
}

/// Run `forward()` once at each position in `0..seq_len`, repeating the whole
/// loop `iters` times. Returns total elapsed.
fn run_decode_loop(
    config: &Config,
    seq_len: usize,
    iters: usize,
) -> std::time::Duration {
    let mut rng = katgpt_types::Rng::new(42);
    let weights = TransformerWeights::new(config, &mut rng);
    let mut ctx = ForwardContext::new(config);
    let mut cache = MultiLayerKVCache::new(config);

    // Warmup: one full decode loop to populate caches / branch predictors.
    for pos in 0..seq_len {
        let token = pos % config.vocab_size;
        let _ = forward(&mut ctx, &weights, &mut cache, token, pos, config);
    }

    let start = std::time::Instant::now();
    for _ in 0..iters {
        // Reset the cache position so each iteration decodes the same sequence.
        // We don't rebuild ctx/cache — they're pre-allocated and just get
        // overwritten. advance_pos inside forward() handles the cursor.
        for pos in 0..seq_len {
            let token = pos % config.vocab_size;
            let logits = forward(&mut ctx, &weights, &mut cache, token, pos, config);
            black_box(logits.as_ptr());
        }
    }
    start.elapsed()
}

#[test]
#[ignore]
fn prof_forward_phase_breakdown() {
    let config = medium_config();
    println!();
    println!("═══ Forward-path profiling harness ═══");
    println!(
        "config: n_embd={}, n_layer={}, n_head={}, head_dim={}, vocab={}, mlp_hidden={}",
        config.n_embd,
        config.n_layer,
        config.n_head,
        config.head_dim,
        config.vocab_size,
        config.mlp_hidden
    );
    println!("iters per measurement: {ITERS}");
    println!();

    // Measure a full decode loop at increasing sequence lengths.
    // The per-token cost at seq_len=1 is matmul-dominated; the per-token cost
    // at seq_len=N minus the seq_len=1 cost is the attention cost at depth N.
    for &seq_len in &[1usize, 8, 32, 64, 128] {
        let total = run_decode_loop(&config, seq_len, ITERS);
        let per_iter_ns = total.as_nanos() as f64 / ITERS as f64;
        let per_token_ns = per_iter_ns / seq_len as f64;
        println!(
            "seq_len={seq_len:>3}: {per_token_ns:>10.1} ns/token  ({per_iter_ns:>12.1} ns/iter)",
        );
    }

    println!();
    println!("── Interpretation ──");
    println!("  ns/token at seq_len=1  ≈ matmul-dominated cost (embedding + QKV +");
    println!("                             attn[1] + output proj + MLP + lm_head)");
    println!("  (ns/token at seq_len=N) - (ns/token at seq_len=1) ≈ attention scan cost at depth N");
    println!("  If attention cost grows linearly with N and dominates at large N,");
    println!("  attention is the bottleneck for long contexts. If the seq_len=1");
    println!("  cost dominates, matmuls are the bottleneck (weight-bandwidth-bound).");
}
