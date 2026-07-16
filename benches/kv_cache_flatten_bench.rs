//! KV Cache Flatten Benchmark — init latency + hot-path store/dequant latency.
//!
//! Measures the cumulative impact of the 8 KV cache codec flattenings
//! (commits 83b6221b + a834a779 + dfdb09d1):
//!   1. turboquant       (katgpt-quant)
//!   2. planar_quant     (katgpt-quant)
//!   3. iso_quant        (katgpt-quant)
//!   4. octopus          (katgpt-quant)
//!   5. hybrid_oct_pq    (katgpt-quant)
//!   6. kvarn            (katgpt-kv)
//!   7. spectral_kv_cache (katgpt-spectral)
//!   8. shard_kv         (katgpt-kv)
//!
//! Uses `std::time::Instant` + `harness = false` (repo convention — criterion
//! is not a katgpt-rs dev-dep; see `benches/faithfulness_probe_bench.rs`).
//!
//! ## A/B Comparison Methodology
//!
//! ```bash
//! # Run on develop (new flattened code):
//! cargo run --release --bench kv_cache_flatten_bench \
//!   --features "turboquant planar_quant iso_quant octopus hybrid_oct_pq kvarn spectral_quant shard_kv" \
//!   > /tmp/kv_bench_develop.txt
//!
//! # Run on origin/develop (old Vec<Vec<Vec<...>>> code):
//! git worktree add /tmp/katgpt-rs-origin origin/develop
//! cp benches/kv_cache_flatten_bench.rs /tmp/katgpt-rs-origin/benches/
//! cd /tmp/katgpt-rs-origin
//! # add the [[bench]] entry to Cargo.toml (see root Cargo.toml for the entry)
//! cargo run --release --bench kv_cache_flatten_bench \
//!   --features "turboquant planar_quant iso_quant octopus hybrid_oct_pq kvarn spectral_quant shard_kv" \
//!   > /tmp/kv_bench_origin.txt
//!
//! diff /tmp/kv_bench_origin.txt /tmp/kv_bench_develop.txt
//! ```

#![cfg(feature = "turboquant")]

use std::time::Instant;

// ─── Config scale presets ─────────────────────────────────────────────────────

struct Scale {
    name: &'static str,
    n_layers: usize,
    kv_dim: usize,
    max_seq_len: usize,
}

const SMALL: Scale = Scale {
    name: "small (L4 D64 S256)",
    n_layers: 4,
    kv_dim: 64,
    max_seq_len: 256,
};

const MEDIUM: Scale = Scale {
    name: "medium (L12 D128 S2048)",
    n_layers: 12,
    kv_dim: 128,
    max_seq_len: 2048,
};

const LARGE: Scale = Scale {
    name: "large (L32 D256 S4096)",
    n_layers: 32,
    kv_dim: 256,
    max_seq_len: 4096,
};

// ─── Deterministic data generation ────────────────────────────────────────────

fn gen_vector(kv_dim: usize, pos: usize, salt: u64) -> Vec<f32> {
    let mut state = (pos as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(salt);
    let mut v = Vec::with_capacity(kv_dim);
    for _ in 0..kv_dim {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let f = ((state >> 33) as f32) / (1u64 << 31) as f32 - 0.5;
        v.push(f * 2.0);
    }
    v
}

// ─── Result type ──────────────────────────────────────────────────────────────

const STORE_POSITIONS: usize = 64;
const LATENCY_ITERS: usize = 1000;

struct CodecResult {
    name: &'static str,
    init_us: f64,
    store_ns: f64,
    dequant_ns: f64,
}

impl CodecResult {
    fn print(&self) {
        println!(
            "  {:<22} | init={:>10.1}µs | store={:>8.0}ns/pos | dequant={:>8.0}ns/pos",
            self.name, self.init_us, self.store_ns, self.dequant_ns
        );
    }
}

/// Generic benchmark loop: construct cache, measure init latency, then measure
/// store_key + store_value and dequantize_key_into + dequantize_value_into
/// latency per position on layer 0.
///
/// `$make` — expression producing a fresh cache each call
/// `$kv_dim` — the kv_dim (for data generation)
/// `$store_key`, `$store_val`, `$dequant_key`, `$dequant_val` — method paths
macro_rules! bench_codec {
    ($name:expr, $make:expr, $kv_dim:expr, $store_key:path, $store_val:path, $dequant_key:path, $dequant_val:path) => {{
        let kv_dim = $kv_dim;

        // ── Init latency (average of 3 constructions) ──
        let mut init_times: [f64; 3] = [0.0; 3];
        for t in &mut init_times {
            let start = Instant::now();
            let cache = $make;
            *t = start.elapsed().as_secs_f64() * 1e6;
            drop(cache);
        }
        let init_us = init_times.iter().sum::<f64>() / init_times.len() as f64;

        // ── Construct for latency measurement ──
        let mut cache = $make;

        // Pre-generate deterministic data
        let keys: Vec<Vec<f32>> = (0..STORE_POSITIONS)
            .map(|p| gen_vector(kv_dim, p, 0x4B45_5900))
            .collect();
        let vals: Vec<Vec<f32>> = (0..STORE_POSITIONS)
            .map(|p| gen_vector(kv_dim, p, 0x5641_4C00))
            .collect();
        let mut key_out = vec![0.0f32; kv_dim];
        let mut val_out = vec![0.0f32; kv_dim];

        // ── Warm up ──
        for p in 0..8.min(STORE_POSITIONS) {
            $store_key(&mut cache, 0, p, &keys[p]);
            $store_val(&mut cache, 0, p, &vals[p]);
        }
        for p in 0..8.min(STORE_POSITIONS) {
            $dequant_key(&mut cache, 0, p, &mut key_out);
            $dequant_val(&mut cache, 0, p, &mut val_out);
        }

        // ── Store latency ──
        let store_start = Instant::now();
        for _ in 0..LATENCY_ITERS {
            for p in 0..STORE_POSITIONS {
                $store_key(&mut cache, 0, p, &keys[p]);
                $store_val(&mut cache, 0, p, &vals[p]);
            }
        }
        let store_ns = store_start.elapsed().as_nanos() as f64
            / (LATENCY_ITERS * STORE_POSITIONS) as f64;

        // ── Dequant latency ──
        let dequant_start = Instant::now();
        for _ in 0..LATENCY_ITERS {
            for p in 0..STORE_POSITIONS {
                $dequant_key(&mut cache, 0, p, &mut key_out);
                $dequant_val(&mut cache, 0, p, &mut val_out);
            }
        }
        let dequant_ns = dequant_start.elapsed().as_nanos() as f64
            / (LATENCY_ITERS * STORE_POSITIONS) as f64;

        CodecResult {
            name: $name,
            init_us,
            store_ns,
            dequant_ns,
        }
    }};
}

// ─── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  KV Cache Flatten Benchmark — Init + Hot-Path Latency");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    for scale in [&SMALL, &MEDIUM, &LARGE] {
        let Scale { name, n_layers, kv_dim, max_seq_len } = *scale;

        println!("┌─ {} ", name);
        println!("│ n_layers={}, kv_dim={}, max_seq_len={}", n_layers, kv_dim, max_seq_len);
        println!("│ store/dequant: {} positions, layer 0, {} iters", STORE_POSITIONS, LATENCY_ITERS);
        println!("├──────────────────────────────────────────────────────────────────");

        let mut results: Vec<CodecResult> = Vec::new();

        // ── turboquant ──
        #[cfg(feature = "turboquant")]
        {
            use katgpt_quant::turboquant::{TurboQuantKVCache, TurboQuantKVCacheConfig};
            results.push(bench_codec!(
                "turboquant",
                TurboQuantKVCache::with_config(&TurboQuantKVCacheConfig {
                    n_layers, kv_dim, max_seq_len, seed: 42, key_bits: 3, val_bits: 3,
                }),
                kv_dim,
                TurboQuantKVCache::store_key,
                TurboQuantKVCache::store_value,
                TurboQuantKVCache::dequantize_key_into,
                TurboQuantKVCache::dequantize_value_into
            ));
        }

        // ── planar_quant ──
        #[cfg(feature = "planar_quant")]
        {
            use katgpt_quant::planar_quant::{PlanarQuantKVCache, PlanarQuantConfig};
            results.push(bench_codec!(
                "planar_quant",
                PlanarQuantKVCache::with_config(&PlanarQuantConfig {
                    n_layers, kv_dim, max_seq_len, seed: 42, key_bits: 3, val_bits: 3,
                }),
                kv_dim,
                PlanarQuantKVCache::store_key,
                PlanarQuantKVCache::store_value,
                PlanarQuantKVCache::dequantize_key_into,
                PlanarQuantKVCache::dequantize_value_into
            ));
        }

        // ── iso_quant ──
        #[cfg(feature = "iso_quant")]
        {
            use katgpt_quant::iso_quant::{IsoQuantKVCache, IsoQuantConfig, IsoQuantMode};
            results.push(bench_codec!(
                "iso_quant",
                IsoQuantKVCache::new(&IsoQuantConfig {
                    n_layers, kv_dim, max_seq_len, seed: 42,
                    mode: IsoQuantMode::Full, key_bits: 3, val_bits: 3,
                }),
                kv_dim,
                IsoQuantKVCache::store_key,
                IsoQuantKVCache::store_value,
                IsoQuantKVCache::dequantize_key_into,
                IsoQuantKVCache::dequantize_value_into
            ));
        }

        // ── octopus ──
        #[cfg(feature = "octopus")]
        {
            use katgpt_quant::octopus::{OctopusKVCache, OctopusConfig};
            results.push(bench_codec!(
                "octopus",
                OctopusKVCache::with_config(&OctopusConfig {
                    seed: 42, n_layers, kv_dim, max_seq_len,
                    val_bits: 3, key_bits: 3,
                    use_qjl_residual: false, use_joint_rounding: true,
                }),
                kv_dim,
                OctopusKVCache::store_key,
                OctopusKVCache::store_value,
                OctopusKVCache::dequantize_key_into,
                OctopusKVCache::dequantize_value_into
            ));
        }

        // ── hybrid_oct_pq ──
        #[cfg(feature = "hybrid_oct_pq")]
        {
            use katgpt_quant::hybrid_oct_pq::{HybridOctPqKVCache, HybridOctPqConfig};
            results.push(bench_codec!(
                "hybrid_oct_pq",
                HybridOctPqKVCache::with_config(&HybridOctPqConfig {
                    seed: 42, n_layers, kv_dim, max_seq_len,
                    key_bits: 3, val_bits: 3, use_joint_rounding: true,
                }),
                kv_dim,
                HybridOctPqKVCache::store_key,
                HybridOctPqKVCache::store_value,
                HybridOctPqKVCache::dequantize_key_into,
                HybridOctPqKVCache::dequantize_value_into
            ));
        }

        // ── kvarn ──
        #[cfg(feature = "kvarn")]
        {
            use katgpt_kv::kvarn::kv_cache::{KVarNConfig, KVarNKVCache};
            use katgpt_kv::kvarn::var_norm::VarNormConfig;
            results.push(bench_codec!(
                "kvarn",
                KVarNKVCache::with_config(&KVarNConfig {
                    n_layers, kv_dim, max_seq_len, bits: 3, tile_size: 128,
                    var_norm: VarNormConfig::default(), hadamard: false,
                    #[cfg(feature = "targeted_precision")]
                    precision_budget: None,
                }),
                kv_dim,
                KVarNKVCache::store_key,
                KVarNKVCache::store_value,
                KVarNKVCache::dequantize_key_into,
                KVarNKVCache::dequantize_value_into
            ));
        }

        // ── spectral_kv_cache ──
        #[cfg(feature = "spectral_quant")]
        {
            use katgpt_spectral::spectral_kv_cache::SpectralQuantKVCache;
            use katgpt_spectral::types::SpectralQuantKVCacheConfig;

            let sq_config = SpectralQuantKVCacheConfig {
                seed: 42, n_layers, kv_dim, max_seq_len,
                lloyd_max_iter: 20, calibration_samples: 64, qjl_dim: 16,
                avg_bits: 3.0, min_tail_bits: 1, max_bits: 8,
                wf_min_bits: 1, wf_max_bits: 6, use_water_fill: true,
            };
            let key_samples: Vec<Vec<f32>> = (0..64)
                .map(|i| gen_vector(kv_dim, i, 0x5351_5F00))
                .collect();
            let val_samples = key_samples.clone();

            results.push(bench_codec!(
                "spectral_kv_cache",
                SpectralQuantKVCache::from_keys(&sq_config, &key_samples, &val_samples),
                kv_dim,
                SpectralQuantKVCache::store_key,
                SpectralQuantKVCache::store_value,
                SpectralQuantKVCache::dequantize_key_into,
                SpectralQuantKVCache::dequantize_value_into
            ));
        }

        // ── shard_kv ──
        #[cfg(feature = "shard_kv")]
        {
            use katgpt_kv::shard_kv::{ShardCalibration, ShardConfig, ShardKVCache};
            use katgpt_spectral::spectral::participation_ratio;

            let config = ShardConfig {
                avg_bits_k: 4.0, avg_bits_v: 2.0, min_tail_bits: 1, max_bits: 8,
                n_layers, kv_dim, head_dim: kv_dim, max_seq_len,
                sink_tokens: 4, window_tokens: 4, seed: 42,
                v_vq_group_size: 4, v_vq_codebook_size: 256, decode_stream_bits: 8,
            };
            let head_dim = kv_dim;
            let mut eigenvectors = vec![0.0f32; head_dim * head_dim];
            for i in 0..head_dim {
                eigenvectors[i * head_dim + i] = 1.0;
            }
            let eigenvalues: Vec<f32> = (0..head_dim)
                .map(|i| 10.0 * 0.8f32.powi(i as i32))
                .collect();
            let d_eff = participation_ratio(&eigenvalues);
            let cal = ShardCalibration {
                k_eigenvectors: eigenvectors, k_eigenvalues: eigenvalues,
                k_d_eff: d_eff, head_dim,
            };

            results.push(bench_codec!(
                "shard_kv",
                ShardKVCache::from_calibration(&config, &(0..n_layers).map(|_| cal.clone()).collect::<Vec<_>>()),
                kv_dim,
                ShardKVCache::store_key,
                ShardKVCache::store_value,
                ShardKVCache::dequantize_key_into,
                ShardKVCache::dequantize_value_into
            ));
        }

        for r in &results {
            r.print();
        }
        println!("└──────────────────────────────────────────────────────────────────");
        println!();
    }

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Compare against origin/develop:");
    println!("    git worktree add /tmp/katgpt-rs-origin origin/develop");
    println!("    cp benches/kv_cache_flatten_bench.rs /tmp/katgpt-rs-origin/benches/");
    println!("    # add [[bench]] entry to worktree Cargo.toml");
    println!("═══════════════════════════════════════════════════════════════════");
}
