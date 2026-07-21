# katgpt-kv

KV-cache namespace — KV-cache compression, compaction, projection-sharing, and
quantization backends. Spun out of `katgpt-rs/src/{kv_share, osc_kv,
cs_kv_probe, shard_kv, sp_kv, still_kv, kvarn, targeted_precision}` (Issue 015
Phase 3) plus Phase 5 absorptions (`cache_prune`, `segment_checkpoint`,
`async_qdq`).

## Overview

All KV-cache backends extracted from the root crate. Each backend is gated by
its historical feature flag, preserving pre-extraction semantics 1:1. The root
`katgpt-rs` crate re-exports each sub-module behind its feature flag as
`katgpt_rs::{kv_share, osc_kv, ...}`, preserving back-compat with all existing
call sites.

## Key types / modules

| Module | Feature | Origin | Plan |
|---|---|---|---|
| `kv_share` | `kv_share` | `src/kv_share.rs` | Plan 185 — Q-K=V projection sharing (50% cache reduction) |
| `osc_kv` | `osc_kv` | `src/osc_kv.rs` | Plan 189 — Oscillatory KV cache, IMEX discretization |
| `cs_kv_probe` | `cs_kv_probe` | `src/cs_kv_probe/` | Plan 280 — Compressed-sensing KV importance probe |
| `shard_kv` | `shard_kv` | `src/shard_kv/` | Plan 147 — ShardKV asymmetric K/V compression |
| `sp_kv` | `sp_kv` | `src/sp_kv/` | Plan 070 — SP-KV self-pruned key-value attention |
| `still_kv` | `still_kv` | `src/still_kv/` | Plan 245 — StillKV perceiver-based compaction |
| `kvarn` | `kvarn` | `src/kvarn/` | Research 159 — KVarN variance-normalized quantization |
| `targeted_precision` | `targeted_precision` | `src/targeted_precision.rs` | Plan 227 Phase 2 — per-head bit allocation |
| `cache_prune` | `cache_prune` | `src/cache_prune/` | Plan 140 — SAT + rolling hash + sensitivity masking |
| `segment_checkpoint` | `segment_checkpoint` | `src/segment_checkpoint/` | Plan 223b — GRM segment caching |
| `async_qdq` | `async_qdq_overlap` | `src/async_qdq.rs` | Plan 227 Phase 6 — double-buffered KV dequantize |

## Feature flags

`default = []`. All flags are empty tracking flags mirroring the historical
root feature surface — they gate `#[cfg(feature = "...")]` branches inside
the code, not external deps. The root crate forwards each one via
`feature = ["katgpt-kv/feature"]`.

| Feature | Description |
|---|---|
| `kv_share` | Q-K=V projection sharing — 50% KV cache reduction (Plan 185). |
| `osc_kv` | Oscillatory KV cache with IMEX discretization (Plan 189 Phase 2). |
| `cs_kv_probe` | CS-KV-Importance Probe + Density-Budget Interpolator (Plan 280). |
| `shard_kv` | ShardKV asymmetric K/V compression (Plan 147, Research 109). Requires `katgpt-spectral/spectral_quant + turboquant` (callers enabling directly MUST also enable those). |
| `sp_kv` | SP-KV self-pruned key-value attention (Plan 070). |
| `still_kv` | StillKV perceiver-based KV cache compaction (Plan 245). |
| `kvarn` | KVarN variance-normalized KV-cache quantization (Research 159). |
| `targeted_precision` | Targeted Precision Budget — per-head bit allocation (Plan 227 Phase 2). |
| `cache_prune` | SAT + rolling hash + sensitivity masking (Plan 140, Research 101). Self-contained. |
| `segment_checkpoint` | GRM segment caching (Plan 223b). DEFAULT-ON in root. |
| `ssc_spec_draft` | SSC Sparse Speculative Drafting — opt-in sub-module of `segment_checkpoint` (Plan 223c). Implies `segment_checkpoint`. |
| `async_qdq_overlap` | Async Q/DQ Overlap — double-buffered KV dequantize for GPU pipeline (Plan 227 Phase 6). |

## Dependencies

- `katgpt-core` — SIMD kernels, `types::*` re-export (`Rng`, `Config`,
  `kv_dim`, `QuantizedKVCache`).
- `katgpt-types` — `QuantizedKVCache` trait (Issue 015 Phase 1).
- `katgpt-spectral` — `spectralquant::*` re-export (shard_kv K-path + kvarn
  via targeted_precision).
- `half`, `bytemuck`, `rayon`, `serde`, `fastrand`, `blake3`.

## License

MIT. Part of the [katgpt-rs](https://github.com/katopz/katgpt-rs) project.
