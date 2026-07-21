# katgpt-spectral

Spectral quantization substrate — calibrated eigenbasis KV cache compression,
Lloyd-Max / water-fill bit allocation, outlier-aware guard. Shared by
`katgpt-kv` (ShardKV, KVarN) and non-KV consumers (`funcattn_compose`,
`chiaroscuro`, `benchmark`). Spun out of `katgpt-rs/src/spectralquant/`
(Issue 015 Phase 2).

## Overview

Calibrated eigenbasis KV cache compression (Plan 078):

- **Offline calibration**: covariance → eigendecomposition → eigenbasis.
- **Two-regime allocation**: semantic (high-energy) + tail dimensions.
- **Water-fill**: per-dim bit allocation proportional to eigenvalue.
- **Lloyd-Max**: optimal non-uniform scalar quantizer per regime.

Compresses KV cache from f32 to ~3 bits/coordinate with minimal MSE.

### Why a separate crate (Issue 015 Phase 2)

The substrate has both KV consumers (`shard_kv`, `kvarn`) and non-KV consumers
(`funcattn_compose`, `chiaroscuro`, `benchmark/infrastructure`). Folding it
into `katgpt-kv` would have forced non-KV modules to depend on a
`-kv`-named crate — wrong direction. This is a standalone foundational
quantization crate that `katgpt-kv` depends on.

The root `katgpt-rs` crate re-exports this as `katgpt_rs::spectralquant` for
back-compat (Issue 015 Phase 5).

## Key types / modules

- `spectral` — calibrated eigenbasis KV cache compression core.
- `nonuniform_quant` — Lloyd-Max / water-fill per-dim bit allocation.
- `spectral_kv_cache` — `SpectralKVCache` with parallel dequant kernels.
- `spectral_rotation` — random rotation pre-processing (the TurboQuant
  base layer that several codecs share).
- `forward` — forward-pass helpers for spectral-quantized weights.
- `types` — shared spectral types (`OutlierGuardConfig`, calibration structs).
- Phase 4 absorptions (each gates its own feature, see below):
  gauge-invariant composition (`Plan 270`), manifold power-iteration router
  (`Plan 279`), quantile-balance router (`Plan 455`), off-principal retrieval
  (`Plan 264`), spectral budget router (`Plan 253`), orthogonal Procrustes
  (`Issue 001`), PEIRA distill (`Plan 153`).
- Phase 12 absorption: `hla_eigenbasis` — per-NPC HLA windowed eigenbasis
  recovery (`Issue 001`).

## Feature flags

`default = []`. All flags are empty tracking gates mirroring the historical
root feature surface; the root crate forwards them via
`feature = ["katgpt-spectral/feature"]`.

| Feature | Description |
|---|---|
| `spectral_quant` | Core spectral quantization (historical default-on at root). |
| `outlier_guard` | Outlier-aware quantization guard (Plan 224). |
| `stiff_anomaly` | Cross-check with `stiff_anomaly` eigenvalue distribution. |
| `maxsim` | MaxSim late-interaction forward kernel. |
| `dual_gram_pca` | Dual-gram PCA eigenbasis calibration. |
| `turboquant` | TurboQuant random rotation export gate. |
| `gauge_invariant` | Gauge-invariant adapter composition (Plan 270, Research 238). Forwards `katgpt-core/newton_schulz`. |
| `manifold_power_iter_router` | Manifold Power Iteration MoE Router (Plan 279, Research 246). Pulls `blake3`. |
| `quantile_balance_router` | Quantile Balancing MoE Router (Plan 455, Research 447). Promoted to **DEFAULT-ON** at root (2026-07-17). |
| `off_principal_retrieval` | Off-Principal Task Vector Retrieval (Plan 264 Phase 2). Forwards `katgpt-core/newton_schulz` + pulls `blake3` + enables local `newton_schulz`. |
| `spectral_budget` | Spectral Budget Router — layer-adaptive NS depth (Plan 253, Research 222). |
| `orthogonal_procrustes` | Orthogonal Procrustes cross-frame alignment (Issue 001). |
| `river_valley` | River-valley diagnostic metrics (Plan 152, Research 114). |
| `newton_schulz` | Newton-Schulz orthogonalization passthrough. Implied by `gauge_invariant`, `off_principal_retrieval`, `spectral_budget`. |
| `peira_distill` | PEIRA modelless distillation (Plan 153, Research 115). Forwards `katgpt-core/peira_distill`. |
| `hla_eigenbasis_recovery` | Per-NPC HLA windowed eigenbasis recovery (Issue 001). Pulls `uuid` (`Uuid::now_v7()`) + `blake3`. |
| `spectral_rewire` | Spectral Rewiring — weight delta purification via base SVD projection (Plan 423, Research 406). **STAYS OPT-IN** — mechanism gates PASS but the spectral-concentration assumption fails at NPC scale. |

## Dependencies

- `katgpt-core` — SIMD kernels + types re-export.
- `katgpt-types` — shared types (no KV-coupling; `OutlierGuardConfig` is
  re-exported from here via the local `outlier_guard` module).
- `katgpt-transformer` — `TransformerWeights` referenced by `outlier_guard`.
- `rayon`, `half`, `bytemuck`, `serde`, `log`, `fastrand`.
- `blake3` *(optional)* — snapshot hashing in `manifold_power_iter_router`
  + `off_principal`. Gated by those features.
- `uuid` *(optional)* — `Uuid::now_v7()` for `EigenbasisProvenance.window_id`.
  Gated by `hla_eigenbasis_recovery`.

## License

MIT. Part of the [katgpt-rs](https://github.com/katopz/katgpt-rs) project.
