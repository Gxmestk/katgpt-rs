# katgpt-quant

Quantization codecs — TurboQuant, PlanarQuant, IsoQuant, OCTOPUS, Hybrid
OCT-PQ. KV cache compression primitives shared across the workspace.
Extracted from `katgpt-rs/src/` (Proposal 003 Phase 1).

## Overview

Five codecs covering different points on the speed/quality/MSE trade-off for
KV-cache compression:

- **TurboQuant** (`turboquant`) — random rotation + uniform codebook. The
  legacy baseline.
- **PlanarQuant** (`planar_quant`) — 2D Givens rotation. `O(d)` vs TQ `O(d²)`.
- **IsoQuant** (`iso_quant`) — 4D quaternion rotation. `O(d)`, 512 FMAs for
  `d=128`.
- **OCTOPUS** (`octopus`) — octahedral triplet codec. Data-oblivious,
  dominates SQ.
- **Hybrid OCT-PQ** (`hybrid_oct_pq`) — OCT encoding + PQ rotation.

All codecs depend on `katgpt-core` for SIMD kernels + shared types. The
inter-codec dependency chain is:

```text
turboquant (base) ← planar_quant, iso_quant
octopus (standalone)
hybrid_oct_pq (planar_quant + octopus)
```

## Feature flags

`default = []`. Each codec is an opt-in feature, mirroring the historical root
feature surface.

| Feature | Description |
|---|---|
| `turboquant` | Random rotation + uniform codebook (baseline). |
| `planar_quant` | 2D Givens rotation codec. Implies `turboquant`. |
| `iso_quant` | 4D quaternion rotation codec. Implies `turboquant`. |
| `octopus` | Octahedral triplet codec (standalone). |
| `hybrid_oct_pq` | OCT + PQ hybrid. Implies `planar_quant` + `octopus`. |
| `asymmetric_kv` | Asymmetric K/V cache config helper (Research 081). Gates `TurboQuantKVCache::new_asymmetric`. Implies `turboquant`. |
| `maxsim` | MaxSim late-interaction scoring (composes with `turboquant` + `octopus`). |

## Dependencies

- `katgpt-core` — SIMD kernels, shared types.
- `katgpt-transformer` *(dev-only)* — `TransformerWeights::new(...)` for the
  `turboquant/forward.rs` end-to-end tests.

## License

MIT. Part of the [katgpt-rs](https://github.com/katopz/katgpt-rs) project.
