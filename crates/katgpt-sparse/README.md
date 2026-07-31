# katgpt-sparse

Sparse task-vector family — SOPTV / SPLAT (Plan 264 / 265). Sparse
Off-Principal Task Vector storage + specialist latent projection. Extracted
from `katgpt-rs/src/{sparse_task_vector, specialist_projection}.rs` per
Proposal 003 Phase 11 (2026-07-04).

## Overview

Two siblings:

- `sparse_task_vector` — Sparse Off-Principal Task Vector (SOPTV) storage.
  OPD-grounded sparse delta format. The cluster's foundation.
  **DEFAULT-ON** in root (Plan 264 Phase 1, Research 231, GOAT G1–G2 PASS:
  2.9–5.7× storage reduction).
- `specialist_projection` — SPLAT specialist latent projection (Plan 265
  Phase 2). Consumes `sparse_task_vector::SparseTaskVector` (intra-crate)
  and `katgpt_band::band_conditioner::ComputeTarget` (cross-crate).
  **DEFAULT-ON** (T5.3, 2026-07-02): G4–G6 pass.

The cross-crate edge is clean: sparse depends on band (for the `ComputeTarget`
5-variant enum), never the reverse.

## Key types / modules

- `sparse_task_vector` — `SparseTaskVector`, sparse delta storage, OPG-grounded
  composition, gauge-invariant composition (Plan 270).
- `specialist_projection` — SPLAT specialist latent projection.

## Feature flags

`default = ["sparse_task_vector", "specialist_projection"]`.

| Feature | Default | Description |
|---|---|---|
| `sparse_task_vector` | yes | SOPTV sparse delta storage (Plan 264 Phase 1, GOAT G1–G2 PASS 2.9–5.7× storage reduction). |
| `specialist_projection` | yes | SPLAT specialist latent projection (Plan 265 Phase 2, G4–G6 pass). Implies `sparse_task_vector` + `katgpt-band/band_conditioner`. |
| `gauge_invariant` | no | Gauge-invariant adapter composition (Plan 270, LoRA-Muon distillation). Gates the `compose_gauge_invariant` impl block on `SparseTaskVector`. Self-contained — the parity test uses `katgpt-spectral` (dev-dep only). |

## Dependencies

- `katgpt-core` — `simd::simd_sum_sq` (sparse_task_vector relative_norm_vs)
  and `sigmoid` (specialist_projection). Always-on.
- `katgpt-band` *(optional)* — `band_conditioner::ComputeTarget` consumed by
  `specialist_projection`. Pulled in by the `specialist_projection` feature.
- `katgpt-spectral` *(dev-only)* — gauge-invariant parity test in
  `sparse_task_vector.rs` (`test_compose_gauge_invariant_matches_full_compose`).
  Production `compose_gauge_invariant` is self-contained.
- `fastrand` *(dev-only)* — deterministic RNG for sparse_task_vector unit
  tests.

## License

MIT. Part of the [katgpt-rs](https://github.com/katopz/katgpt-rs) project.
