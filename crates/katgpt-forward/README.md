# katgpt-forward

Forward-pass composition context (`ForwardContext`) — the top-tier join point
above `katgpt-transformer` + `katgpt-pruners` (Issue 007 Phase F). Also hosts
the moved forward trio (`forward`, `forward_base`, `forward_coda`, Plan 385)
and the moved forward-cycle cluster (`drafter_lora`, `dflash`, `verifier`,
`step`, `prefill`, Plan 394).

## Overview

`ForwardContext` is the topmost type in the inference DAG. It composes:

- transformer substrate buffers (`x`, `q`, `k`, `v`, `attn_out`, …) +
  `WallPrefixState` (from `katgpt-transformer`), and
- pruner handles `CnaModulator` / `SubstrateMask` / `HydraSkipPlan` (from
  `katgpt-pruners`, gated by `cna_steering` / `substrate_gate` /
  `hydra_budget`).

It cannot live in `katgpt-transformer` (would force transformer → pruners,
but pruners already depends on transformer → cycle) nor in `katgpt-pruners`
(would invert the layering — a pruner crate shouldn't own forward-pass
buffers). This crate sits ABOVE both, breaking the composition-layer pin so
the 34 composition files (`dash_attn/forward.rs`, `gdn2/forward.rs`,
`hla/forward.rs`, `speculative/*`, `sleep/consolidation.rs`, …) can migrate
out of the root crate into their respective leaves.

Fields are `pub`: `ForwardContext` is a pre-allocated scratch buffer accessed
directly by forward-pass functions.

## Key types / modules

- `ForwardContext` — pre-allocated buffers for zero-alloc forward passes.
  Composes `TransformerWeights`, KV caches, and pruner handles.
- `forward` / `forward_base` / `forward_coda` — the three forward-pass
  composition functions moved from `katgpt-rs/src/transformer.rs` (Plan 385).
  Gated by various tracking features (see below).
- `drafter_lora` — speculative drafter LoRA save/load with BLAKE3 integrity
  envelope (Plan 394).
- `dflash` — DFlash zero-alloc `_with` cores (Issue 013 Phase B). Generic
  over `DflashCtx` + `DflashCache` + `forward_fn` closure.
- `verifier` / `step` / `prefill` — the speculative forward-cycle cluster
  moved from `katgpt-rs/src/speculative/` (Plan 394).

## Feature flags

`default = []`. Each flag is an empty tracking flag here — it gates a
field/impl in this crate. The root crate forwards its feature to this crate
so the cfg blocks resolve when the root feature is enabled. Upstream crates
(`katgpt-transformer`, `katgpt-pruners`) get their own feature forwarded by
the root independently.

| Feature | Description |
|---|---|
| `cna_steering` | CNA contrastive neuron attribution runtime modulator (Plan 087). |
| `sparse_mlp` | Sparse MLP active-indices/value buffers (Plan 022). |
| `substrate_gate` | SubstrateGate per-sequence capability mask for dual sparsity (Plan 216). Implies `sparse_mlp`. |
| `delta_routing` | Delta routing block-delta accumulation (Plan 097). |
| `coda_fusion` | CODA fused kernels: partial RMS accumulation buffer (Plan 103). |
| `mls_aggregate` | MLS Multi-Layer Sum aggregation (Plan 104). |
| `tiled_attention` | Tiled attention repacking buffers (Plan 115). |
| `tf_loop` | Training-free loop window buffers (Issue 091). |
| `hydra_budget` | Hydra Adaptive Layer Budget pre-computed skip plan (Plan 165). |
| `wall_attention` | Wall Attention per-head prefix sum state (Plan 173). |
| `turboquant` | TurboQuant incremental dequant reset alias (back-compat). |
| `domain_latent` | Free Transformer mid-layer domain conditioning (Plan 038). |
| `kog_cpu_fusion` | Kog AI monokernel CPU fusion — RMSNorm gamma folding + QKV interleaving (Plan 160). |
| `dense_mesh` | DenseMesh latent node network (Plan 266, Plan 385 T6). |
| `decode_specialize` | Stage-specialized decode paths for speculative decoding (Plan 102). |
| `gated_mlp` | SwiGLU gated MLP variant (Issue 377). |
| `speculative_generator` | Forward-cycle cluster (Plan 394 — `drafter_lora`, `dflash`, `verifier`, `step`, `prefill`). |
| `stability_metrics` | `log::debug!` stability metrics in `step.rs` (Plan 394). |
| `weaver_runtime` | Weaver runtime (forward-cycle cluster). |

See `Cargo.toml` for the full feature list (additional Plan 386 / 394 / 396
tracking flags consumed by the moved modules).

## Dependencies

- `katgpt-core` — SIMD kernels, `types::*` re-export.
- `katgpt-types` — `Config`, `kv_dim`, `DepthTier`, `simd`.
- `katgpt-transformer` — `WallPrefixState`, `TransformerWeights`.
- `katgpt-pruners` — `CnaModulator`, `SubstrateMask`, `HydraSkipPlan`.
- `katgpt-hla` — HLA cache types + streaming kernels (Issue 007 Phase F.4b).
- `katgpt-speculative` — `DflashCtx<TransformerWeights>` impl for
  `ForwardContext` (Issue 007 Phase F).
- `rayon` — parallel forward kernels.
- `blake3` — `drafter_lora.rs` save/load integrity envelope (Plan 394).
- `log` — `step.rs` stability metrics (Plan 394).

## License

MIT. Part of the [katgpt-rs](https://github.com/katopz/katgpt-rs) project.
