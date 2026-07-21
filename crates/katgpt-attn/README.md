# katgpt-attn

Attention stack primitives — GDN2 recurrent attention kernels, DashAttention
sparse routing, Chiaroscuro DCT spectral-entropy routing, RAT+ recurrence
bridge, Energy-Gated Attention, Static Calibration tables, DiagonalGate
abstraction, and the FuncAttn composition layer (freeze/thaw, spectral
pre-rotate, chiaroscuro blend). Extracted from `katgpt-rs/src/` per Proposal
003 Phase 2.

## Overview

This crate owns the attention *kernel* and *types* layers above `katgpt-core`.
The base attention primitives (`attention`, `parallax_attn`, `set_attention`,
`funcattn`) live in `katgpt-core` and are NOT moved here — moving them would
invert the dependency DAG (`katgpt-core` cannot depend on `katgpt-attn`). This
crate adds the root-level attention modules that sit above `katgpt-core` in the
stack.

Some composition layers (e.g. `gdn2/forward.rs`, `dash_attn/forward.rs`) moved
here from the root crate alongside their kernels because they depend on
`ForwardContext` / `TransformerWeights` (Issue 007 Phase F.4a, 2026-07-02),
both of which now live in leaf crates.

## Key types / modules

- `gdn2` — GDN2 recurrent attention kernel + types + forward composition layer.
  Gates `gdn2_attention`.
- `diagonal_gate` — shared DiagonalGate abstraction used by both GDN2 and Wall
  attention. Gates `diagonal_gate`.
- `dash_attn` — DashAttention sparse routing kernels + Phase 12 VortexFlow
  cluster (`vortex_flow`, `block_topk`, `channel_aware`, `entmax_router`,
  `meta_router`, `sat_analysis`, `msa_*` sub-features) + forward composition
  layer. Gates `dash_attn` (+ sub-features).
- `chiaroscuro` — per-token DCT spectral entropy operator routing. Gates
  `chiaroscuro` (pulls `rustfft`).
- `rat_bridge` — RAT+ recurrence bridge — dilated inference via GDN2 state.
  Gates `rat_plus_bridge`.
- `ega_attn` — Energy-Gated Attention — spectral salience gating. Gates
  `ega_attn`.
- `static_cal` — pre-computed per-head attention scales. Gates
  `static_cal_tables` (pulls `blake3`).
- `funcattn_compose` — FuncAttn composition layer (Plan 286 Phase 5): freeze/thaw
  snapshots, spectral pre-rotate, chiaroscuro blend. Gates
  `funcattn_freeze_thaw` / `funcattn_spectral_pre_rotate` /
  `funcattn_chiar_blend`.

Additional integration modules (forward composition paths):

- `forward_hga` — Hierarchical Global Attention three-stage chunk→group→token
  routing (Plan 397). Requires both `hga` and `dash_attn`.

## Feature flags

`default = []`. Every module is opt-in.

| Feature | Description |
|---------|-------------|
| `gdn2_attention` | GDN2 recurrent attention kernel + types + forward composition. Implies `katgpt-forward` + `katgpt-transformer` + `blake3`. |
| `diagonal_gate` | Shared DiagonalGate trait (GDN2 + Wall). |
| `dash_attn` | DashAttention sparse routing + VortexFlow cluster + forward composition. Implies `katgpt-forward` + `katgpt-transformer` + `katgpt-pruners/bandit` + `katgpt-kv/cache_prune` + `serde`. |
| `vortex_flow` | Composable sparse routing (Plan 196). Sub-feature of `dash_attn`. |
| `msa_sparse` / `msa_per_group` / `msa_kv_outer` / `msa_adaptive_k` | MSA blockwise sparse distillation GOAT gates (Plan 256). Each implies `vortex_flow`. |
| `chiaroscuro` | Per-token DCT spectral entropy routing. Pulls `rustfft`. |
| `rat_plus_bridge` | RAT+ recurrence bridge. |
| `ega_attn` | Energy-Gated Attention. |
| `static_cal_tables` | Pre-computed per-head attention scales. Pulls `blake3`. |
| `gdn_tree_verify` | GDN2 cache ↔ tree verify bridge adapter (Plan 424 T4.2). Implies `gdn2_attention` + `katgpt-core/gdn_tree_verify`. |
| `gdn_hola_tree_verify` | Dual-path GDN × HOLA tree verify bridge (Plan 430). Implies `gdn_tree_verify` + `hippocampal_cache`. |
| `funcattn_freeze_thaw` | FuncAttn freeze/thaw snapshots (Plan 286 Phase 5). Pulls `blake3` + `serde`. |
| `funcattn_spectral_pre_rotate` | FuncAttn spectral pre-rotate. Pulls `katgpt-spectral`. |
| `funcattn_chiar_blend` | FuncAttn chiaroscuro blend. Implies `chiaroscuro`. |
| `sparse_mlp` | Sparse MLP path in `gdn2/forward.rs`. Forwards to `katgpt-core/sparse_mlp` + `katgpt-forward/sparse_mlp`. |
| `hippocampal_cache` | HOLA Hippocampal Exact KV Cache integration (Plan 395). Forwards to `katgpt-core/hippocampal_cache`. |
| `hga` | Hierarchical Global Attention forward path (Plan 397). Forwards to `katgpt-core/hga`. |
| `gated_mlp` | SwiGLU gated MLP variant (Issue 377). Forwards to `katgpt-forward/gated_mlp` + `katgpt-transformer/gated_mlp`. |

## Dependencies

- `katgpt-core` — always-on (SIMD kernels, shared traits, base attention).
- `katgpt-forward` *(optional)* — `gdn2_attention` / `dash_attn` composition
  layers (`ForwardContext`).
- `katgpt-transformer` *(optional)* — `TransformerWeights` / `MultiLayerKVCache`
  for the forward composition layers.
- `katgpt-spectral` *(optional)* — `funcattn_spectral_pre_rotate`
  (`calibrate_eigenbasis`).
- `katgpt-pruners` + `katgpt-kv` *(optional)* — Phase 12 VortexFlow cluster
  (`meta_router` bandit + `sat_analysis` SummedAreaTable).
- `rustfft` *(optional)* — DCT spectral entropy operator for Chiaroscuro.
- `blake3` / `serde` *(optional)* — FuncAttn freeze/thaw + StaticCal tables.

## License

MIT. Part of the [katgpt-rs](https://github.com/katopz/katgpt-rs) project.
