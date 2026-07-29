# Issue 200 — f16 Weight Quantization for `forward_base`

## Status: IN PROGRESS (2026-07-29)

## Origin

Third-pass `rust-optimize` session (commit `2e65af8d`) profiled the production
`forward_base` path and found it is **95% matmul (GEMV), not attention**. The
dot kernel (`simd_dot_f32`) is at the research frontier (4-way FMA accumulator
unrolling). f32 GEMV arithmetic intensity is 0.5 FLOP/byte → firmly
memory-bandwidth-bound → no kernel-level optimization can beat the bandwidth
ceiling.

The only actionable path to ~2× speedup: **halve weight bandwidth by storing
weights as f16**. `matmul_f16` already exists in `katgpt-types/src/math.rs`
(L283-291) and dispatches to `simd_matmul_f16_f32_rows` (f16 weights × f32
activations, dequant-on-load). `WeightDtype::F16` exists in config but is
unwired for `forward_base`.

## Scope

Add a parallel f16 weight path for `forward_base`, following the established
`forward_gemma2_f16` pattern (riir-engine Plan 095):

1. **`TransformerWeightsF16` + `LayerWeightsF16`** — parallel structs in
   `katgpt-transformer/src/weights.rs`, mirroring the f32 layout but with
   `Vec<half::f16>` for projection weights. RMSNorm gamma / embeddings handled
   per the f16 path (convert-to-f32 at use site for tiny vectors; f16 direct
   for matmul inputs).
2. **`TransformerWeights::to_f16()`** — one-time conversion at load time.
3. **`forward_base_f16()`** in `katgpt-forward/src/forward.rs` — parallel
   forward function. Default-feature path only (no kog_cpu_fusion, no
   gated_mlp, no sparse_mlp, no wall_attention, no domain_latent). Falls back
   to f32 `forward_base` if those features are enabled + dtype is F16 (so the
   dtype config never produces wrong results, just falls back when the f16
   path doesn't support the feature combo).
4. **`forward_f16()`** public entry point — non-breaking addition. The caller
   (inference engine) chooses `forward()` vs `forward_f16()` based on their
   weight storage. Does NOT change `forward()`'s signature.
5. **GOAT gate** (`prof_forward.rs` extension) — G1 approximate-correctness
   (logits within f16 epsilon, NOT bit-identical — f16 is lossy by design),
   G2 perf (≥1.5× speedup at seq=1, the bandwidth-bound regime), G3
   no-regression (all existing tests pass), G4 alloc-free steady state.

## Design Decisions

- **Parallel struct, not enum-wrapped WeightMatrix.** The enum approach
  (changing every `Vec<f32>` weight field to `WeightMatrix`) would touch 5+
  forward variant files with 50+ call sites. The parallel-struct approach is
  additive (zero breakage), follows the `forward_gemma2_f16` precedent, and
  keeps the f16 path focused on the default feature combination. If f16 +
  sparse_mlp is ever needed, a follow-up can add it.
- **Default-feature path only for f16.** The f16 path targets the
  bandwidth-bound default path. Feature-gated paths (sparse_mlp,
  wall_attention, etc.) are specialized optimizations that are already
  opt-in; if they're enabled, the dtype config is ignored (f32 forward_base
  runs). This keeps `forward_base_f16` at ~150 lines instead of ~400 with 10
  feature gates duplicated.
- **G1 is approximate, not bit-identical.** f16 has ~3 decimal digits of
  precision. The GOAT gate checks `|logit_f16 - logit_f32| < epsilon` per
  vocab element, where epsilon accounts for f16 rounding. Bit-identical is
  impossible by design (f16 is lossy). The gate's purpose is to confirm the
  f16 dequant path is correct (no bugs in the conversion/matmul), not that
  f16 == f32.

## Non-Goals

- BF16 support (the `WeightDtype::BF16` variant exists but bf16 dequant kernels
  don't — defer to a follow-up).
- f16 for attention variants (dash_attn, gdn2, tree_gdn2) — defer.
- f16 + feature-gated paths (sparse_mlp, wall_attention, kog_cpu_fusion, etc.)
  — defer; fall back to f32.
- Changing `forward()`'s public signature — non-breaking addition only.

## Tasks

- [ ] T1: `LayerWeightsF16` + `TransformerWeightsF16` structs
- [ ] T2: `TransformerWeights::to_f16()` conversion
- [ ] T3: `forward_base_f16()` implementation (default-feature path)
- [ ] T4: `forward_f16()` public entry point + dispatch helper
- [ ] T5: GOAT gate — G1 approximate correctness
- [ ] T6: GOAT gate — G2 perf speedup measurement
- [ ] T7: GOAT gate — G3 no-regression (cargo test --all)
- [ ] T8: GOAT gate — G4 alloc-free steady state
- [ ] T9: Commit + update this issue

## GOAT Promotion Criteria

If G1-G4 pass AND the speedup is ≥1.5× at seq=1 → the f16 path is a
modelless perf gain. Promote consideration: wire `forward()` to auto-dispatch
to f16 when `config.weight_dtype == WeightDtype::F16` (requires the caller to
hold f16 weights). Default stays F32 (no quality regression for users who
don't opt in).
