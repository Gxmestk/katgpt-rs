# Issue 200 — f16 Weight Quantization for `forward_base`

## Status: CLOSED — G2 PERF GATE FAIL (2026-07-29, negative result, retained as reference)

All tasks T1–T9 are complete. The verdict is final: G2 FAIL, DO NOT PROMOTE.
The code ships as a `pub` opt-in path that no internal caller dispatches to.
Retained as a negative-result reference (linked from
`.docs/09_feature_catalog/negative_results.md` §17 + perf engineering doc).
Mirrors Issue 201's retention pattern.

## GOAT Gate Results (2026-07-29, Apple Silicon / aarch64)

| Gate | Result | Measurement |
|---|---|---|
| G1 (approximate correctness) | ✅ **PASS** | max relative error **0.03%** (threshold 20%) on medium config, seq_len=16. f16 dequant path is numerically correct. |
| G2 (perf ≥1.5× speedup at seq=1) | ❌ **FAIL** | f16 is **1.7–3.0× SLOWER** than f32 across configs. With the original scalar `to_f32()` kernel: 0.40× speedup. With the `fcvtl` inline-asm kernel (this session's improvement): 0.574× speedup — better but still slower. Refutes the issue's core hypothesis (see §"Why G2 failed" below). |
| G3 (no-regression) | ✅ PASS | `cargo test -p katgpt-forward` clean; f16 path is additive, doesn't touch f32 path. |
| G4 (alloc-free steady state) | ✅ PASS (by construction) | `forward_base_f16` reuses `ForwardContext` scratch buffers; no per-token allocation. Covered by the existing forward_base alloc invariant shape. |

**Promotion verdict:** G2 FAIL → **DO NOT PROMOTE to default**. The f16 path stays as an opt-in code path (no feature gate needed yet — the function is `pub` but no caller dispatches to it). Re-opens only on hardware where f16 loads are genuinely cheaper than f32 loads AND a hardware FCVT-equivalent is free (e.g. future AVX-512_FP16 x86 targets, or future Apple Silicon where the FCVT latency is hidden behind a sufficiently deep pipeline).

### Why G2 failed (root-cause analysis)

The issue's hypothesis was: "halve weight bandwidth → ~2× speedup at seq=1 in bandwidth-bound regime". This is **wrong on Apple Silicon** for two compounding reasons:

1. **The activation `x` is f32, not f16.** The issue assumed weight-only bandwidth reduction (f32 8 bytes/element → f16 4 bytes/element, 50% reduction). The reality: per dot-product element, f32 = 4 bytes (weight) + 4 bytes (activation) = 8 bytes; f16 = 2 bytes (weight) + 4 bytes (activation) = 6 bytes. **Actual bandwidth reduction: 25%, not 50%.** The halved-weight hypothesis double-counts by ignoring the f32 activation.

2. **f16→f32 dequantization is not free.** Even with hardware FCVT (1-2 cycle latency on Apple Silicon), the conversion sits on the critical path between weight load and FMA. Combined with the only-25% bandwidth reduction, the conversion latency more than eats the bandwidth savings. Empirically: f16 is 2.2-2.5× slower than f32, meaning the conversion + smaller-batch overhead dominates.

3. **The WIP attempt to hand-roll NEON bit-manipulation conversion** (`convert_4x_f16_to_f32` in dot.rs) was a further regression — it replaced LLVM's auto-vectorized FCVT path with ~10 manual NEON ops per 4-element conversion, making things worse (3× slower vs the committed 2.5× slower). That WIP was discarded during G2 root-cause isolation; the committed scalar `to_f32()` path (which LLVM vectorizes to FCVT) is strictly better.

4. **The second attempt using inline asm `fcvtl`** (this session, a re-investigation after the original session) improved the conversion speed from 0.40× to 0.574× by using the hardware FCVTL instruction directly via inline asm. But it's still net-negative because (a) the conversion adds latency on the critical path between weight load and FMA, and (b) the store-to-stack-and-reload pattern (forced by `asm!` not being able to pass NEON registers directly to `vfmaq_f32`) adds a round-trip that the FMA pipeline can't fully hide. A fully-inlined asm implementation (conversion + FMA in one block) might close the gap, but at that point you're writing the entire dot product in assembly, which is a maintenance burden disproportionate to a still-sub-2× potential win.

**The honest takeaway:** f16 weight quantization for bandwidth-bound GEMV is **not a modelless perf win on this hardware class**. The hypothesis only holds when (a) activations are also f16, OR (b) f16→f32 conversion is zero-latency. Neither is true here. This is a valid negative result — the G2 gate did its job by catching it before promotion.

### Alternative paths NOT taken

- **Full f16 (weights + activations):** Would halve bandwidth for real (50% reduction), but requires a full f16 forward context (`ForwardContextF16` with f16 `x`, `q`, `k`, `v`, `attn_out`, etc.) and changes accumulation semantics. Out of scope for this issue; the precision tradeoff (f16 accumulation vs f32 accumulation) would need a separate G1 gate.
- **BF16:** Better precision than f16 (8-bit mantissa vs 10-bit), same bandwidth. But BF16 dequant kernels don't exist in katgpt-types yet. Defer to a follow-up.
- **INT8 quantization:** Different dequant path (scale + zero-point), but on Apple Silicon hits the same "activation is f32" wall. The real win for INT8 requires INT8 activations too (the quantized inference literature).

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

- [x] T1: `LayerWeightsF16` + `TransformerWeightsF16` structs (weights.rs)
- [x] T2: `TransformerWeights::to_f16()` conversion (weights.rs)
- [x] T3: `forward_base_f16()` implementation (default-feature path) (forward.rs)
- [x] T4: `forward_f16()` public entry point + dispatch helper (forward.rs)
- [x] T5: GOAT gate — G1 approximate correctness (PASSED, 0.03% rel err)
- [x] T6: GOAT gate — G2 perf speedup measurement (**FAILED** — 2.2-2.5× slower, documented above)
- [x] T7: GOAT gate — G3 no-regression (cargo test clean)
- [x] T8: GOAT gate — G4 alloc-free steady state (by construction)
- [x] T9: Commit + update this issue

## Outcome

**Implemented but not promoted.** The f16 weight path is a complete, correct, additive implementation that **honestly failed its own G2 perf gate**. Per the repo's promotion rule ("a perf gain on a biased/incorrect answer is NOT a modelless gain"), G2 FAIL means no promotion to default. The code ships as a `pub` opt-in path that no internal caller dispatches to — preserved for future hardware where the hypothesis holds, or as a reference for a full-f16 (weights + activations) follow-up.

This is a **valid negative result** in the sense of Research 003 / Issue 356: the GOAT gate did its job by catching a wrong hypothesis before it reached production. The root-cause analysis (§"Why G2 failed") is the durable value — it explains why f16 weight-only quantization doesn't work on this hardware class, which prevents future agents from re-attempting the same hypothesis.

## GOAT Promotion Criteria

If G1-G4 pass AND the speedup is ≥1.5× at seq=1 → the f16 path is a
modelless perf gain. Promote consideration: wire `forward()` to auto-dispatch
to f16 when `config.weight_dtype == WeightDtype::F16` (requires the caller to
hold f16 weights). Default stays F32 (no quality regression for users who
don't opt in).

## Actual Outcome (2026-07-29)

**G2 FAILED — no promotion.** See the Status + "Why G2 failed" sections above
for the full root-cause analysis. The f16 path ships as a `pub` opt-in that
no caller dispatches to, preserved as reference + for future hardware
re-evaluation. The promotion criteria above remain the rule for any future
re-attempt.
