# Issue 136 — Weaver f16 Weight Optimization (Issue 131 G4 follow-up)

> **Spawned from:** Issue 131 G4 (latency) — "f16 weights (future)" path
> **Date:** 2026-07-14
> **Type:** optimization (perf — memory bandwidth)
> **Severity:** MEDIUM — the parallel path (7.05 ms) is a MARGINAL PASS on G4;
> f16 targets ~3.5 ms (closer to the <1 ms paper target but still above it)
> **Status:** ❌ GOAT FAIL (f16 is 0.78× — SLOWER than f32). Code stays opt-in.
> See `.benchmarks/136_weaver_f16_latency.md` for the honest negative result.

## Context

Issue 131 G4 measured the Weaver forward at:
- Allocating path: 20.9 ms
- Scratch path: 20.6 ms (1.01× speedup — confirms compute-bound, not alloc-bound)
- **Parallel path (rayon): 7.05 ms** (2.96× speedup on M3 Max 12 P-cores)

The paper's GPU-measured target is <1 ms. The parallel path makes Weaver
practical on CPU (break-even at ~3 verifier steps), but three further
optimization paths were noted:

1. **GPU port** via riir-gpu CubeCL — the path to <1 ms. Bigger scope.
2. **f16 weights** — halve memory traffic, ~2× additional speedup. ← **this issue**
3. **Production-scale retraining** — separate concern (riir-train).

## What was done

Added an f16 weight storage path that halves the memory bandwidth for weight
matrix reads — the dominant cost in the matmul-heavy Weaver forward (8 matmuls
per position × 5 positions = 40 matmuls per forward call).

### New SIMD primitive

`simd_fused_scale_acc_f16(dst, src_f16, scale, len)` in
`katgpt-types/src/simd/research.rs` — the f16 sibling of
`simd_fused_scale_acc`. Converts f16→f32 during the FMA loop:

- **aarch64 NEON**: scalar `.to_f32()` conversion (compiles to hardware `fcvt`
  on Apple Silicon — mirrors the existing `neon_dot_f16_f32` pattern in
  `dot.rs`). Avoids the unstable `stdarch_neon_f16` feature gate
  (`vcvt_f32_f16` / `vld1_f16`).
- **x86_64 AVX2+FMA**: scalar `.to_f32()` conversion (compiles to `vcvtph2ps`
  on F16C-capable CPUs).
- **Scalar fallback**: `mul_add` with single-rounding FMA parity.

### New Weaver types

| Type | File | Purpose |
|---|---|---|
| `WeaverWeightsF16` | `katgpt-speculative/src/weaver.rs` | f16 weight storage (norm scales + pos_emb stay f32) |
| `WeaverCorrectorF16` | same | Wrapper with `correct_parallel` method |
| `matmul_vec_f16` | same | f16×f32 GEMV (AXPY pattern, mirrors `matmul_vec`) |
| `weaver_forward_parallel_f16` | same | f16 parallel forward (mirrors `weaver_forward_parallel`) |

### Design: sibling variant (no DRY violation)

Following the established pattern (`weaver_forward` / `_into` / `_parallel`),
the f16 path is a sibling variant, not a generic. The f32 path is preserved
bit-identical — zero API churn for existing callers. Callers explicitly opt in
via `WeaverCorrectorF16::from_f32(&corrector_f32)`.

The f16 forward duplicates the ~160-line parallel forward body. This is
acceptable because:
1. It's a hot-path optimization (perf > DRY here)
2. The sibling-variant pattern is already established
3. The function is well-documented

### What stays f32

- **Norm scales** (`norm_cond`, `norm_attn`, `norm_mlp`): only `[hidden]` each
  (2304 elements = 9 KB). Negligible bandwidth.
- **Position embeddings** (`pos_emb`): `[max_depth * hidden]` = 8 × 2304 =
  18 KB. Small.
- **Embedding table** (`WeaverInput.embedding`): passed by the caller, not
  part of the corrector's weight budget. The top-K gather only does K=32 dot
  products per depth — tiny compared to the matmuls.

### Memory savings

| Weight matrix | f32 size | f16 size | Reduction |
|---|---|---|---|
| w_c, w_q, w_k, w_v, w_o (5× h×h) | 5 × 2304² × 4 = 106 MB | 53 MB | 2× |
| w_gate, w_up (2× h×d_ff) | 2 × 2304 × 5824 × 4 = 107 MB | 54 MB | 2× |
| w_down (d_ff×h) | 5824 × 2304 × 4 = 54 MB | 27 MB | 2× |
| **Total weight budget** | **267 MB** | **134 MB** | **2×** |

The real checkpoint is 219 MB (smaller than the theoretical max because the
real config uses k=512 not the full hidden×hidden). f16 would reduce it to
~110 MB.

## Acceptance criteria (GOAT gate)

- [x] **G1 (correctness)**: f16 corrected probs sum to 1.0, no NaN/Inf.
      **DONE** — `f16_corrected_probs_sum_to_one_no_nan` test passes.
- [x] **G1 (no-harm)**: f16 zero weights produce zero residual.
      **DONE** — `f16_zero_weights_produce_zero_residual` test passes.
- [x] **G3 (precision)**: f16 probs match f32 within f16 precision (<10% abs diff).
      **DONE** — `f16_matches_f32_within_precision` test passes (test config,
      small weights). **The real-checkpoint precision validation needs the
      benchmark below.**
- [x] **G3 (no-regression)**: `weaver_runtime` OFF → no change (f16 code not compiled).
      **DONE** — clippy clean with feature OFF.
- [x] **G4 (wrapper)**: `WeaverCorrectorF16::correct_parallel` matches raw forward.
      **DONE** — `f16_corrector_wrapper_matches_forward` test passes.
- [ ] **G2 (latency gain)**: f16 parallel forward ≥ 1.5× faster than f32 parallel
      forward on the real config (hidden=2304, d_ff=5824, seq_len=5).
      **FAILED** — f16 is 0.78× (a regression). Root cause: the forward is
      compute-bound (FMA throughput), not memory-bound. The f16→f32 conversion
      overhead exceeds the bandwidth savings. See `.benchmarks/136_weaver_f16_latency.md`.
- [ ] **G2 (quality)**: f16 corrected marginals produce acceptance length within
      5% of f32 marginals on the real checkpoint.
      **NOT RUN** — blocked on the latency gate (G2 above). Since the latency
      gate failed, the quality benchmark is moot for promotion. The f16 path
      remains correct (tests pass) but provides no benefit.

## Why this is NOT modelless-promotable (same as Issue 131)

The f16 path is an optimization of a trained artifact. The feature stays opt-in
under `weaver_runtime`. Promotion to default-on is not applicable — Weaver
itself requires trained weights (Issue 131 §"Why this is NOT modelless-promotable").

## Cross-references

- [Issue 131](131_weaver_runtime_integration.md) — the parent integration (G4
  latency criterion lists f16 as path #3)
- [Plan 433](../.plans/433_weaver_dflash_pipeline_wiring.md) — DFlash ↔ Weaver
  pipeline wiring (DONE)
- [Plan 434](../.plans/434_spec_step_weaver_call_site_wiring.md) — spec step
  Weaver call-site wiring (DONE)
- `katgpt-types/src/simd/research.rs` — `simd_fused_scale_acc_f16`
- `katgpt-speculative/src/weaver.rs` — `WeaverWeightsF16`, `WeaverCorrectorF16`,
  `weaver_forward_parallel_f16`
