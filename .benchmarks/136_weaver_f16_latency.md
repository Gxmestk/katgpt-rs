# Benchmark 136 — Weaver f16 Weight Latency (GOAT FAIL — honest negative result)

> **Date:** 2026-07-14
> **Config:** hidden=2304, d_ff=5824, seq_len=5, K=32 (Gemma2-2B scale)
> **Hardware:** M3 Max (12 P-cores, aarch64 NEON)
> **Toolchain:** rustc 1.93.0 (254b59607 2026-01-19), stable channel

## Result: ❌ G2 FAIL — f16 is SLOWER than f32 (0.78×)

| Path | Median | P99 | vs f32 baseline |
|---|---|---|---|
| f32 parallel (AXPY) | 7.83 ms | 9.50 ms | baseline |
| f16 parallel (AXPY) | 11.07 ms | 12.22 ms | **0.71× (regression)** |
| f16 parallel (dot-product, transposed) | 10.06 ms | 10.23 ms | **0.78× (regression)** |

## Root Cause: Compute-bound, not memory-bound

Weaver's forward pass is **compute-bound** (FMA throughput), not memory-bound.
The weight matrices at h=2304 are ~21 MB (f32) — at M3 Max's ~400 GB/s memory
bandwidth, the memory access time is ~0.05 ms, negligible vs the ~7.8 ms FMA
compute time (~200M FMAs across all 8 weight matrices × 5 positions).

f16 weights halve memory traffic but **add f16→f32 conversion overhead**:
- Each 4-element chunk needs 4 scalar `to_f32()` conversions (compiles to `fcvt`)
- On aarch64, the `vcvt_f32_f16` intrinsic would do this in 1 instruction, but
  it's behind the unstable `stdarch_neon_f16` feature gate (not on stable)
- The scalar conversion + stack load/store pattern adds ~25-40% overhead

For a **memory-bound** workload, the 50% bandwidth reduction would dominate.
For a **compute-bound** workload like Weaver, the conversion overhead dominates.

## Two approaches tested

### 1. AXPY pattern (simd_fused_scale_acc_f16)

`output[j] += input[i] * weight_f16[i][j]` — iterates over input dimension,
AXPYs weight rows into output.

- **Bandwidth analysis:** Weight reads halved (f16), but output RMW stays f32.
  Theoretical reduction: 10/12 = 17%. Not enough.
- **Result:** 0.71× (worst — output RMW traffic offsets bandwidth savings)

### 2. Dot-product pattern (simd_dot_f16_f32, transposed weights)

`output[o] = dot(weight_t[o], input)` — iterates over output dimension,
each output element is a single dot product. Weight stored transposed
`[out_dim, in_dim]`.

- **Bandwidth analysis:** Weight reads halved (f16), input stays in L1 cache.
  Theoretical reduction: 6/8 = 25%. Better than AXPY.
- **Result:** 0.78× (better than AXPY but still a regression — conversion
  overhead exceeds the 25% bandwidth savings on this compute-bound workload)

## Conversion cost

| Metric | Value |
|---|---|
| f32→f16 conversion (AXPY, no transpose) | 30.6 ms |
| f32→f16 conversion (dot-product, with transpose) | 138.3 ms |

The transpose adds ~108 ms to the one-time load cost. This is acceptable
(once per session), but the per-call forward overhead is the blocker.

## What WOULD make f16 faster

1. **Native f16 FMA** — `vfmlaq_f16` on ARMv8.6-A does f16×f16→f32 FMA in one
   instruction, avoiding the conversion overhead entirely. Requires the
   `stdarch_neon_f16` feature to stabilize on Rust stable.

2. **Larger matrices** — at h=8192+ (Qwen3.5-27B scale), the weight matrices
   exceed L2 cache (~50 MB) and the workload becomes memory-bound. At that
   scale, the 50% bandwidth reduction would dominate. Weaver at h=2304
   (Gemma2-2B) is too small for this to matter.

3. **GPU port** — the GPU has native f16 support (half-precision tensor cores).
   This is the path to <1 ms (Issue 131 G4 path #1).

## Decision

The f16 code stays **opt-in** (not promoted to default-on). The negative result
is shipped honestly. The code is correct (all tests pass, precision within f16
tolerance) but provides no latency benefit at the Gemma2-2B scale on CPU.

The `simd_fused_scale_acc_f16` primitive added to `katgpt-types` is kept — it's
a valid primitive that may benefit future **memory-bound** workloads at larger
scales.

## Run

```bash
CARGO_TARGET_DIR=/tmp/136_weaver_f16 \
  cargo bench -p katgpt-speculative --features weaver_runtime \
    --bench bench_136_weaver_f16_latency -- --nocapture
```
