# Benchmark 581 — Ternary Group Matvec AVX2 Kernel GOAT

**Date:** 2026-08-11
**Status:** PASS — Issue 578 AVX2 follow-up complete
**Feature:** `ternary_group_scale` (opt-in; AVX2 path is now auto-selected on `x86_64` hosts with AVX2+FMA)
**Validated on:** RTX 4090 host (AMD Ryzen-class x86_64, AVX2+FMA confirmed via `cpuid`)
**Host toolchain:** rustc 1.93.0 (254b59607 2026-01-19), `--release` profile
**Related issue:** [`katgpt-rs/.issues/578`](../.issues/578_ternary_group_scale_q2_0_g128_tier.md) — AVX2 deviation (closed by this benchmark)

## TL;DR

Issue 578 shipped the `Q2_0_g128` ternary tier with scalar + NEON kernels but
**deferred the AVX2 port**: the development machine was aarch64 (M3 Max), so
a hand-written AVX2 intrinsics kernel could not be executed or validated, and
an unvalidated AVX2 kernel is worse than a correct scalar one.

The 4090 host (x86_64 + AVX2 + FMA, GPU occupied by Plan 320 run5) was the
correct machine for the deferred work. This benchmark records the result.

| Gate | Threshold | Measured | Verdict |
|---|---|---|---|
| **G1 (correctness)** | AVX2 ≈ scalar, rel < 1e-6 | rel ≤ 1.7e-7 across 6 shapes | **PASS** |
| **G2 (perf)** | AVX2 ≥ 2× vs scalar | **5.98× – 6.13×** across 3 shapes | **PASS** |
| **G3 (no-regression)** | existing tests stay green | 166/166 lib tests pass on AVX2 host | **PASS** |
| **G4 (alloc-free)** | 0 allocs in steady state | 0 allocs / 1000 calls (matvec) + 0 / 200 (batch) | **PASS** |

**Promotion:** the AVX2 path is now auto-selected on `x86_64` hosts that
report AVX2+FMA at runtime via [`simd_level`]. No feature flag change — the
`ternary_group_scale` opt-in gate is unchanged (it still needs the G1
llama.cpp logit match on a real Bonsai tensor to flip default-on, which
stays in [riir-train Plan 333 T3.2](../../riir-train/.plans/333_bitnet_ternary_moe_neuro_symbolic_poc.md)).

## G1 — AVX2 vs scalar reference (correctness)

Shapes cover exact multiples of `GROUP_SIZE` (128), ragged tails (partial
group / partial block), and a sub-4 scalar tail. Same fixture family as the
existing `neon_matches_scalar_reference` test.

| (rows, cols) | rel error (max across rows) |
|---|---|
| (4, 128) | ≤ 1e-7 |
| (3, 256) | ≤ 1e-7 |
| (2, 300) | ≤ 1e-7 |
| (5, 65) | ≤ 1e-7 |
| (1, 7) | exact (small enough for both paths to scalar-fallback much of the row) |
| (2, 129) | ≤ 1.7e-7 |

All within the documented 1e-6 agreement band. The scalar-vs-SIMD divergence
is the same arithmetic-association effect already documented in
[`simd/ternary_group.rs`] §"Scalar vs SIMD agreement": scalar applies the
per-group scale once (`Σ_g scale_g · (Σ sign·x)`), AVX2 folds it per element
(`Σ (scale_g·sign)·x`). Equal in exact arithmetic, ~1e-6 in f32.

## G2 — AVX2 vs scalar (throughput)

Median of 9 reps × 20 inner iterations, `--release`, single-threaded,
interleaved so cache state is symmetric. Shapes mirror the M3 NEON table in
Issue 578 so the cross-arch ratio is directly comparable.

| Shape | AVX2 (ns/call) | Scalar (ns/call) | Speedup |
|---|---|---|---|
| 512×512 | 35 270 | 213 425 | **6.05×** |
| 1024×1024 | 141 195 | 865 125 | **6.13×** |
| 512×5120 | 362 655 | 2 166 880 | **5.98×** |

Min across shapes: **5.98×**. Gate: ≥ 2× (the threshold at which hand-written
SWAR+FMA intrinsics beat what LLVM's auto-vectorizer recovers from the scalar
form). Passes by ~3×.

**Why ~6× and not the theoretical 8× (AVX2 f32 lane count):** the SWAR
sign-decoding pipeline (`and → cmpgt → sub → cvt → mul → fmadd`) is ~6 ops
per 8 elements, vs scalar's ~3 ops per element. The 8-wide SIMD therefore
delivers roughly `8 / (6/3)` ≈ 4× the raw arithmetic throughput, plus
memory-prefetch / pipeline wins bring the measured ratio to ~6×. Consistent
with the binary AVX2 kernel's measured range.

## G3 — no regression

`cargo test -p katgpt-types --features ternary_group_scale,plasma_path --lib`
on the 4090 host: **166/166 tests pass.** The pre-existing scalar + NEON
paths and the parallel / batch entry points are unchanged.

The `parallel_matches_serial_bit_identical` test continues to hold — both
the serial and the parallel paths dispatch to the same AVX2 row-range
function, so the per-row accumulation order is identical.

## G4 — alloc-free

`bench_578_ternary_group_goat.rs` `g4_matvec_allocates_nothing_in_steady_state`:
**0 allocs / 1000 calls** on the AVX2 path. `g4_small_batch_is_alloc_free`:
**0 allocs / 200 calls** for sub-threshold batch.

**Pre-existing test-harness flakiness (NOT a kernel bug):** when run with
default test parallelism, G2's `median_ns` helper heap-allocates a
`samples: Vec<f64>` that can bleed into G4's `CountingAllocator` window,
producing false-positive counts (26 / 73 allocs observed). This is identical
on baseline (without the AVX2 patch) — `cargo test --release` on the same
file fails G4 the same way. The fix is `--test-threads=1`, or hoisting
`median_ns`'s `samples` to a stack array. Recorded here as a harness
artifact; the kernel itself is allocation-free (confirmed by the
`tmp_alloc_probe` experiment, since removed: 0 allocs after `simd_level`
warm-up across 100 calls).

## Honest caveats

1. **Single-host measurement.** Validated on one x86_64 host (the 4090 box).
   The kernel uses only `avx2,fma` target features (no AVX-512, no VNNI), so
   the result should generalize to any Haswell (2013)+ x86_64 CPU, but the
   absolute numbers will vary with cache hierarchy and clock.
2. **AVX2 / AVX10 / AVX-512 future.** This port targets AVX2 specifically —
   wider ISAs (AVX-512 FP16, AVX10) could roughly double throughput again
   by going 16-wide. Out of scope for Issue 578; tracked as a possible
   future optimization if a real consumer needs it.
3. **No real-model end-to-end measurement.** The Bonsai GGUF is on the M3,
   not on the 4090 host, so the row-parallel + AVX2 path was not exercised
   against the real 27B model here. Bench 582 (riir-ai) measures the M3
   NEON path against the real model; an equivalent x86_64 measurement
   waits on a real consumer (Issue 578 promotion gate).

## How to reproduce

```bash
# Correctness (G1) + perf (G2) — this benchmark file:
cargo test -p katgpt-types --features ternary_group_scale,plasma_path \
  --test bench_578_avx2_goat --release -- --nocapture --test-threads=1

# Pre-existing Issue 578 gates (now dispatch to AVX2 on x86_64):
cargo test -p katgpt-types --features ternary_group_scale,plasma_path \
  --test bench_578_ternary_group_goat --release -- --nocapture --test-threads=1

# No-regression sweep:
cargo test -p katgpt-types --features ternary_group_scale,plasma_path --lib
```

## Sign-computation note (why this differs from binary AVX2)

The binary AVX2 kernel ([`simd/binary.rs`]
`fma_scaled_byte8_avx2`) exploits the two-state identity: every weight is
either `+scale` or `−scale`, so `scaled_sign = fmadd(neg_2scale, bs_f, neg_scale)`
recovers both signs in one FMA. **Ternary cannot use this trick** — the zero
state (neither pos nor neg set) must produce `0`, not `±scale`. The ternary
AVX2 helper therefore computes the sign explicitly:

```text
pos_set = cmpgt(and(pos_byte, mask), 0)  → −1 where set, 0 where clear
neg_set = cmpgt(and(neg_byte, mask), 0)  → −1 where set, 0 where clear
sign    = cvt(neg_set − pos_set)         → +1 (pos), −1 (neg), 0 (neither)
scaled  = sign · scale_v
acc    = fmadd(scaled, x_v, acc)
```

Two `cmpgt` + one `sub` + one `cvt` + one `mul` per 8-element chunk, vs the
binary kernel's one `cmpgt` + one `fmadd`. This is the structural cost of
the zero state — exactly the cost Issue 578 §"Deviations" anticipated.
