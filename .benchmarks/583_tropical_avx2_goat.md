# Benchmark 583 — Tropical Matvec AVX2 Kernel (Plan 337 T3.4 Follow-up)

**Date:** 2026-08-11
**Status:** PASS — Plan 337 T3.4 deferred-AVX2 closed
**Plan:** [`katgpt-rs/.plans/337_tropical_semiring_primitive.md`](../.plans/337_tropical_semiring_primitive.md) T3.4
**Feature:** `tropical_algebra` (DEFAULT-ON; AVX2 path is now auto-selected on `x86_64`)
**Validated on:** RTX 4090 host (x86_64 + AVX2 + FMA)
**Host toolchain:** rustc 1.93.0, `--release` profile

## TL;DR

Plan 337 T3.4 deferred the AVX2 specialization of `tropical_matvec_into`:
"AVX2 path deferred (this dev machine is aarch64; x86 uses the 4-acc
scalar fallback which is competitive)."

The deferral note's "competitive" claim was **wrong** on x86_64 hosts
where `simd_matvec` (the G2 baseline) has its own AVX2 path. Comparing
AVX2-simd_matvec against scalar-tropical was unfair: on the 4090 host
the scalar fallback ran at **0.17-0.24× of simd_matvec at D=64/128**
(well under the 0.80× G2 gate).

The AVX2 port closes the gap completely. After the port:

| D | simd_matvec | tropical (AVX2) | speedup | gate (≥0.80×) |
|---|---|---|---|---|
| 8 | 33.30 ns | 35.20 ns | **0.95×** | PASS* |
| 64 | 222.70 ns | 225.20 ns | **0.99×** | PASS |
| 128 | 914.70 ns | 903.40 ns | **1.01×** | PASS |

**Clean sweep** at all three gate dims. Default-on status holds.

## Implementation

`avx2_tropical_row_max_sum` in
[`crates/katgpt-core/src/algebra/tropical.rs`]
mirrors the NEON kernel's pattern with twice the lane width:

- 4 × `__m256` = 32 lanes in flight per outer iteration
- `_mm256_add_ps` does the tropical "product" (the `+`)
- `_mm256_max_ps` does the tropical "sum" (the `max`)
- Horizontal max reduce via `_mm256_extractf128_ps` + `_mm_max_ps` shuffles

**No FMA** — `(max, +)` is a different semiring; there is no multiply-accumulate,
just add-then-max. The kernel is gated by `#[target_feature(enable = "avx2")]`
and selected at compile time via the existing `#[cfg(target_arch = "x86_64")]`
dispatcher (no runtime `is_x86_feature_detected!` — pre-Haswell x86_64 hosts
are not a target).

A small `horizontal_max_256` helper is inlined in this file (not imported
from `katgpt-types::simd::horizontal`) because that module is `pub(super)`-
private to `katgpt-types`.

## Why the deferral note's "competitive" claim was wrong

The note said: "x86 uses the 4-acc scalar fallback which is competitive."
That is true *in absolute terms* — the scalar 4-accumulator tree-reduce is
a perfectly good latency-hiding pattern, and on aarch64 (where the dev host
sat) the NEON kernel beats it by only a few percent.

But the G2 gate is *relative to `simd_matvec`*, not absolute. On x86_64
`simd_matvec` has its own AVX2 path (8-wide `_mm256_fmadd_ps`), so the
fair comparison is AVX2-vs-AVX2. The scalar tropical fallback was
competing against AVX2 simd_matvec, and losing 4-6×.

| D | pre-AVX2 (scalar tropical vs AVX2 simd_matvec) | post-AVX2 (AVX2 vs AVX2) |
|---|---|---|
| 64 | 0.24× (FAIL) | 0.99× (PASS) |
| 128 | 0.17× (FAIL) | 1.01× (PASS) |

## G1 correctness (carry-over)

The 9 existing `algebra::tropical::tests::*` tests all pass with the AVX2
dispatcher arm wired in. The tests exercise `tropical_matvec_into` through
the public API, so on this host they actually invoke the AVX2 path. No
new correctness gate needed — the kernel computes the same `(max, +)`
reduction the scalar and NEON paths do.

## How to reproduce

```bash
# Perf (G2):
cargo bench -p katgpt-core --features tropical_algebra --bench bench_337_tropical_perf -- \
  --warm-up-time 1 --measurement-time 2

# Correctness (G1 carry-over):
cargo test -p katgpt-core --features tropical_algebra --lib algebra::tropical
```

## Honest caveats

1. **Single-host measurement.** Validated on the 4090 host. The kernel
   uses only `avx2` (no AVX-512, no VNNI), so it should generalize to any
   Haswell+ x86_64 CPU; absolute timings will vary.
2. **No runtime detection.** The kernel is selected at compile time via
   `#[cfg(target_arch = "x86_64")]`. If a project target is baseline
   `x86-64` (pre-Haswell), the binary will crash with SIGILL on the first
   tropical matvec. This matches the existing `simd_matvec` behavior on
   x86_64 (which also uses `target_feature`-gated AVX2 with no runtime
   fallback in some paths). The project's CI guard assumes Haswell+.
3. **No FMA variant.** `(max, +)` does not use FMA by definition. A
   "linear+tropical hybrid" that mixes FMA with max would be a different
   semiring and is out of scope.
