# Benchmark 582 — Plan 309 T3.2 Latent Steering SIMD SAXPY (Honest Negative)

**Date:** 2026-08-11
**Status:** HONEST NEGATIVE — gate fails, code kept for portability
**Plan:** [`katgpt-rs/.plans/309_latent_field_steering_primitive.md`](../.plans/309_latent_field_steering_primitive.md) T3.2
**Validated on:** RTX 4090 host (x86_64 + AVX2 + FMA, the host Plan 309 was deferred to)
**Host toolchain:** rustc 1.93.0, `--release` profile

## TL;DR

Plan 309 T3.2 deferred the SIMD-vs-scalar speedup measurement because the
development host is aarch64 (M3 Max) — the `#[cfg(target_arch = "x86_64")]`
AVX2 SAXPY backend is compiled out on ARM, so `apply_latent_steering` routes
to the scalar fallback and the measurement was scalar-vs-scalar (0.0ns/call,
NaN speedup — below timer resolution).

The 4090 host (x86_64 + AVX2 + FMA) is the correct machine for the deferred
measurement. **Verdict: SIMD SAXPY does NOT pass the T3.2 gate on x86_64.**
The plan author predicted exactly this ("Phase 3 as 'likely a no-op' because
LLVM already auto-vectorizes the scalar SAXPY at `-O3`") — the measurement
confirms it.

## Measurement (d=8 / d=16)

The original bench in `tests/latent_steering_t3_simd_vs_scalar.rs` measured
one call per timer read, which at d=8 (8 SAXPY ops ≈ few ns) was below
`Instant::now()` resolution (~100ns QPC floor on Windows). The bench was
patched to batch 4096 calls per timer read so per-call cost rises above the
timer floor.

| d | scalar (ns/call) | SIMD (ns/call) | speedup | gate | verdict |
|---|---|---|---|---|---|
| 8 | 1.6 | 1.9 | **0.82×** | ≥ 2.0× | **FAIL** |
| 16 | 1.7 | 2.0 | **0.88×** | ≥ 1.5× | **FAIL** |

The explicit AVX2 path is **slightly slower** than the auto-vectorized
scalar loop. The per-call dispatcher overhead (`is_x86_feature_detected!`)
costs more than the explicit AVX2 saves at this small dimension — LLVM's
auto-vectorizer emits equivalent AVX2 instructions for the scalar form
at `-O3`, and the extra branch in the dispatcher is pure overhead.

## G4 carry-over (crowd-scale)

The crowd-scale path (5000×8) still PASSES:

| Path | p50 (µs) | gate | verdict |
|---|---|---|---|
| `apply_field_to_crowd` (SIMD dispatcher) | 6.2 | < 1000 | **PASS** |

At crowd scale the dispatcher fires once for 5000 NPCs, so its overhead
amortizes to invisible.

## Verdict and recommendation

**T3.2 GATE FAILS.** Per Plan 309 §T3.2's own pre-registered recommendation:
"do NOT promote T3.2 but keep T3.1 (the code is correct and may help on
targets where auto-vec is disabled, e.g. `RUSTFLAGS=-C target-cpu=x86-64`
baseline builds)."

The T3.1 AVX2 SAXPY backend **stays in place** as a correctness / portability
asset. It does not earn promotion as a perf win on x86_64 hosts with a
modern auto-vectorizing LLVM, but it remains the explicit non-relied-on
form for builds where auto-vec is off or unavailable.

This is the second validated instance of the "explicit SIMD loses to
auto-vec on small dimensions" pattern (the first was Plan 227 Phase 5
Channel SIMD's debug-mode 1.02× — which turned into a real 6-8× win in
release mode *only because the operation count was high enough*).
At d=8/16 the operation count is too small for the explicit SIMD path to
ever beat auto-vec — the dispatcher overhead alone exceeds the savings.

## Honest caveats

1. **Single-host, single-toolchain.** Validated on the 4090 host with
   rustc 1.93.0. Newer LLVM versions may auto-vectorize more aggressively,
   widening the loss; older ones may auto-vectorize less, narrowing it.
   The qualitative finding (auto-vec wins at small d) is robust.
2. **No FMA.** Plan 309 required non-FMA mul+add for bit-identity with
   scalar. A FMA-based variant would have one fewer instruction per
   element but would break the bit-equality invariant — out of scope.
3. **Timer-floor correction.** The original bench's 0.0ns/call was a
   measurement artifact, not a real result. The patched bench (4096-call
   batches) produces real numbers; the bench file is updated in the same
   commit as this benchmark.

## How to reproduce

```bash
cargo test --features latent_field_steering --release \
  --test latent_steering_t3_simd_vs_scalar -- --nocapture --test-threads=1
```
