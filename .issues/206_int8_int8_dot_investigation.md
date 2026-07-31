# Issue 206 — int8×int8 Dot Product Investigation (the unexplored 4th path)

## Status: ✅ T1+T2 PASS — proceeding to WASM port + conv2d_int8

## Origin

Bench 563 (Issue 201 f16×f16 FHM negative result) explicitly identified at
L146-149:

> "The only remaining f16-style path with a plausible ≥1.5× win on this hardware
> would be **INT8 with INT8 activations** (the quantized-inference literature's
> regime), which is a different dequant path entirely and out of scope for both
> Issue 200 and 201. Filed as a non-goal in both issues; not pursued."

Nobody picked it up. The prior PUCT WASM session concluded "no further
defensible work" after ruling out im2col (FPU-saturated), smaller net
(needs riir-train), and WebGPU (needs deliberate decision). This issue
picks up the explicitly-filed-as-unexplored INT8 path.

## Why int8×int8 is fundamentally different from f16

| Property | f32 (current) | f16 (Issues 200/201) | **int8 (this issue)** |
|---|---|---|---|
| Bytes per weight | 4 | 2 | **1** |
| Weight footprint (Moka 105K params) | 420 KB (L2) | 210 KB (L2) | **105 KB (L1)** |
| Multiplies per SIMD instruction | 4 (f32x4 FMA) | 8 (FHM widening) | **16 (i8x16.dot_s)** |
| Native WASM dot instruction? | No (mul+add separate) | No (convert to f32) | **Yes (i8x16.dot_s)** |
| Accumulation type | f32 (exact) | f32 (drift at long vecs) | **i32 (exact, no drift)** |
| Result of prior investigation | baseline | 1.31× (under 1.5× gate) | **UNTESTED** |

The critical differences:
1. **`i8x16.dot_s` does 16 multiplies + 4 accumulates in ONE instruction** — 4×
   the arithmetic density of f32x4 FMA. The "FPU saturated" finding (Bench 205)
   was about the *f32 FPU specifically*; int8 uses the *integer SIMD unit* (or
   the same unit at 4× density).
2. **Weight footprint drops to 105 KB** — fits in L1 cache (128 KB on Apple
   Silicon). f32 weights at 420 KB spill to L2. This is a cache-level change,
   not just a bandwidth change.
3. **WASM f32 has NO FMA** — the current WASM kernel uses separate `f32x4_mul`
   + `f32x4_add`. `i8x16.dot_s` fuses multiply+accumulate, which is an even
   bigger relative win on WASM than on native.

## The Moka weight situation

Moka's weights are stored as **int8 on disk** (`go-model.bin`, format
`"million-go-int8"`) with per-output-channel f32 scales. The current code
(`moka.rs` L63-76, `moka_net.rs` L92-111) **immediately dequantizes to f32 at
load time**:

```rust
fn load_dequantized(...) -> Vec<f32> {
    ...
    out.push((bytes[base + k] as i8) as f32 * scale);
}
```

So the int8 weights are **never used directly in the dot product**. This
investigation tests keeping them as int8 and using `i8x16.dot_s`.

## Decision gate results (2026-07-31, Apple Silicon aarch64, dotprod)

See [Bench 565](../.benchmarks/565_int8_int8_sdot_positive.md) for full results.

- [x] **T1: Dot-only microbenchmark** — ✅ PASS (7/7 sizes ≥2×). Best: 6.3× at
      size=288. The SDOT instruction uses a different execution unit than f32
      FMA, so the "FPU saturated" finding doesn't apply.
- [x] **T2: Amortized conv microbenchmark** — ✅ PASS (6/7 sizes ≥1.5×). Best:
      4.4× at size=288/324. Only size=16 fails (1.38× — quantization overhead
      dominates at tiny sizes).
- [x] **T3: Accuracy** — ✅ PASS (all sizes <3% rel error).
- [ ] **T4: WASM port + V8 JIT verification** — pending. The `i8x16.dot_s`
      WASM instruction maps to `sdot` on aarch64 V8 JIT, so the native result
      should transfer. Needs explicit measurement.
- [ ] **T5: Full conv2d_int8 implementation** — pending T4.
- [ ] **T6: GOAT gate** — pending T5.

## Methodology

Native aarch64 (Apple Silicon) release build with `target-cpu=native` +
`target-feature=+dotprod`. If native passes, port to WASM (`i32x4_dot_i8x16_s`
via `core::arch::wasm32`) and re-measure under Node V8 JIT.

## Non-goals

- Full conv2d_int8 implementation (deferred until T1+T2 pass)
- WASM measurement (deferred until native passes)
- Activation function int8 (ReLU quantization is trivial; deferred)
- Training (this is modelless inference — katgpt-rs scope)
