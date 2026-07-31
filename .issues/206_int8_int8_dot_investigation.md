# Issue 206 — int8×int8 Dot Product Investigation (the unexplored 4th path)

## Status: ✅ T1-T6 DONE — int8 forward path GOAT on native aarch64 (1.39×); ❌ FAIL on WASM V8 JIT (0.88×, quantization overhead dominates)

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
- [x] **T4: WASM port + V8 JIT verification** — ✅ PASS (5/7 sizes ≥1.5×).
      The extmul approach (stable Rust — `i32x4_dot_i8x16_s` is NOT exposed
      in Rust stdarch, even on nightly) delivers 1.76-2.19× at sizes 108-324.
      Native SDOT is 2.5-6.3×; WASM extmul is ~2× — lower but still clears gate.
- [x] **T5: Full conv2d_int8 implementation** — ✅ DONE. `moka_int8.rs` ships
      `MokaWeightsInt8` + `MokaScratchInt8` + `conv2d_int8_into` +
      `linear_int8_into` + `forward_int8_with_scratch`. Platform-dispatched
      int8 dot kernel (SDOT inline asm on aarch64, extmul on wasm32, scalar
      fallback). 3 new GOAT gate tests.
- [x] **T6: GOAT gate** — ✅ PASS on native aarch64, ❌ FAIL on WASM V8 JIT:
      - **Native aarch64 (Apple M3 Max, SDOT)**: G1-G4 ALL PASS.
        - G1: argmax matches f32 on 4/4 boards; value diff < 0.053.
        - G2: 1.39× forward speedup (gate 1.3×).
        - G3: 18/18 tests pass.
        - G4: scratch capacities stable (alloc-free).
      - **WASM V8 JIT (Node.js)**: int8 is **0.88× — SLOWER** than f32.
        - Root cause: quantization overhead dominates on V8 JIT. The extmul
          dot kernel IS fast (wasm-opt even emits native `i8x16.dot_s`), but
          the per-layer activation quantization (scalar max-abs + scale loop)
          is not vectorized well enough by V8 to be free.
        - The microbenchmark (T4) was misleading: isolated dots with hot data
          showed ~2×, but the real forward pass has different memory patterns.
      - **Verdict**: int8 forward path is a GOAT on native aarch64 but NOT
        on WASM. The PUCT WASM latency floor remains ~30ms/move. Stays
        opt-in (`PuctPlayer::with_int8` / `WasmPuctPlayerInt8`) — promoted
        to default-on only if a future SIMD-accelerated quantization fixes
        the WASM regression.

## Methodology

Native aarch64 (Apple Silicon) release build with `target-cpu=native` +
`target-feature=+dotprod`. If native passes, port to WASM (`i32x4_dot_i8x16_s`
via `core::arch::wasm32`) and re-measure under Node V8 JIT.

## Non-goals

- Full conv2d_int8 implementation (deferred until T1+T2 pass)
- WASM measurement (deferred until native passes)
- Activation function int8 (ReLU quantization is trivial; deferred)
- Training (this is modelless inference — katgpt-rs scope)
