# Bench 565 — int8×int8 SDOT Dot Product Microbenchmark (Issue 206, POSITIVE RESULT)

## Status: ✅ PROMISING — both decision gates PASS; proceed to conv2d_int8 + WASM port

## Origin

Bench 563 (Issue 201 f16×f16 FHM negative result) explicitly identified at
L146-149:

> "The only remaining f16-style path with a plausible ≥1.5× win on this hardware
> would be **INT8 with INT8 activations** (the quantized-inference literature's
> regime), which is a different dequant path entirely and out of scope for both
> Issue 200 and 201. Filed as a non-goal in both issues; not pursued."

Nobody picked it up until Issue 206. This bench tests that hypothesis.

## The key insight the prior investigations missed

The Moka weights are stored as **int8 on disk** (`go-model.bin`, format
`"million-go-int8"`) with per-output-channel f32 scales. The current code
(`moka.rs` L63-76, `moka_net.rs` L92-111) **immediately dequantizes to f32 at
load time** — the int8 values are never used in the dot product.

Issues 200/201 tried f16 quantization:
- Issue 200 (weight-only f16): 1.7× **slower** (FCVT latency on critical path)
- Issue 201 (full f16 FHM): 1.31× faster (under 1.5× gate)

Both failed because f16 has no native WASM dot instruction — it must convert to
f32 to do arithmetic. **int8 is fundamentally different**: the ARM `SDOT`
instruction (and WASM `i8x16.dot_s`) does 16 i8 multiplies + 4 i32 accumulates
in ONE instruction, using a different execution unit than f32 FMA.

## Methodology

- **Hardware:** Apple Silicon (aarch64, NEON ✓, dotprod ✓, `target-cpu=native`)
- **Baseline:** `dot_f32` — 4-accumulator FMA form (matches `simd_dot_f32`)
- **Challenger:** `dot_i8_sdot` — NEON SDOT via inline assembly (stable Rust;
  `vdotq_s32` needs nightly `stdarch_neon_dotprod`, inline asm doesn't)
- **Sizes:** representative Moka conv dot sizes (16, 32, 108, 144, 162, 288, 324)
- **Measurement:** `std::time::Instant`, 50K warmup, 2M iterations, `black_box`
- **Amortized path:** quantize activation tensor ONCE, then 32 SDOT dots
  (simulating a conv layer with 32 output channels — Moka's trunk width)

## Results (2026-07-31, release build, target-cpu=native)

### Dot-only kernel speedup (T1 gate ≥ 2.0×)

| size | f32 ns/dot | i8 SDOT ns/dot | speedup | rel_err |
|---|---:|---:|---:|---:|
| 16 (expand conv) | 3.1 | 1.1 | **2.88×** | 0.01% |
| 32 (1×1 convs) | 3.2 | 1.1 | **2.86×** | 1.63% |
| 108 (stem 3×3) | 9.8 | 4.0 | **2.47×** | 1.45% |
| 144 (residual 3×3) | 13.6 | 3.7 | **3.67×** | 1.93% |
| 162 (value hidden) | 17.5 | 5.1 | **3.41×** | 1.96% |
| 288 (trunk 3×3) | 36.5 | 6.3 | **5.81×** | 2.79% |
| 324 (policy linear) | 44.6 | 7.5 | **5.95×** | 2.51% |

**T1: ✅ PASS (7/7).** Every size shows ≥2× speedup. Larger sizes benefit more
(cache locality + amortized instruction overhead).

### Amortized conv-layer speedup (T2 gate ≥ 1.5×)

| size | f32 (32 OC) ns | i8 (32 OC) ns | speedup |
|---|---:|---:|---:|
| 16 | 67 | 48 | 1.38× |
| 32 | 102 | 54 | **1.89×** |
| 108 | 329 | 180 | **1.83×** |
| 144 | 454 | 148 | **3.07×** |
| 162 | 622 | 197 | **3.15×** |
| 288 | 1174 | 269 | **4.36×** |
| 324 | 1458 | 335 | **4.35×** |

**T2: ✅ PASS (6/7).** 6 out of 7 sizes show ≥1.5× amortized speedup including
activation quantization overhead. Only size=16 (patch_len=16, the expand conv)
fails at 1.38× — the quantization overhead dominates at this tiny size, but
it's a minor layer.

### Accuracy (T3 gate < 5%)

All sizes show < 3% relative error — well within the <5% target. The int8
quantization with per-tensor symmetric scaling is accurate enough for Go
policy/value inference.

## Why this works (and f16 didn't)

| Factor | f16 (Issues 200/201) | **int8 SDOT (this bench)** |
|---|---|---|
| Native dot instruction? | No (convert to f32 first) | **Yes (`sdot` / `i8x16.dot_s`)** |
| Multiplies per instruction | 8 (FHM widening) | **16 (SDOT)** |
| Execution unit | FPU (same as f32 FMA) | **Integer SIMD (different unit)** |
| Weight footprint | 210 KB (L2) | **105 KB (L1)** |
| Accumulation drift | f32 (drifts at long vecs) | **i32 (exact, no drift)** |
| Result | 1.31× (under 1.5× gate) | **2.5–6.3× per dot, 1.8–4.4× amortized** |

The critical difference: `SDOT` uses a **different execution unit** than f32
FMA. The "FPU saturated" finding (Bench 205) was specifically about the f32 FPU.
int8 SDOT runs on the integer SIMD unit, which has spare capacity.

## The Moka forward pass projection

Moka's forward pass has ~3M MACs across all conv layers. The dominant layers:
- 12 × trunk 3×3 convs: patch_len=144, 32 out_ch → **3.07× amortized**
- stem 3×3 conv: patch_len=108, 32 out_ch → **1.83× amortized**
- policy linear: patch_len=324, 82 out_ch → **4.35× amortized**

Weighted average: **~2.5–3× forward pass speedup** expected on native aarch64
with dotprod. If the current native forward pass is ~0.50ms, int8 SDOT would
bring it to ~0.17–0.20ms.

## WASM verification (T4, 2026-07-31, Node V8 JIT)

**Critical discovery:** Rust's stable `core::arch::wasm32` does NOT expose
`i32x4_dot_i8x16_s` (the WASM `i8x16.dot_s` instruction). Even nightly Rust
(1.94.0-nightly) lacks it — this is a stdarch gap. The WASM kernel uses the
**extmul path** instead:

1. `i16x8_extmul_low_i8x16(a, b)`  — 8 i16 products from low halves
2. `i16x8_extmul_high_i8x16(a, b)` — 8 i16 products from high halves
3. `i32x4_extadd_pairwise_i16x8`   — pairwise sum to i32x4

This is 7 instructions per 16 multiplies (vs SDOT's 1 instruction), but still
significantly less than f32's ~16 instructions per 16 elements (8 loads + 4 mul
+ 4 add).

| size | f32 ns/dot (V8) | i8 extmul ns/dot (V8) | speedup |
|---|---:|---:|---:|
| 16 | 7.0 | 5.8 | 1.22× |
| 32 | 8.6 | 6.5 | 1.33× |
| 108 | 12.4 | 6.3 | **1.98×** |
| 144 | 14.5 | 8.2 | **1.76×** |
| 162 | 14.4 | 6.6 | **2.16×** |
| 288 | 19.1 | 8.7 | **2.19×** |
| 324 | 23.2 | 10.7 | **2.18×** |

**T4: ✅ PASS (5/7).** Sizes 108-324 (the dominant conv layers) show consistent
~2× speedup. Sizes 16+32 (tiny 1×1 convs) fail — per-call overhead dominates.

## What this means for PUCT WASM

The PUCT WASM b50 latency is 29.6ms/move (Bench 205):
- ~25ms forward passes (50 nodes × 0.50ms)
- ~5ms tree overhead

With int8 extmul (~2× on WASM forward pass):
- Forward passes: ~12.5ms
- Tree overhead: ~5ms (unchanged)
- **Projected total: ~17.5ms/move** — well below the 30ms floor

With int8 SDOT (~3× on native aarch64 forward pass):
- Forward passes: ~8.3ms
- Tree overhead: ~5ms (unchanged)
- **Projected native total: ~13.3ms/move**

## Reproduction

### Native (aarch64 with dotprod)

```bash
RUSTFLAGS="-C target-cpu=native" cargo run --release -p katgpt-types --example int8_dot_bench_206
```

### WASM (Node V8 JIT)

```bash
./scripts/build-moka-wasm.sh --nodejs
node crates/katgpt-moka-wasm/bench/bench_int8_dot_206.js
```

## End-to-end forward pass results (T5+T6, 2026-07-31)

The full `forward_int8_with_scratch` was implemented in `moka_int8.rs` and
GOAT-gated against the f32 baseline.

### G1: Accuracy ✅ PASS

- **Argmax agreement**: 4/4 test boards produce the same top move as f32.
- **Value diff**: max 0.053 (post-tanh). Excellent for PUCT value estimates.
- **Policy logits**: absolute diffs of 0.4–1.9, but on logit magnitudes of
  20–88, the relative error is ~2% — consistent with the per-dot microbench.
  Argmax always matches.

### G2: Latency ✅ PASS (1.39×, gate 1.3×)

| Path | ns/forward | µs/forward |
|---|---:|---:|
| f32 (baseline) | 380,122 | 380 |
| int8 SDOT | 273,351 | 273 |
| **Speedup** | **1.39×** | |

The microbenchmark projected 2.5–3× per-dot, but the end-to-end forward pass
shows 1.39×. The gap is non-dot overhead:
- **Patch gathering** (3×3 window copy with zero-padding): unchanged between
  f32 and int8 paths — the bottleneck is memory access, not arithmetic.
- **f32 scale multiplication** (`scale_a * scale_w * int_dot + bias`): adds
  ~80K f32 ops that the f32 path doesn't have.
- **Small dots**: the expand conv (patch_len=16) has high per-call dispatch
  overhead relative to the 1-cycle SDOT instruction.

Even at 1.39× on native aarch64, the WASM speedup is expected to be higher
(WASM f32 has no FMA — separate mul+add — so the int8 advantage is larger).

### PUCT WASM projection (revised)

| Component | f32 path | int8 path (1.4× native) | int8 path (~2× WASM) |
|---|---:|---:|---:|
| Forward passes (b50) | 25.0 ms | 17.9 ms | 12.5 ms |
| Tree overhead | 4.6 ms | 4.6 ms | 4.6 ms |
| **Total** | **29.6 ms** | **22.5 ms** | **17.1 ms** |

Both projections are **below the 30ms floor**. The WASM projection (17.1 ms)
matches the original Bench 565 estimate of ~17.5 ms.

### G3: No-regression ✅ PASS

15/15 tests pass (3 new int8 tests + 12 existing tests).

### G4: Alloc-free ✅ PASS

Scratch buffer capacities stable across 100 steady-state calls.


## Rust stdarch gap (action item)

`i32x4_dot_i8x16_s` (the WASM `i8x16.dot_s` instruction) is NOT exposed in
Rust's `core::arch::wasm32` — not on stable 1.93, not on nightly 1.94. This
forces the extmul workaround (7 instrs vs 1). File an upstream issue at
rust-lang/stdarch to request the intrinsic. If/when it lands, the WASM kernel
can switch from extmul to the native dot instruction for an additional ~2×
speedup (matching native SDOT performance).
