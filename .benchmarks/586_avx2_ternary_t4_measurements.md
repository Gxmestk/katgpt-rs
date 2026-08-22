# Bench 586 — AVX2 legs of Issues 582 + 583 (the deferred T4 measurements)

**Date:** 2026-08-12
**Hardware:** 4090 host — 13th Gen Intel(R) Core(TM) i7-13700K (x86_64, AVX2+FMA confirmed)
**Host toolchain:** `cargo test --release`, single-threaded (`--test-threads=1`), box quiet (no sibling cargo/rustc running)
**Isolated target dir:** `CARGO_TARGET_DIR=/tmp/i583_avx2` (removed after measurement)
**Resolves:** Bench 582 T4 (AVX2 trit-pack), Bench 583 T4 (AVX2 scale-hoist) — both previously DEFERRED 2026-08-12 with the note "x86 CPU measurement on the 4090 is out of focus" / "the 4090 was unreachable at the time"

## Why this bench exists

Benches 582 and 583 shipped their NEON legs on the M3 Max (aarch64) and wrote
their AVX2 legs (kernels + harnesses) into the tree, but explicitly **deferred
measurement** because:

- The development host was aarch64; Rosetta 2 does not implement AVX2, so
  `is_avx2_fma_available()` is false and the AVX2 path was unreachable.
- Bench 583's own caveat: *"AVX2's `horizontal_sum_256` is more expensive than
  NEON's single `vaddvq`, and Bench 611 already produced one Metal-vs-CUDA case
  where a latency assumption inverted between architectures. The x86_64
  dispatch keeps the folded AVX2 path until a 4090 run says otherwise."*

This bench is that 4090 run. **Both deferrals are now resolved.**

---

## Result 1 — Bench 583 T4: hoisted vs folded on AVX2 → **FAILS G2, fold stays**

`simd_ternary_group_matvec_hoisted` vs `simd_ternary_group_matvec_folded` on
the same `TernaryGroupWeights` container (so decode cost cancels and the only
difference is where the group scale is applied). Median of 9 reps × 20 calls.

| Shape | hoisted ns | folded ns | speedup (fold/hoist) |
|---|---|---|---|
| 512×512 | 32 820 | 34 980 | **1.07×** |
| 1024×1024 | 134 350 | 142 670 | **1.06×** |
| 512×5120 | 354 850 | 362 645 | **1.02×** |
| **median** | | | **1.06×** |

**G2 gate was ≥ 1.10×. Measured 1.06×. FAIL.** The x86_64 dispatch in
`simd_ternary_group_matvec` keeps the folded path (the status quo). The hoisted
path stays compiled-in for reproducibility but is not the x86_64 dispatch
target.

### Why it flips: AVX2's hsum cost cancels the vmul saving

The arithmetic trade that motivates hoisting is:

| | per 128-weight group |
|---|---|
| fold | 32 `vmulps` (scale folded into every sign vector) |
| hoist | 1 `vmulps` + 1 horizontal sum |

On NEON the horizontal sum is a single `vaddvq_f32` (1 cycle), so hoisting
nets ~30 ops → wins 1.11×. On AVX2 the horizontal sum is
`horizontal_sum_256` (a sequence of shuffles + adds to collapse a 256-bit
`__m256` to a scalar), which costs enough to roughly cancel the 31-`vmulps`
saving. The 1.02–1.07× residual is inside the WDDM-class noise the 13700K's
frequency scaling introduces under single-threaded microbenchmarks.

**This is exactly the architecture-dependent flip Bench 583's caveat
predicted.** The prediction holds; no dispatch change on x86_64.

### Cross-architecture summary for Issue 583

| Arch | hoisted vs folded | G2 verdict (≥1.10×) | x86_64/aarch64 dispatch |
|---|---|---|---|
| aarch64 (NEON, M3 Max) | 1.11–1.12× hoisted-wins | **PASS — PROMOTED** | `neon_row_range_hoisted` is the aarch64 dispatch target |
| x86_64 (AVX2, i7-13700K) | 1.02–1.07× hoisted-wins | **FAIL — STAYS FOLDED** | `avx2_row_range_folded` remains the x86_64 dispatch target |

G1 (hoisted matches scalar <1e-6) and G4 (0 allocs / 1000 calls) both PASS on
AVX2 — the kernel is correct and allocation-free; it simply doesn't win enough
to displace the fold.

---

## Result 2 — Bench 582 T4: trit-pack vs bit-plane on AVX2 → **trit slower on SIMD, footprint win stands**

`simd_ternary_trit_matvec` (trit, base-3 packed) vs the bit-plane kernel
(`simd_ternary_group_matvec` folded) on the same logical weights.

### G2b latency (SIMD, cache-resident)

| Shape | trit ns | bit-plane ns | trit/plane |
|---|---|---|---|
| 512×512 | 45 735 | 34 810 | **1.31×** (trit slower) |
| 1024×1024 | 177 150 | 153 575 | **1.15×** (trit slower) |
| 512×5120 | 429 715 | 368 245 | **1.17×** (trit slower) |

**On AVX2 the trit SIMD kernel is 15–31% slower than the bit-plane SIMD kernel.**
This is the **opposite** of NEON, where trit was 1.10–1.15× faster. Reject
bound is 2.00× — PASSes the gate, but the latency advantage that held on NEON
does NOT transfer to AVX2.

### Attribution (scalar-vs-scalar isolates decode from scale placement)

| Shape | trit scalar ns | bit-plane scalar ns | ratio |
|---|---|---|---|
| 512×512 | 132 480 | 210 140 | **0.63×** (trit faster) |
| 1024×1024 | 530 920 | 850 900 | **0.62×** (trit faster) |
| 512×5120 | 1 350 980 | 2 087 700 | **0.65×** (trit faster) |

**Scalar trit is 35–38% faster than scalar bit-plane.** The base-3 LUT decode
is genuinely cheaper than SWAR bit-extraction at the scalar level — this
matches the NEON finding (~10pp of the SIMD win traced to decode).

### Why SIMD flips but scalar doesn't

On NEON (4-wide), the LUT approach maps well: one `tbl` per 4 lanes, cheap
hsum. The bit-plane SWAR extraction needs more operations per 4 lanes.

On AVX2 (8-wide), the calculus inverts: SWAR bit-extraction is extremely
efficient (the `fma_nibble8_avx2_unscaled` helper extracts 8 sign bits per
instruction and FMA-accumulates in one pipe), while the trit LUT lookup +
8-wide horizontal sum doesn't map as cleanly to the wider lanes. **The wider
the SIMD lane, the more SWAR bit-extraction wins over LUT decode.**

### G2c streaming regime (32768×5120, both past L2)

| | ms/call | trit/plane | effective bandwidth |
|---|---|---|---|
| trit (34.5 MiB) | 28.647 | **1.198×** (trit slower) | 1.3 GB/s |
| bit-plane (42.5 MiB) | 23.918 | — | 1.9 GB/s |

On AVX2 the streaming regime **amplifies** trit's disadvantage (1.198× vs
1.17–1.31× cache-resident), instead of washing it out like it does on NEON.
Both arms run ~1.3–1.9 GB/s against a memory roofline ~50 GB/s (DDR5-5600
theoretical) — **CPU ternary GEMV is compute-bound on x86_64 too**, so the 18.8%
traffic saving does not convert to latency here either.

### G2d row-parallel (4096×5120, 24 threads)

| | trit ms | plane ms | trit/plane |
|---|---|---|---|
| serial | 3.513 | 2.849 | 1.23× |
| parallel | 0.541 | 0.453 | 1.19× |

Thread scaling: trit 6.49×, plane 6.28×. The trit/plane ratio holds under
parallelism (the row-parallel kernel is a pure `par_chunk_mut` partition).

### Verdict for trit-pack on x86_64

- **Footprint win stands** (1.725 vs 2.125 bits/weight, 18.8% smaller) — this
  is arithmetic and architecture-independent.
- **Latency win does NOT transfer**: on AVX2, bit-plane is 15–31% faster in all
  regimes. A consumer adopting trit-pack on x86_64 trades 18.8% of footprint
  for a 15–31% latency regression. **The right default on x86_64 CPU remains
  the bit-plane tier** (`ternary_group_scale`); `ternary_trit_pack` stays
  opt-in and is a footprint-tier, not a speed-tier, on x86_64.
- No dispatch change needed: `simd_ternary_trit_matvec` already dispatches to
  `avx2_trit_row_range` when AVX2 is available, which is correct for the
  consumer who explicitly chose the trit container. The point is that no one
  should reach for trit on x86_64 for speed — they should reach for it for
  capacity.

### Cross-architecture summary for Issue 582

| Arch | trit vs bit-plane (SIMD latency) | Footprint | Right CPU default |
|---|---|---|---|
| aarch64 (NEON, M3 Max) | trit 1.10–1.15× **faster** | trit 18.8% smaller | trit is the better default |
| x86_64 (AVX2, i7-13700K) | trit 1.15–1.31× **slower** | trit 18.8% smaller | bit-plane is the better default; trit is footprint-only |

The interesting GPU leg (riir-ai Issue 628) remains the place where the
footprint win should convert to throughput: the Metal GEMV sits at 45% of
roofline and the CUDA dp4a kernel at 88.9% of HBM peak — those are
bandwidth-bound regimes where 18.8% fewer bytes should directly buy throughput.
The CPU measurement (NEON + AVX2) confirms both are compute-bound, so neither
is the place to look for the bandwidth win.

---

## Sanity check — Bench 578 AVX2 re-confirmed

Re-running `bench_578_avx2_goat` (folded AVX2 vs scalar, the gate Bench 581
closed on 2026-08-11) on this session's same host:

| Shape | AVX2 ns | Scalar ns | Speedup |
|---|---|---|---|
| 512×512 | 35 665 | 209 745 | **5.88×** |
| 1024×1024 | 140 590 | 844 165 | **6.00×** |
| 512×5120 | 359 635 | 2 171 735 | **6.04×** |

Min 5.88× (Bench 581 measured 5.98×). Same gate (≥2×), same verdict, within
run-to-run variance. The folded AVX2 kernel that Bench 582 T4 + Bench 583 T4
compare against is the same kernel Bench 581 validated.

---

## What this changes in the tree

**Nothing in the dispatch.** Both deferred T4 tasks resolve as **negative
results that confirm the status-quo dispatch**:

- `simd_ternary_group_matvec` keeps the folded AVX2 path (Bench 583 T4 FAIL).
- `simd_ternary_trit_matvec` keeps dispatching to `avx2_trit_row_range` for
  consumers who opt into `ternary_trit_pack` — but the tier doc now records
  that on x86_64 this is a footprint choice, not a speed choice (Bench 582 T4).

**The deferral notes in Benches 582 and 583 are resolved.** Future agents
reading those files will find a cross-reference to this bench instead of an
open "UNMEASURED on AVX2" caveat.

## Reproduce

```bash
# Bench 583 T4 — hoisted vs folded on AVX2 (G2 FAILS at 1.06×)
CARGO_TARGET_DIR=/tmp/i586_avx2 cargo test -p katgpt-types \
  --features "ternary_group_scale plasma_path" --release \
  --test bench_583_scale_hoist_goat -- --nocapture --test-threads=1

# Bench 582 T4 — trit vs bit-plane on AVX2 (SIMD + scalar + streaming + parallel)
CARGO_TARGET_DIR=/tmp/i586_avx2 cargo test -p katgpt-types \
  --features "ternary_trit_pack" --release \
  --test bench_582_trit_pack_goat -- --nocapture --test-threads=1
# Streaming regime (allocates ~80 MB):
CARGO_TARGET_DIR=/tmp/i586_avx2 cargo test -p katgpt-types \
  --features "ternary_trit_pack" --release \
  --test bench_582_trit_pack_goat -- g2c_streaming_regime_out_of_cache \
  --nocapture --test-threads=1 --ignored

# Bench 578 — AVX2 vs scalar baseline (sanity check)
CARGO_TARGET_DIR=/tmp/i586_avx2 cargo test -p katgpt-types \
  --features "ternary_group_scale plasma_path" --release \
  --test bench_578_avx2_goat -- --nocapture --test-threads=1
```

`--release` is mandatory — debug timings of SWAR/LUT kernels are meaningless.
`--test-threads=1` is mandatory — `median_ns` heap-allocates and bleeds into
G4's `CountingAllocator` window under parallel test dispatch.

## Honest caveats

1. **Single x86_64 host.** Validated on one i7-13700K. The kernel uses only
   `avx2,fma` (no AVX-512 / AVX10 / VNNI), so the result should generalize to
   any Haswell (2013)+ x86_64 CPU, but absolute numbers will vary with cache
   hierarchy and clock. The **direction** (hoist loses on AVX2, trit loses on
   AVX2) is the load-bearing finding and is architecture-level, not
   microarchitecture-level.
2. **Single-threaded microbenchmark.** No real-model end-to-end. The consumers
   that would feel the difference (riir-ai's GPU forward) run GEMV on GPU, not
   CPU. The CPU path is the portable fallback, where a 10–30% kernel difference
   is worth ~0.1 tok/s at the ~1 tok/s CPU fallback rate.
3. **The 13700K has P-cores + E-cores.** The microbenchmark pins to whatever
   core the OS scheduler picks. On a P-core the AVX2 numbers above hold; on an
   E-core the scalar path is relatively less bad (E-cores have narrower SIMD).
   Not investigated further — the gate compares AVX2 to scalar **on the same
   run**, so whichever core it lands on, the ratio is internally consistent.
4. **No code change, no dispatch change.** This bench records two negative
   results. The kernel code is unchanged. The only edits are documentation
   (this file + cross-references in Benches 582 + 583 + the tier doc).
