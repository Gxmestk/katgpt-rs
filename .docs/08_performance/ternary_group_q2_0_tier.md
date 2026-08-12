# `TernaryGroupWeights` — the `Q2_0_g128` plasma tier

> **Status:** LANDED, **opt-in** (`ternary_group_scale`). G1–G4 GOAT gate **ALL
> PASS** as of 2026-08-12. Stays opt-in *by policy*, not because a gate fails —
> see [Why it stays opt-in](#why-it-stays-opt-in).
> **Supersedes** `.issues/578_ternary_group_scale_q2_0_g128_tier.md` (closed and
> removed 2026-08-12 per the noise-reduction rule).
> **Filed by** riir-train [Plan 333](../../../riir-train/.plans/333_bitnet_ternary_moe_neuro_symbolic_poc.md) T3.1a.

## What it is

The third plasma tier: ternary `{-1, 0, +1}` bit-planes with **one f16 scale per
128 weights** — the container shape of GGUF `Q2_0_g128`, which is how
[prism-ml/Ternary-Bonsai-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf)
ships. Neither pre-existing tier could hold it:

| Tier | File | Alphabet | Scale granularity | Verdict |
|---|---|---|---|---|
| `TernaryWeights` | `crates/katgpt-types/src/ternary.rs` | ✅ `{-1, 0, +1}` | ❌ `row_scale` — one per **row** | wrong scale |
| `BinaryWeights` | `crates/katgpt-types/src/binary.rs` | ❌ `{-1, +1}`, no zero state | ✅ `group_scale: Vec<f16>`, `GROUP_SIZE = 128` | wrong alphabet |
| **`TernaryGroupWeights`** | `crates/katgpt-types/src/ternary_group.rs` | ✅ `{-1, 0, +1}` | ✅ per-128 f16 | **this tier** |

`BinaryWeights::from_ternary_no_zeros` is *not* a usable bridge — it returns
`None` the moment any weight is zero, and Bonsai's zeros are the whole point
(ternary sparsity is why it holds 94.6% of FP16 quality at 1.71 bits/weight).

```rust
pub struct TernaryGroupWeights {
    pub rows: usize,
    pub cols: usize,
    pub blocks64: usize,       // cols.div_ceil(64)
    pub groups_per_row: usize, // cols.div_ceil(GROUP_SIZE), GROUP_SIZE = 128
    pub pos_bits: Vec<u64>,    // [rows * blocks64]  bit set -> +1
    pub neg_bits: Vec<u64>,    // [rows * blocks64]  bit set -> -1
    pub group_scale: Vec<f16>, // [rows * groups_per_row]
}
```

**Representation invariant:** `pos_bits & neg_bits == 0` per block; a position
clear in both planes is the zero state. Checked on `set`, by
`invariant_holds()`, and on load.

**Footprint is a repack, not an expansion.** 2 bit-planes + 16 bits per 128
weights = **2.125 bits/weight** = 7.17 GB at 27B params, matching Bonsai's
documented ~7.2 GB deployed size. `ternary_group_scale` implies `binary_plasma`
so `GROUP_SIZE` stays a single source of truth.

## Shipped surface

| Piece | Where |
|---|---|
| Container (`new`/`set`/`get`/`scale_at`/`set_scale`/`checksum`/`encoded_bytes`/`invariant_holds`/`quantize_from_f32`/`from_ternary`) | `katgpt-types/src/ternary_group.rs` |
| Kernels — scalar, NEON, AVX2+FMA, batch, row-parallel | `katgpt-types/src/simd/ternary_group.rs` |
| Loader `load_ternary_group_bits` (`TGPLSMA1` magic) | `katgpt-transformer/src/contiguous.rs` |
| GOAT harnesses | `katgpt-types/tests/bench_578_ternary_group_goat.rs`, `bench_578_avx2_goat.rs`, `katgpt-core/tests/ternary_group_parallel_alloc_check.rs` |

`from_ternary` widens a row-scale `TernaryWeights` by broadcasting `row_scale`
across the row's groups — always succeeds, unlike `from_ternary_no_zeros`.

## GOAT gate — final status (all four PASS)

| Gate | Status |
|---|---|
| **G1 correctness** | **PASS.** Scalar/NEON/AVX2 parity, invariants, quantizer, loader (incl. 3 corruption modes), widening — 12 unit tests + 2 loader tests. The load-bearing half — **a real Bonsai tensor reproducing llama.cpp** — closed by riir-ai Issue 594: ` Paris` rank 1 @ p=0.69 vs llama.cpp ref ~0.69, 5/5 top-5 recovery. See [riir-ai `bonsai_ternary_throughput.md`](../../../riir-ai/.docs/09_performance/bonsai_ternary_throughput.md). |
| **G2 perf** | **PASS.** Now **1.16–1.21×** the row-scale kernel (ceiling 1.5×) — improved from 1.29–1.31× by Issue 583's scale hoist. Row-parallel kernel adds **7.21×** over serial on real Bonsai geometry. AVX2 5.98–6.13× vs scalar ([Bench 581](../../.benchmarks/581_ternary_group_avx2_goat.md)). |
| **G3 no-regression** | **PASS.** `katgpt-types` 157 tests with the feature / 127 default; `katgpt-transformer` 25. Clean under default, `--no-default-features`, feature-on, and feature-on-without-`plasma_path`. |
| **G4 alloc-free** | **PASS.** 0 allocs / 1000 calls on NEON, scalar, and AVX2 matvec; 0 / 200 sub-threshold batch; 0 / 50 row-parallel — under `CountingAllocator`. |

**Modelless by construction** — a container plus closed-form kernels, no
training.

### G2 numbers (M3 Max, aarch64/NEON, `--release`, median of 9 × 20 calls)

| Shape | group-scale ns | row-scale ns | dense f32 ns | vs row (gate ≤1.5×) | vs dense (info) |
|---|---|---|---|---|---|
| 512×512 | 41 500 | 31 719 | 14 467 | **1.31×** | 0.35× |
| 1024×1024 | 163 673 | 127 227 | 60 773 | **1.29×** | 0.37× |
| 512×5120 | 415 094 | 319 777 | 162 410 | **1.30×** | 0.39× |

The ~1.3× was the extra `vmulq` per 4 lanes folding the group scale into the sign
vector — stable across a 10× shape range, so arithmetic cost, not a cache
artifact. **Issue 583 removed most of it**: hoisting the scale to one `vmulq` +
one `vaddvq` per group (instead of 32 `vmulq`) is 1.11× faster and brought the
ratio to 1.16–1.21×. The fold trick is retired on aarch64 and retained only as
the A/B baseline (`simd_ternary_group_matvec_folded`); the AVX2 dispatch still
folds, pending a 4090 measurement. See [Bench 583](../../.benchmarks/583_scale_hoist_goat.md).

**Ternary is slower than dense f32 on NEON and always has been** (0.35–0.45×,
reproducing [Bench 044](../../.benchmarks/044_plasma_path_goat.md) independently);
its win is memory traffic. The original G2 clause "≥2× the dense f16 matvec"
was **struck on arrival 2026-08-09** — it was copied from Plan 333's G-PERF,
which misread Bench 044 (16.12 Gop/s = 0.45× FP32 NEON's 36 Gop/s, documented
there as *fundamental*: SWAR bit-decoding has higher opcode count than pure
load+FMA). No group-scale kernel can satisfy a gate the tier it extends already
fails.

### Row-parallel kernel (`simd_ternary_group_matvec_parallel`, 2026-08-10)

The G2 table is single-threaded, and the only rayon entry point
(`simd_ternary_group_matmul_batch`) parallelizes over **batch** — so at
`batch = 1`, every autoregressive decode step, there was no parallelism at all.
Measured on real Ternary-Bonsai-27B geometry (riir-ai
[Bench 582](../../../riir-ai/.benchmarks/582_ternary_bonsai_decode_throughput_preflight.md),
M3 Max, 16 threads):

| | s/token | tok/s | GMAC/s |
|---|---|---|---|
| serial | 3.995 | 0.25 | 6.41 |
| **row-parallel** | 0.554 | **1.80** | **46.22** — 7.21× |

Rows are independent, so the `par_chunks_mut` split is a pure partition and the
result is bit-identical to serial (tested across ragged row counts and a zeroed
row); up to 7.77× on a 248320-row `lm_head`. Below `PARALLEL_ROW_MIN = 256` it
delegates to serial — Bonsai's 48-row `ssm_alpha`/`ssm_beta` measure 1.01×, so
the threshold is exercised by the real model, not only by tests. The 0-alloc
gate is what let riir-engine's `forward_ternary` adopt it without breaking its
own 0-alloc-per-token contract.

## Format verification (2026-08-09) — read from the real file

Verified against `PrismML-Eng/llama.cpp` @ `9ca265a` and the real
`Ternary-Bonsai-27B-Q2_0.gguf` (7,165,121,600 B) via HTTP range requests.

```c
#define QK2_0 128
typedef struct {
    ggml_half d;            // f16 scale
    uint8_t   qs[QK2_0/4];  // 2 bits/weight -> 32 B
} block_q2_0;               // 34 B per 128 weights = 2.125 bits/weight
```

Exactly `TernaryGroupWeights`' geometry; 7.165 GB / (2.125/8) = **26.97B
params**. The fork also defines `block_q1_0` = `f16 + qs[128/8]` = 1.125
bits/weight, which is exactly `BinaryWeights` — the two older plasma tiers map
1:1 onto the fork's two custom types (Issue 145 modelled `Q1_0`, this tier
models `Q2_0`). Type IDs: `GGML_TYPE_Q1_0 = 41`, `GGML_TYPE_Q2_0 = 42`,
`GGML_TYPE_COUNT = 43`. Decode, 2 bits LSB-first per byte:

```c
q = (qs[j/4] >> ((j%4)*2)) & 0x03;   y[j] = ((int)q - 1) * d;
```

### ⚠️ The format has a fourth state two bit-planes cannot represent

`11` decodes to **+2**, not +1 — and `TernaryGroupWeights` can hold exactly
`{-1, 0, +1}`. Two facts make this safe for *this* checkpoint but not
unconditionally:

1. **Unreachable via the reference encoder.** `quantize_row_q2_0_ref` sets
   `d = amax`, so `round(w*id)+1 ∈ {0,1,2}` and the `if (q > 3) q = 3` clamp is
   dead code. Any encoder choosing `d < amax` (e.g. an error-minimising fit)
   *would* reach it.
2. **Measured absent.** Scanning 30,720,000 weights across two structurally
   different tensors: `-1` 35.118%, `0` 29.927%, `+1` 34.955%, **`+2` 0
   (0.000%)** (`blk.32.ffn_up.weight` and `output.weight`, 0 each).

**Any parser MUST reject code `11` loudly** rather than silently folding it to
`+1` — a future Prism-ML checkpoint, or the `Q2_g64` / `PQ2_0` variants in the
same repo, may use it.

**`Ternary-Bonsai-27B-Q2_g64.gguf` exists in the same repo** — group size
**64**, not 128. `GROUP_SIZE` is a compile-time constant shared with
`binary_plasma`; supporting g64 needs it parameterised. Not done.

## Known deviations (intentional)

1. **Scalar and NEON are close, not bit-identical** (~1e-6 relative). They
   associate the scale differently: scalar applies it once per group (matching
   how `Q2_0_g128` is defined and how llama.cpp dequantizes), NEON folds it per
   element so the 4 accumulators span the whole row without per-group horizontal
   sums — the trick Issue 145's binary kernel uses. Documented at the top of
   `simd/ternary_group.rs`.
2. **AVX2 computes the sign explicitly** (`cvt(neg_set − pos_set)` → +1/0/−1)
   rather than via the binary kernel's two-state FMA identity — ternary's zero
   state requires it. Single-host measurement (the 4090 box); no real-Bonsai
   end-to-end on x86_64 yet. See [Bench 581](../../.benchmarks/581_ternary_group_avx2_goat.md).
3. **`ternary_group_scale` implies `binary_plasma`** to share `GROUP_SIZE`.

### Batch bug found and fixed while writing the G4 harness

`simd_ternary_matmul_batch` was **broken for every `batch >= 2`** — both its
sub-threshold loop and its rayon path passed an open-ended `&x[x_off..]` into a
matvec asserting `x.len() == w.cols`, panicking on the length check. `batch == 1`
happened to work (offset 0, exact length), which is why it survived since Plan
148. `simd_binary_matmul_batch` had the same bug in its sub-threshold loop only.
Fixed in all three tiers (exact slicing + `zip(par_chunks)`), covered by
`regression_sub_threshold_batch_does_not_panic` across `batch` 1..=5 through the
`PARALLEL_BATCH_MIN = 4` boundary.

## Why it stays opt-in

G1–G4 all pass, so promotion is *permitted* by the GOAT rule — but it is
declined for two structural reasons:

1. **It would transitively promote `binary_plasma`**, which is opt-in **by
   deliberate decision** (Issue 145 Phase 2 T2.4: promotion required *both*
   ≥1.5× latency and ≥1.5× storage; measured 1.22× latency FAIL + 1.82× storage
   PASS → conjunction fails). Promoting this tier would silently overturn that.
2. **The tier is a model-specific container, not a general-purpose win.** It is
   1.3× *slower* than row-scale ternary and 2.6–2.9× slower than dense f32 on
   NEON; it pays for itself only when the weights are actually `Q2_0_g128`.
   Consumers opt in explicitly and always know they are loading Bonsai.

Consumers enable it by name: `riir-engine/q2_0_ternary_bridge`,
`riir-gpu/ternary_gemv`, and transitively `deltanet_ternary_inference`.

## Consumers (all unblocked as of 2026-08-12)

| Consumer | Status | Measured |
|---|---|---|
| riir-ai CubeCL `TernaryDeltanetGpuForward` (Issue 604) | ✅ PRODUCTION on M3 Metal + 4090 | **9.3 tok/s** on M3 Metal (9.7× over CPU's 0.96; steady-state A/B, Bench 616) and **15.00 tok/s** on the 4090. `bonsai-go-gpu-resident` example feature. See [`ternary_gpu_forward.md`](../../../riir-ai/.docs/09_performance/ternary_gpu_forward.md) |
| riir-ai cudarc `TernaryDeltanetGpuForwardCudarc` (Issue 615) | ✅ PRODUCTION on RTX 4090, T8 gate PASS | **30.02 tok/s** / 33.31 ms/token — 2.0× the CubeCL path, ~110× CPU, clears the 26.72 target with 1.12× headroom (Bench 615, commit `fe0bbab5`) |
| riir-ai Issue 594 (qwen35 DeltaNet ternary port) | ✅ DONE, G1 PASS | ` Paris` rank 1 @ p=0.69 vs llama.cpp ref |
| riir-train Plan 333 T3.1b/T3.1c/T2.2/T3.3a/b | ✅ unblocked | `q2_0.rs` dequant + `ternary_lora_forward.rs` |

The CPU pure-ternary SIMD path (0.96–1.05 tok/s) is the portable fallback; see
[`bonsai_ternary_throughput.md`](../../../riir-ai/.docs/09_performance/bonsai_ternary_throughput.md)
for the CPU-vs-GPU headline table.

### What the container costs the consumers

Reference point: llama.cpp (PrismML fork `9ca265a`) does **13.33 tok/s** on the
same M3 Max and **86.74 tok/s** on the 4090. So we are 0.70× on Metal and 0.35×
on CUDA end-to-end — *even though* our dp4a GEMV kernel is **1.44× faster than
llama.cpp's** (895.8 GB/s = 88.9% of 4090 HBM peak vs their 621.5 GB/s).

The end-to-end gap is therefore **not** the GEMV and **not** this container's
arithmetic. Two things dominate:

1. **Non-projection sequential compute** — 55.9 of M3 Metal's 107.5 ms/token is
   recurrence, conv1d, attention, norms, gating. Real dependent compute, not
   launch overhead (Bench 616 proved fusion buys nothing on Metal: 9.33 baseline
   vs 9.28 fused; the Metal command scheduler already hides cheap elementwise
   dispatches behind the next GEMV).
2. **Dispatch count** — ~200/token on cudarc, ~1200/token on CubeCL Metal.

**The one gap that *is* this container's scope** is the last ~10% of the Metal
GEMV: we store two separate `pos_bits`/`neg_bits` arrays, llama.cpp stores one
array of 2-bit codes. Two-array means two loads and two popcount chains per
word where they have one. Closing it means changing the container layout
(interleave the planes, or store native 2-bit codes and decode in-register) —
which would also close the 2.125 → 1.71 bits/weight footprint gap. Not
attempted; it is a real, bounded, katgpt-rs-side optimization with a measured
~10% Metal ceiling as its prize, and it would touch every kernel + the
`TGPLSMA1` wire format.

## Non-goals

- **Native 2-bit-slot storage / plane interleaving.** ~~A separate
  optimization.~~ **DONE, better than specified** — [Issue 582](../../.issues/582_ternary_trit_packed_footprint_tier.md)
  ships `TernaryTritWeights`: base-3 packing, 5 trits per byte, **1.725
  bits/weight** (below the 1.71-ish "ideal" this doc called unreachable), Bonsai
  5.82 GB instead of 7.16 GB, and — against the prediction — **1.22–1.28×
  faster** than this tier on NEON, not slower. See
  [Bench 582](../../.benchmarks/582_trit_pack_goat.md). The GPU leg (single load
  stream on Metal/CUDA) is riir-ai Issue 628.
- **Changing `TernaryWeights` or the `CIOTBIT1` format.** Additive tier only —
  adding fields to `TernaryWeights` would break `load_ternary_bits`' struct
  literal and force an on-disk version bump, and carrying both `row_scale` and
  `group_scale` on one struct creates a which-wins ambiguity no caller wants.
- **`Q2_g64`.** Needs `GROUP_SIZE` parameterised.
- **Training on this container.** Frozen base; the LoRA delta is riir-train's
  concern (Plan 333 T3.3).
