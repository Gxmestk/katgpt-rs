# Bench 583 — Hoisting the group scale out of the bit-plane inner loop: 1.11× (NEON), PROMOTED

**Date:** 2026-08-12
**Hardware:** M3 Max, aarch64/NEON, `--release`
**Harness:** `crates/katgpt-types/tests/bench_583_scale_hoist_goat.rs`
**Issue:** 583 — closed + removed 2026-08-12 (T1–T3 done, T4 deferred); this file is the record
**Origin:** [Bench 582](582_trit_pack_goat.md) §Attribution
**Affects:** the shipped `ternary_group_scale` tier (Issue 578) — no container or
wire-format change

## Executive summary

`simd_ternary_group_matvec` folded the group scale into every sign vector
(`±1 → ±scale`) so its 4 accumulators could span a whole row with no per-group
reset and no per-group horizontal sum. Issue 578 chose that deliberately, on the
reasoning that a reset + hsum serializes the pipeline, and measured the resulting
kernel at **1.29–1.31×** the row-scale kernel.

Counting the ops says the fold is a bad trade at `GROUP_SIZE = 128`:

| | per 128-weight group |
|---|---|
| fold | **32** `vmulq` |
| hoist | **1** `vmulq` + **1** `vaddvq` |

Measured on the same container, so decode cost cancels and the *only* difference
is where the scale is applied:

| Shape | hoisted ns | folded ns | speedup | run 2 | run 3 | run 4 |
|---|---|---|---|---|---|---|
| 512×512 | 35 596 | 39 890 | **1.12×** | 1.12× | 1.12× | 1.08× |
| 1024×1024 | 144 496 | 161 085 | **1.11×** | 1.11× | 1.12× | 1.12× |
| 512×5120 | 363 206 | 402 462 | **1.11×** | 1.09× | 1.11× | 1.11× |
| **median** | | | **1.11×** | 1.11× | 1.12× | 1.11× |

**G2 PASS** (gate ≥ 1.10×). **PROMOTED**: `neon_row_range_hoisted` is now the
aarch64 dispatch target for `simd_ternary_group_matvec` *and*
`simd_ternary_group_matvec_parallel` (both must move together or the
parallel-vs-serial bit-identity gate breaks — it still passes).

### Knock-on: Issue 578's own G2 improved

Re-running `bench_578_ternary_group_goat` unchanged, the group-scale kernel's
overhead over the row-scale kernel dropped:

| Shape | vs row-scale before (Issue 578) | vs row-scale now |
|---|---|---|
| 512×512 | 1.31× | **1.21×** |
| 1024×1024 | 1.29× | **1.16×** |
| 512×5120 | 1.30× | **1.18×** |

The group-scale tier now costs 16–21% over per-row scaling instead of 29–31%.
Issue 578's ≤1.5× ceiling passes with more room, and its stated cause of the
overhead ("the extra `vmulq` per 4 lanes that folds the group scale into the sign
vector") is now confirmed as the dominant term — by removing it.

## G1 — correctness

`g1_hoisted_matches_scalar_and_folded` over 7 shapes (ragged group, ragged block,
sub-4 tail, and 512×5120):

- hoisted vs `ternary_group_matvec_scalar`: **< 1e-6 relative**, typically ~1e-7.
- hoisted vs folded: < 1e-5 relative.
- `parallel_matches_serial_bit_identical` (the existing Issue 578 test) still
  passes — the row-parallel kernel delegates to the same row-range function.
- All 31 `ternary`-matching lib tests + 25 `katgpt-transformer` tests pass; clippy
  clean on default / feature-on / `--all-features` / `--all-targets`.

### A claim in the issue that the measurement refuted

Issue 583 was filed asserting hoisting would also move the NEON path *closer* to
the scalar reference, since one-scale-per-group is how `Q2_0_g128` is defined.
**That is not what happens.** The harness prints a note whenever the hoisted
result is further from scalar than the folded one is, and on 512×5120 it fires on
dozens of rows: the folded kernel is frequently **bit-exact** against scalar
(rel 0.0) where hoisted sits at ~1e-7.

Both are far inside the 1e-6 gate, so it changes nothing operationally — but the
"closer to the reference" argument was wrong and is withdrawn. The case for
hoisting rests on the 1.11×, not on numerics. (f32 rounding is not monotone in
association order; matching the reference's *grouping* does not imply matching
its *result*.)

## G4 — alloc-free

0 allocations / 1000 calls at 512×5120 under a thread-local `CountingAllocator`.
Per-group accumulators are registers; nothing heap-side changed.

## T4 — the AVX2 leg: MEASURED on i7-13700K, G2 FAILS, fold stays

> **RESOLVED 2026-08-12 (Bench 586).** The AVX2 leg was measured on the 4090
> host (i7-13700K, AVX2+FMA). **G2 FAILS at 1.06× median (gate ≥1.10×).** The
> fold stays as the x86_64 dispatch target. The architecture-flip predicted in
> the original caveat below was confirmed by measurement: AVX2's
> `horizontal_sum_256` cost cancels the 31-`vmulps` saving.

`avx2_row_range_hoisted` + `fma_nibble8_avx2_unscaled` ship as the x86_64 mirror
(1 `_mm256_mul_ps` + 1 horizontal sum per group instead of 16 `_mm256_mul_ps`),
reachable through `simd_ternary_group_matvec_hoisted`.

**The x86_64 dispatch is NOT switched.** The 4090 measurement (Bench 586)
confirms the original prediction: AVX2's `horizontal_sum_256` is more expensive
than NEON's single `vaddvq`, enough to cancel the vmul saving — exactly the
Metal-vs-CUDA-style architecture-flip Bench 611 warned about. Measured 1.02–1.07×
on the i7-13700K vs 1.11–1.12× on the M3 Max. See [Bench 586](586_avx2_ternary_t4_measurements.md)
for the full AVX2 table.

| Arch | hoisted vs folded | G2 (≥1.10×) | x86_64/aarch64 dispatch |
|---|---|---|---|
| aarch64 (NEON, M3 Max) | 1.11–1.12× | **PASS — PROMOTED** | `neon_row_range_hoisted` |
| x86_64 (AVX2, i7-13700K) | 1.02–1.07× | **FAIL — STAYS FOLDED** | `avx2_row_range_folded` (status quo) |

## Honest caveats

1. **One host, one microarchitecture.** M3 Max only. The gain is small enough
   (1.11×) that a different core could plausibly erase it, though the op-count
   argument (32 → 2 per group) is architecture-independent in direction.
2. **Synthetic matvecs.** No real-model end-to-end. The consumers that would feel
   it (riir-ai's forward at 9.3 tok/s M3) run their GEMV on GPU; the CPU path is
   the portable fallback at ~1.0 tok/s, where an 11% kernel gain is worth ~0.1
   tok/s at best.
3. **The folded kernel is retained, not deleted** — it is half of this A/B, and
   deleting it would make the claim unreproducible.
4. **`GROUP_SIZE`-dependent.** The trade flips as the group shrinks: at
   `GROUP_SIZE = 8` the fold would pay 2 `vmulq` to avoid a per-8-element hsum
   and the hoist would likely lose. The conclusion is specific to 128 (and to the
   g64 variant it would need re-measuring — see Issue 578's `Q2_g64` note).

## Reproduction

```bash
CARGO_TARGET_DIR=/tmp/t583r cargo test --release -p katgpt-types \
  --features ternary_group_scale,plasma_path \
  --test bench_583_scale_hoist_goat -- --nocapture

# The knock-on effect on Issue 578's own gate
CARGO_TARGET_DIR=/tmp/t583r cargo test --release -p katgpt-types \
  --features ternary_group_scale,plasma_path \
  --test bench_578_ternary_group_goat -- --nocapture
```
