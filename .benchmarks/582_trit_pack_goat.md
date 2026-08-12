# Bench 582 — Base-3 trit-packed ternary tier: GOAT gate ALL PASS (and the perf prediction was wrong)

**Date:** 2026-08-12
**Hardware:** M3 Max, aarch64/NEON, `--release`
**Harness:** `crates/katgpt-types/tests/bench_582_trit_pack_goat.rs`
**Feature:** `ternary_trit_pack` (opt-in)
**Issue:** [582](../.issues/582_ternary_trit_packed_footprint_tier.md)
**Tier doc:** [`.docs/08_performance/ternary_group_q2_0_tier.md`](../.docs/08_performance/ternary_group_q2_0_tier.md)

## Executive summary

`TernaryTritWeights` packs ternary `{-1, 0, +1}` weights **5 per byte in base 3**
(`3^5 = 243 ≤ 256`) instead of two 1-bit planes. It was filed as a *footprint*
tier — the expectation, written into the issue and the kernel docs before
measuring, was that it would trade latency for bytes.

**It does not trade. It wins both.**

| Gate | Result | Verdict |
|---|---|---|
| **G1 correctness** | scalar **bit-identical** to `ternary_group_matvec_scalar`; `from_group`→`to_group` lossless; NEON within 1e-6 | **PASS** |
| **G2 footprint** | **1.725 bits/weight vs 2.125** — ratio 0.8118–0.8162 (gate ≤ 0.83) | **PASS** |
| **G2b latency** | **0.78–0.82×** the bit-plane kernel — i.e. **1.22–1.28× faster** (gate was only "≤ 2× reject bound") | **PASS, unexpectedly** |
| **G3 no-regression** | default / `--no-default-features` / feature-on / `--all-features` clippy-clean; **155** lib tests with the feature, **130** default, 0 failures | **PASS** |
| **G4 alloc-free** | 0 allocs / 1000 calls, SIMD and scalar | **PASS** |

Ternary-Bonsai-27B (26.97B params): **5.82 GB instead of 7.16 GB.**

## G2 — footprint (deterministic, not a timing)

| Shape | trit B | bit-plane B | ratio | bits/weight |
|---|---|---|---|---|
| 512×512 | 56 832 | 69 632 | 0.8162 | 1.7344 |
| 1024×1024 | 226 304 | 278 528 | 0.8125 | 1.7266 |
| 512×5120 | 565 248 | 696 320 | 0.8118 | 1.7250 |

`8/5` bits per weight + `16/128` bits of f16 scale = **1.725**, asymptotically.
Small shapes round up slightly on the row-tail byte. **−18.8%**, not the −17.6%
the issue was filed with: the first estimate wrongly rounded each 128-weight
group up to a whole 26 bytes, but rows are packed contiguously, so only the *row*
tail rounds. The gate (≤ 0.83) was written before this correction and holds
either way.

## G2b — latency (the wrong prediction)

Median of 9 × 20 calls, three independent runs:

| Shape | trit ns | bit-plane ns | ratio | run 2 | run 3 |
|---|---|---|---|---|---|
| 512×512 | 33 131 | 40 631 | **0.82×** | 0.81× | 0.82× |
| 1024×1024 | 126 700 | 160 944 | **0.79×** | 0.80× | 0.79× |
| 512×5120 | 318 533 | 403 831 | **0.79×** | 0.79× | 0.79× |

Stable to ±0.01 across runs — far outside the 15% noise band this repo treats as
meaningless. **The trit tier is 1.22–1.28× faster than the bit-plane tier while
being 18.8% smaller.**

And note *where* it was measured: every shape here is **cache-resident**, which
is the regime that should favour the incumbent. The footprint advantage has
nothing to pay its decode cost with, and it still wins.

### Attribution — two causes, and one of them is free money for the shipped tier

The trit SIMD kernel differs from the bit-plane one in two ways at once:

- **(a) decode** — a 2 KB LUT (byte → 5 signed values, padded to 8 lanes for an
  aligned 8-byte store) instead of SWAR bit extraction (splat a pos/neg byte,
  mask each bit into its own lane).
- **(b) scale placement** — one horizontal sum + one scale multiply *per group*,
  instead of folding the group scale into every sign vector (Issue 578's trick,
  costing a `vmulq` per 4 lanes).

The **scalar** paths of both tiers already share (b) — both accumulate a group
then multiply once — so scalar-vs-scalar isolates (a):

| Shape | trit scalar ns | bit-plane scalar ns | ratio |
|---|---|---|---|
| 512×512 | 168 333 | 186 842 | **0.90×** |
| 1024×1024 | 678 183 | 752 308 | **0.90×** |
| 512×5120 | 1 703 150 | 1 871 725 | **0.91×** |

So the ~21% SIMD win splits roughly evenly: **~10pp from base-3 decode being
genuinely cheaper than bit extraction**, and **~10pp from hoisting the scale out
of the inner loop**.

**The second half is available to the existing `ternary_group_scale` tier for
free — no format change.** Issue 578 documented the fold-into-the-sign-vector
trick as the reason its accumulators "span the entire row with no per-group reset
and no per-group horizontal sum, keeping the out-of-order pipeline fed", and
measured the resulting kernel at 1.29–1.31× the row-scale kernel. This
measurement says the fold **costs more than the hsum it avoids** at
`GROUP_SIZE = 128` — one hsum per 128 weights is cheap, one `vmulq` per 4 lanes
is 32 of them. Filed as Issue 583.

## G1 — correctness

- `ternary_trit_matvec_scalar` is **bit-identical** (`assert_eq!` on f32, not a
  tolerance) to `ternary_group_matvec_scalar` on the same logical weights, at
  128 / 256 / 300 / 133 / 13 columns and at all three benchmark shapes. Both
  apply one scale per group over an in-order sum, so exact equality is the right
  gate and it holds.
- `from_group` → `to_group` round-trips `pos_bits`, `neg_bits` and `group_scale`
  bit-identically, including `cols % 5 != 0` and `cols % 128 != 0`.
- `quantize_from_f32` agrees with the bit-plane tier on **every weight and every
  scale** — same mean-abs scale, same `0.5·scale` threshold, same error carry.
- NEON vs scalar within 1e-6 relative (summation order only).
- Non-canonical bytes (`≥ 243`) are detected by `is_canonical()` and decode to
  zero rather than garbage.

### The bug this gate caught

The first implementation placed a group's first weight at a **negative** offset
in the decode scratch (`FRONT_PAD - off`). It is a forward offset (`off`): the
group's first byte begins at weight `5·b_start ≤ w_start`, so decoding that whole
byte puts weight `w_start` at `off = w_start % 5` lanes *into* the scratch.

The failure mode is exactly why the cross-tier equality test earns its keep —
**single-group shapes still passed**. Only shapes with a second group diverged
(3×256 produced `[-7.09, 5.31, 9.25]` against the reference's
`[4.72, -1.72, -0.59]`). A tolerance-based self-consistency test would have
passed; comparing against the shipped reference did not.

## G4 — alloc-free

0 allocations / 1000 calls for both `simd_ternary_trit_matvec` and
`ternary_trit_matvec_scalar` at 512×5120, under a thread-local
`CountingAllocator`. The per-group decode scratch is a `[i8; 160]` stack array.

## Honest caveats

1. **NEON + scalar only.** No AVX2 kernel for this tier yet, so x86_64 hosts get
   the scalar path (still ~0.90× the bit-plane scalar path, so not a regression
   — but it forgoes the SIMD win). The bit-plane tier has AVX2 (Bench 581).
2. **No real-model end-to-end.** The measurement is synthetic matvecs. The
   consumers that would benefit (riir-ai's Metal/CUDA forwards) run their GEMV on
   GPU, where this container is not yet implemented — that is riir-ai Issue 628.
3. **Cache-resident regime only.** The claimed *reason* for the tier (18.8% less
   RAM traffic) is untested here, because the tier wins before that effect can
   even appear. The streaming regime should widen the gap, not narrow it, but
   that is a prediction, not a measurement — and this benchmark is a reminder
   that my predictions in this area have been wrong once already today.
4. **`from_group` is O(rows·cols) scalar get/set.** Fine at load time
   (one pass, no hot path), but it is not a fast repack; a block-wise version
   would be needed if it ever landed on a latency path.

## Promotion verdict — stays opt-in

All four gates pass, and unlike Issue 578 the perf axis is a *gain*, not a
tolerated overhead. Promotion is still declined, for the same structural reason:
`ternary_trit_pack` implies `ternary_group_scale` implies `binary_plasma`, which
is opt-in **by deliberate decision** (Issue 145 T2.4). Promoting a third tier
would silently overturn that, and no default-path consumer holds ternary weights.

What *should* change is which tier a ternary consumer reaches for: **on CPU,
trit-packed is now the better default choice** — smaller and faster, with
bit-identical scalar results. Recorded in the tier doc.

## Reproduction

```bash
CARGO_TARGET_DIR=/tmp/t582r cargo test --release -p katgpt-types \
  --features ternary_trit_pack --test bench_582_trit_pack_goat -- --nocapture

# G1 unit tests (10, incl. the cross-tier bit-identity + repack round-trip)
cargo test -p katgpt-types --features ternary_trit_pack --lib ternary_trit
```

`--release` is mandatory — a debug timing of a SWAR/LUT kernel is meaningless.
