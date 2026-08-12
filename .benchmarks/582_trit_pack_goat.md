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
| **G2b latency** | **0.87–0.91×** the bit-plane kernel — i.e. **1.10–1.15× faster** (gate was only "≤ 2× reject bound"). Was 0.78–0.82× against the *pre-Issue-583* baseline; the bit-plane kernel then got 11% faster, so the remaining gap is decode-only — exactly what the attribution predicted | **PASS, unexpectedly** |
| **G2c streaming** | ratio **0.872×** at 32768×5120 (34.5 MiB trit vs 42.5 MiB planes, both past L2) — **the same as cache-resident**, so the traffic saving does *not* compound. Prediction refuted | measured |
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

Median of 9 × 20 calls, three independent runs **against the bit-plane kernel as
it stood when this tier landed** (scale folded into the sign vector):

| Shape | trit ns | bit-plane ns | ratio | run 2 | run 3 |
|---|---|---|---|---|---|
| 512×512 | 33 131 | 40 631 | **0.82×** | 0.81× | 0.82× |
| 1024×1024 | 126 700 | 160 944 | **0.79×** | 0.80× | 0.79× |
| 512×5120 | 318 533 | 403 831 | **0.79×** | 0.79× | 0.79× |

Stable to ±0.01 across runs — far outside the 15% noise band this repo treats as
meaningless. **The trit tier is 1.22–1.28× faster than the bit-plane tier while
being 18.8% smaller.**

### Re-measured after Issue 583 (the moving baseline)

The attribution below sent ~10pp of that win back to the bit-plane tier: Issue
583 hoisted its group scale and made it 1.11× faster. Re-running the same gate
against the improved baseline:

| Shape | trit ns | bit-plane ns (hoisted) | ratio |
|---|---|---|---|
| 512×512 | ~34 000 | ~37 400 | **0.91×** |
| 1024×1024 | ~127 000 | ~140 000 | **0.90×** |
| 512×5120 | 319 975 | 368 677 | **0.87×** |

**0.87–0.91×, i.e. 1.10–1.15× faster.** This is the number to quote now, and it
is a *confirmation*, not a retreat: 0.79 × 1.11 ≈ 0.88, so the win that survived
the baseline improvement is precisely the decode half the attribution isolated
(scalar-vs-scalar, 0.90–0.91×). The two independent measurements agree.

## G2c — the streaming regime, and why the traffic story is wrong on CPU

The caveat in the first version of this benchmark said the cache-resident shapes
"structurally favour the bit-plane tier" and that the streaming regime "should
widen the gap". **It does not.** At 32768×5120 — 34.5 MiB of trits vs 42.5 MiB of
bit-planes, both far past the M3 Max's 16 MB L2, so every call pulls the weights
from RAM:

| | ms/call | ratio | effective bandwidth |
|---|---|---|---|
| trit | 20.513 | **0.872×** | 1.8 GB/s |
| bit-plane | 23.515 | — | 1.9 GB/s |

Same ratio as cache-resident (0.87–0.91×). The reason is in the last column:
**1.8 GB/s against an M3 Max roofline of ~400 GB/s — we are ~200× below the
memory limit.** Single-threaded ternary matvec runs at ~8 GMAC/s, and at
2.125 bits/weight that only demands ~2 GB/s. Even the 16-thread row-parallel
kernel (46 GMAC/s) would ask for ~12 GB/s. **CPU ternary GEMV is compute-bound,
not bandwidth-bound, so bytes-per-weight cannot buy latency there** — the 18.8%
is a *capacity* win (what fits in RAM/VRAM), and the 1.10–1.15× is a *decode*
win. They are unrelated effects that happened to arrive together.

This matters beyond this tier: it is the reason riir-ai Issue 628's GPU leg is
the interesting one. The Metal GEMV sits at 45% of its roofline and the CUDA dp4a
kernel at 88.9% of HBM peak — **those** are bandwidth-bound regimes where 18.8%
fewer bytes should convert directly into throughput. The CPU measurement here
says nothing against that; it says the CPU simply is not the place to look for it.

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

1. **AVX2 kernel written but UNMEASURED.** `avx2_trit_row_range` ships (one
   `_mm256_cvtepi8_epi32` unpacks 8 decoded trits per instruction), compile-
   verified via `--target x86_64-apple-darwin`, clippy clean. It cannot be
   *executed* here: Rosetta 2 does not implement AVX2, so
   `is_avx2_fma_available()` is false and the path is unreachable on this host.
   Queued for the 4090 alongside Issue 583 T4. Until then the x86_64 claim is
   compile-only.
2. **No real-model end-to-end.** The measurement is synthetic matvecs. The
   consumers that would benefit (riir-ai's Metal/CUDA forwards) run their GEMV on
   GPU, where this container is not yet implemented — that is riir-ai Issue 628.
3. **~~Cache-resident regime only.~~ Now measured (G2c) — and the prediction was
   wrong again.** Streaming a 42 MiB matrix gives the same 0.87× ratio as a
   cache-resident one, because CPU ternary GEMV runs ~200× below the memory
   roofline. Two predictions in this benchmark, both refuted by their own
   harness: the tier would be slower (it is faster), and streaming would widen
   its lead (it does not move). The footprint claim is arithmetic and stands; the
   *causal story* attached to it did not survive contact with a measurement.
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

```bash
# G2c streaming (allocates ~80 MB, #[ignore]d by default)
CARGO_TARGET_DIR=/tmp/t582r cargo test --release -p katgpt-types \
  --features ternary_trit_pack --test bench_582_trit_pack_goat \
  g2c -- --nocapture --ignored
```

`--release` is mandatory — a debug timing of a SWAR/LUT kernel is meaningless.

## Test-methodology fix found by G2c

The cross-tier comparison originally divided the error by `max(|want|, 1.0)`.
G2c's 32768 rows exposed that as wrong: row 190 came out at −0.3007803 vs
−0.3007903, a 1e-5 absolute difference that the metric scored as 3e-5 relative
and failed. But a row of a 5120-column matvec is the sum of **40 group sums of
magnitude ~3**, so a row landing near zero is catastrophic cancellation — 1e-5
absolute is ~5e-7 of the work actually done.

Dividing by the final value punishes rare near-zero rows and lets large rows off
lightly, exactly backwards. Both benches now use `assert_close_rms`: denominator
`max(|want|, rms(want))`, tolerance tightened to **1e-6**. Stricter than the old
metric in the common case, correct in the cancelling case. The old form only
looked fine because 512-row shapes rarely hit a cancellation; at 32768 rows it is
near-certain.
