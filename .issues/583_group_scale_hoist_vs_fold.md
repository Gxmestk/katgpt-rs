# Issue 583: Hoist the group scale out of the bit-plane inner loop (retire the fold trick)

**Date:** 2026-08-12
**Status:** **T1–T3 DONE, hoist PROMOTED on aarch64 (2026-08-12). T4 (AVX2) implemented but UNMEASURED.**
See [Bench 583](../.benchmarks/583_scale_hoist_goat.md).
**Feature flag:** none — this is a change to the shipped `ternary_group_scale`
NEON kernel, gated by measurement, not by a flag
**Filed by:** [Bench 582](../.benchmarks/582_trit_pack_goat.md) §Attribution
**Related:** Issue 578 (closed, the tier), Issue 582 (the trit tier that exposed
this), [Bench 581](../.benchmarks/581_ternary_group_avx2_goat.md) (AVX2, same
fold structure)

---

## The finding

`simd_ternary_group_matvec` folds the group scale **into the sign vector**
(`±1 → ±scale`, one `vmulq_f32` per 4 lanes) so that its 4 accumulators can span
the whole row without a per-group reset or a per-group horizontal sum. Issue 578
documented that as the reason the kernel "keeps the out-of-order pipeline fed",
and measured the result at **1.29–1.31×** the row-scale kernel — an overhead
attributed to "the extra `vmulq` per 4 lanes".

Bench 582 built a second kernel (the trit tier) that does the opposite —
accumulate a group unscaled, one horizontal sum, one scale multiply per group —
and measured it **1.22–1.28× faster** than the bit-plane kernel. Attribution via
the scalar paths (which both already hoist) put **~10pp of that on the decode**
and **~10pp on the scale placement**.

That second half is a free win for the shipped tier: **no format change, no new
container, same weights.** The arithmetic says it should be:

| | per 128-weight group |
|---|---|
| fold (shipped) | **32** `vmulq` (one per 4 lanes) |
| hoist (proposed) | **1** `vmulq` + **1** `vaddvq` horizontal sum |

At `GROUP_SIZE = 128` the fold pays 32 multiplies to avoid one horizontal add.
The "keeps the pipeline fed" reasoning is sound in the abstract — a per-group
accumulator reset does serialize — but it was never A/B'd, and one hsum per 128
elements is far too infrequent to matter.

**~~Bonus: hoisting also moves the NEON path closer to the scalar reference.~~**
**REFUTED by measurement (2026-08-12).** The argument was that
`ternary_group_matvec_scalar` applies one scale per group — how `Q2_0_g128` is
defined — so matching that grouping should narrow the ~1e-6 divergence Issue 578
had to document. It does not: the *folded* kernel is frequently bit-exact against
scalar where hoisted sits at ~1e-7. Both are far inside the 1e-6 gate, so nothing
operational changes, but the claim is withdrawn — f32 rounding is not monotone in
association order. The case for hoisting is the 1.11×, nothing else.

## Tasks

- [x] **T1** Added `fmla_nibble8_unscaled` + a hoisted NEON row-range kernel beside
      the shipped one (both retained so the A/B is reproducible).
- [x] **T2** A/B'd at 512×512 / 1024×1024 / 512×5120, median of 9×20, ≥ 3
      runs — the same harness shape Bench 582 used.
- [x] **T3** Hoist won (1.11× median over 4 runs) → promoted. If hoist wins: make it the NEON path for
      `simd_ternary_group_matvec` (and therefore for the row-parallel + batch
      kernels, which delegate to the same row-range function). If it loses,
      record the negative result and **keep the fold** — Issue 578's reasoning
      would then be vindicated by measurement rather than assumed.
- [~] **T4** Same question for the AVX2 kernel (Bench 581), which mirrors the
      fold structure. **Cannot be measured on this M3 host** — file forward for a
      4090 run rather than guessing.

## GOAT gate

- **G1** the hoisted kernel stays within 1e-6 relative of
  `ternary_group_matvec_scalar` on every shape the existing
  `neon_matches_scalar_reference` test covers (6 shapes incl. ragged group /
  block / sub-4 tail), and the row-parallel kernel remains **bit-identical** to
  serial (it partitions rows; both must use the same row-range function).
- **G2** ≥ **1.10×** over the fold on the 3-shape median, stable across 3 runs.
  Below that, not worth touching a shipped kernel — record and stop.
- **G3** `katgpt-types` + `katgpt-transformer` tests unchanged; Issue 578's own
  G2 bench re-run (the `vs row-scale` ratio should *improve* from 1.29–1.31×).
- **G4** unchanged: 0 allocations per call.

**Modelless** — a kernel restructuring, no training, no numerics change beyond
association order (~1e-7, well inside the 1e-6 G1 gate; see the refuted "closer
to the reference" claim above).

## Non-goals

- **Changing the container or the wire format.** Same `TernaryGroupWeights`,
  same `TGPLSMA1`.
- **Touching the row-scale ternary kernel** (`plasma_path`), which has one scale
  per row and therefore nothing to hoist.
- **Metal/CUDA kernels.** riir-ai owns those; the equivalent question there is
  Issue 628.

---

## Measured outcome (2026-08-12)

| Gate | Result |
|---|---|
| G1 | **PASS** — hoisted within 1e-6 of the scalar reference (typ. ~1e-7) on 7 shapes; `parallel_matches_serial_bit_identical` still holds; 31 lib + 25 transformer tests pass |
| G2 | **PASS** — 1.11× median (1.08–1.12× per shape) over 4 runs, gate ≥ 1.10× |
| G3 | **PASS** — clippy clean default / feature-on / `--all-features` / `--all-targets`; Issue 578's own G2 *improved* from 1.29–1.31× to **1.16–1.21×** vs the row-scale kernel |
| G4 | **PASS** — 0 allocs / 1000 calls |

**Promoted on aarch64.** `neon_row_range_hoisted` is the dispatch target for both
`simd_ternary_group_matvec` and `simd_ternary_group_matvec_parallel` — they must
move together or the parallel-vs-serial bit-identity breaks. The folded kernel is
retained as `simd_ternary_group_matvec_folded` because it is half of the A/B.

**T4 (AVX2) is the one loose end.** `avx2_row_range_hoisted` +
`fma_nibble8_avx2_unscaled` are implemented and reachable via
`simd_ternary_group_matvec_hoisted`, but **the x86_64 dispatch was deliberately
left on the folded path**: this host is aarch64, AVX2's `horizontal_sum_256` is
more expensive than NEON's single `vaddvq`, and Bench 611 already produced one
case where a latency assumption inverted between architectures. Needs a 4090 run;
until then the claim stands only for NEON.
