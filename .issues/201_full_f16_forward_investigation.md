# Issue 201 — Full f16 (Weights + Activations) Forward Investigation

## Status: CLOSED — Phase 1 decision gate FAILED (2026-07-29)

See [Bench 563](../.benchmarks/563_issue201_f16_f16_fhm_negative.md) for the full
measurement + root-cause analysis. This file is removed per the noise-reduction
rule once the verdict is recorded; the negative result is preserved in Bench 563
+ git history (commit `f1a1a282`).

## Verdict

**Phase 1 T7 decision gate FAILED.** Best L3-exceeding speedup of
`simd_dot_f16_f16` (FHM widening FMA) vs `simd_dot_f32` = **1.31×**, under the
1.5× gate. Issue 201's hypothesis is refuted on Apple Silicon (M3 Max).

Phase 2 (full forward path) is NOT pursued — it depends on the Phase 1 gate,
which failed.

## Origin (historical)

Successor to [Issue 200](./200_f16_weight_quantization_forward_base.md), which
honestly closed the weight-only f16 path (G2 FAIL: 1.7× slower than f32 on
Apple Silicon). Issue 201 picked up the "full f16 (weights + activations)" path
that Issue 200 §"Alternative paths NOT taken" explicitly identified.

The hypothesis: widening FMA (`fmlalb`/`fmlalt`, a.k.a. `fmlal`/`fmlal2`) does
f16×f16→f32 in a single instruction, eliminating the explicit FCVT from Issue
200's critical path while achieving the full 50% bandwidth reduction.

## Why it failed (summary — full detail in Bench 563)

1. f32 is already near the bandwidth ceiling (~95–110 GB/s); halving bandwidth
   yields only ~25–30% (not 50%) because the kernel isn't purely bandwidth-bound.
2. FHM FMA throughput + accumulator-reduction overhead eat the rest of the
   theoretical gain.
3. f16 accumulation drift grows with vector length (6.2% rel_err at 16M) — even
   a marginal perf pass would have needed a separate precision gate.

Result: weight-only f16 (Issue 200) is 1.7× slower; full f16 (Issue 201) is
1.3× faster but short of the 1.5× promotion gate. **f32 stays the production
dtype** for `forward_base` GEMV on Apple Silicon.

## Toolchain note

FHM is inaccessible on stable Rust 1.93.0 (intrinsics unstable #136306; LLVM
21.1.8 assembler rejects the `fmlalb`/`fmlalt` mnemonic in every arrangement
form). The Phase 1 measurement was done via the nightly toolchain's unstable
intrinsics (`vfmlalq_low_f16` / `vfmlalq_high_f16`), verified correct on a known
input. Production code on stable would need verified `.inst` encodings — moot
given the gate failed.

## Outcome

Valid negative result (Research 003 / Issue 356 sense). The GOAT gate did its
job a second time on the f16 line (Issue 200 weight-only, Issue 201 full-f16),
preventing a perf-regressing "optimization" from reaching production. The
remaining f16-style path with a plausible ≥1.5× win would be INT8 with INT8
activations (different dequant path, out of scope, filed as a non-goal).
