# Issue 654 — MOP solver sparse-row fast path (one-hot zone-KG kernels)

**Filed:** 2026-08-15, from riir-ai [Bench 680](../../riir-ai/.benchmarks/680_mop_runtime_goat.md) §"G2 solve: honest FAIL analysis".
**Target:** `crates/katgpt-core/src/mop/solve.rs`
**Type:** optimization (bit-identical — no behavior change)
**Status:** OPEN

## Problem

`MopSolver::solve`'s inner product is the dense `simd_dot_f32(&p[i][k], &scratch.ln_z, N)`.
The first real consumer (riir-ai Plan 538 `mop_runtime`, zone-KG abstraction) feeds **one-hot
sparse rows**: a deterministic zone-level kernel has exactly 1 nonzero of N per `(s, a)` row
(blended abstractions have a handful). At N=64 the dense dot wastes 63 of 64 multiplies:

| Fixture (riir-ai Bench 680, release, M3 Max) | Solve | Iters | Effective MAC rate |
|---|---|---|---|
| Gridworld N=64/A=4 (one-hot) | 1.70 ms | 297 | 4.6 GFLOP/s |
| Stride-ring N=64/A=16 (one-hot) | 4.29 ms | 271 | 4.1 GFLOP/s |
| Dense reference (Bench 638, N=256) | 71 µs/iter | — | 14 GFLOP/s |

The primitive is at its dense optimum; the consumer's structure is sparse. riir-ai's G2-solve
gate (≤ 1 ms @ N=64/A=16) FAILs on this mismatch (honestly recorded — not papered over).

## Proposed fix

A sparse-row accumulation in `solve`'s two dot sites (the LSE sweep + the final
materialization) + `state_conditional_entropy`'s row walk: iterate only nonzero `p[i][k][j]`
entries (for one-hot rows: find the single `j`). Row sparsity is a property of the caller's
kernel — a runtime check per row (early scan or a caller hint) picks dense-SIMD vs sparse-scalar.

## Bit-identity argument (why this is NOT a re-derivation)

Skipping zero entries preserves the f32 result **exactly**:

- `0.0 · x = ±0.0` exactly for every finite `x` (sign per `signbit(x)` — irrelevant next);
- `acc + (±0.0) == acc` for every finite accumulator (the only edge, `+0.0 + (−0.0)`,
  still yields `+0.0`);
- all `ln_z` entries are finite by construction (pinned states hold 0.0; the iteration is a
  γ-scaled LSE of finite terms).

Therefore the sparse path computes the same f32 fixed point with the same iteration count —
the G1 golden-parity gate (Bench 638) must still pass **bit-identically**, and `priorities` in
`MopSolution` are untouched. The re-gate is the standard Bench 638 protocol (G1 parity + G2
re-bench + G3/G4) plus the riir-ai parity harness (6 tests) as a second oracle.

## Acceptance

- [ ] Sparse path in `solve.rs` (+ `state_conditional_entropy` walk) with a per-row structure
      check; dense-SIMD retained for dense rows.
- [ ] G1: Bench 638 golden parity bit-identical; riir-poc parity harness 6/6 unchanged.
- [ ] G2: re-bench — the riir-ai Bench 680 fixtures (gridworld + stride-ring) close toward the
      1 ms bound; record the new numbers in a katgpt-rs bench addendum + update riir-ai
      Bench 680's follow-up note.
- [ ] G3/G4: default suite + alloc-free witness unchanged.
- [ ] No new deps; no public-API change (`MopScratch`/`MopSolution` shapes untouched).

## Priority

Low-urgency, well-understood: riir-ai's promotion is blocked on G8 (civ arena) regardless, and
the solve is cold-tier (kernel-refresh cadence — 4.29 ms = 0.0086% of a 20 Hz budget at 1
refresh/s). But it IS the identified path to un-sticking the G2-solve gate before any promotion
evaluation.
