# Issue 187 — KARC Householder+QL Parallelization (close the d_h=18_720 gap)

## TL;DR

Issue 186 T1-T5 PASS: the Householder+QL primitive is algorithmically correct
and 7-14× faster than Jacobi at `n ≤ 512`. But T6 (the `d_h=18_720` timing
trial) is **projected infeasible single-threaded** — cubic extrapolation from
the n=512 measurement (794 ms) gives ~10 hours wall at `d_h=18_720`, ~20×
over the ≤30-min feasibility target.

The parallelism is in the inner loops, not the outer iteration:

- **Householder tridiagonalization** has three O(n²)-per-reflection loops
  (matrix-vector multiply, symmetric rank-2 update, Q accumulation), each
  **row-parallel**. With `n-2` reflections → `O(n³)` parallel work.
- **QL eigenvector accumulation** rotates columns `i, i+1` of z across all
  `n` rows per Givens rotation — **row-parallel**. With `O(n²)` total
  rotations → `O(n³)` parallel work.

**Row-parallel rayon preserves bit-identity** because each row's work is
fully sequential; only the assignment of rows to threads varies. Using
fixed-chunk `par_chunks_mut` makes the result deterministic regardless of
thread count.

## The numbers (from Issue 186 T4 GOAT gate)

Single-threaded, release build:

| n | Householder+QL | Jacobi | Speedup |
|---|---|---|---|
| 64 | 310 µs | 2.5 ms | 7.92× |
| 128 | 3.4 ms | 36.6 ms | 10.62× |
| 256 | 73.5 ms | 687 ms | 9.35× |
| 512 | 794 ms | 10.9 s | 13.69× |

Effective throughput at n=512: ~0.5 GFLOPS single-threaded. Modern CPUs
sustain 5-50 GFLOPS single-threaded with FMA + SIMD. **This suggests the
current loops are not auto-vectorizing well** (strided access patterns, no
explicit `#[inline]` hints). Two opportunities:

1. **Rayon row-parallel** (8-16× expected on commodity hardware)
2. **Cache-friendly blocked access + auto-vec hints** (2-4× expected)

Combined 16-64× would bring `d_h=18_720` from ~10 hours → ~10-40 min wall.

## Decision (recorded 2026-07-20)

**Pursue rayon row-parallel first** as the highest-expected-value, lowest-
risk win. Blocked/tiled access is a follow-up if rayon alone doesn't close
the gap.

The feature `karc_householder_eig_par` (implies `karc_householder_eig`) gates
the parallel path. Both paths share the same public API and produce the same
numerical result (modulo thread-scheduling variance, which we eliminate by
using `par_chunks_mut` with a fixed chunk size — see §Determinism below).

## Acceptance criteria

- [ ] **T1** — File this issue.
- [ ] **T2** — Add `karc_householder_eig_par` feature (default-off, implies
  `karc_householder_eig`). Parallel path lives behind `#[cfg(feature =
  "karc_householder_eig_par")]`.
- [ ] **T3** — Parallel implementation of the three Householder hot loops
  (matrix-vector `p = β·A_sub·v`, symmetric rank-2 update `A_sub -= vwᵀ + wvᵀ`,
  Q accumulation `Q[i, block_start..n] -= s·v`) + the QL eigenvector
  rotation inner loop. All use `par_chunks_mut` with a fixed chunk size
  (default 16 rows).
- [ ] **T4** — Bit-identity preserved: identical input → identical output
  when the parallel feature is on. Same `tests.rs` parity tests pass under
  both features.
- [ ] **T5** — Benchmark: ≥ 4× speedup at n ≥ 256 (criterion; 4-core minimum
  expected from row-parallel work).
- [ ] **T6** — `d_h=18_720` timing trial passes within ≤30 min wall (or
  honest verdict if not — defer to T7).
- [ ] **T7** — Promotion decision: if T6 passes, promote
  `karc_householder_eig_par` to default-on; if `karc_forecaster` G1 also
  passes (Issue 185 T3 / Plan 308 T4.5), promote `karc_forecaster` to
  default-on.

## Determinism

`par_chunks_mut(chunk_size)` slices the target into disjoint `[T]` chunks
of exactly `chunk_size * stride` elements. Each chunk's work is fully
sequential. The chunks are deterministic (same input → same chunks → same
work per chunk). **The result is independent of the rayon thread count**.

Caveat: if `n % chunk_size != 0`, the last chunk is smaller. This is still
deterministic per-call — same input, same chunking, same output.

Caveat: chunk_size is a tuning parameter, NOT a numerical one. Changing
chunk_size does NOT change the numerical result (each chunk does the same
work it would do in the serial path).

## Out of scope

- SIMD intrinsics (`std::arch::x86_64`): the inner loops are simple enough
  that explicit intrinsics are likely not needed once rayon unblocks. Revisit
  if T6 still misses the target after rayon.
- GPU offload: would require a new dep (wgpu/cubecl). Out of scope for this
  issue.
- Lanczos / divide-and-conquer eigensolvers: bigger algorithmic change;
  revisit if even parallel Householder+QL is insufficient.

## Dependencies

- **Issue 186** — closed T1-T5; this issue picks up T6 (and T7, transitively).
- **Issue 185** — source issue; T3-T5 will close when T6 here closes.
- **Plan 308** — T4.5-T4.7 (G1 + GOAT + promotion) blocked on this.

## Re-evaluation triggers

- If T5 fails to show ≥4× speedup: investigate why (likely cache thrashing
  at large n) before deciding on T6.
- If T6 fails (parallel path still >30 min at d_h=18_720): file follow-up
  for cache-blocking + explicit SIMD before re-attempting.
- If T4 fails (bit-identity breaks): investigate the scheduling-dependent
  path before promoting.

## Related

- Issue 186 — Path B deliberation + T1-T5 closure.
- Issue 185 — T1+T2 implementation of the consumer (`low_rank_fit_jacobi_bstep`).
- Plan 308 — parent plan; T4.5-T4.7 promotion gate.
