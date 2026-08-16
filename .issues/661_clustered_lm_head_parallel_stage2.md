# Issue 661 — Clustered LM head: parallelize stage 2

**Status:** Open
**Opened:** 2026-08-16
**Owner:** katgpt-rs
**Blocks:** Plan 574 promotion (re-gate after this)
**Evidence:** `.benchmarks/658_clustered_lm_head_admissible_goat.md`

## Problem

The admissible stop proves the exact argmax after touching **7.30%** of the
vocabulary — a **13.70× FLOP reduction**. Measured wall-clock is only
**2.1–2.9×**, and on the unstructured control it inverts to **0.08×** (a 12×
regression).

The gap is not the method. `standard_lm_head` dispatches through
`matmul_parallel` (rayon across all `vocab` rows); `clustered_lm_head_bounded`
walks scattered token IDs on **one thread**. So the comparison is
serial-vs-parallel, and any regime where pruning is weak loses outright.

## Fix

Parallelize stage 2. Two candidates, both modelless:

1. **Rayon over selected clusters.** Simplest, but the admissible stop is
   *sequential by construction* — it needs `best_exact` from cluster `i` to
   decide whether to visit `i+1`. Parallelizing forfeits the early stop unless
   done in waves (evaluate a batch, sync `best_exact`, re-check the next
   batch's bounds). Wave size is a tunable.
2. **Gather then `matmul_parallel`.** Copy the selected clusters' `lm_head` rows
   into a contiguous scratch buffer and reuse the existing parallel matmul. Pays
   a `active_tokens × n_embd` copy to get rayon; worth it only when the active
   fraction is small enough that the copy is cheaper than the saved work.

Option 1 preserves the early stop; option 2 reuses shipped substrate. Measure
both — the crossover is an active-fraction threshold, and that threshold is what
should gate the primitive on/off at load time.

## Tasks

- [ ] Wave-parallel admissible stop (option 1), wave size swept.
- [ ] Gather + `matmul_parallel` (option 2).
- [ ] Add both to `tests/bench_657_clustered_lm_head_bound.rs` under the same
      interleaved protocol.
- [ ] Report the active-fraction crossover below which the clustered path beats
      the full head. That number is the load-time enable condition.
- [ ] Re-gate Plan 574 G3 on the winner.

## Note

The 0.08× control result is the *useful* half of this issue: it establishes that
the primitive must be **conditionally enabled**, not unconditionally on. Even
after parallelization there will be an active-fraction above which the full head
wins; the deliverable is that threshold, not just a speedup.
