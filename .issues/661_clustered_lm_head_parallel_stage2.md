# Issue 661 — Clustered LM head: parallelize stage 2

**Status:** **RESOLVED — the proposed fix does not work, and the measurement
says why.** Wave-parallelism is a wash; the bottleneck is memory locality, not
thread count. The crossover this issue asked for is measured. Successor: Issue
666 (row permutation).
**Opened:** 2026-08-16
**Resolved:** 2026-08-17
**Owner:** katgpt-rs
**Blocks:** Plan 574 promotion — now via Issue 666 + Issue 662
**Evidence:** `.benchmarks/658_clustered_lm_head_admissible_goat.md`,
`tests/bench_657_clustered_lm_head_bound.rs` §661a/§661b

## Outcome

### 661a — wave-parallelism is a wash (option 1 FAILS)

`ClusterStop::Admissible { wave }` shipped: `wave` clusters are evaluated per
round before the bound is re-checked, with rayon inside a wave above a
512-token cutoff. Exactness is wave-independent (unit-tested). Structured
fixture, `active%` is deterministic, speedup is the interleaved per-pair median:

| wave | active% | vs wave 1 | speedup |
|---|---|---|---|
| 1 | 7.30% | 1.00× | 2.96× |
| 2 | 7.46% | 1.02× | 1.69× |
| 4 | 7.84% | 1.07× | 2.24× |
| 8 | 8.59% | 1.18× | **3.01×** |
| 16 | 10.05% | 1.38× | 2.68× |
| 32 | 14.60% | 2.00× | 1.43× |
| 64 | 26.45% | 3.62× | 1.01× |

`wave: 8` (3.01×) barely beats `wave: 1` (2.96×), and the run-to-run spread on
these ratios is wider than the gap. Beyond 16 it degrades outright: a coarser
stop check over-computes faster than rayon can absorb it (`wave: 64` touches
3.62× the tokens for no net gain). **Parallelism is not the lever.**

### The real bottleneck: locality, and the arithmetic closes

From the measured `wave: 8` row (structured, `standard 0.6213 ms`,
`admissible 0.2805 ms`, 8.59% active):

| path | bytes touched | time | effective bandwidth |
|---|---|---|---|
| full head (contiguous stream) | 67.1 MB | 0.6213 ms | **108.0 GB/s** |
| admissible (scattered rows) | 5.8 MB | 0.2805 ms | **20.6 GB/s** |

A **5.26× locality penalty**. And:

```text
11.64x (FLOP ratio, 1/0.0859)  /  5.26x (locality)  =  2.21x  ==  measured 2.21x
```

The gap between the FLOP reduction and the wall-clock is *fully* accounted for
by scattered access. Cluster members are non-contiguous token IDs, so stage 2
gathers ~2 800 separate 2 KB rows instead of streaming one 67 MB block. No
amount of threading fixes that — hence **Issue 666**, which permutes `lm_head`
rows into cluster order at load time so each cluster is one contiguous span.

### 661b — the crossover (the deliverable)

Walking the fixture's separability dial produces the curve Benchmark 658 could
only bracket by its two endpoints. `wave: 8`, exactness asserted at every point:

| group spread | active% | speedup | |
|---|---|---|---|
| 0.05 | 8.59% | **2.88×** | win |
| 0.07 | 13.74% | **1.71×** | win |
| 0.09 | 21.41% | **1.14×** | win |
| 0.11 | 34.30% | 0.16× | LOSS |
| 0.13 | 46.43% | 0.35× | LOSS |
| 0.15 | 60.21% | 0.26× | LOSS |
| 0.30 | 100.00% | 0.11× | LOSS |

**Crossover: between 21.4% and 34.3% active.**

Load-time enable condition, stated conservatively because the margin at 21%
is only 1.14×:

- **< 15% active → enable** (≥1.7×, comfortably clear of noise).
- **15–25% → marginal**, not worth the complexity at current locality.
- **> 25% → use the full head.** The loss below the crossover is severe
  (0.11–0.35×), so the default must be off and the check must be measured, not
  assumed.

This is a property of the *current* layout. Issue 666 should move the crossover
substantially to the right; re-measure after it.

---

## Original proposal (preserved)

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

- [x] Wave-parallel admissible stop (option 1), wave size swept.
      **Shipped and measured — it is a wash.** See §661a.
- [-] Gather + `matmul_parallel` (option 2). **Deferred as mis-aimed.** The
      implementation already gathers into a contiguous scratch and parallelizes
      above 512 tokens; handing that buffer to `matmul_parallel` instead would
      change the thread-scheduling shape, not the *gather*, which is the 5.26×
      cost. Issue 666 attacks the gather itself.
- [x] Add to `tests/bench_657_clustered_lm_head_bound.rs` under the same
      interleaved protocol (§661a, §661b).
- [x] Report the active-fraction crossover. **21.4%–34.3%**; enable below ~15%.
- [-] Re-gate Plan 574 G3 on the winner. Deferred to after Issue 666 — re-gating
      now would pin a threshold that the layout fix is expected to move.

## Note

The 0.08× control result is the *useful* half of this issue: it establishes that
the primitive must be **conditionally enabled**, not unconditionally on. Even
after parallelization there will be an active-fraction above which the full head
wins; the deliverable is that threshold, not just a speedup.
