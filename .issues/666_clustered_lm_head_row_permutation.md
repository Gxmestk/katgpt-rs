# Issue 666 — Clustered LM head: permute `lm_head` rows into cluster order

**Status:** Open
**Opened:** 2026-08-17
**Owner:** katgpt-rs
**Blocks:** Plan 574 promotion (with Issue 662)
**Supersedes:** Issue 661 (wave-parallelism — measured as a wash)
**Evidence:** `.benchmarks/658_clustered_lm_head_admissible_goat.md`,
Issue 661 §"The real bottleneck"

## Problem

The admissible stop proves the exact argmax after touching **8.59%** of the
vocabulary — an **11.64× FLOP reduction** — but measures **2.21×**. Issue 661
established that threading is not the missing factor, and that the gap is
entirely accounted for by memory locality:

| path | bytes | time | effective bandwidth |
|---|---|---|---|
| full head (contiguous) | 67.1 MB | 0.6213 ms | **108.0 GB/s** |
| admissible (scattered) | 5.8 MB | 0.2805 ms | **20.6 GB/s** |

```text
11.64x (FLOP)  /  5.26x (locality)  =  2.21x  ==  measured
```

Cluster members are non-contiguous token IDs, so stage 2 gathers ~2 800
separate 2 KB rows rather than streaming one span. `simd_dot_f32` per row is
already optimal; the loss is upstream of it, in the access pattern.

## Fix

Permute the LM-head rows into cluster order **once, at load time**:

```text
lm_head_permuted[r] = lm_head[token_of_row[r]]
```

Cluster `c` then owns the contiguous row range `[offset_c, offset_c + len_c)`,
and stage 2 becomes a `matmul_parallel` over a contiguous sub-block per cluster
instead of a scattered gather. Two extra artifacts, both `Vec<usize>` sized by
vocabulary:

- `token_of_row: Vec<usize>` — row → token ID, to scatter logits back.
- `cluster_offsets: Vec<(usize, usize)>` — per-cluster `(start, len)`, replacing
  `Vec<Vec<usize>>` (which is also `num_clusters` separate heap allocations
  walked per call).

Deterministic from shipped weights — **modelless**, no training.

## Why this should work

The full head already achieves 108 GB/s on the same hardware with the same
`simd_dot_f32`; the only difference is contiguity. If the permuted layout
recovers even half the gap, the 8.59%-active operating point moves from 2.21×
toward 5–8×, and — more importantly — the Issue 661 crossover (currently
**21.4%–34.3%** active) moves substantially right, widening the band where the
primitive is enabled at all.

The `Vec<Vec<usize>>` → flat-offsets change is worth doing regardless: it
removes `num_clusters` pointer chases per decode step and makes `cluster_map`
one allocation instead of ~2 000.

## Cost and risk

- One extra `vocab × n_embd` copy at load time (~500 MB at Qwen scale, transient
  if the original can be dropped) and `2 × vocab` `usize`.
- **Does not compose with weight sharing.** If `lm_head` is tied to `wte`, the
  permuted copy cannot alias the embedding table — it is a genuine second
  allocation. Measure whether that is acceptable before promoting; on a tied
  2 B model it doubles the largest tensor.
- The permutation must be applied consistently to `lm_head`, the classifier, and
  the radii, or the bound stops being admissible. Assert it.

## Tasks

- [ ] `cluster_layout_from_map()` → `(permuted_lm_head, token_of_row, offsets)`.
- [ ] `ClusterHeadView` variant taking the flat layout.
- [ ] Contiguous stage 2: `matmul_parallel` per cluster span, scatter via
      `token_of_row`.
- [ ] Unit test: permuted path is **bit-identical** to the scattered path on
      every computed logit, and admissibility still holds.
- [ ] Re-run `tests/bench_657_clustered_lm_head_bound.rs` §661a/§661b. Report
      the new effective bandwidth and the new crossover.
- [ ] If the crossover moves past ~50% active, re-gate Plan 574 G3 and
      reconsider promotion (still gated on Issue 662's real checkpoint).

## Note

Do not skip the bit-identity test. Issue 657's fix passed every plausibility
check and still turned out to be attacking the wrong defect; the only thing that
caught it was measuring the alternative side by side. Same discipline here — keep
the scattered path runnable as the A-side.
