# Issue 666 — Clustered LM head: permute `lm_head` rows into cluster order

**Status:** **RESOLVED — shipped and measured. It worked, and by the predicted
mechanism.** Speedup 2.2× → **8.3×**; effective bandwidth 20.5 → **74.7 GB/s**
(71% of the full head's 105.0). Crossover moved 21–34% → **60–100% active**.
**Opened:** 2026-08-17
**Resolved:** 2026-08-17
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

## Outcome (2026-08-17)

Everything this issue predicted, measured over 3 runs — see
`.benchmarks/658` §Addendum.

| | scattered | **packed** | full head |
|---|---|---|---|
| effective bandwidth | 20.5 GB/s | **74.7 GB/s** | 105.0 GB/s |
| speedup @ 8.59% active | 2.22× | **8.27×** | — |
| share of the 11.64× theoretical | 19% | **71%** | — |
| crossover | 21.4–34.3% active | **60.2–100%** | — |
| random control | 0.11× | **0.52×** | — |

Wave size stays a wash under the packed layout (`wave` 1–8 all 8.8–11.0×),
independently re-confirming Issue 661: threading was never the lever.

**Enable condition, revised: below ~50% active** (was ~15%). The primitive now
wins at every structured spread tested, including 60% active.

## Tasks

- [x] `cluster_layout_from_map()` → `ClusterLayout { permuted, token_of_row,
      offsets }`, with `TiedPolicy` / `LayoutRefusal`.
- [x] `PackedHeadView` taking the flat layout.
- [x] Contiguous stage 2 — `simd_matmul_rows_parallel` per cluster span,
      scatter via `token_of_row`. Stage 1 (`rank_clusters`) and the wave loop
      (`wave_plan`) are **shared** with the scattered path, so the part that
      decides exactness cannot drift between them.
- [x] Unit tests: `packed_layout_is_bit_identical_to_scattered` (logits, cost,
      and visited-cluster list, over `TopK` and `Admissible`),
      `layout_covers_every_token_exactly_once` (spans tile the row space; each
      permuted row is its token's original row),
      `tied_embeddings_are_refused_unless_explicitly_accepted`.
- [x] Re-ran the bench §661a/§661b with the new layout. New bandwidth and
      crossover recorded above.
- [x] Crossover moved past 50% active — but **promotion is still declined**.
      Two reasons, neither of them quality:
      1. **Issue 662** — no real checkpoint measured. The enable condition is
         weaker now but it is still a condition, and it is still unchecked.
      2. **Memory.** 100% of `lm_head` is not a default-on cost. `TiedPolicy`
         refuses tied weights precisely so this cannot be inherited silently.

## Note on the tie guard

The check is on **storage identity** (`as_ptr()` + `len()`), not content
equality — that is what "tied" means at runtime, and a content compare over
`vocab × n_embd` would cost more than the layout it guards. Two separate
allocations holding equal values are already paying for two copies, so
permuting one adds nothing new and is correctly allowed through.

## Note

Do not skip the bit-identity test. Issue 657's fix passed every plausibility
check and still turned out to be attacking the wrong defect; the only thing that
caught it was measuring the alternative side by side. Same discipline here — keep
the scattered path runnable as the A-side.
