# Benchmark 574: measured-vs-predicted retrieval break point (Issue 580 T4)

**Date:** 2026-08-10
**Issue:** [580 T4](../.issues/580_limit_style_adversarial_retrieval_fixture.md)
**Research:** [472](../.research/472_Embedding_Retrieval_Dimension_Capacity_Limit.md) (arXiv:2508.21038, ICLR'26)
**Harness:** `crates/katgpt-types/tests/capacity_break_point.rs` (feature `sigmoid_margin`)
**Reproduce:**
```bash
cargo test -p katgpt-types --features sigmoid_margin \
    --test capacity_break_point -- --nocapture
```
No GPU, no training, no external deps (inline xorshift64* PRNG, seeded).

---

## Question

Theorem 1 says a `d`-dimensional single-vector index cannot realize every top-`k`
subset above `dim_capacity_ceiling(d, k, γ)` documents. That is a **necessary**
condition — it does not claim a real embedding scheme reaches the ceiling. The
paper itself measured a **4.5× gap** between its theoretical floor and what
free-embedding optimization actually required (`d=4` predicted vs `d>18` measured
at `n=100`).

**What is our gap?** Sweep `n` upward on the LIMIT construction (all `C(n,2)`
pairs must be realizable as exact top-2) until it breaks, and compare.

## Setup

- **Documents:** `n` random unit vectors in `R^d`, seeded, **never optimized**.
- **Queries:** three modelless arms, no training, no autodiff.
  - `centroid` — normalized sum of the two relevant docs.
  - `rocchio` — relevant centroid minus the non-relevant mean.
  - `perceptron` — margin perceptron on the pairwise constraints, started from
    *both* heuristics. Iteration 0 evaluates the unmodified heuristic under the
    same success criterion, so this arm dominates the other two by construction
    (asserted). It finds a separating query whenever one exists, so it measures
    **realizability given the document geometry** rather than heuristic weakness.
- **Success:** cosine top-2 returns exactly the target pair, for **every** pair.
- 8 random corpora per `d`; the reported break point is the mean.

## Results (γ = 0.1)

| `d` | k-free floor | predicted ceiling (k=2) | centroid | mult | rocchio | mult | **perceptron** | **mult** |
|---|---|---|---|---|---|---|---|---|
| 2 | 6 | 16 | 2.0 | 8× | 2.2 | 7× | **3.0** | **5×** |
| 3 | 10 | 52 | 2.5 | 21× | 2.8 | 19× | **3.8** | **14×** |
| 4 | 13 | 171 | 3.8 | 46× | 3.8 | 46× | **4.8** | **36×** |
| 5 | 17 | 568 | 2.8 | 207× | 3.2 | 175× | **7.2** | **78×** |
| 6 | 20 | 1,882 | 3.2 | 579× | 4.0 | 470× | **6.8** | **279×** |
| **8** | **27** | **20,706** | 3.8 | 5522× | 5.0 | 4141× | **9.8** | **2124×** |

`mult` = predicted ceiling ÷ measured break point.

## Verdict — the capacity ceiling is not our binding constraint

**At `d = 8`, unoptimized embeddings break at `n ≈ 10`, against a predicted
ceiling of 20,706 — a 2124× gap.** The paper's own theory-vs-practice gap was
4.5×; ours is nearly three orders of magnitude larger, and it *widens* with `d`
(5× at d=2 → 2124× at d=8) because the ceiling grows superlinearly while the
measured break point grows roughly linearly.

**Where the loss actually comes from.** The perceptron arm beats the best fixed
heuristic by only ~2× at `d=8` (9.8 vs 5.0). So query construction accounts for a
factor of ~2, and the remaining ~1000× is **document geometry** — random unit
vectors simply do not arrange themselves so that every pair is jointly
top-2-separable. Optimizing queries harder cannot recover it; only changing how
document embeddings are *constructed* can.

This reframes the priority set by Research 472. The honest ordering of concerns
for an 8-D modelless index is:

1. **Document embedding construction** (~1000× headroom, dominant).
2. **Query construction** (~2×).
3. **The Theorem 1 ceiling** — real, provable, and ~2124× away from where we
   actually operate. It is a *long-run* constraint, not the current bottleneck.

Concretely: `ItemEmbedIndex` sits 213× past its k=5 ceiling, which Research 472
flagged as the most exposed artifact. This benchmark says the ceiling is the
*second-order* worry there — the first-order question is whether schema-centroid
initialization produces geometry good enough for the realized qrels, which is
what Issue 580 T5 (seed recall against known ground truth) should measure.

## Honest scope limits

- **Upper bound on the gap, not an estimate.** Neither documents nor queries are
  optimized, whereas the paper optimizes both. A trained embedder would land
  between our numbers and the ceiling.
- **Worst-case × worst-case.** The LIMIT construction demands *all* `C(n,2)`
  pairs be realizable (adversarial qrel) against *random* documents (unstructured
  geometry). Real workloads have benign, structured qrels — which is exactly why
  our shipped GOAT gates pass. This measures headroom, not observed failure.
- **`k=2` only.** Larger `k` was not swept (the sweep is `O(n³·iters)` per point).
  The ceiling falls steeply with `k` (20,706 → 44 at k=8 for d=8), so the gap
  should narrow considerably at larger `k` — untested, and worth a follow-up.
- Break points are small integers over random corpora, so adjacent `d` sit within
  noise of one another. Only the end-to-end trend (d=2 → d=8) is asserted; an
  earlier pairwise-monotonicity assertion failed for exactly this reason and was
  replaced.

## Invariants asserted (regression value)

1. No arm ever exceeds the Theorem 1 ceiling — it is a necessary condition, so a
   violation would mean the bound or the harness is wrong.
2. `perceptron ≥ rocchio` and `perceptron ≥ centroid` (dominance by construction).
3. A theory-to-practice gap exists at every `d`.
4. `dim_capacity_floor ≤ dim_capacity_ceiling(d, 2, γ)`.
5. The perceptron break point grows from `d=2` to `d=8`.
