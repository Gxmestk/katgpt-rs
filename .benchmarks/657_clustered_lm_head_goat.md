# Benchmark 657 — Plan 574 T4: Clustered LM Head GOAT

**Date:** 2026-08-16
**Harness:** `tests/bench_574_clustered_lm_head_goat.rs`
**Config:** vocab=32768, n_embd=512, cluster_size=128 (256 clusters), 200 probes
**Verdict:** **G2a PASS · G2b FAIL · G3 PASS → PROMOTION BLOCKED.** Stays opt-in.

---

## Results — recall at matched active budget

Argmax recall = fraction of probes where the pruned head returns the same token
the full LM head would have. Compared at **equal active fraction**, not equal
`topk` (see Methodology).

### Structured, groups == clusters (favourable case — the verdict regime)

| budget | k-means (topk) | round-robin (topk) |
|---|---|---|
| 2% | **0.0700** (4) | 0.0150 (5) |
| 5% | **0.1600** (15) | 0.0550 (12) |
| 10% | **0.2300** (88) | 0.1200 (25) |
| 25% | **0.6750** (102) | 0.4900 (64) |

### Structured, 64 groups vs 256 clusters (split penalty)

| budget | k-means | round-robin |
|---|---|---|
| 10% | **0.4900** (6) | 0.3250 (25) |
| 25% | **0.9350** (223) | 0.5450 (64) |

### Random control (no structure)

| budget | k-means | round-robin |
|---|---|---|
| 25% | **0.5000** (64) | 0.4200 (64) |

### G3 latency (structured, topk=32)

| | ms |
|---|---|
| standard | 1.6806 |
| clustered | 0.3588 |
| **speedup** | **4.68×** |

## Gate outcomes

- **G2a relative — PASS.** K-means beats round-robin at every budget in every
  regime, by 1.4×–4.7×.
- **G2b absolute — FAIL.** Plan 574 requires recall ≥ 0.99. Best observed is
  **0.675**, and only **0.16** at a usable 5% budget.
- **G3 perf — PASS.** 4.68× faster than the full head.
- **G1 correctness** — covered by the unit test: bit-identical to
  `standard_lm_head` when `topk >= num_clusters`.

**Promotion BLOCKED on G2b.** AGENTS.md: a perf gain on a wrong answer is not a
modelless gain. A 4.68× speedup that returns the wrong argmax ~84% of the time
at a 5% budget is exactly the prohibited case. `mtp_cluster_*` weights stay
unloaded by default.

## Diagnosis → Issue 657

At a 25% budget the selector already admits **102 of 252 clusters** and still
misses the argmax ~32% of the time. Were the hidden state uninformative,
admitting 40% of clusters would not leave a third of the mass unreachable — so
the defect is the **scoring objective**, not the input.

Stage 1 ranks by `dot(hidden, centroid_c)` = the cluster's **mean** logit, but
the question is which cluster holds the **max**. A cluster with one spike among
many low values scores poorly and is pruned despite owning the argmax.

Fix proposed in **Issue 657**: add `radius_c = max‖w_t − centroid_c‖` and score
with the Cauchy–Schwarz upper bound
`max_t logit ≤ dot(h, centroid_c) + ‖h‖·radius_c`. That is *admissible* — keep
every cluster whose bound beats the best exact logit found and the argmax
cannot be missed. One extra f32 per cluster. Still modelless.

**Issue 658** tracks the multi-layer / FUNCATTN predictor as a second lever,
explicitly ordered *after* 657 because more layers cannot repair a wrong
objective.

## Methodology — three errors that each moved the verdict

Recorded because each produced a *different, confident, wrong* answer:

1. **Compared at equal `topk`.** K-means clusters are uneven, so its top-`k`
   covered 7.82% of the vocabulary while round-robin's covered 25.00% — handing
   the baseline 3× the compute. This inverted the verdict to "round-robin wins".
2. **Coarse geometric budget grid.** Sweeping `topk ∈ {1,2,4,…,96,128,…}` stepped
   straight over k-means' optimum at `topk=102`, understating it as 0.40 vs the
   true 0.675.
3. **Recomputed ground truth inside the sweep.** `standard_lm_head` does not
   depend on `topk`; calling it per swept value repeated a full-vocabulary
   matmul hundreds of times and pushed the bench past a 10-minute timeout.

Final method: ground truth computed once; per budget, **binary search** the
largest `topk` whose active fraction fits (`active(topk)` is monotonic, so
bisection is exact in ~8 evaluations).

## Caveats

- Synthetic LM heads (planted-Gaussian and uniform-random), not a real
  checkpoint. Real output embeddings may cluster better or worse.
- The random control also shows k-means winning (0.50 vs 0.42). With no
  structure to find, this is probably a **norm effect** — k-means groups
  high-norm rows and high-norm tokens are likelier to be argmax. So part of the
  "structured" win is not semantic. Issue 657's radius bound would make that
  signal explicit rather than incidental.
- 200 probes ⇒ recall resolution is ±0.005; the gap to 0.99 is far larger than
  that, so the FAIL is not a sampling artifact.
