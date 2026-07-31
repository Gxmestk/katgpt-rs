# Benchmark 437: recos GOAT Gate — G1 FAIL (do NOT promote)

**Date:** 2026-07-14
**Plan:** [437_recos_rearrangement_bound_similarity](../.plans/437_recos_rearrangement_bound_similarity.md)
**Research:** [421_Recos_Rearrangement_Bound_Similarity](../.research/421_Recos_Rearrangement_Bound_Similarity.md)
**Paper:** [arXiv:2602.05266](https://arxiv.org/abs/2602.05266) — Ai (2026), "Beyond Cosine Similarity"
**Feature flag:** `recos` (opt-in, stays opt-in per G1 FAIL verdict)
**Status:** G1 FAIL — recos does NOT beat cosine on OUR embedding regime for retrieval.

## TL;DR

recos fails the GOAT G1 quality gate on synthetic d=8 retrieval modeling the HLA /
style_weights regime. recos is a **better matching metric** (wider capture range —
inflates correct-pair scores), but a **worse retrieval metric** (Corollary 2 inflates
distractor scores too, reducing discrimination, and noise breaks the ordinal structure
recos depends on). The primitive stays opt-in as a diagnostic; **do NOT promote**.

## Method

Synthetic d=8 retrieval (1000 shards, 200 queries, 12 seeds):

- **Each shard** = its own random base vector in [-2, 2]⁸ + Gaussian noise (σ=0.1).
  This gives each shard a distinct ordinal structure (8! = 40320 possible orderings).
- **Query** = correct shard transformed by a random power-law `sign(v)·|v|^p` for
  `p ∈ [0.5, 2.0]` (preserves ordinal structure, breaks linear correlation) +
  Gaussian perturbation (σ=0.3, modeling observation noise).
- **Scorers**: cosine ranking `(cos)²·sgn(cos)` vs recos ranking `(dot/bound)²·sgn(dot)`.
- **Metrics**: recall@1, recall@5, win rate (recos > cosine per seed).

This is the corrected regime (each shard has its OWN base vector). The plan's original
design (shared base across all shards) was also tested and failed harder — all shards
shared ordinal structure, making recos score ~1.0 for every pair (zero discrimination).

## G1 Results (quality gate)

```
  seed      r1_cos     r1_rec     r5_cos     r5_rec    r1_win   r5_win
  ──── ────────── ────────── ────────── ──────────  ──────── ────────
     0      0.9550     0.7750     0.9950     0.9800   -0.1800❌  -0.0150❌
     1      0.9400     0.7700     0.9950     0.9700   -0.1700❌  -0.0250❌
     [12 seeds, all ❌ except 2 ties at r@5]
    11      0.9550     0.8150     0.9950     0.9800   -0.1400❌  -0.0150❌

  Mean recall@1: cosine=0.9475  recos=0.7829  Δ=-0.1646
  Mean recall@5: cosine=0.9967  recos=0.9850  Δ=-0.0117
  Win rate: r@1=0.0%  r@5=0.0%  (bar ≥80%)
```

**G1 verdict: FAIL.** recos is worse on both recall@1 (-16.5pp) and recall@5 (-1.2pp),
with 0% win rate across 12 seeds (bar was ≥80%).

## G2 Results (latency, informational)

```
  Single-pair:   cosine=0.3ns  recos=13.2ns   ratio=41-48×
  3-pair rerank: cosine=0.3ns  recos=41.5ns   ratio=156-158×  overhead=41.2-41.7ns
```

recos is ~40-160× slower than cosine due to the two d=8 sorts per call. The plan's
§"Open optimizations" notes a d=8 branchless sorting network could close this gap, but
is moot given G1 FAIL.

## Root cause analysis

**Why recos loses despite the paper's 98.6% win rate on STS:**

The paper's gain is on **semantic textual similarity (STS)** — a *matching* task where
the goal is to score HOW SIMILAR a pair is. recos's wider capture range (saturates at
1.0 under ordinal concordance) directly helps: it recognizes monotonic relationships
that cosine misses.

Our use case (`ShardIndex::query`) is **retrieval** — the goal is to rank the correct
item ABOVE distractors. This is a *discrimination* task. Here recos's wider capture range
works against it via two mechanisms:

1. **Corollary 2 inflation of distractor scores.** `|recos| ≥ |cos|` holds for ALL pairs,
   including distractors. recos inflates both correct and distractor scores; the net
   discrimination effect is not guaranteed to improve.

2. **Noise sensitivity.** recos relies on ordinal structure (ranking of components).
   Gaussian noise flips the order of close-valued components, breaking ordinal concordance
   on the correct pair. Cosine is more robust to noise because it measures linear
   correlation, which degrades gracefully.

**Diagnostic confirmation (clean case, no noise):** With `query = shard_a²` (exact
power-law, no noise) and `shard_b` a distractor with different ordering:
- recos discrimination = 0.3318, cosine discrimination = 0.3214 → recos is *slightly better*
- But this advantage vanishes with any realistic noise (σ ≥ 0.1).

## Promotion decision

**G1 FAIL → do NOT promote `recos` to default.** Keep opt-in as a diagnostic metric.

Per Plan 437 §"Promotion / demotion rules":
> If G1 fails: keep `recos` opt-in as a diagnostic; do NOT promote; document the
> negative result. The primitive still ships (zero cost unless called) for future
> embeddings where it may help.

**Phase 3 (cold MAG) and Phase 4 (hot ShardIndex) are BLOCKED** — there is no modelless
gain to wire into consumers. The primitive stays shipped behind `recos` feature flag for
future embeddings where ordinal concordance is the dominant signal and noise is low.

## UQ floor check

N/A — recos is NOT a UQ-bearing primitive (no probability/interval/coverage claim).
The "Report the Floor" rule (Issue 010) does not apply.

## Reproduction

```bash
cargo run --release --features recos --example recos_goat
```

Output is deterministic (fixed seeds 0-11).
