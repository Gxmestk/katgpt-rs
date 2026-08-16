# Plan 574 — Modelless Clustered LM Head (activate the `mtp_cluster_*` substrate)

**Status:** Active — Phase 1
**Date:** 2026-08-16
**Owner:** katgpt-rs
**Related Research:** 026 (Gemma 4 MTP), 078 (MTP Cluster Top-K Efficient Embedder),
407 (Trees from Marginals — DFlash-TfM)
**Related Benchmarks:** 656 (MTP Metal batch-width floor)
**Feature flag:** `cluster_lm_head` (opt-in until GOAT passes)

---

## Why

`clustered_lm_head()` is **fully implemented and wired** into both forward call
sites (`forward.rs:752`, `forward.rs:1323`) but has **never run in production**,
because `TransformerWeights::mtp_cluster_classifier` and `mtp_cluster_map` are
always `None`. The only producer, `cluster_map_from_embeddings`
(`forward.rs:372`), is a **stub** — it ignores its `wte` argument and delegates
straight to `cluster_map_round_robin`. There is no classifier producer at all;
the three tests that exercise the path hand-build one.

Its in-source TODO requires a plan before implementation, and records that the
previously-cited "Plan 056 / riir-burner" owner was **bogus** (Plan 056 is
Bomber game-state MCTS; the `riir-burner` crate was never created). This plan is
that owner. `.docs/02_inference/mtp_threshold.md` still repeats the stale
riir-burner claim and must be corrected.

### Why now

Benchmark 656 measured `lm_head` as the **dominant matrix** in decode
(500 MB, 1.7–2.5 ms — larger than `attn_qkv` and `ffn_up` combined). A width-N
speculative verify computes `N × vocab` logits, so the LM head is also where
MTP/DDTree costs scale worst. Research 026 puts the reduction at **~100× at BPE
scale**; Research 078 records Gemma 4's shipping configuration as
`num_centroids=2048, top_k=32, active_tokens=4096/262144` (1.6%).

This is **modelless** — no gradients. Both artifacts are deterministic functions
of weights the model already ships.

## Design

`logit[t] = dot(hidden, lm_head[t])`. Define cluster `c`'s classifier row as the
centroid of its members' `lm_head` rows:

```text
centroid_c = mean(lm_head[t] for t in c)
dot(hidden, centroid_c) = mean(logit[t] for t in c)
```

So the stage-1 score is exactly the cluster's **mean logit** — a principled
proxy for "does the argmax live here". K-means over `lm_head` rows minimises
within-cluster variance, which is precisely what tightens that proxy. Both steps
are deterministic; nothing is trained.

**Clustering target is `lm_head`, not `wte`.** The stub's signature takes `wte`,
but the quantity being pruned is the `lm_head` matmul. When weights are tied
these coincide; when untied, `wte` is the wrong matrix. The signature changes.

### Cost problem and the fix

Naive k-means assignment is `O(vocab × k × d × iters)`. At Qwen scale
(`vocab=152k, k=2048, d=5120`) that is ~1.6e12 FLOPs *per iteration* — hours.
Fix: a deterministic **Johnson–Lindenstrauss random projection** to `d' = 64`
before clustering (seeded, reproducible), then compute final centroids in the
**full** `d` space for the classifier. Assignment drops to
`152k × 2048 × 64 ≈ 2e10` — seconds with SIMD.

**Reuse decision (revised during T1 — the original plan was wrong).** The intent
was to reuse the proven `kmeans` in `katgpt-speculative/src/distill/ilc.rs:150`.
That is not reachable: it is a private `fn`, and its `distill` module is gated
behind `ilc_distill`/`trd_refined_draft`. Reusing it would have forced
`cluster_lm_head → ilc_distill` feature coupling on a would-be default-on
primitive. A local implementation ships in `cluster_build.rs` instead — it is
not pure duplication, since it needs projected-space assignment with
**full-space** centroid emission, which `ilc::kmeans` does not do.

## GOAT gate

The load-bearing metric is **argmax recall**: does the true `argmax(logits)`
token survive cluster pruning? Top-1 cluster selection can miss it — that is why
Gemma ships top-32-of-2048, not top-1.

- **G1 correctness** — with `topk >= num_clusters`, output is bit-identical to
  `standard_lm_head` (no pruning ⇒ no change).
- **G2 quality** — argmax recall ≥ 0.99 vs full LM head at the chosen
  `(num_clusters, topk)`. K-means must beat round-robin at equal `topk`, or the
  clustering adds nothing and round-robin ships instead.
- **G3 perf** — measurable LM-head latency reduction at BPE scale; report the
  active-token fraction alongside.
- **G4 no-regression** — default-off path unchanged.
- **G5 alloc** — builders may allocate (one-time, load-time); the *hot path*
  (`clustered_lm_head`) must stay alloc-free, as it already is.

Promotion to default-on requires G1–G5 **and** G2 beating round-robin — a
speedup on a wrong argmax is not a modelless gain.

## Tasks

- [x] **T1** Real k-means in `cluster_map_from_embeddings` (JL projection +
      reuse `ilc::kmeans`); retarget signature from `wte` to `lm_head`.
- [x] **T2** `cluster_classifier_from_map()` — full-space centroids
      `[num_clusters, n_embd]`.
- [x] **T3** Unit tests: determinism (same input ⇒ same map), full-coverage
      (every token in exactly one cluster), G1 bit-identity at
      `topk >= num_clusters`.
- [ ] **T4** GOAT bench: argmax recall + latency, k-means vs round-robin, swept
      over `(num_clusters, topk)`.
- [x] **T5** Correct the stale riir-burner/Plan-056 claim in
      `.docs/02_inference/mtp_threshold.md`.
- [ ] **T6** If G1–G5 pass and k-means beats round-robin → promote; else record
      the negative result and keep round-robin.

## Non-goals

- Loading Qwen-native `nextn`/`mtp.*` tensors (separate work).
- Batched / graph-fused forward (Benchmark 656 prerequisites).
- Any trained component. Weaver-style conditional restoration is riir-train.
