# Plan 574 — Modelless Clustered LM Head (activate the `mtp_cluster_*` substrate)

**Status:** Phase 2 COMPLETE — **G2b now PASSES (0.675 → 1.0000)**, still NOT
promoted to default-on. Blocked on Issue 661 (serial stage 2 → 0.08× on flat
data) and Issue 662 (no real checkpoint measured).
Follow-ups: Issue 661, Issue 662. Issue 657 RESOLVED (its diagnosis was refuted
by its own fix); Issue 658 (multi-layer predictor) is now moot — see T7.
**Date:** 2026-08-16
**Owner:** katgpt-rs
**Related Research:** 026 (Gemma 4 MTP), 078 (MTP Cluster Top-K Efficient Embedder),
407 (Trees from Marginals — DFlash-TfM)
**Related Benchmarks:** 656 (MTP Metal batch-width floor), 657 (Phase 1 GOAT —
the recorded FAIL), 658 (Phase 2 re-gate — supersedes 657)
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
- [x] **T4** GOAT bench — `.benchmarks/657_clustered_lm_head_goat.md`,
      `tests/bench_574_clustered_lm_head_goat.rs`.
- [x] **T5** Correct the stale riir-burner/Plan-056 claim in
      `.docs/02_inference/mtp_threshold.md`.
- [x] **T6** **NOT PROMOTED — negative result recorded.** G2a PASS (k-means beats
      round-robin 1.4×–4.7× at every matched budget), G3 PASS (4.68×), but
      **G2b FAIL**: best argmax recall **0.675** vs the 0.99 target, and 0.16 at
      a usable 5% budget. Per AGENTS.md a speedup on a wrong argmax is not a
      modelless gain, so `mtp_cluster_*` stays unloaded by default.
      Root cause diagnosed as the *scoring objective* (mean logit vs max logit)
      — **that diagnosis was wrong, see T7**.

## Phase 2 — Issue 657 (2026-08-16)

- [x] **T7** Radius bound + admissible stop shipped
      (`cluster_radii_from_map`, `cluster_head::clustered_lm_head_bounded`,
      `ClusterStop::{TopK, Admissible}`). **G2b PASSES: 0.675 → 1.0000 at a 2%
      active budget** — but Benchmark 658's 2×2 attribution shows the bound did
      not cause it. The defect was a degenerate k-means seeding (strided init at
      `stride = vocab/k` drew every centre from two planted groups); D² seeding
      alone yields 1.0000, while the bound *alone* makes recall worse (0.410 at
      25%). `ClusterInit::Dsquared` is now the default; `::Strided` survives only
      so the bench can keep attributing.

      The bound's value is the **admissible** stop: exact argmax after touching
      **7.30%** of the vocabulary on a clustered head, recall 1.0 by
      construction. On the unstructured control it needs **99.99%**.

      This also **moots T8/Issue 658** (multi-layer FUNCATTN predictor). It was
      ordered after 657 on the reasoning that "more layers cannot repair a wrong
      objective" — but the objective was never wrong, and a single-layer
      centroid score now achieves recall 1.0. There is no residual quality gap
      for a deeper predictor to close.

- [ ] **T8** **NOT PROMOTED (still).** Blocked on two measurements, not on
      quality. Both are cost questions; G1/G2 are settled:
      1. **Issue 661 — RESOLVED 2026-08-17, and the fix it proposed does not
         work.** Wave-parallelism is a wash (`wave: 8` 3.01× vs `wave: 1`
         2.96×, inside the noise). The shortfall is **locality**: the scattered
         gather runs at 20.6 GB/s where the full head streams at 108.0 GB/s, and
         `11.64× FLOP / 5.26× locality = 2.21× measured` closes exactly.
         Successor is **Issue 666** (permute `lm_head` rows into cluster order
         at load time). The crossover asked for *was* delivered: the clustered
         path wins only below **21.4–34.3% active**; enable below ~15%.
      2. **Issue 662** — both fixtures are synthetic extremes (planted-Gaussian
         vs uniform-random) with opposite verdicts. A real checkpoint decides
         it. Owned by riir-ai: katgpt-rs has no GGUF reader and must not
         path-depend on a private sibling.

## Non-goals

- Loading Qwen-native `nextn`/`mtp.*` tensors (separate work).
- Batched / graph-fused forward (Benchmark 656 prerequisites).
- Any trained component. Weaver-style conditional restoration is riir-train.
