# Issue 580: LIMIT-style adversarial retrieval fixture — measure the deferred 8-D recall ceiling

**Date:** 2026-08-10
**Type:** poc + proof
**Research:** [472](../.research/472_Embedding_Retrieval_Dimension_Capacity_Limit.md)
**Cross-ref:** riir-neuron-db Plan 324 / Benchmark 324, riir-ai Plan 524
**Status:** In progress — **T1, T3, T4, T5 DONE; T2 public legs done** 2026-08-10 ([Bench 574](../.benchmarks/574_retrieval_capacity_break_point.md), [Bench 576](../.benchmarks/576_limit_fixture_recall.md)). Remaining: T2's private index legs.

---

## Problem

Every retrieval quality number in the stack is measured on a **benign** qrel matrix:

- `ItemEmbedIndex` G1 — type-centroid queries against a schema-centroid-initialised catalogue, i.e. items of the same `ItemType` cluster *by construction*.
- `fast_knn` G1 — recall 1.0000 vs brute-force cosine (fidelity, not correctness).
- `DenseEmbedIndex` G1b — recall@10 100% on synthetic chunks.
- `riir-rag` G5 — a transitive-caller query with zero lexical overlap (a structural win, not a capacity stress).

`riir-neuron-db/src/dense_embed/mod.rs:10-14` states the gap outright: the Tier-2 two-stage rerank has a **"recall ceiling by 8-D stage-1"**, and Benchmark 324 defers measuring it ("measured when Plan 524 wires the full two-stage"). We have never measured what a *combinatorially dense* relevance structure does to the 8-D path.

Research 472 supplies the exact construction protocol. This is a documented-limitation hit, so it is actionable per skill §1.55.2.

## Construction (from arXiv:2508.21038 §5.2)

The paper's recipe, adapted — no LLM, no training, fully modelless:

1. Choose `n` docs so `C(n,2)` is just above the query count. Paper uses **46 docs → C(46,2)=1035 → 1000 queries**.
2. Assign each query a random attribute pair; give each of its 2 relevant docs that attribute. Pad all docs to equal attribute count with random non-query attributes.
3. Two corpus variants: **small** (only the 46 relevant docs) and **full** (46 + ~50k distractors).
4. A **synonym** variant that strips lexical overlap — this is what exposes BM25 (paper: −89.2%).

Expected shape of results, if the bound is real at `d=8`: single-vector 8-D cosine should fail LIMIT-small badly (the paper's 1024–4096-dim SOTA models get 19–54% recall@2); `diverse_retrieval` and `smooth_min_similarity` should help but not solve; BM25 should near-solve the lexical variant and collapse on the synonym variant.

## Tasks

- [x] **T1 DONE (2026-08-10)** — `crates/katgpt-core/src/limit_fixture/` behind an opt-in
      `limit_fixture` feature: `LimitConfig` (with `paper_small` / `paper_full`
      presets and `with_synonyms`), `build_limit`, `recall_at_k`,
      `modelless_embed_8`, `cosine`. Seeded, deterministic, no network, no new deps.
      **11 construction tests** assert it is genuinely the paper's adversarial
      structure: ground truth is exactly the attribute carriers, no two queries
      share a relevant set, attribute counts uniform, filler drawn from a disjoint
      slot range, synonym variant preserves relevance byte-identically.

      **Deviation, deliberate:** the embedder is a *reference* modelless embedder
      matching riir-ai's `ModellessEmbedder` in shape (BLAKE3 + DFT + sigmoid →
      `[f32; 8]`, weightless) but **not bit-exact**. Importing the real one would
      drag private game IP into the public engine, so the fixture lives in
      `katgpt-core` with a pluggable embedding and riir-ai can re-run it against
      the production embedder from its side. Flagged in Bench 576's scope limits.
- [~] **T2 PARTIAL (2026-08-10)** — the **public** legs are measured in
      [Bench 576](../.benchmarks/576_limit_fixture_recall.md), reported per leg and
      per k, not averaged:

      | leg / variant | R@2 | R@10 | R@20 |
      |---|---|---|---|
      | dense 8-D / plain | **0.045** | 0.229 | 0.434 |
      | dense 8-D / synonym | 0.038 | 0.209 | 0.436 |
      | lexical / plain | **1.000** | 1.000 | 1.000 |
      | lexical / synonym | 0.043 | 0.220 | 0.438 |

      **Headline: the modelless 8-D dense leg is statistically indistinguishable
      from random ranking** (0.045 vs a 0.043 chance floor) on *both* variants. It
      has no channel by which a shared attribute token can raise cosine — it is not
      weak on LIMIT, it is absent. That quantifies riir-rag's own
      "structural, not semantic" caveat.

      **Still open:** the index-specific legs — `ShardIndex::query` (3-candidate
      path), `query_k_nearest_cosine` (`fast_knn`), `retrieve_diverse` (wedge span),
      `ItemEmbedIndex::query_top_k`, `smooth_min_similarity` — all live in private
      repos and need a riir-neuron-db test consuming `katgpt-core/limit_fixture`.
      What is already settled is the paradigm they share: since plain 8-D cosine
      over a modelless embedding is at chance, none of them can inherit quality
      from the embedding — any gain must come from their own mechanism
      (diversification, multi-token aggregation, lexical fusion).
- [x] **T3 DONE (2026-08-10)** — synonym variant shipped; the paper's asymmetry is
      **CONFIRMED, and more extreme than the paper's**:

      | | plain | synonym | drop |
      |---|---|---|---|
      | lexical (ours) | 1.000 | 0.043 | **−95.7%** |
      | BM25 (paper) | 97.8 | 10.6 | −89.2% |
      | dense (ours) | 0.045 | 0.038 | −15.6% |
      | Qwen3 (paper) | 19.0 | 11.6 | −38.9% |

      Ours drops harder because the leg is pure token overlap with no idf
      smoothing. **The dense leg's smaller drop is vacuous** — it looks robust only
      because it never worked; a robustness ratio is meaningless without an
      absolute baseline.

      **Caveat:** measured with an idealised token-overlap leg, not the real
      `Bm25Index` (k1=1.2, b=0.75, `CodeTokenizer`), which ships in riir-neuron-db.
      Re-measuring there is folded into the remaining T2 work.
- [x] **T4 DONE (2026-08-10)** — harness `crates/katgpt-types/tests/capacity_break_point.rs`,
      results in [Benchmark 574](../.benchmarks/574_retrieval_capacity_break_point.md).

      **Answer: the measured break point does NOT track the theorem — it is ~2124×
      below it at `d=8`** (breaks at `n ≈ 10` vs a 20,706 ceiling), and the gap
      *widens* with `d` (5× at d=2 → 2124× at d=8). The paper's own
      theory-vs-free-embedding gap was 4.5×.

      Three arms isolate where the loss lives: `centroid` (3.8), `rocchio` (5.0),
      and a margin `perceptron` started from both heuristics (9.8), which finds a
      separating query whenever one exists. Perceptron beats the best heuristic by
      only ~2×, so **query construction costs ~2× and the remaining ~1000× is
      document geometry** — random unit vectors are simply not arranged so that
      every pair is jointly top-2-separable. Optimizing queries cannot recover it.

      **This reframes Research 472's priority ordering** for an 8-D modelless
      index: (1) document embedding construction, ~1000× headroom, dominant;
      (2) query construction, ~2×; (3) the Theorem 1 ceiling, real and provable
      but ~2124× from where we operate — a long-run constraint, not today's
      bottleneck. So `ItemEmbedIndex` being 213× past its k=5 ceiling is a
      second-order worry; the first-order question is whether schema-centroid init
      yields good enough geometry for the *realized* qrels, which is T5.

      Scope limits recorded honestly in the benchmark: upper bound on the gap (not
      an estimate — neither docs nor queries are optimized); worst-case ×
      worst-case (adversarial all-pairs qrel against unstructured random docs);
      `k=2` only, and the ceiling falls steeply with `k`, so the gap should narrow
      at larger `k` — untested.
- [x] **T5 DONE (2026-08-10)** — [Bench 576](../.benchmarks/576_limit_fixture_recall.md).
      The conditional in this task resolved to its second branch: **the lexical leg
      is the only one that survives plain LIMIT**, which is an argument for making
      `riir-rag` fusion weight workload-conditional.

      This **contradicts riir-rag's stated priority** — `riir_rag.md` makes the
      latent 8-D leg "the primary path", calls BM25 "the fallback … never the
      primary path", and folds additively so the latent term stays dominant. On
      this workload that ordering is inverted, and additive fusion with a dominant
      latent term would bury the only working leg.

      Kept honest two ways: LIMIT is *designed* to be attribute-matching, maximally
      favourable to lexical, and is not representative of riir-rag's actual target
      (structural code retrieval, where `graph_score` answers what neither leg can);
      and the synonym variant shows lexical alone is equally unsafe. The conclusion
      is not "promote BM25" but "the fusion weight should be workload-conditional",
      which `riir_rag.md` caveat 4 already flags as a future knob. This supplies the
      evidence to act.

      The wedge-diverse / smooth-min promotion question stays open — those legs are
      part of the remaining T2 work.

## Why this is worth the cost

The fixture is cheap (46–50k synthetic docs, no training, no GPU) and it converts "8-D feels small" into a measured break point with a known theoretical reference. It also becomes a permanent regression check: any future change to the embedder, the index, or the fusion weights can be re-run against it.

## Done when

The fixture is committed and deterministic, all five retrieval legs have recall@k numbers on both variants, and T4's measured-vs-predicted multiplier is recorded in `.benchmarks/`.
