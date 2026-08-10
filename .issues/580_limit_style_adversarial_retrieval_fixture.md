# Issue 580: LIMIT-style adversarial retrieval fixture — measure the deferred 8-D recall ceiling

**Date:** 2026-08-10
**Type:** poc + proof
**Research:** [472](../.research/472_Embedding_Retrieval_Dimension_Capacity_Limit.md)
**Cross-ref:** riir-neuron-db Plan 324 / Benchmark 324, riir-ai Plan 524
**Status:** Open

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

- [ ] **T1** Build the fixture as a modelless generator (seeded, deterministic, no network): `n`, `k`, attribute count and distractor count parameterised. Emit 8-D embeddings via the existing `ModellessEmbedder` path (`riir-ai/crates/riir-rag/src/embedder.rs:74`) so the test measures *our* embedder, not a stand-in.
- [ ] **T2** Measure recall@{2,10,20} for: `ShardIndex::query` (3-candidate path), `query_k_nearest_cosine` (`fast_knn`), `retrieve_diverse` (wedge span), `ItemEmbedIndex::query_top_k`, and `smooth_min_similarity` aggregation. Report each separately — do not average.
- [ ] **T3** Add the BM25 leg (`bm25.rs`) and the synonym variant. Confirm or refute the paper's asymmetry (lexical near-solves plain, collapses on synonyms) on our tokenizers (`CodeTokenizer` default).
- [ ] **T4** Sweep `n` upward until recall breaks, and compare the empirical break point against `dim_capacity_ceiling` from Issue 579 T2. **This is the real deliverable**: does our measured break point track the theorem, and by what multiplier? The paper measured 4.5× between theory and free-embedding best case; our modelless BLAKE3+DFT embedder should be worse still. Report the multiplier.
- [ ] **T5** Record results in `.benchmarks/`. If the wedge-diverse or smooth-min legs materially beat plain cosine on the fixture, that is a promotion argument for the corresponding riir-* gates; if BM25 is the only leg that survives, that is an argument for raising its weight in `riir-rag` fusion (see Issue 579 context and the additive-fusion caveat in `riir_rag.md`).

## Why this is worth the cost

The fixture is cheap (46–50k synthetic docs, no training, no GPU) and it converts "8-D feels small" into a measured break point with a known theoretical reference. It also becomes a permanent regression check: any future change to the embedder, the index, or the fusion weights can be re-run against it.

## Done when

The fixture is committed and deterministic, all five retrieval legs have recall@k numbers on both variants, and T4's measured-vs-predicted multiplier is recorded in `.benchmarks/`.
