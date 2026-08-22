# Benchmark 576: LIMIT fixture recall — the modelless dense leg is at chance (Issue 580 T1–T3, T5)

**Date:** 2026-08-10
**Issue:** 580 T1, T2 (public legs), T3, T5 (removed 2026-08-10 per noise-reduction rule — DONE; full content preserved in this benchmark + [Bench 574](574_retrieval_capacity_break_point.md) + riir-neuron-db [Bench 476](../../riir-neuron-db/.benchmarks/476_limit_recall_five_legs.md))
**Research:** [472](../.research/472_Embedding_Retrieval_Dimension_Capacity_Limit.md) (arXiv:2508.21038, ICLR'26)
**Prior:** [Benchmark 574](574_retrieval_capacity_break_point.md) (measured-vs-predicted break point)
**Harness:** `crates/katgpt-core/src/limit_fixture/` + `crates/katgpt-core/tests/limit_recall.rs` (feature `limit_fixture`)
**Reproduce:**
```bash
cargo test -p katgpt-core --features limit_fixture --lib limit_fixture
cargo test -p katgpt-core --features limit_fixture --test limit_recall -- --nocapture
```
No GPU, no training, no network, no new deps.

---

## Results — LIMIT-small (46 relevant docs, 1000 queries, k=2)

Chance floor at k=2 is **0.043** (`k / n_docs`).

| leg / variant | R@2 | R@10 | R@20 |
|---|---|---|---|
| dense 8-D / plain | **0.045** | 0.229 | 0.434 |
| dense 8-D / synonym | **0.038** | 0.209 | 0.436 |
| lexical / plain | **1.000** | 1.000 | 1.000 |
| lexical / synonym | **0.043** | 0.220 | 0.438 |

Reported per leg and per `k`, never averaged across legs.

## Finding 1 — the modelless 8-D dense leg is statistically indistinguishable from random ranking

`0.045` against a `0.043` chance floor, on **both** variants. The modelless
embedder (BLAKE3 + DFT + sigmoid → `[f32; 8]`) has **no channel through which a
shared attribute token can raise cosine similarity**. It is not weak on LIMIT; it
is absent.

This is the quantification of a caveat `riir-rag` already documents —
*"modelless embedding is structural, not semantic … two structurally-similar but
semantically-different functions will score high cosine"* — and it is a sharper
statement than Benchmark 574's "breaks at `n ≈ 10`". On an attribute-matching
task the dense leg contributes **nothing at all**.

## Finding 2 — the lexical asymmetry reproduces, and is more extreme than the paper's

| | plain | synonym | drop |
|---|---|---|---|
| lexical (ours) | 1.000 | 0.043 | **−95.7%** |
| BM25 (paper) | 97.8 | 10.6 | −89.2% |
| dense (ours) | 0.045 | 0.038 | −15.6% |
| Qwen3 Embed (paper) | 19.0 | 11.6 | −38.9% |

The lexical leg **solves plain LIMIT outright** (1.000 — the task is
linguistically trivial by construction, so a token match reads the answer off)
and **falls to exactly the chance floor** on synonyms. Our drop is steeper than
BM25's because our lexical leg is pure token overlap with no idf smoothing, so it
has nothing at all to fall back on.

**The dense leg's smaller drop is vacuous.** It looks more "robust" to synonyms
only because it never worked in the first place — a reminder that a robustness
ratio is meaningless without an absolute baseline.

## Finding 3 — this contradicts `riir-rag`'s stated fusion priority

`riir_rag.md` specifies the latent 8-D leg as *"the primary path"* with
*"BM25 … the fallback for exact symbol matching, never the primary path"*, and
folds scores additively so the *"latent path's contribution [stays] dominant by
default"*.

On this workload class that ordering is **inverted**: the latent leg scores at
chance and the lexical leg scores 1.000. Additive fusion with a dominant latent
term would actively bury the only leg that works.

Two qualifications keep this honest:

- LIMIT is *designed* to be attribute-matching — maximally favourable to lexical.
  It is not representative of `riir-rag`'s actual target (structural code
  retrieval), where `graph_score` answers queries neither leg can, and where
  `ModellessEmbedder`'s structural similarity is the point.
- The synonym variant shows lexical alone is equally unsafe. Neither leg is a
  default; the correct conclusion is that **the fusion weight should be
  workload-conditional**, which `riir_rag.md` caveat 4 already flags as
  *"a future tuning knob"*. This benchmark supplies evidence to act on it.

## Finding 4 — distractor mass does not help

Growing the corpus 46 → 2,046 documents left the lexical leg at 1.000 (exact
token match is unambiguous regardless of corpus size) and could not improve any
leg. Asserted as an invariant, since a recall *increase* from adding irrelevant
documents would indicate a harness bug.

## Fixture fidelity (T1)

11 construction tests assert the fixture really is the paper's adversarial
structure, not merely a random corpus:

- `C(46,2) = 1035` is the smallest binomial above 1000 — the paper's sizing rule.
- Ground truth is **exactly** the set of documents carrying the query attribute.
- **No two queries share a relevant set** (otherwise the qrel matrix is less
  dense than LIMIT requires).
- Attribute counts are uniform across documents, so length carries no signal.
- Padding tokens are drawn from a slot range disjoint from query attributes, so
  filler can never accidentally satisfy a query.
- The synonym variant preserves the relevance structure **byte-identically**
  while removing all lexical overlap — that is what makes the two runs comparable.
- Distractors are relevant to no query; generation is seed-deterministic.

## Scope limits

- **The dense leg here is a *reference* modelless embedder**, matching riir-ai's
  `ModellessEmbedder` in shape (BLAKE3 + DFT + sigmoid → `[f32; 8]`, weightless)
  but **not bit-exact**. The real one is private; importing it would drag game IP
  into the public engine. So this characterises *the modelless 8-D regime*, not
  riir-ai's exact production embedder. Re-running against the real one is a
  riir-ai-side follow-up.
- **The lexical leg is idealised token overlap, not `Bm25Index`.** It exists to
  establish the asymmetry; the real BM25 measurement (with `CodeTokenizer`,
  k1=1.2, b=0.75) belongs in riir-neuron-db where the index ships.
- **The index-specific legs are not yet measured** — `ShardIndex::query` (the
  3-candidate path), `query_k_nearest_cosine` (`fast_knn`), `retrieve_diverse`
  (wedge span), `ItemEmbedIndex::query_top_k`, and `smooth_min_similarity`
  aggregation all live in private repos. **Issue 580 T2 DONE** — those legs measured in riir-neuron-db [Bench 476](../../riir-neuron-db/.benchmarks/476_limit_recall_five_legs.md).
  What *is* settled is the paradigm they share: plain 8-D single-vector cosine
  over a modelless embedding is at chance on LIMIT, so none of them can inherit
  quality from the embedding — any gain they show must come from their own
  mechanism (diversification, multi-token aggregation, lexical fusion).
- The paper's full variant uses ~50k distractors; the runtime-bounded test uses
  2,000. The invariant tested (distractors cannot help) is size-independent.
