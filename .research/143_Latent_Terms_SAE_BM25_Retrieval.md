# Research 143: Latent Terms — Dense Retrievers Contain Extractable BM25-Ready Vocabularies

**Paper:** arXiv:2605.29384 (Clavié et al., 2026)
**Date:** 2026-05-31
**Verdict:** ❌ NO GAIN — Validates existing architecture, no actionable distillation

> **PASS-Redirects (synthesis):** Qdrant Team [https://qdrant.tech/blog/qdrant-1.19.x/ "Qdrant 1.19 - TurboQuant Datatype & Memory Tiers"] (2026-08-05) — vector-DB release. Per-Tenant IDF Statistics, Prefix Matching on keyword fields, and Slice Filter are **lexical-DB optimizations** that sit on the BM25-fallback side of our latent-first retrieval substrate (`riir-rag` + `riir-neuron-db::Bm25Index`). Our primary retrieval path is 8-D vector KNN (Clifford-wedge on `hla_moments`) + KG triples (k-hop BFS); BM25 is the explicit fallback for exact symbol matching, never the primary path. Per-zone vocabulary rarity (the game-context analog of per-tenant IDF) is modeled as a **latent** signal (zone embedding dot-product → NPC zone-attention), not a lexical IDF segmentation. Prefix matching is the lexical concern our CodeTokenizer + post-C13 AST chunker is planned to handle on the code-RAG path. Slice filter (deterministic disjoint subsets) maps to our `zone_hash`-keyed partitioning — zones ARE the disjoint subsets. §1.55.2 reverse-grep found ZERO documented gaps in `Bm25Index`/`riir-rag` mapping to any of these features. PASS: latent-first architecture handles the same problems via different primitives, no actionable gap.

---

## Paper Summary

**Latent Terms** shows that dense retrieval models (single-vector and multi-vector) learn representations that can be decomposed into sparse, BM25-searchable features via Sparse Autoencoders (SAEs) trained with only reconstruction loss. Key results:

1. SAE extracts a ~32K latent vocabulary with Zipfian collection statistics from frozen retrievers
2. BM25 over SAE features outperforms the base model's cosine similarity (Nomic+LT: 0.526 vs Nomic: 0.521 nDCG@10)
3. MaxSim scoring remains stronger than Latent Terms for multi-vector models (GTE-MC: 0.547 vs GTE-MC+LT: 0.500)
4. The latent vocabulary contains ~⅓ lexical + ~⅔ semantic features
5. No retrieval-specific training, no sparse supervision, no hard negatives needed

---

## GOAT Verdict: ❌ NO GAIN

### Why No Gain

| Angle | Assessment | Reason |
|-------|-----------|--------|
| **ScreeningPruner** | ❌ No mapping | BM25 requires corpus-level IDF statistics; at decode time there's no "corpus" of tokens. Per-token SAE projection + BM25 scoring would need pre-computed document frequencies over a token vocabulary — essentially entropy-based weighting, which Plan 061 already covers. |
| **MaxSim** | ✅ Validates existing | Paper confirms MaxSim > Latent Terms for multi-vector (0.547 vs 0.500). Plan 080 was the right call. No action needed. |
| **Embedding Router** | ❌ Marginal, high cost | riir-ai Plan 024 already has three-tier fallback routing. Adding SAE layer would add latency (SAE encode = matmul into 32K dims per token) for marginal quality improvement. |
| **Modelless distillation** | ❌ Not modelless | SAE training requires A100 GPU, 30B tokens, 2 hours per run. This is model-based infrastructure, not modelless distillation. |
| **ConstraintPruner** | ❌ No mapping | ConstraintPruner.is_valid() is binary syntactic validation. SAE→BM25 produces continuous relevance scores, which is ScreeningPruner's domain. And ScreeningPruner already has better relevance signals. |
| **RTPurbo** | ❌ Orthogonal | RTPurbo uses learned 16-dim projections on attention heads for sparse decode. Different mechanism (top-p token selection), different domain (attention patterns, not representation structure). |

### δ-Mem Lesson Applied

Research 024 (δ-Mem) showed that corrections derived from representational structure are often **too small to flip branch ordering in DDTree**. The same applies here: SAE features would provide marginal relevance adjustments that don't change which branches the tree explores. The ScreeningPruner already gets strong relevance signals from BanditPruner Q-values and WASM validators.

### When This Would Be Gain

If the codebase ever builds a **document retrieval / RAG layer** (e.g., for the NPC Dialog Engine in riir-ai), Latent Terms would be directly applicable. The current focus (speculative decoding, constraint pruning, game AI) doesn't have a retrieval component where this technique helps.

---

## Distillation Map

| Distillation | Target | Applicable? |
|-------------|--------|------------|
| D1: SAE encoder as ScreeningPruner | Token relevance scoring | ❌ No corpus for IDF |
| D2: Latent vocabulary for routing | Embedding Router | ❌ Latency cost > quality gain |
| D3: Zipfian weighting for entropy scoring | Entropy Anomaly Detection | ❌ Already covered by Plan 061 |
| D4: BM25 over sparse features for RAG | NPC Dialog Engine | ⏳ Future — not in scope now |

---

## Key Takeaway

The paper's insight — "dense representations contain more structure than their scoring interface exposes" — is architecturally aligned with the codebase's philosophy (ScreeningPruner exposes more signal than binary ConstraintPruner, MaxSim exposes more than cosine). But the specific technique (SAE→BM25) is document-retrieval-specific and doesn't transfer to the token-level speculative decoding pipeline. **The best outcome from this research is validation that MaxSim was the right scoring choice (Plan 080).**
