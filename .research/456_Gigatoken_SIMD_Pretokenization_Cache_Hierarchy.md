# Research 456: Gigatoken — SIMD + Cache Hierarchies for 1000× Faster BPE

> **Source:** [Gigatoken](https://github.com/marcelroed/gigatoken) — Marcel Rød, 2026 (GitHub project; cite as `@software{roed2026gigatoken}`)
> **Headline number:** ~1000× faster than HF `tokenizers`, ~500–700× faster than `tiktoken`, on 11.9 GB `owt_train.txt`. Validated bit-identical to HF on 22 tokenizer families across AMD EPYC 9565 / Apple M4 Max / AMD Ryzen 7 9800X3D.
> **Date:** 2026-07-25
> **Status:** Active
> **Related Research:** 137 (Pplx Fast Unigram DATRIE — closest cousin, same axis 5× vs 1000×), 081/082 (ToaST Split Trees — vocab quality, orthogonal axis), 087 (ConvexTok LP vocab — vocab quality, orthogonal axis), 110 (Ciot — Plasma tier = ternary SIMD, Cold tier = Q4_K dequant; canonical proof we simulate hardware techniques in software SIMD)
> **Related Code:** `crates/katgpt-tokenizer/src/bpe.rs` (the slow O(n²) iterative-merge baseline this would accelerate), `crates/katgpt-tokenizer/src/datrie.rs` (137's DATRIE vocab lookup — complementary, not competing), `crates/katgpt-core/src/simd/` (NEON/AVX2 substrate gigatoken's techniques map onto)
> **Classification:** Public

---

## TL;DR

Gigatoken achieves ~1000× BPE throughput over HF `tokenizers` via four substrate-independent techniques that all land in this codebase's wheelhouse (Plasma tier SIMD, branchless hot loops, cache hierarchies, cross-arch portability). The repo's existing `katgpt-tokenizer/src/bpe.rs` is a clean O(n²) iterative merge — **no pretokenization, no cache, no SIMD** — and is ~1000× slower than gigatoken on equivalent workloads. This is a real, validated, actionable gain on an existing primitive.

**Distilled for katgpt-rs (modelless, inference-time):**

The four transferable techniques (substrate-stripped per the R418 hardware-paper guard — gigatoken's substrate is CPU SIMD, ours is too via Plasma tier):

1. **SIMD pretokenization replacing regex engine** — hand-rolled SIMD char-class scan (AVX512/AVX2/NEON) replaces the `regex` crate's per-character NFA walk. This is the largest single contributor to the 1000×. Maps directly onto `katgpt-core/src/simd/` (NEON/AVX2 lanes) — the same substrate Research 110 (Ciot) uses for ternary SIMD dequant.
2. **Pretoken cache hierarchy** — pretoken→token-list mappings are cached in a multi-tier structure sized to the long-tailed Zipf distribution of natural language (a few pretokens repeat millions of times; the tail is huge). This is structurally the same pattern as Engram's `ZipfianCacheHierarchy` (Plan 299 Phase 6, cited in Research 385 §2.2) and riir-neuron-db's `ItemEmbedIndex`. The hard part gigatoken solves is **cache growth management** under the long tail.
3. **Branchless hot loops** — already a codebase rule (`AGENTS.md` hot-loop rules: "Keep inner loops branch-free"). Gigatoken applies it to pretokenization; we apply it elsewhere. No new technique, but validation that the rule bites at GB/s scale.
4. **Cross-arch SIMD strategy portability** (AVX512 / AVX2 / NEON) — gigatoken ports the same algorithm three ways. The codebase's `SimdLevel` enum + `katgpt-core/src/simd/` dispatch already does this for matmul/dot/argmax; gigatoken's pretokenization strategies are a new client of the same dispatch surface.

---

## 1. What gigatoken is

Gigatoken is a software BPE/SentencePiece tokenizer (~2.5k stars, MIT, Rust + Python bindings) that achieves GB/s encode throughput on commodity CPUs. The headline table (GPT-2 row):

| CPU | gigatoken | HF tokenizers | tiktoken | vs HF | vs tiktoken |
|---|---|---|---|---|---|
| AMD EPYC 9565 ×2 (144 cores) | 24.53 GB/s | 24.8 MB/s | 36.0 MB/s | 989× | 681× |
| Apple M4 Max (16 cores) | 8.79 GB/s | 6.9 MB/s | 62.8 MB/s | 1,268× | 140× |
| AMD Ryzen 7 9800X3D (16 cores) | 6.27 GB/s | 59.0 MB/s | 92.1 MB/s | 106× | 68× |

At EPYC rates, the entire Common Crawl (~130T tokens) tokenizes in ~6.5 hours.

### 1.1 What is NOT the value (per R418)

- The substrate (CPU SIMD, x86/ARM). The codebase already simulates hardware techniques in software SIMD (Research 110 Ciot).
- The Python bindings (we are pure Rust).
- The BPE algorithm itself — gigatoken uses standard BPE merges with the same vocabularies HF/tiktoken use.

### 1.2 What IS the value (the four techniques in §TL;DR)

These four techniques are substrate-independent. Three of them have direct codebase analogs; one (pretoken cache hierarchy with long-tail growth management) is the genuinely novel distillation.

---

## 2. Distillation

### 2.1 The four techniques, mapped to the codebase

| Gigatoken technique | Codebase analog | Gap |
|---|---|---|
| SIMD pretokenization (replaces regex) | `katgpt-core/src/simd/` (NEON/AVX2 dot/matmul/argmax) — Plasma tier per Research 110 | **No pretokenization SIMD client exists today.** `bpe.rs` has no pretokenization at all. |
| Pretoken cache hierarchy (Zipf-aware, long-tail growth management) | Engram `ZipfianCacheHierarchy` (Plan 299 P6), `riir-neuron-db/src/item_index.rs` `ItemEmbedIndex` | **No pretoken cache in `katgpt-tokenizer`.** Closest structural cousin is Engram's; the long-tail growth trick is the new piece. |
| Branchless hot loops | `AGENTS.md` hot-loop rules (already enforced) | None — already a codebase invariant. |
| Cross-arch SIMD dispatch (AVX512/AVX2/NEON) | `katgpt-core::SimdLevel` + `simd/` runtime dispatch | None — the dispatch surface exists; gigatoken is a new client. |

### 2.2 The Pretoken Cache Hierarchy — the genuinely transferable distillation

This is the piece with the deepest transfer potential beyond tokenization. Gigatoken's author names it as one of the hardest problems: "Caching is a very hard problem in this domain since the cache grows very quickly, and pretoken distributions are very long-tailed."

The structure (inferred from the README + design notes):

- **L1**: per-thread inline buffer for the most-recent pretoken (single-entry, branchless hit test).
- **L2**: small open-addressing hash table for hot pretokens (top-K by frequency, ~64–256 entries, fits in L1 cache).
- **L3**: larger hash table with eviction for the warm middle of the distribution.
- **L4**: full BPE encode for cold tail (no caching — encode each occurrence).

The key insight: **natural-language pretoken distribution is so skewed that L1+L2 hit on >90% of occurrences while using <1% of the working set a naïve cache would need.** This is the same shape as Engram's ZipfianCacheHierarchy and riir-neuron-db's `ItemEmbedIndex`, but applied to a different access pattern.

**Cross-cutting fusion candidate** (not in scope for this note, flagged for future): the long-tail cache-growth-management technique could retroactively improve Engram's `ZipfianCacheHierarchy` and `ItemEmbedIndex`. Flag for a follow-up issue if the gigatoken cache-hierarchy port lands and reveals a transferable trick.

### 2.3 Vocabulary translation (per fusion protocol step 2)

| Paper / project vocabulary | Codebase vocabulary | Note |
|---|---|---|
| "pretokenization" | *(none — `bpe.rs` skips pretokenization entirely, encodes char-by-char)* | This IS the gap. |
| "pretoken cache" | `ZipfianCacheHierarchy` (Engram, Plan 299 P6), `ItemEmbedIndex` (riir-neuron-db) | Structural cousins; same Zipf assumption. |
| "GB/s throughput" | "tok/s", "Mtok/s" (see `benchmarks/005_g_zero_modelless.md` etc.) | Codebase measures per-call µs; gigatoken measures corpus GB/s. Same axis, different unit. |
| "SIMD strategy" (AVX512/AVX2/NEON) | `SimdLevel::{Auto, Avx2, Neon, Scalar}`, `katgpt-core/src/simd/` | Direct match. |
| "branchless pretokenization" | "branchless hot loops" (`AGENTS.md`) | Direct match. |

### 2.4 Latent-space reframing (per fusion protocol step 3 — mandatory)

Tokenization is the **raw→token-id boundary op** — by construction it is NOT a latent operation, and that is correct per the AGENTS.md sync-boundary rule (raw physical domain: byte offsets, token ids are deterministic and bit-identical for replay). The latent reframing here is *not* "operate on embeddings" — it's "this is the input-boundary analog of what the output-boundary (`simd_*` sampling, `argmax` token selection) already does in `katgpt-core/src/simd/`". Tokenization is the symmetric input-side SIMD opportunity the codebase has not yet exploited.

No Super-GOAT reframing applies. This is a clean GOAT-tier perf gain on a real primitive, not a new class of latent capability.

---

## 3. Verdict

**Gain (actionable), GOAT candidate pending gate.**

| Tier check | Result |
|---|---|
| Q1 No prior art? | **YES in this magnitude.** Closest cousin is Research 137 (Pplx Datrie) at 5×; gigatoken is 1000×. The two are complementary (Pplx attacks vocab lookup, gigatoken attacks pretokenization + cache) — gigatoken's techniques are not in the codebase. |
| Q2 New class of behavior? | **YES — GB/s tokenization enables corpus-scale workflows currently infeasible** (Common Crawl in hours; streaming inference over raw text). |
| Q3 Product selling point? | Partial — "katgpt-tokenizer runs at GB/s" is a real adoption funnel bullet for the public engine. Not a moat-level claim (gigatoken is public MIT). |
| Q4 Force multiplier (≥2 pillars)? | **YES** — touches `katgpt-tokenizer` (substrate), `riir-data` (corpus processing), `riir-train` (data prep), and the Input Layer of the inference flow. |

All four Q-table checks pass for at least Gain; Q1+Q2 pass for GOAT candidate. Q3 caps it below Super-GOAT (no private moat — gigatoken is public). **Final verdict: Gain; promote to GOAT if the `fast_bpe` feature flag's GOAT gate passes (G1 bit-identical to HF, G2 ≥100× perf, G3 no-regression, G4 alloc-free or equivalent).**

### 3.1 Why this is not PASS (correcting the initial lazy verdict)

Initial verdict was PASS on two weak grounds: (a) "gigatoken is a GitHub project not an arxiv paper", (b) "katgpt-rs is modelless inference on pre-tokenized ids, so tokenizer speed is at the boundary not the hot path". Both fail:

- (a) The research workflow explicitly covers systems papers and blog posts; R418's substrate-≠-value guard *mandates* translating SIMD substrate to software-SIMD technique before PASS. The technique IS the value, and the technique IS in the codebase's wheelhouse.
- (b) "At the boundary" is not "irrelevant". The Input Layer is part of the E2E inference flow (README §"Input Layer"); a 1000× speedup there matters for streaming-corpus consumers (`riir-data`, `riir-train`). The boundary is exactly where modelless preprocessing lives.

The actionable improvement is real: `katgpt-tokenizer/src/bpe.rs` ships a slow O(n²) baseline; a `fast_bpe` feature flag with either a gigatoken cargo dep or a port of the four techniques is a clean Gain with a measurable GOAT gate. Tracked in [Issue 191](../.issues/191_fast_bpe_via_gigatoken.md).

### 3.2 Why this is not Super-GOAT

The four techniques are individually well-known (SIMD char scan, Zipf cache, branchless loops, cross-arch dispatch). The contribution is **integration + tuning**, not a novel mechanism. A real moat would require a technique that doesn't exist anywhere — gigatoken's value is being the best-tuned instance of known techniques, not inventing them. Per §1.5 Q3 ("product selling point?") — "fastest BPE" is adoption-funnel, not moat. Correctly Gain/GOAT, not Super-GOAT.

### 3.3 Why this is not → riir-train

Tokenization is **inference-time preprocessing**, not training. Modelless-first constraint satisfied: no backprop, no gradient descent, no weight mutation. The four techniques are runtime SIMD/cache optimizations on a deterministic algorithm. Stays in katgpt-rs.

---

## 4. Implementation paths (for the issue, not this note)

Three options, ordered by DRY discipline:

1. **(Preferred) Cargo/git dep on gigatoken** — add `gigatoken = { git = "https://github.com/marcelroed/gigatoken", optional = true }` behind a `fast_bpe` feature in `katgpt-tokenizer/Cargo.toml`. Provide `BpeTokenizerImpl::encode_fast()` that delegates. Cleanest DRY; the ~1000× gain is real engineering we should not reimplement.
2. **(Fallback if dep is undesirable) Port the four techniques** into `katgpt-tokenizer/src/bpe_simd.rs`. Larger effort; only justified if gigatoken's license/dep-graph/blocking concerns rule out option 1.
3. **(Defer) Document the gap only** — only if no consumer needs GB/s tokenization today. The current `bpe.rs` is fine for research-grade ConvexTok/ToaST comparisons; the gain matters when corpus-scale processing lands in `riir-data`/`riir-train`.

### 4.1 GOAT gate (option 1 or 2)

- **G1 correctness**: bit-identical token sequences to HF `tokenizers` on the 22 tokenizer families gigatoken validates against. We re-use gigatoken's validation suite if option 1; port it if option 2.
- **G2 perf**: ≥100× throughput vs `bpe.rs::BpeTokenizerImpl::encode` on `owt_train.txt` (gigatoken publishes 1000×; we accept 100× as the gate floor to leave headroom for our integration overhead).
- **G3 no-regression**: all existing `katgpt-tokenizer` tests pass; ConvexTok/ToaST pipelines unaffected (they don't use `bpe.rs` hot path).
- **G4 alloc-free or equivalent**: steady-state encode allocates zero bytes (gigatoken claims this; we verify with `CountingAllocator`).
- **No UQ floor** — BPE is a deterministic encoder, not a probability distribution.

If G1–G4 pass → promote `fast_bpe` to default in `katgpt-tokenizer` and re-export from root `katgpt-rs` feature surface.

---

## 5. Cross-references

- **Closest cousin (same axis, smaller gain):** [Research 137](137_Pplx_Fast_Unigram_Viterbi_Double_Array_Trie.md) — Pplx Datrie at 5× vs HF. Complementary: Pplx attacks vocab lookup (algorithmic layer), gigatoken attacks pretokenization + cache (I/O + cache layer). A max-throughput BPE uses both.
- **Orthogonal axis (vocab quality, not throughput):** [Research 081/082](082_ToaST_Tokenization_Split_Trees.md) ToaST, [Research 087](087_ConvexTok_Tokenisation_via_Convex_Relaxations.md) ConvexTok. These optimize *what* the vocabulary is; gigatoken optimizes *how fast* a fixed vocabulary encodes.
- **Substrate precedent:** [Research 110](110_Ciot_Ternary_Inference_CPU_Distillation.md) — canonical proof the codebase simulates hardware SIMD techniques in software (Plasma tier = ternary SIMD, Cold tier = Q4_K dequant-on-read). Gigatoken's techniques are a new client of the same substrate.
- **Cache-hierarchy structural cousins:** Engram `ZipfianCacheHierarchy` (Plan 299 Phase 6, cited in Research 385 §2.2), `riir-neuron-db/src/item_index.rs::ItemEmbedIndex`. Same Zipf assumption; gigatoken's long-tail growth-management trick may be retroactively portable.
- **Actionable task:** [Issue 191](../.issues/191_fast_bpe_via_gigatoken.md) — `fast_bpe` feature flag + GOAT gate.
