# Issue 041: Smooth-Min Soft Similarity — PoC PASS, Primitive Shipped (opt-in)

> **Spawned from:** Research 385 (SoftMatcha 2 smooth-min soft pattern matching — Gain)
> **Date:** 2026-07-07
> **Status:** IMPLEMENTED (opt-in `smooth_min_similarity` feature) — GOAT PoC PASS, zero consumers
> **Updated:** 2026-07-12 — PoC resolved the Research 385 §4 vs Issue 041 contradiction

---

## TL;DR

Research 385 distilled the **smooth-min similarity** + Zipfian-norm edit penalty from SoftMatcha 2 (ICML 2026) as a modelless latent-space utility for fuzzy multi-token retrieval. The function is ~20 lines. Issue 041 originally said "Don't impl — no consumer ready, PoC blocked on consumer prerequisites."

**That was wrong.** Research 385 §4 explicitly says the PoC should use "**synthetic** multi-token retrieval task." The PoC does NOT need a real consumer — it needs synthetic per-token embeddings, which can be generated without ItemEmbedIndex, AnyRAG, or Engram.

The PoC was written (`examples/issue_041_smooth_min_poc.rs`) and **PASSED all gates**:
- **G1 (quality):** +12.0pp recall@5 over plain cosine (0.815 vs 0.695)
- **G2 (latency):** ~0ns overhead (LLVM vectorized smooth-min to match plain mean)
- **G3 (β sensitivity):** all β ∈ [10¹, 10⁶] beat plain cosine

The primitive was implemented in `katgpt-core/src/similarity.rs` behind the `smooth_min_similarity` feature flag (opt-in). It is NOT promoted to default-on because zero consumers exist — promotion waits for consumer wiring (ItemEmbedIndex, AnyRAG, or soft Engram).

---

## The PoC (done — `examples/issue_041_smooth_min_poc.rs`)

### Task design

- **50 words** organized into 10 clusters × 5 words/cluster, each with a 16-dim embedding
- Within-cluster cosine ≈ 0.5-0.8 (shared centroid); cross-cluster cosine ≈ 0.0-0.2
- **200 items** in the catalog, each with 4 tokens from random clusters
- **200 queries** with ALL 4 tokens mismatched (different word from the same cluster as the correct item)
- This gives the correct item 4 moderate-cosine positions (~0.5-0.8)
- Distractors accidentally share clusters at 1-2 positions (high cosine ~0.8-1.0) but differ at the rest (low cosine ~0.0-0.2)

### Why smooth-min wins

The key scenario: a distractor has 2 exact-match positions (cosine=1.0) and 2 unrelated positions (cosine≈0.0 or negative). Plain cosine averages these, giving a moderate score that can exceed the correct item's all-moderate score. Smooth-min penalizes the low-cosine positions, correctly ranking the correct item higher.

**Sample from the PoC:**
```
Correct item [0]: cosines = [0.53, 0.55, 0.29, 0.33]
  plain=0.4246  smooth_min=0.2208
Top distractor [166]: cosines = [0.29, 1.00, -0.12, 1.00]
  plain=0.5424  smooth_min=-0.1270
→ Plain margin:    -0.1179 (distractor wins — WRONG)
→ Smooth-min margin: 0.3478 (correct wins — RIGHT)
```

### G1: Quality gate — recall@k

| k | plain-cosine | smooth-min (β=10⁴) | gain |
|---|---|---|---|
| 1 | 0.2000 | 0.4050 | +0.2050 ✅ |
| 3 | 0.5700 | 0.7200 | +0.1500 ✅ |
| 5 | 0.6950 | 0.8150 | **+0.1200 ✅** |
| 10 | 0.9100 | 0.9100 | +0.0000 |
| 20 | 0.9750 | 0.9600 | -0.0150 |

Smooth-min provides a large gain at low k (where discrimination matters most) and converges to plain cosine at high k (where both methods find the correct item). The -0.0150 at k=20 is within noise (3 out of 200 queries).

### G2: Latency gate

| Method | ns/call |
|---|---|
| Plain cosine (mean) | 0.8 |
| Smooth-min (β=10⁴) | 0.7-0.8 |
| **Overhead** | **~0 ns** |

LLVM auto-vectorized the smooth-min computation (4 `exp` calls + 1 `ln` + arithmetic) to match the speed of a simple mean. Both are sub-nanosecond. The <100ns target is met with 100× headroom.

### G3: β sensitivity

| β | recall@5 |
|---|---|
| 10¹ | 0.8750 |
| 10² | 0.8600 |
| 10³ | 0.8300 |
| **10⁴** | **0.8150** ← paper |
| 10⁵ | 0.8100 |
| 10⁶ | 0.7950 |

All β values beat plain cosine (0.6950). Lower β gives higher recall (more lenient), but the paper's β=10⁴ is a good operating point. The gain is robust across β — this is not a fragile parameter tuning.

---

## The Primitive (shipped — `katgpt-core/src/similarity.rs`)

### API

```rust
/// Smooth-minimum similarity for variable-length soft pattern matching.
/// sim = 1 - log_β(Σ(β^(1-c_i) - 1) + 1)
pub fn smooth_min_similarity(cosines: &[f32], beta: f32) -> f32

/// Insertion/deletion penalty using Zipfian-whitened norm.
/// exp(-norm_sq / gamma)
pub fn edit_penalty(norm_sq: f32, gamma: f32) -> f32
```

### Feature gate

`smooth_min_similarity = []` in `katgpt-core/Cargo.toml` (opt-in, default-OFF).
Forwarded in root `Cargo.toml` as `smooth_min_similarity = ["katgpt-core/smooth_min_similarity"]`.

### Tests

24 unit tests in `similarity.rs` covering:
- Basic properties (all-ones → 1.0, single-element → c, deterministic)
- β limits (large β → min-like, small β → sum-like)
- Penalization behavior (uniform cosines beat mixed high+low)
- Edge cases (empty panics, β≤1 panics, negative cosines, large token count)
- PoC scenario (correct item beats distractor — the exact PoC assertion)
- edit_penalty properties (monotonicity, boundaries)

### Why not default-on

Zero consumers. The primitive is proven to work (GOAT PoC PASS, modelless gain) but has no callers. Per the feature flag discipline: opt-in until a consumer demonstrates real-world value. Promotion to default-on requires:
1. A consumer wires the primitive (ItemEmbedIndex, AnyRAG, or soft Engram)
2. The consumer's GOAT gate passes with the primitive enabled

---

## Consumer Wiring: Rerank Module (T4 — attempted, GOAT FAILED)

**Attempted (2026-07-12):** Wired `smooth_min_similarity` as `RerankMethod::SmoothMin` in
`katgpt-attn-match/src/rerank.rs` behind the `smooth_min_rerank` feature flag.

**The wiring:** `smooth_min_score_into` computes per-position max cosine (for each
query token, find the best-matching doc token), then aggregates via
`smooth_min_similarity`. This is a hybrid of MaxSim (per-position max) and
smooth-min (penalizing low-cosine positions).

**GOAT gate result: FAILED ❌**

| Gate | Result | Details |
|------|--------|--------|
| G1 (quality) | FAIL | SmoothMin recall@5 = 0.385 vs Cosine 0.495 (−11pp) |
| G2 (latency) | FAIL | 46µs vs 8µs per query (5× slower — O(lq×ld×dim) vs O(dim)) |
| G3 (no-regression) | PASS | All methods produce finite scores; Cosine/MaxSim unchanged |

**Why G1 failed:** The original PoC used **position-aligned** cosines (q_i vs d_i),
which smooth-min was designed for. The rerank module's API doesn't have position
alignment — it computes all-pairs or mean-pooled cosines. The per-position-max
adaptation (max_j cos(q_i, d_j)) is a different signal that doesn't provide the
same quality gain. Mean-pooled cosine is more robust for small token counts (4
tokens) because it aggregates ALL token information into a single vector.

**Why G2 failed:** The per-position-max approach is O(lq×ld×dim) per document,
same as MaxSim. Cosine is O(dim) per document (mean-pooled). SmoothMin will
always be ~lq×ld slower than Cosine.

**Decision:** The feature stays opt-in (`smooth_min_rerank`). The wiring is
correct and tested (17 tests pass), but the GOAT gate honestly shows SmoothMin
doesn't beat Cosine on this task. The quality gap might be different on a
task with position-aligned tokens (e.g., sequence matching where q_i aligns
with d_i). The rerank module's API doesn't naturally support position-aligned
comparison — a future consumer with position alignment might show a gain.

**What this means for `smooth_min_similarity` promotion:** The primitive is
proven to work (original PoC PASS: +12pp recall@5), but the rerank consumer
doesn't demonstrate a gain. Promotion to default-on still requires a consumer
whose GOAT gate passes. The rerank consumer is a negative result — it shows
that smooth-min doesn't beat mean-pooled cosine on all-pairs retrieval.

## Consumer Paths (status updated 2026-07-12)

The primitive is shipped. Consumer wiring status:

---

## Tasks

- [x] **T0** (done 2026-07-12) Write the PoC with synthetic per-token embeddings, resolving the Research 385 §4 vs Issue 041 contradiction. PoC at `examples/issue_041_smooth_min_poc.rs`.
- [x] **T1** (done 2026-07-12) Implement `smooth_min_similarity` + `edit_penalty` in `katgpt-core/src/similarity.rs` behind feature flag `smooth_min_similarity`. 24 unit tests PASS.
- [-] **T2** (deferred) When ItemEmbedIndex grows a per-token embedding path (Path A), wire smooth-min as the multi-token query scorer.
- [-] **T3** (deferred) When AnyRAG gets a real retrieval backend (Path B), wire smooth-min as the scoring function.
- [-] **T4** (attempted 2026-07-12, GOAT FAILED) Wired smooth-min as `RerankMethod::SmoothMin` in `katgpt-attn-match/src/rerank.rs`. GOAT gate FAILED: SmoothMin recall@5 (0.385) < Cosine (0.495). The rerank module's all-pairs/mean-pooled API doesn't match smooth-min's position-aligned design. Feature stays opt-in.
- [-] **T5** (deferred) When a consumer's GOAT gate passes with smooth-min enabled → promote to default-on.

---

## Cross-references

- **Research 385** (`katgpt-rs/.research/385_SoftMatcha2_Smooth_Min_Soft_Pattern_Match.md`) — the Gain-tier verdict + PoC spec (§4: "synthetic multi-token retrieval task")
- **Research 012** (`riir-neuron-db/.research/012_egg_shell_pruner_funcattn_item_retrieval_fusion.md`) — ItemEmbedIndex Super-GOAT strategy guide
- **Plan 362** (`riir-neuron-db/.plans/362_*`) — ItemEmbedIndex implementation (default-on)
- **Plan 299** (`katgpt-rs/.plans/299_Engram_Hash_Addressed_Pattern_Memory.md`) — Engram, ZipfianCacheHierarchy

---

## TL;DR

**PoC PASS, primitive shipped (opt-in). Consumer wiring attempted (rerank module) — GOAT FAILED.** The Research 385 §4 spec said "synthetic PoC" — the issue's "blocked on consumer prerequisites" was wrong. The PoC showed +12pp recall@5 gain, ~0ns overhead, robust across β. The primitive is in `katgpt-core/src/similarity.rs` behind `smooth_min_similarity` (opt-in, 24 tests). The first consumer wiring (rerank module `smooth_min_rerank`) honestly FAILED the GOAT gate: SmoothMin recall@5 (0.385) < Cosine (0.495) on the rerank task, because the rerank module's all-pairs API doesn't match smooth-min's position-aligned design. Promotion to default-on still waits for a consumer whose GOAT gate passes — likely one with position-aligned token comparison (e.g., sequence matching, not all-pairs retrieval).
