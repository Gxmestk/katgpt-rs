# Issue 650: PkmEpisodicStore IDF write gate — non-interference slot selection

**Research:** [katgpt-rs/.research/481_Sparse_Memory_Finetuning_TFIDF_Slot_Ranking.md](../.research/481_Sparse_Memory_Finetuning_TFIDF_Slot_Ranking.md)
**Source paper:** [arXiv:2510.15103](https://arxiv.org/abs/2510.15103) — "Continual Learning via Sparse Memory Finetuning" (Meta FAIR + Berkeley, Oct 2025)
**Target:** `katgpt-rs/crates/katgpt-core/src/product_key_memory/episodic.rs` (+ benches)
**Status:** Open
**Date:** 2026-08-14

---

## Problem

`PkmEpisodicStore::write` / `write_weighted` select write targets by raw retrieval activation (top-k(q)) — **TF-only ranking**. The paper's ablation (§6, Fig 6) shows TF-only selection learns comparably but **forgets significantly more** than TF-IDF selection, and the gap **widens as the write set shrinks** (t=50 vs t=500). Our per-event write top-k (≤64) is the smallest write set in the stack — the regime where the paper measures the largest IDF benefit.

Concretely: successive episodic writes to a shared PKM table overwrite slots that general/background queries rely on, eroding prior recall. Nothing in any of our write paths downweights generally-hot slots.

## Fix (distilled, modelless)

Rank write targets by `retrieval_weight × idf(slot)` where `idf` comes from a **static background access-count table** built once from a consumer-supplied background query corpus (mirrors the paper's static background indices stored in the checkpoint). The δ-rule update itself is unchanged.

```rust
pub struct BackgroundAccessStats<const N: usize> {
    n_batches: u32,
    slot_batch_counts: [u32; N],   // doc-frequency: # background batches that retrieved slot i
}

// idf(i) = ln((|B| + 1) / (1 + slot_batch_counts[i]))   — same math as riir-neuron-db Bm25Index IDF, applied to slot selection
// write_idf: score(idx) = weight[idx] * idf(idx) → top-t by score → V[idx] += gate·(target − V[idx])
```

Properties to preserve: u32 counts; zero-alloc (fixed arrays / existing `PkmScratch`); selection stays local to the write path; only the existing BLAKE3 table commitment crosses the sync boundary. Selection score is a ranking statistic, **not** a probability — not UQ-bearing, conformal floor N/A.

## Tasks

- [ ] **T1** — `BackgroundAccessStats<const N: usize>` type: `new`, `record_batch(&[slot indices])` (bumps counts for the distinct slots of one background batch), `build_background_stats(corporum of background queries, batch_size)` constructor, `idf(slot) -> f32`. Unit tests: idf monotone decreasing in count; idf = ln(|B|+1) for never-accessed slots.
- [ ] **T2** — `PkmEpisodicStore::write_idf` + `write_weighted_idf` variants (same signatures as existing + `&BackgroundAccessStats` param): fold `idf(idx)` into the selection score, take top-t by score, apply the unchanged δ-rule. Keep plain `write`/`write_weighted` untouched (baseline arms for the bench).
- [ ] **T3** — GOAT bench `benches/bench_481_idf_write_gate.rs` (feature `product_key_memory_episodic`):
  - **G1 (interference/retention — the load-bearing gate):** write fact set A, then fact set B (shared table, overlapping key space); measure recall(A) after B for three arms — IDF vs TF-only vs random-t control — at matched learning(B) (same final recall(B) within ε, tuned via `gate`). PASS: IDF recall(A) ≥ TF-only by a clear margin (target ≥ +10pp absolute; paper's regime suggests more). This is the defend-wrong PoC — the 11%-forgetting figure is from a 1.3B LLM, not our substrate; we re-measure.
  - **G2 (latency):** per-write overhead of the idf fold vs plain write — must stay O(k) multiplies + one top-t selection (µs-scale; report ns/write both arms).
  - **G4 (alloc-free):** 0 allocations across 1000 steady-state `write_idf` calls (counts table preallocated; scratch reused).
- [ ] **T4** — G3 no-regression: `bench_408_pkm_goat` + `bench_408_pkm_episodic_fusion` unchanged (plain write path untouched).
- [ ] **T5** — If G1–G4 PASS: make `write_idf` the documented recommended write path in the module docs (paper + our PoC evidence); evaluate promoting the whole `product_key_memory_episodic` module toward default at its next scheduled re-gate. If G1 FAILS: record the negative result in `.benchmarks/481_idf_write_gate.md` (honest-defend-wrong artifact) and keep IDF opt-in.

## Out of scope (follow-ups, cross-ref Research 481 §2.5)

- F3: consolidation-level batch aggregation (aggregate TF over a consolidation sleep-cycle pass, then rank, then write) — riir-neuron-db; gates on Research 387 F5's retention question (≥80% recall after 5 domain shifts).
- F2: hierarchical branch(RIZZ)×slot(IDF) non-interference — riir-ai.
- F4: background-relative SPEFT salience — riir-train.
