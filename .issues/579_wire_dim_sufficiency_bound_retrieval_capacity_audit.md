# Issue 579: `dim_sufficiency_bound` ships with zero production call sites — wire it + audit the 8-D indexes

**Date:** 2026-08-10
**Type:** refactor + proof
**Research:** [472](../.research/472_Embedding_Retrieval_Dimension_Capacity_Limit.md), [123](../.research/123_TopK_Dimensionality_Barrier_Retrieval.md)
**Plans:** [157](../.plans/157_sigmoid_margin_loss.md)
**Status:** Open

---

## Problem

`dim_sufficiency_bound(k, n)` shipped in Plan 157, is GOAT-proven (Benchmark 048, 7/7 proofs), is exported all the way up (`katgpt-types/src/simd/research.rs:176` → `simd/mod.rs:124` → `katgpt-core/src/lib.rs:584`) — and is **called from nothing but its own tests**. Every call site is in `katgpt-types/src/simd/tests.rs`. It is a validated orphan.

Meanwhile Research 123's "Theoretical Sufficiency Check" table audits `MaxSim` (d=64), `RtTurbo` (d=16), `EmbeddingRouter` (d=64) and `GoStyleEncoder` (d=32) — and **omits every default-on 8-D index in the stack**, which are precisely the most exposed:

| Index | d | Location | Default-on |
|---|---|---|---|
| `ShardIndex` (`hla_moments`) | 8 | `riir-neuron-db/src/shard/mod.rs:16,225` | always |
| `ItemEmbedIndex` | 8 | `riir-neuron-db/src/item_index.rs:54` | yes |
| `riir-rag` `LatentQuery.direction` | 8 | `riir-ai/crates/riir-rag/src/query.rs:11` | yes |
| `NeuronVessel` / `ExperienceNode` | 8 | `neuron_vessel.rs:55`, `experience_graph/node.rs:35` | yes |

Per Research 472, Theorem 1 at γ=0.1 gives `d=8` a ceiling of `n ≤ 20,482` (k=2), `267` (k=4), `44` (k=8). `ItemEmbedIndex` runs a **k=5** GOAT gate over **25,943** real Seal items, where the bound demands `d ≥ 19.2` — **2.4× under**.

This does **not** invalidate the passing gates (they measure a benign, centroid-clustered realized qrel matrix). It means there is no worst-case headroom, and nothing in the build tells us when we cross the line.

## Tasks

- [ ] **T1** Extend Research 123's sufficiency table with the four default-on 8-D indexes above, using both bounds: Research 123's `Θ(k log n)` and Research 472's exact `d ≥ ln C(n,k)/ln(1+1/γ)`. Record γ used.
- [ ] **T2** Add a worst-case variant alongside the existing bound, e.g. `dim_capacity_ceiling(d, k, γ) -> usize` returning max representable `n` (inverse of Theorem 1), zero-alloc, `const`-friendly. Feature-gate under the existing `sigmoid_margin` flag; do not add a new default.
- [ ] **T3** Wire a **debug-only** capacity assertion at index construction (`ShardIndex::from_shards`, `ItemEmbedIndex` build, `riir-rag` retriever init): if `n > dim_capacity_ceiling(d, k_typical, γ)`, emit a one-line warning naming the index, `n`, `k`, `d`, and the ceiling. Must be zero-cost in release — no hot-path checks.
- [ ] **T4** Document in `riir-neuron-db/.docs/04_consolidation_retrieval/` that `fast_knn`'s "recall@k = 100% within ε=1e-4" and `DenseEmbedIndex`'s "recall@10 = 100%" are **fidelity to the cosine ranking**, not retrieval correctness, and therefore confer no immunity to the capacity bound. This is the single most likely misreading.
- [ ] **T5** Note in the same docs that `ShardIndex::query` (`index/mod.rs:257`) scores only 3 candidates via a binary search on `embedding[0]`, so it is strictly weaker than true cosine top-1 — the capacity bound is an *upper* bound on what that path can do. Cross-link the existing tests that document it (`tests/hebbian_bridge_t44_compat.rs:197,212`).

## Non-goals

- **Do not raise `BELIEF_DIM` from 8.** `NeuronShard` is a frozen `#[repr(C)]` Pod (~368 bytes) with BLAKE3 commitments, Lean-proofed offsets and chain-committed layout. That is a sync-boundary + proof-invariant change, not a research follow-up.
- No new default-on feature. This issue adds instrumentation and documentation only.

## Done when

The sufficiency table covers every default-on retrieval index, the ceiling function is benched and tested, the debug assertion fires on a deliberately over-capacity fixture, and the two misreading risks (T4, T5) are documented where the recall numbers are published.
