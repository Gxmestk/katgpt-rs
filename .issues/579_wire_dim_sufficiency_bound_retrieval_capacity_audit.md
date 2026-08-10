# Issue 579: `dim_sufficiency_bound` ships with zero production call sites — wire it + audit the 8-D indexes

**Date:** 2026-08-10
**Type:** refactor + proof
**Research:** [472](../.research/472_Embedding_Retrieval_Dimension_Capacity_Limit.md), [123](../.research/123_TopK_Dimensionality_Barrier_Retrieval.md)
**Plans:** [157](../.plans/157_sigmoid_margin_loss.md)
**Status:** In progress — **T1, T2, T3, T4, T5 DONE 2026-08-10** (riir-rag call site remains)

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

Per Research 472, Theorem 1 at γ=0.1 gives `d=8` a ceiling of `n ≤ 20,706` (k=2), `269` (k=4), `122` (k=5), `44` (k=8) — and a minimum over *all* `k` of just **30 documents**. `ItemEmbedIndex` runs a **k=5** GOAT gate over **25,943** real Seal items: **213× past** the k=5 ceiling, 865× past the k-free floor, and `d ≥ 19.2` required vs 8 shipped.

This does **not** invalidate the passing gates (they measure a benign, centroid-clustered realized qrel matrix). It means there is no worst-case headroom, and nothing in the build tells us when we cross the line.

## Tasks

- [x] **T1 DONE (2026-08-10)** — added a second sufficiency table to Research 123 covering `ShardIndex` (k=2 and k=8), `ItemEmbedIndex`, `riir-rag` `LatentQuery` and `ExperienceGraph.task_embedding`, with both bounds side by side (γ=0.1 recorded) plus the k-free floor. Notably the two bounds *disagree in an informative way*: the positive Θ(k log n) bound flags `ShardIndex` k=2 as under-provisioned where Theorem 1 says it is fine, and vice versa at larger k. Original text: Extend Research 123's sufficiency table with the four default-on 8-D indexes above, using both bounds: Research 123's `Θ(k log n)` and Research 472's exact `d ≥ ln C(n,k)/ln(1+1/γ)`. Record γ used.
- [x] **T2 DONE (2026-08-10)** — shipped in `katgpt-types/src/simd/research.rs` behind the existing
      `sigmoid_margin` flag (no new default), re-exported via `katgpt-types::simd` and `katgpt-core`:
      - `ln_binomial(n, k) -> f64` — log-space `C(n,k)`; a direct evaluation overflows almost at once.
      - `dim_capacity_required(n, k, γ) -> usize` — Theorem 1 forward. **Reproduces the paper's Table 1
        cell-for-cell** (17 cells pinned in tests) — the strongest available correctness evidence.
      - `dim_capacity_ceiling(d, k, γ) -> usize` — Theorem 1 inverted (max representable `n`).
      - `dim_capacity_floor(d, γ) -> usize` — the O(1) k-free floor `d·log₂(1+1/γ)` (≈3.46·d at γ=0.1).
      11 tests, clippy clean.

      **Two findings from implementing it, both of which corrected this issue's own earlier text:**
      1. **γ must be `f64`, not `f32`.** Boundary cases are razor-thin: at `d=8, k=2, γ=0.1` the ceiling
         20,706 clears the cap by only `7.5e-8` nats while `0.1f32` perturbs the cap by `1.1e-7` —
         enough to flip the answer to 20,705. A test caught this.
      2. **The ceiling is U-shaped in `k`, not monotone decreasing** (as this issue originally claimed).
         It falls to a minimum near `k ≈ n/2`, then *rises* because `C(n,k) = C(n,n−k)` makes
         near-complete subsets easy. At `d=8, γ=0.1`: 30 at k=16, then 31 (k=20), 40 (k=32), 47 (k=40).
         The rising branch is the degenerate "retrieve almost everything" regime and must **not** be
         cited as headroom. The practical claim survives intact for `k ≪ n`, and the minimum yields a
         cleaner k-free statement: **a d=8 index can never represent all top-k subsets of more than 30
         documents, for any k.**
- [x] **T3 DONE (2026-08-10)** — `riir-neuron-db/src/capacity_audit.rs` + the
      `audit_retrieval_capacity!` macro, wired at `ShardIndex::from_shards` (k=8)
      and `ItemEmbedIndex::from_entries` (k=5). New opt-in `capacity_audit` feature
      pulling `katgpt-core/sigmoid_margin` (no new default, no new external deps).

      **Design change vs the task as written:** the trigger is the **k-free floor**
      `dim_capacity_floor(d, γ)` rather than `dim_capacity_ceiling(d, k_typical, γ)`.
      Keying on a guessed `k_typical` would make the warning wrong whenever a caller
      passes a different `k`; the floor is the minimum over all `k`, so it cannot be
      wrong about the configuration. The per-`k` ceiling is still reported in the
      message as context.

      Zero-cost: the macro body is `#[cfg(all(debug_assertions, feature = "capacity_audit"))]`,
      so release and default builds expand it to nothing. Even when active it is
      once-per-call-site (`AtomicBool` latch) and only at construction, never on a
      query path. Verified clippy-clean at default / `capacity_audit` / `--all-features`,
      6 tests, and the warning observed firing:
      `[capacity_audit] ...: n=25943 at d=8 exceeds the k-free capacity floor 27 by 961x (γ=0.1). At k=5 the ceiling is 122. ...`

      Remaining: the **`riir-rag` retriever init** call site (riir-ai repo) — separate
      repo, not wired here.
- [x] **T4 DONE (2026-08-10)** — new `riir-neuron-db/.docs/04_consolidation_retrieval/retrieval_capacity_bound.md` ("Misreading 1"), plus an anchor note at each site that publishes a recall number: `shard_index.md` (fast_knn) and `dense_embed_index.md` (G1b row, now footnoted). Original text: Document in `riir-neuron-db/.docs/04_consolidation_retrieval/` that `fast_knn`'s "recall@k = 100% within ε=1e-4" and `DenseEmbedIndex`'s "recall@10 = 100%" are **fidelity to the cosine ranking**, not retrieval correctness, and therefore confer no immunity to the capacity bound. This is the single most likely misreading.
- [x] **T5 DONE (2026-08-10)** — same doc, "Misreading 2", plus the note appended to `shard_index.md`; cross-links `tests/hebbian_bridge_t44_compat.rs:197,212`. Original text: Note in the same docs that `ShardIndex::query` (`index/mod.rs:257`) scores only 3 candidates via a binary search on `embedding[0]`, so it is strictly weaker than true cosine top-1 — the capacity bound is an *upper* bound on what that path can do. Cross-link the existing tests that document it (`tests/hebbian_bridge_t44_compat.rs:197,212`).

## Non-goals

- **Do not raise `BELIEF_DIM` from 8.** `NeuronShard` is a frozen `#[repr(C)]` Pod (~368 bytes) with BLAKE3 commitments, Lean-proofed offsets and chain-committed layout. That is a sync-boundary + proof-invariant change, not a research follow-up.
- No new default-on feature. This issue adds instrumentation and documentation only.

## Done when

The sufficiency table covers every default-on retrieval index, the ceiling function is benched and tested, the debug assertion fires on a deliberately over-capacity fixture, and the two misreading risks (T4, T5) are documented where the recall numbers are published.
