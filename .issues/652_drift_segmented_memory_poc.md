# Issue 652: DriftSegmentStore — training-free drift-segmented multi-state memory PoC

**Research:** [katgpt-rs/.research/482_Dynamic_Linear_Attention_Drift_Segmented_Memory.md](../.research/482_Dynamic_Linear_Attention_Drift_Segmented_Memory.md)
**Source paper:** [arXiv:2606.10650](https://arxiv.org/abs/2606.10650) — "Dynamic Linear Attention" (OSU + UMich + ByteDance Seed, Jun 2026)
**Target:** `katgpt-rs/crates/katgpt-kv/src/drift_segment/` (new module beside `segment_checkpoint/`) + benches
**Status:** DONE — GOAT PASS ([Bench 635](../.benchmarks/635_drift_segment_goat.md), 2026-08-15)
**Date:** 2026-08-14 (filed) / 2026-08-15 (executed)

---

## Problem

Every bounded multi-state memory in the stack uses **content-blind** boundaries and **evict**-based capacity: `SegmentStore` cuts fixed 128-token segments and LFU-evicts; HOLA heap-evicts min β·‖e‖; nothing anywhere opens a memory slot on drift, and nothing merges adjacent slots by information density. DLA's theorem (Research 482 §1) says fixed blocking is strictly suboptimal on non-stationary streams — the regime game NPC streams and long contexts actually live in. The published open lane is the **training-free post-hoc** variant (no prior art found), which is exactly what our substrate can test without any GPU training.

## Fix (distilled, modelless)

`DriftSegmentStore<const K: usize, const D: usize>` — a chronologically ordered ring of K slots `{state accumulator, n_tokens: u32, info_sum: f32}`:

- **Boundary:** per-token score = `TemporalDerivativeKernel::surprise_norm()` on the key stream (O(d); dual-EMA already shipped in `katgpt-types::temporal`). `sigmoid_gate(score, τ)` ≥ threshold ⇒ append new slot (capacity full ⇒ merge first). Optional small-rank arm: relative-Frobenius state delta (paper-faithful, affordable at δ-Mem rank 8).
- **Capacity:** when full, merge the **adjacent** pair with `argmin (info_sum_i + info_sum_j)/(n_i + n_j)` (O(K) scan, K≈30), preserving chronological order.
- **Readout:** existing GRM-style sigmoid gating over slot summaries (query-dependent; sigmoid, never softmax; no learned λ).
- **Zero-alloc:** fixed arrays, no per-token heap; merge reuses slot storage in place.

Not UQ-bearing (recall metric, no probability/interval claim) — conformal floor N/A.

## Tasks

- [x] **T1** — Type + unit tests: boundary fires at planted change points (synthetic key-stream regime switches); capacity invariant (never > K slots); chronological order preserved across merges; density accounting (info_sum/n monotone bookkeeping); zero-alloc over 1000 steady tokens. — DONE: 7/7 unit tests in `drift_segment/mod.rs` (G4 zero-alloc asserted at bench level via CountingAllocator; the struct is heap-free by construction — fixed arrays only). Two empirical lessons baked into the tests: (a) random unit directions can be nearly parallel on unlucky seeds → Gram–Schmidt-orthogonalized planted change points; (b) a noise floor above τ permanently disarms the rising-edge detector (hysteresis trap).
- [x] **T2** — **G1 PoC bench (load-bearing, defend-wrong §3.6):** DONE — [Bench 635](../.benchmarks/635_drift_segment_goat.md): **drift 0.578 vs fixed-LFU 0.117 (change-point, +46.09pp) and 0.930 vs 0.180 (stationary, +75.00pp)**, both targets exceeded (+10pp / −2pp); also beats single-state (0.172/0.133). 16 paired seeds, matched budget (K=30 slots × identical representation × identical readout). The Area-7 training-free claim is now measured, not just argued.
- [x] **T3** — G2 latency: DONE — 12 ns/token observe (O(d + K), d=32 K=30), 307 ns/query readout; all arms reported.
- [x] **T4** — G3 no-regression: DONE — `segment_checkpoint` lib tests unchanged (38 PASS); `--all-features` combo clean (253 PASS); clippy 0 warnings (module + bench).
- [x] **T5** — Verdict recording: DONE — PASS recorded in [Bench 635](../.benchmarks/635_drift_segment_goat.md) (bench number allocated per `.benchmarks/.highwater` discipline — 634→635; the issue's prescribed 482 is the Research number and was never allocated as a bench). **Stays opt-in** per this task's wording: promotion is evaluated at the next re-gate (synthetic-only validation, no consumer wiring yet — the same bar as segment_checkpoint's blocked real-model NIAH). F2 + F3 filed: [riir-ai Issue 677](../../riir-ai/.issues/677_per_npc_episodic_belief_memory.md) + [riir-neuron-db Issue 595](../../riir-neuron-db/.issues/595_consolidation_density_aware_adjacent_merge.md).

## Out of scope (Research 482 §2.4)

- F2: riir-ai per-NPC episodic belief memory (surprise opens episodic slot, K per NPC, 20 Hz) — file after PoC passes.
- F3: riir-neuron-db consolidation density-aware adjacent merge (wake buffer / hope_compactor adjacency variant) — file after PoC passes.
- Learned λ readout (requires training — riir-train only if T5 fails and the trained path is ever pursued).
