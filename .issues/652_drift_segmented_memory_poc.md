# Issue 652: DriftSegmentStore — training-free drift-segmented multi-state memory PoC

**Research:** [katgpt-rs/.research/482_Dynamic_Linear_Attention_Drift_Segmented_Memory.md](../.research/482_Dynamic_Linear_Attention_Drift_Segmented_Memory.md)
**Source paper:** [arXiv:2606.10650](https://arxiv.org/abs/2606.10650) — "Dynamic Linear Attention" (OSU + UMich + ByteDance Seed, Jun 2026)
**Target:** `katgpt-rs/crates/katgpt-kv/src/drift_segment/` (new module beside `segment_checkpoint/`) + benches
**Status:** Open
**Date:** 2026-08-14

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

- [ ] **T1** — Type + unit tests: boundary fires at planted change points (synthetic key-stream regime switches); capacity invariant (never > K slots); chronological order preserved across merges; density accounting (info_sum/n monotone bookkeeping); zero-alloc over 1000 steady tokens.
- [ ] **T2** — **G1 PoC bench (load-bearing, defend-wrong §3.6):** synthetic non-stationary streams (planted regime changes + needles to recall), three arms at **matched budget** (same slot count / bytes): (a) single-state accumulator (vanilla linear attention), (b) `SegmentStore` policy (fixed 128-token segments + LFU evict), (c) `DriftSegmentStore`. PASS: (c) beats (b) by a clear margin on change-point streams (target ≥ +10pp needle recall) AND is ≥ (b) on stationary streams (no regression where fixed blocking is fine). This measures the Area-7 training-free claim ourselves — the paper only proves the co-trained version.
- [ ] **T3** — G2 latency: per-token overhead stays O(d + K) (score + argmin scan); report ns/token all arms. G4 alloc-free (covered by T1 assertion in bench form).
- [ ] **T4** — G3 no-regression: existing `segment_checkpoint` benches/tests unchanged (new module, opt-in feature `drift_segment`).
- [ ] **T5** — Verdict recording: PASS → `.benchmarks/482_drift_segment_goat.md`, evaluate promotion at next re-gate, file F2 (riir-ai per-NPC episodic memory) + F3 (riir-neuron-db consolidation policy) follow-ups. FAIL → honest negative result in the same bench doc (co-training was load-bearing; training-free transfer does not hold), keep opt-in, note the riir-train Path-0.5 follow-up (DLA-style soft-gate co-training recipe) in the bench doc.

## Out of scope (Research 482 §2.4)

- F2: riir-ai per-NPC episodic belief memory (surprise opens episodic slot, K per NPC, 20 Hz) — file after PoC passes.
- F3: riir-neuron-db consolidation density-aware adjacent merge (wake buffer / hope_compactor adjacency variant) — file after PoC passes.
- Learned λ readout (requires training — riir-train only if T5 fails and the trained path is ever pursued).
