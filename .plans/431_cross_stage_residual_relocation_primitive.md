# Plan 431: Cross-Stage Residual Relocation Operator + Permeation-Map Diagnostic

**Date:** 2026-07-13
**Research:** [katgpt-rs/.research/417_Knowing_Using_Gap_Cross_Stage_Residual_Relocation.md](../.research/417_Knowing_Using_Gap_Cross_Stage_Residual_Relocation.md)
**Source paper:** [arxiv 2607.08393](https://arxiv.org/abs/2607.08393) — Dai, Rao, Wang et al., "Towards Mechanistically Understanding Why Memorized Knowledge Fails to Generalize in LLM Finetuning" (HKUST-GZ / HKUST, NeurIPS 2026 submission)
**Target:** `katgpt-rs/crates/katgpt-core/src/cross_stage_relocation/` (new module) + Cargo feature `cross_stage_relocation`
**Status:** Phase 1–4 COMPLETE. Phase 1–2 shipped (34 unit tests PASS, clippy clean). Phase 4 GOAT gate G1–G6 ALL PASS for katgpt-rs scope (see `.benchmarks/431_cross_stage_relocation_goat.md`). Phase 3 defend-wrong PoC DONE — verdict: **REFUTE the fixed-pair `LateEarly` default** (CLOBBERS in 2/4 clean configs because op_b overwrites op_a's recovery); the mechanism itself (single-op relocation) works. Primitive stays opt-in behind `cross_stage_relocation`; promotion to default blocked on a real-domain PoC with CUSTOM relocation (not the fixed default).

---

## Goal

Ship two modelless primitives distilled from the Knowing-Using Gap paper:

1. **Permeation-Map Diagnostic** — `permeation_scan` that scans `(source_stage, target_stage)` pairs and reports the paper's L×L heatmap. Reuses `direct_effect_importance` (Plan 358) as the cell score. **Safe half — purely diagnostic, no behavior change.**

2. **Cross-Stage Residual Relocation Operator** — `RelocateOp` that snapshots an anchor's residual state at one stage and overwrites at another during a forward pass. Ships with the paper's fixed two-pair default `(0.82L→0.45L) + (0.10L→0.45L)` as `RelocatePair::LateEarly`. **Risky half — applied operator, needs PoC.**

**GOAT gate rule:** opt-in feature flag `cross_stage_relocation`. Promotion to default requires G1–G6 PASS **AND** the §3.6 defend-wrong PoC (Phase 3) confirming the operator actually recovers capability on a toy domain (not just architectural coverage).

**Honest caveat:** the paper proves 58–75% oracle-headroom recovery on LLMs with knowledge injection. Our substrate (latent functors, HLA, neuron shards) doesn't have the same "early MLP / late MLP" structure. The PoC must verify the mechanism transfers; if it doesn't, the primitive stays opt-in diagnostic-only.

---

## Phase 1 — Permeation-Map Diagnostic Skeleton (CORE)

The safe half. Reuses Plan 358's `direct_effect_importance` as the cell score; adds the 2D scan + two-cluster classification. **No forward-pass machinery of its own** — the caller supplies the patched-forward closure (same contract as `causal_head_importance::patching`).

### Tasks

- [x] **T1.1** Create `katgpt-rs/crates/katgpt-core/src/cross_stage_relocation/mod.rs` with module docstring citing Research 417 + arxiv 2607.08393.
- [x] **T1.2** Define `PermeationMap` struct:
  ```rust
  /// L_src × L_dst matrix of `direct_effect_importance` cell scores.
  /// Cell `(i, j) > 0` means: snapshotting the anchor's state at source stage `i`
  /// and overwriting at destination stage `j` increases the readout by `cell(i,j)`.
  #[derive(Clone, Debug)]
  pub struct PermeationMap {
      /// Row-major `[n_src * n_dst]`. `cell(i, j) = rows[i * n_dst + j]`.
      pub cells: Vec<f32>,
      pub n_src: usize,
      pub n_dst: usize,
  }
  ```
- [x] **T1.3** Define `permeation_scan_into` — the zero-alloc scan loop:
  ```rust
  /// Scan all (src_stage, dst_stage) pairs, calling `patched_readout` for each.
  /// `patched_readout(src_stage, dst_stage)` must run the forward pass with the
  /// anchor's state at `src_stage` copied to `dst_stage`, and return `(m_patched,)`.
  /// `m_clean` and `m_corrupt` are the unpatched and fully-corrupted readouts.
  ///
  /// Writes into `out.cells` (caller pre-allocates). Cell score is
  /// `direct_effect_importance(m_clean, m_corrupt, m_patched)`.
  pub fn permeation_scan_into<F>(
      m_clean: f32,
      m_corrupt: f32,
      patched_readout: F,
      out: &mut PermeationMap,
  ) where
      F: FnMut(usize, usize) -> f32;
  ```
- [x] **T1.4** Define `ClusterClass` enum + `classify_two_cluster` method on `PermeationMap`:
  ```rust
  #[derive(Clone, Copy, Debug, PartialEq, Eq)]
  pub enum ClusterClass { EarlyToMid, LateToMid, Both, None }

  impl PermeationMap {
      /// Classify per the paper's two-cluster pattern (§5.4 / Figure 5).
      /// Quadrants: early = src < n_src/3, mid = n_src/3 ≤ src < 2*n_src/3,
      ///            late = src ≥ 2*n_src/3 (same for dst).
      /// Returns the dominant effective-patch cluster.
      pub fn classify_two_cluster(&self) -> ClusterClass { /* ... */ }
  }
  ```
- [x] **T1.5** G1 correctness test: synthetic 4-stage chain with a known-stranded representation. Construct a `PermeationMap` by hand, verify `classify_two_cluster` returns the expected class.
- [x] **T1.6** G4 zero-alloc test: `permeation_scan_into` writes into a pre-allocated `PermeationMap` without growing any `Vec`.
- [x] **T1.7** G3 latency test (criterion bench, opt-in): scan with a no-op `patched_readout` closure — overhead must be ≤ 5% vs a hand-rolled loop. **PASS — scan is 10–25% FASTER than hand-rolled loop with closure + IE arithmetic** (see `.benchmarks/431_cross_stage_relocation_goat.md`).

**Phase 1 exit criterion:** the module compiles standalone, `permeation_scan_into` is instantiable in a unit test, `classify_two_cluster` is correct on hand-constructed cases.

---

## Phase 2 — Cross-Stage Relocation Operator (the risky half)

The applied operator. **Cannot ship without Phase 3 PoC confirmation** that the mechanism transfers to our substrate.

### Tasks

- [x] **T2.1** Define `RelocateOp`:
  ```rust
  /// A single cross-stage residual relocation. During a forward pass with this
  /// op active, the anchor token's state at `src_stage` is snapshotted and
  /// overwrites the anchor's state at `dst_stage`.
  ///
  /// The host's forward pass must implement `RelocatingForward` to expose the
  /// snapshot/overwrite hooks.
  #[derive(Clone, Copy, Debug)]
  pub struct RelocateOp {
      pub src_stage: usize,
      pub dst_stage: usize,
      pub anchor_token_idx: usize,
  }
  ```
- [x] **T2.2** Define `RelocatePair` enum with the paper's fixed default:
  ```rust
  #[derive(Clone, Copy, Debug)]
  pub enum RelocatePair {
      /// The paper's two-pair default (§5.5): (⌊0.82L⌉, ⌊0.45L⌉) +
      /// (⌊0.10L⌉, ⌊0.45L⌉). Recovers 58–75% of oracle headroom on the KU Gap
      /// benchmark across 6 models × 2 domains.
      LateEarly,
      /// Custom pair — caller specifies both source fractions and the shared
      /// destination fraction.
      Custom { src_a: f32, src_b: f32, dst: f32 },
  }

  impl RelocatePair {
      pub fn to_ops(&self, n_stages: usize, anchor_token_idx: usize) -> [RelocateOp; 2] { /* ... */ }
  }
  ```
- [x] **T2.3** Define `RelocatingForward` trait (the host contract):
  ```rust
  /// Host's forward pass with snapshot/overwrite hooks. The primitive itself
  /// does NOT own forward-pass machinery — same contract as Plan 358's
  /// `direct_effect_importance` caller-supplied closure pattern.
  pub trait RelocatingForward {
      /// Snapshot the anchor's residual state at `stage`. Caller-owned buffer.
      fn snapshot_anchor_at(&self, stage: usize, anchor_idx: usize, out: &mut [f32]);
      /// Overwrite the anchor's residual state at `stage` with `state`.
      fn overwrite_anchor_at(&mut self, stage: usize, anchor_idx: usize, state: &[f32]);
      /// Number of stages in the forward pass (e.g., n_layers for an LLM).
      fn n_stages(&self) -> usize;
  }
  ```
- [x] **T2.4** `RelocateOp::apply_into<F: RelocatingForward>` method that orchestrates snapshot → forward-to-dst → overwrite → continue-forward. **No new sync-boundary data** — operates on local activation buffers only.
- [x] **T2.5** G1 unit test: synthetic 4-stage `RelocatingForward` impl where stage 1 holds the answer but stage 3 (the readout) reads from stage 2 (empty). `RelocateOp{ src: 1, dst: 2 }` recovers the answer.
- [x] **T2.6** G4 zero-alloc test: `apply_into` uses a caller-supplied scratch buffer for the snapshot; no `Vec` growth.

**Phase 2 exit criterion:** the operator compiles, applies correctly on the synthetic 4-stage test, zero-alloc.

---

## Phase 3 — Defend-Wrong PoC (MANDATORY before any promotion)

Per Research 417 §3.6, the operator's quality claim ("relocate recovers capability") crosses substrates — the paper proves it on LLMs with knowledge injection; we'd be claiming the latent-functor analog works. **A PoC is mandatory.**

### Tasks

- [x] **T3.1** Created `riir-ai/crates/riir-poc/benches/cross_stage_relocation_modelless_goat.rs` + `riir-ai/crates/riir-poc/src/cross_stage_relocation_poc.rs`. Used `CARGO_TARGET_DIR=/tmp/cross_stage_poc` per AGENTS.md. **DONE — cross-repo (`riir-ai`).**
- [x] **T3.2** Constructed a controlled toy domain: 8-stage residual stream, 4 placement configs (PlanDomain {2,7} per plan text, HeuristicMatch {1,7} = exact heuristic targets, BroadCluster {1,2,6,7}, LateOnly {6,7}). The reasoning circuit reads from stage 4 (⌊0.45L⌉ for L=8). Standard forward pass fails when stage 4 is empty. **DONE.**
- [x] **T3.3** Ran four competitors head-to-head:
  - **(a) Paper's heuristic** — `RelocatePair::LateEarly` (both ops, as shipped).
  - **(a') Late-only single op** — apply only op_a (src=7), skip the clobbering op_b.
  - **(b) No-relocation baseline** — standard forward pass.
  - **(c) Distilled coherence-triggered re-estimation** (R313 analog) — scan all 8 stages when coherence < tau_reest.
  **DONE.**
- [x] **T3.4** Printed verdict table: per-competitor cosine recovery (clean + noise sweep) + latency overhead. **DONE.**
- [x] **T3.5** **Honest recording:** the fixed-pair `LateEarly` heuristic does NOT beat both (b) and (c) — it CLOBBERS in 2/4 clean configs (PlanDomain, LateOnly) because op_b (src=1) overwrites op_a's recovery with an empty stage. The mechanism itself works (the single-op (a') variant recovers in all 4 configs), but the fixed two-pair default is too brittle for our substrate. Verdict recorded as "PoC Addendum" in Research 417. The operator stays opt-in diagnostic-only; `LateEarly` should NOT be promoted to default. **DONE.**

**Phase 3 exit criterion:** verdict table printed; raw numbers recorded in Research 417 §"PoC Addendum". **DONE.**

**Phase 3 verdict: REFUTE the fixed-pair `LateEarly` default.** The mechanism transfers (single-op relocation works), but the paper's fixed two-pair heuristic is brittle to misalignment on our substrate. The diagnostic half (permeation map) + CUSTOM relocation (where the caller uses the diagnostic to pick the right source stage) is the production path.

---

## Phase 4 — GOAT Gate + Promote/Demote Decision

### Tasks

- [x] **T4.1** **G1 (correctness)** — Phase 1 T1.5 + Phase 2 T2.5 unit tests pass. **34 unit tests PASS.**
- [x] **T4.2** **G2 (perf)** — `RelocateOp::apply_into` overhead ≤ 5% vs unpatched forward pass (criterion bench). The snapshot+overwrite is two `memcpy`s; should be near-zero. **PASS — 26ns at D=256; <0.03% of a 100µs+ forward pass. See `.benchmarks/431_cross_stage_relocation_goat.md`.**
- [x] **T4.3** **G3 (no-regression)** — `cargo test --features cross_stage_relocation` clean; `cargo check --all-features` clean. **PASS — 1580/1580 tests; `--all-features` + `--no-default-features --features cross_stage_relocation` both clean.**
- [x] **T4.4** **G4 (zero-alloc)** — Phase 1 T1.6 + Phase 2 T2.6 confirm no `Vec` growth on the hot path. **PASS — CountingAllocator re-check: 0 allocs on all three hot paths.**
- [x] **T4.5** **G5 (feature-isolated)** — `cargo check` (default features) unchanged; the feature compiles standalone. **PASS — `--no-default-features --features cross_stage_relocation` compiles clean; implies `causal_head_importance`.**
- [x] **T4.6** **G6 (modelless)** — no `riir-train` dep, no gradient descent, no training-time analysis leaked into the public repo. The saturation-epoch / gradient-locality findings stay in Research 417's §1 only. **PASS — verified by inspection.**
- [x] **T4.7** **Promote/demote per the §1.6 per-stack ledger:**
  - **DECISION: stays OPT-IN.** G1–G6 PASS for katgpt-rs scope, Phase 3 PoC is now COMPLETE — verdict **REFUTE the fixed-pair `LateEarly` default**: the heuristic CLOBBERS in 2/4 clean configs (PlanDomain, LateOnly) because op_b (src=1) overwrites op_a's recovery. The mechanism itself works (single-op variant recovers in all 4 configs), but the paper's fixed two-pair default is too brittle for our substrate (latent functors / HLA / neuron shards don't have the same early/late MLP structure that the paper's two-cluster pattern relies on). **Promotion blocked.** The diagnostic half (permeation map) + CUSTOM relocation (where the caller uses the diagnostic to pick the right source stage) is the production path. See Research 417 §“PoC Addendum” for raw numbers.
- [x] **T4.8** Record stack slot: **intervention/diagnostic** (alongside `causal_head_importance`, `faithfulness_probe`). Update `katgpt-rs/README.md` Feature Showcase + `katgpt-rs/.docs/01_orientation/overview.md` Feature Flags table. **DONE** — both updated.

**Phase 4 exit criterion:** GOAT gate recorded in `.benchmarks/431_cross_stage_relocation_goat.md` with promote/demote decision and per-stack ledger entry.

---

## Feature Flag

```toml
# katgpt-rs/crates/katgpt-core/Cargo.toml
[features]
cross_stage_relocation = []
```

Default: **off** (opt-in). Promotion gated by Phase 4 T4.7.

---

## Latent vs Raw Boundary

- **Local-only (latent, never synced):** the anchor's residual state at any stage; the `PermeationMap` cells; the `RelocateOp` configuration.
- **No new sync-boundary data.** The operator reads/writes local activation buffers; it does not introduce any new field that would cross `SyncBlock → ChainConsensus`. The anchor token index and the stage indices are configuration blobs (raw `usize`), not gameplay state.

---

## What This Plan Does NOT Do

- **Does NOT claim parity with the paper's 58–75% recovery on our substrate.** That's a quality claim requiring the Phase 3 PoC.
- **Does NOT touch fine-tuning dynamics.** Saturation epochs, gradient locality, alignment-aware training → riir-train.
- **Does NOT ship a private riir-ai guide.** The MOAT gate routes this to `katgpt-rs` only; no pillar multiplication.
- **Does NOT fuse with QK-Restore (R259) in this plan.** Fusion C is noted as a follow-up if the GOAT gate passes.
- **Does NOT promote to default without a real-game-domain PoC.** The synthetic PoC (Phase 3) is the gate for *considering* promotion; a real-domain PoC (deferred to riir-ai) is the gate for *actually* promoting.

---

## Risks

- **Latent substrate mismatch.** Our latent functors and HLA don't have the "early MLP / late MLP" structure that produces the paper's two-cluster pattern. The PoC may show the mechanism doesn't transfer. → Mitigation: Phase 3 is mandatory; the diagnostic half (Phase 1) ships regardless because it reuses `direct_effect_importance` cleanly.
- **Operator may be too thin to justify a feature flag.** If `RelocateOp::apply_into` is just two `memcpy`s + a forward-pass orchestration, consider folding it into the host's forward pass directly. → Revisit after Phase 2.
- **PoC may be hard to construct.** A "stranded representation" toy domain is artificial; designing one that's fair to all three competitors (paper heuristic, baseline, re-estimation) is non-trivial. → Mitigation: start with the simplest possible synthetic (8-stage linear residual stream, ground-truth known); expand only if the simple case is ambiguous.

---

## TL;DR

Ship two modelless primitives from arxiv 2607.08393 (Knowing-Using Gap): (1) **`permeation_scan`** — a 2D `(src_stage, dst_stage)` intervention heatmap reusing `direct_effect_importance` (Plan 358) as the cell score, plus two-cluster classification; (2) **`RelocateOp`** — an applied operator that snapshots an anchor's state at one stage and overwrites at another, with the paper's `(0.82L→0.45L) + (0.10L→0.45L)` fixed default. Both behind `cross_stage_relocation` feature flag, opt-in. **Phase 3 defend-wrong PoC DONE (2026-07-13)** — verdict: **REFUTE the fixed-pair `LateEarly` default**; the mechanism transfers (single-op relocation recovers in all 4 configs) but the fixed two-pair default CLOBBERS in 2/4 clean configs because op_b overwrites op_a's recovery. The diagnostic half survives (clean Plan 358 extension); the operator half stays opt-in. Production path: permeation-map diagnostic + CUSTOM relocation (not the fixed default). GOAT, not Super-GOAT (per Research 417 §3 verdict).
