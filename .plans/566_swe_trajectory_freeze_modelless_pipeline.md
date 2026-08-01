# Plan 566: SWE Trajectory Freeze — Modelless Pipeline (Proposal 011 Layer 4)

**Date:** 2026-08-01
**Proposal:** [katgpt-rs/.proposals/011_rust_swe_bench_latent_space_via_wasm_pruner.md](../.proposals/011_rust_swe_bench_latent_space_via_wasm_pruner.md)
**Status:** ✅ COMPLETE (all substantive work landed via Issues 569-571 + Benches 011-020). The original Phase 0-4 scaffolding was executed as a defend-or-refute issue/bench sequence instead of inline plan tasks — see the cross-reference table below. Phase 5 (real model validation) landed as Benches 012-020 (real Kimi-K3 trajectories). Phase 6 (riir-train fallback) is N/A — G5 PASSED.

> This plan executes **Layer 4** of Proposal 011 — the modelless trajectory-freeze pipeline that composes shipped DEFAULT-ON primitives. It does NOT depend on rubrc (Layer 3's blocker) or Kimi-K3 (Layer 2's blocker). It runs on synthetic trajectories first, then on real model trajectories once P032 Phase 5 ships.
>
> **The core hypothesis:** even when a model proposes zero valid patches on Rust-SWE-bench, the inference loop's trajectory through patch-space has measurable geometry (curvature, drift, oscillation vs committed-wrong) that can be frozen and compared across snapshots. This is the flipped R463 caveat — geometry is always measurable.

---

## Goal

Ship a `SweTrajectoryFreezer` that composes `tf_loop` + `latent_trajectory_geometry` + `committed_field_blend` + `MerkleFrozenEnvelope` into a pipeline that:
1. Runs the recursive forward pass (`tf_loop`) with test-feedback as a ConstraintPruner signal.
2. Measures the latent trajectory geometry (`latent_trajectory_geometry`).
3. Detects compaction points when tests pass (`closed_unit_compaction` / CUCG).
4. Freezes the trajectory summary as a BLAKE3-committed artifact (`committed_field_blend` / `KarcShard` + `MerkleFrozenEnvelope`).
5. Supports self-healing snapshot swap when coherence drops (`reestimation.rs`).

**GOAT gate (Layer 4):** G5 = trajectory geometry discriminates across snapshots/models. Even with zero passing patches, the geometry is freezable + comparable.

**Riir-train fallback (Layer 4b):** If G5 shows insufficient signal, document why modelless was insufficient per §3.5, then defer to riir-train LoRA fine-tune on whatever passing patches exist.

---

## Execution cross-reference (how the plan was actually executed)

The plan was originally drafted as inline Phase 0-6 tasks. The actual execution pivoted to a **defend-or-refute issue/bench sequence** (Issues 569-571 + Benches 011-020), which is the codebase's canonical pattern for empirical questions. The mapping:

| Plan phase | Executed as | Result |
|---|---|---|
| Phase 0 (synthetic scaffolding) + Phase 1 (geometry discriminates?) + Phase 2 (CUCG) + Phase 3 (FAME) | [Issue 569](../.issues/569_swe_trajectory_geometry_synthetic_poc.md) (T5.1-T5.3) + Bench 011 | T5.1+T5.2 PASS, T5.3 CONDITIONAL FAIL (random directions degenerate) |
| Phase 3 fix (data-derived directions) | [Issue 570](../.issues/570_data_derived_directions_fix_t53.md) (T5.3b) | PASS — 100% accuracy with geometry-encoded summaries + data-derived directions |
| Phase 4 (`SweTrajectoryFreezer` impl + GOAT) | Bench 013 (substrate GOAT) + Bench 014 (G5 cross-model) | G1-G5 ALL PASS — 100% accuracy on real Kimi-K3 vs random |
| Phase 5 (real model, value discrimination) | Benches 012-020 (depth trajectory NEGATIVE, sequence trajectory POSITIVE) | Depth trajectory fails value-level discrimination (Bayes-optimal ceiling ~54%); sequence trajectory overcomes it (100% at σ≥0.1, d_M=14.526). **StateMagnitudeEncoder** ported to substrate (Bench 019, Issue 571). |
| Phase 6 (riir-train fallback) | N/A — G5 PASSED | The modelless path is validated. riir-train LoRA remains an ALLOWED fallback if future cross-snapshot discrimination shows insufficient signal, but the primary G5 gate passed. |

The inline task checkboxes below are retained as the original design record; they are all addressed by the issue/bench sequence above.

---

## Phase 0 — Synthetic trajectory POC scaffolding

> The cheapest validation. No Rust-SWE-bench, no rubrc, no Kimi-K3 needed. Tests whether the *pipeline composition* produces signal on synthetic data that mimics SWE attempt patterns.

### Tasks

- [ ] **T0.1** Define synthetic trajectory classes mimicking SWE failure modes:
  - `Oscillation` — high curvature, model can't commit (flips between patch variants)
  - `Drift` — rotating through wrong answers (gradual direction change)
  - `CommittedWrong` — low curvature but wrong (model confidently commits to a bad patch)
  - `Converging` — decreasing curvature toward a valid patch (the positive class, rare)
- [ ] **T0.2** Generate synthetic latent state chains for each class (dim=8, 20-100 steps, matching `latent_trajectory_geometry` Bench 342 scale).
- [ ] **T0.3** Run `latent_trajectory_geometry::from_states()` on each class. Record length, curvature, drift angle, bifurcation ratio.

---

## Phase 1 — POC: does trajectory geometry discriminate? (THE LOAD-BEARING POC)

> **This is the cheapest + most curious POC.** If it fails (no discrimination), Layer 4 is demoted. If it passes, the modelless pipeline is validated in principle. **Even a failed result is valuable** — it documents that trajectory geometry alone is insufficient, motivating the riir-train LoRA path.

### Tasks

- [ ] **T1.1** G1 correctness: verify `latent_trajectory_geometry` produces finite, non-NaN output on all 4 synthetic classes.
- [ ] **T1.2** G3 visible proof: does curvature differ across classes by ≥ 0.5 rad (matching Bench 342 G3.1 threshold)? Does the oscillation class have measurably higher curvature than committed-wrong?
- [ ] **T1.3** G5 decisive: can a simple classifier (dot-product + sigmoid onto a "committed vs oscillating" direction) distinguish the classes above chance? This is the modelless analog of "does the trajectory contain signal."
- [ ] **T1.4** **HONEST NEGATIVE RESULT DOCUMENTATION**: if T1.3 fails, record the raw numbers + the reason (e.g., "all failure trajectories have similar entropy regardless of class"). This motivates Layer 4b (riir-train).

---

## Phase 2 — POC: CUCG closed-unit detection on test-pass events

> Tests whether a test pass qualifies as a CUCG compaction point. The CUCG rubric is (closed-unit ∧ summarizable ∧ progress ∧ ¬stuck). A test pass is clearly "progress", but is it a "closed unit"?

### Tasks

- [ ] **T2.1** Construct synthetic test-pass sequences (e.g., [fail, fail, fail, pass, fail, pass, pass] — mimicking gradual convergence).
- [ ] **T2.2** Run CUCG `evaluate()` on the trajectory at each step. Does `FireRule` fire at the test-pass boundaries?
- [ ] **T2.3** Verify the G7 isomorphism holds: when CUCG fires, the trajectory segment up to that point is freezable as a `MerkleFrozenEnvelope` (BLAKE3 round-trip).
- [ ] **T2.4** If CUCG does NOT fire on test-pass events (the rubric's "closed-unit" predicate rejects them), document whether a custom `FireRule` is needed or whether the stock rubric is sufficient.

---

## Phase 3 — POC: committed_field_blend from failure trajectory

> Tests whether FAME (Plan 321) produces a stable, sampling-invariant blend even with zero positive examples (all-fail trajectory). FAME Prop. 3 says the blend is sampling-invariant — but that was proven on game trajectories.

### Tasks

- [ ] **T3.1** Construct a synthetic all-fail trajectory summary (no converging class, only oscillation + drift + committed-wrong).
- [ ] **T3.2** Run `committed_field_blend::commit()` on the summary. Does it produce a valid BLAKE3-committable blend?
- [ ] **T3.3** Verify sampling invariance (FAME Prop. 3): does dense vs sparse observation of the same trajectory produce identical committed `pi`? (Reuse the G2 test pattern from Plan 321.)
- [ ] **T3.4** Compare two different all-fail trajectories (different failure modes). Do they produce measurably different blends? (This is the discrimination test from Phase 1, but at the blend level.)

---

## Phase 4 — `SweTrajectoryFreezer` impl (gated on Phases 1-3 passing)

> The actual primitive. Composes the validated pieces into a pipeline behind a feature flag.

### Tasks

- [ ] **T4.1** `SweTrajectoryFreezer` struct — owns the `tf_loop` config + `latent_trajectory_geometry` scratch + `committed_field_blend` state.
- [ ] **T4.2** `run_trajectory_freeze(&mut self, attempt_trajectory: &[f32], test_results: &[bool]) -> FreezerOutput` — the main entry point.
- [ ] **T4.3** `FreezerOutput` — contains the frozen trajectory summary (BLAKE3-committed), the geometry measurements, and the CUCG compaction points.
- [ ] **T4.4** Feature flag `swe_trajectory_freeze` (opt-in, NOT default).
- [ ] **T4.5** G1/G2/G4 gates (determinism, perf < 5µs, alloc-free).
- [ ] **T4.6** G5 gate: trajectory geometry discriminates across synthetic snapshots.

---

## Phase 5 — Real model validation (gated on P032 Phase 5 — Kimi-K3 loaded)

> Once Kimi-K3 is loaded, run the pipeline on real Rust-SWE-bench tasks. This is where the hypothesis meets reality.

### Tasks

- [ ] **T5.1** Run `tf_loop` on Rust-SWE-bench tasks with Kimi-K3. Extract trajectory geometry.
- [ ] **T5.2** Compare trajectory geometry across model snapshots (e.g., base Kimi-K3 vs a frozen/thawed variant). Do they produce measurably different failure-trajectory shapes?
- [ ] **T5.3** G5 real-model gate: trajectory geometry discriminates across real snapshots.
- [ ] **T5.4** If G5 FAILS on real data (but PASSED on synthetic) → document the synthetic-real gap. The synthetic POC was necessary but not sufficient.
- [ ] **T5.5** If G5 PASSES → the modelless path is validated. Layer 3 (WASM pruner) becomes an enhancement, not a dependency.

---

## Phase 6 — riir-train fallback (Layer 4b, only if Phase 5 G5 fails)

> Per §3.5 modelless-unblock protocol: document why modelless was insufficient, then defer to riir-train.

### Tasks

- [ ] **T6.1** §3.5 documentation: what modelless paths were checked (Phase 1-5), why each failed (concrete reason per path).
- [ ] **T6.2** File riir-train plan: LoRA fine-tune on whatever passing patches exist in the Rust-SWE-bench subset.
- [ ] **T6.3** Note: this is the explicit allowed fallback. riir-train ships adapter training.

---

## GOAT gate summary

| Gate | Phase | Threshold | Status |
|---|---|---|---|
| G1 correctness | Phase 1 T1.1 | finite, non-NaN geometry on all classes | ✅ PASS — Issue 569 T5.1 |
| G2 perf | Phase 4 T4.5 | < 5µs per trajectory (matches Bench 342) | ✅ PASS — Bench 013 (4582 ns/call); Bench 019 value encoder 51.8µs (D=1024, N=64) |
| G3 no-regression | Phase 4 T4.5 | opt-in feature, no default impact | ✅ PASS — Bench 013 + Bench 019 (1851 lib tests) |
| G4 alloc-free | Phase 4 T4.5 | zero steady-state allocation | ✅ PASS — Bench 014 (after `from_states_into` fix); Bench 019 (value path 0 allocs) |
| **G5 decisive (synthetic)** | **Phase 1 T1.3** | **trajectory geometry discriminates failure modes above chance** | **✅ PASS — Issue 569 T5.1+T5.2, Issue 570 T5.3b (100% accuracy)** |
| G5 decisive (real) | Phase 5 T5.3 | trajectory geometry discriminates across real snapshots | ✅ PASS — Bench 014 (100% cross-model on real Kimi-K3); Benches 018+020 (100% value-level at σ≥0.1) |

---

## Dependencies

- `tf_loop` (Plan 136) — DEFAULT-ON ✅
- `latent_trajectory_geometry` (Plan 342) — shipped ✅
- `closed_unit_compaction` (Plan 333) — DEFAULT-ON ✅
- `committed_field_blend` (Plan 321) — DEFAULT-ON ✅
- `KarcShard` (Plan 308) — shipped ✅
- `MerkleFrozenEnvelope` (riir-neuron-db) — shipped ✅
- `reestimation.rs` (riir-ai latent_functor) — shipped ✅
- P032 Phase 5 (Kimi-K3 safetensors loader) — **BLOCKING Phase 5 only** (Phases 0-4 run on synthetic data)

---

## Honest caveats

1. **Phase 1 is the cheapest + most decisive POC.** If synthetic trajectory geometry can't discriminate failure modes in principle, real data won't either. Run Phase 1 first; if it fails, stop.
2. **A negative result is valuable.** If Phase 1 T1.3 fails, it documents that trajectory geometry alone is insufficient — motivating the riir-train LoRA path (Layer 4b). This is honest science, not failure.
3. **The CUCG G7 isomorphism is the load-bearing prior art.** "Trajectory compaction = shard freeze" is proven (Plan 333 G7). Layer 4 is the application of this proven isomorphism to the SWE domain.
4. **The flipped R463 caveat is the core insight.** Even with zero passing patches, the trajectory has shape. This is strictly more information than the pass/fail-only signal.
5. **Phases 0-4 do NOT need Rust-SWE-bench, rubrc, or Kimi-K3.** They run on synthetic data. This is the cheapest validation path in the entire Proposal 011 portfolio.
