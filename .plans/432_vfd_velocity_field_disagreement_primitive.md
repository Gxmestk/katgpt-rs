# Plan 432: VFD — Velocity-Field Disagreement Epistemic UQ Primitive

**Date:** 2026-07-13
**Research:** [katgpt-rs/.research/420_VFD_Velocity_Field_Disagreement_Epistemic_UQ.md](../.research/420_VFD_Velocity_Field_Disagreement_Epistemic_UQ.md)
**Source paper:** [arxiv 2606.18043](https://arxiv.org/abs/2606.18043) — Römer et al., *Uncertainty Quantification for Flow-Based Vision-Language-Action Models*, §4 (VFD estimator + Theorem 4.1). The SAVE half (§5) is a training method → riir-train, out of scope.
**Target:** `katgpt-rs/crates/katgpt-core/src/velocity_field_disagreement.rs` (new module) + Cargo feature `velocity_field_disagreement`
**Status:** Active — Phase 1 unblocking skeleton

---

## Goal

Ship a modelless epistemic-UQ estimator (`VfdScore`) that consumes M frozen velocity fields and produces a scalar uncertainty score per conditioning input, by computing pairwise velocity-field disagreement along an ODE integration weighted by `κ_s = s/(1−s)`. The primitive extends the existing `VelocityFieldEnsemble<P, D>` (Plan 376, default-on) — it activates that primitive's deferred G7 UQ gate (Plan 376 Phase 6, "primitive still ships as non-UQ") and closes the open `QgfVarianceSignal` "ensemble KL" item in `qgf/adaptive.rs`.

**Two-use substrate:** the same M frozen velocity fields are (a) ridge-combined into a super-forecaster via `VelocityFieldEnsemble::fit_into` (existing), AND (b) measured for pairwise disagreement via `vfd_score_into` (this plan). Two uses, one frozen library, no extra training.

**GOAT gate success → promote to default-on** (closes Plan 376 Phase 6 G7 deferred gate).

---

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [ ] **T1.1** Define `VfdScore` struct + `VfdScratch<M, D>` for zero-alloc batched computation. Layout:
  ```rust
  pub struct VfdScore { pub score: f32 }  // the raw (unnormalized) VFD scalar

  pub struct VfdScratch<const M: usize, const D: usize> {
      pub x_traj:    [[f32; D]; M],   // per-member current ODE state
      pub x_next:    [f32; D],        // ODE step output buffer
      pub v_at_i:    [[f32; D]; M],   // velocity-field j evaluated at member i's state, for all j
      pub drift_buf: [f32; D],        // single-member drift for the ODE step
  }
  ```
  All `M` and `D` as `const` generics (M=2 default, D=8 for HLA). Zero heap allocations on the score path.

- [ ] **T1.2** Define the core scoring function:
  ```rust
  pub fn vfd_score_into<F, const M: usize, const D: usize, R>(
      fields: &[&dyn VelocityField<D>; M],
      y_state: &[f32],            // conditioning input (length = fields' input dim)
      schedule: Schedule,
      n_steps: usize,             // N_s
      batch: usize,               // B (paper default 5)
      scratch: &mut VfdScratch<M, D>,
      rng: &mut R,
  ) -> f32
  where
      F: Fn(&[f32], &mut [f32]),
      R: FnMut() -> f32,
  ```
  Implements paper eq. 7:
  1. Sample `x_0^{(i)} ~ N(0, I_D)` for each member i and each batch sample (B × M initial states).
  2. For each ODE step `ℓ ∈ {0, …, N_s − 1}`:
     - Compute `s_ℓ = ℓ · δ_s`, `δ_s = 1/N_s`.
     - For each member i: forward-integrate its OWN trajectory `x_{s_{ℓ+1}}^{(i)} = x_{s_ℓ}^{(i)} + v_{s_ℓ}^i(x_{s_ℓ}^{(i)}, y) · δ_s` (use `stochastic_interpolant_step_into` with `drift_at_t = v_{s_ℓ}^i(...)` and the optimal-diffusion SDE step).
     - For each member i and each OTHER member j (j ≠ i): evaluate `v_{s_ℓ}^j(x_{s_ℓ}^{(i)}, y)` and accumulate `κ_{s_ℓ} · ‖v_{s_ℓ}^i(x_{s_ℓ}^{(i)}, y) − v_{s_ℓ}^j(x_{s_ℓ}^{(i)}, y)‖²₂` into the running sum.
  3. Normalize by `M (M−1) N_s B` and return.

- [ ] **T1.3** Reuse `Schedule::optimal_diffusion(t)` to derive `κ_s`. The paper's weighting `κ_s = s/(1−s)` corresponds to:
  - `Schedule::Linear` (`α = 1−t, β = t, γ = 1`): `D*_t = (1−t)/t`, so `κ_s = 1/D*_s = t/(1−t) = s/(1−s)`. **Exact match.**
  - `Schedule::Trigonometric` (`α = cos(πt/2), β = sin(πt/2), γ = π/2`): `D*_t = (π/2) cot(πt/2)`, so `κ_s = (2/π) tan(πs/2)`. **Same divergence shape, scaled.**

  Document both in the `vfd_score_into` docstring; default to `Linear` for the κ_s-exact case.

- [ ] **T1.4** Implement `VfdVarianceSignal` wrapper that implements `QgfVarianceSignal`:
  ```rust
  pub struct VfdVarianceSignal {
      pub raw_score: f32,
      pub tau: f32,  // normalization temperature
  }
  impl QgfVarianceSignal for VfdVarianceSignal {
      fn normalized_disagreement(&self) -> f32 {
          // sigmoid normalization to [0, 1] — heuristic, NOT a probability
          let s = self.tau * self.raw_score;
          if s.is_nan() { return 1.0; }  // NaN → max disagreement (defensive)
          1.0 / (1.0 + (-s).exp())
      }
  }
  ```
  This closes the `qgf/adaptive.rs:131-133` docstring's "ensemble KL" open item.

- [ ] **T1.5** Write the G1 mechanics test:
  ```rust
  #[test]
  fn test_vfd_approximates_known_kl_2d_gaussian() {
      // Two linear velocity fields on a 2D Gaussian toy.
      // Analytic KL between two Gaussians N(μ1, I) and N(μ2, I) is 0.5‖μ1−μ2‖².
      // With μ1=(0,0), μ2=(1,0): KL = 0.5.
      // Set up fields v^1(x,y) = μ1 - x (constant target), v^2(x,y) = μ2 - x.
      // Run VFD with N_s=20, B=20. Assert |VFD - 0.5| < 0.1 (loose tolerance for
      // the discrete approximation; the analytic-exact case is the bound, not the value).
  }
  ```
  Synthetic, no training. Verifies the math wiring.

- [ ] **T1.6** Write the M=2 sufficiency smoke test: with M=2 on the same toy, VFD's normalized score is in `[0, 1]` and varies monotonically with `‖μ1 − μ2‖` (more disagreement → higher score).

- [ ] **T1.7** Add the `velocity_field_disagreement` feature flag in `crates/katgpt-core/Cargo.toml`:
  ```toml
  [features]
  velocity_field_disagreement = ["velocity_field_ensemble"]  # depends on Schedule + stochastic_interpolant_step_into
  ```
  And the module wiring in `crates/katgpt-core/src/lib.rs`:
  ```rust
  #[cfg(feature = "velocity_field_disagreement")]
  pub mod velocity_field_disagreement;
  ```

- [ ] **T1.8** Zero-allocation audit: `vfd_score_into` uses ONLY caller-provided `VfdScratch`. The `x_traj`, `x_next`, `v_at_i`, `drift_buf` buffers are all stack-allocated via const generics. Verify with a debug-assertions build that no hidden allocation sneaks in (e.g., via `Vec` inside `stochastic_interpolant_step_into` — there are none, but audit anyway).

- [ ] **T1.9** File-size check: target < 800 lines for `velocity_field_disagreement.rs` (well under the 2048 line ceiling).

---

## Phase 2 — GOAT Gate (Benchmarks + UQ Floor)

### Tasks

- [ ] **T2.1** **G1 (mechanics)** — Run `test_vfd_approximates_known_kl_2d_gaussian` from T1.5. PASS criterion: VFD approximates the analytic KL on the 2D Gaussian toy within tolerance. Failure means a math wiring bug, not a fundamental issue.

- [ ] **T2.2** **G2 (UQ floor — MANDATORY per Issue 010 / "Report the Floor" rule)** — Benchmark VFD-calibrated prediction intervals against `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` (Plan 340, m=1) on:
  - (a) The same AR(1) corpus Plan 376 Phase 6 used (CRPS / coverage / Winkler).
  - (b) A 2D flow-matching toy (bimodal Gaussian target, M=2 members trained on different modes) — the paper's Appendix C.1 setup.
  
  PASS criterion: VFD beats the conformal-naive floor on **at least 2 of 3 metrics** on at least one corpus. Document verdict in `.benchmarks/432_vfd_uq_floor.md`. **If VFD cannot beat the floor → drop the UQ claim** (the primitive still ships as a non-UQ disagreement score for downstream consumers like CLR L1 gating, but no calibrated-UQ claim).

- [ ] **T2.3** **G3 (no regression)** — `cargo check --features velocity_field_disagreement` (single feature) adds zero warnings; `cargo check --all-features` (combo check per AGENTS.md) passes; `cargo clippy -D warnings` clean.

- [ ] **T2.4** **G4 (latency)** — Microbench: M=2, D=8, N_s=10, B=5 → `vfd_score_into` ≤ 50µs. Justification: plasma-tier budget. If VFD exceeds this, document the actual perf and propose mitigations (smaller N_s, async off-hot-path computation, per-NPC throttling). Failure here does NOT block ship — it constrains the deployment regime.

- [ ] **T2.5** **G5 (QGF integration smoke)** — `VfdVarianceSignal` impl + a smoke test: feed a synthetic VFD score into `adaptive_guidance_weight(signal_confidence)` and verify the weight responds. Closes the docstring open item.

---

## Phase 3 — Promotion Decision

### Tasks

- [ ] **T3.1** Aggregate G1–G5 verdicts into `.benchmarks/432_vfd_goat.md`. Document per-gate PASS/FAIL with raw numbers.

- [ ] **T3.2** If all gates PASS (especially G2 UQ floor): promote `velocity_field_disagreement` to `default = [...]` in `crates/katgpt-core/Cargo.toml`. This **closes Plan 376 Phase 6 G7 deferred gate** — the velocity-field ensemble primitive becomes UQ-bearing.

- [ ] **T3.3** If G2 FAILS (VFD does not beat the floor): keep `velocity_field_disagreement` opt-in. Document in `.benchmarks/432_vfd_goat.md` that the primitive ships as a non-UQ disagreement score (still useful for CLR L1 gating, sleep-time prioritization, runtime failure detection — but no calibrated-UQ claim). Update Research 420 §4 verdict accordingly.

- [ ] **T3.4** Update `crates/katgpt-core/src/lib.rs` doc comment on the velocity-field ensemble module to note that VFD activates its UQ axis (cross-link to Research 420 + Plan 432).

- [ ] **T3.5** Update Plan 376 Phase 6 G7 status from "deferred" to "ACTIVATED by Plan 432" (or "still deferred — VFD did not promote" if G2 fails).

---

## Phase 4 — Optional: Heterogeneous-D VFD (Cross-Resolution fusion)

**Deferred.** Only relevant if a use case emerges requiring velocity fields with different output dimensions `d_i`. Fuse with Plan 310 (Cross-Resolution Spectral Transport) — project each member's velocity to common D first, then run VFD. **Not in this plan.**

---

## Phase 5 — Optional: LatCal Commitment of VFD Threshold (riir-chain)

**Deferred.** Commit per-target conformal-calibrated VFD threshold as one extra LatCal fixed-point scalar alongside the existing `EtaCommitBatch` (riir-chain/.research/008). Anti-cheat use: prevents nodes from silently lowering their abstention bar. **Not in this plan.**

---

## Phase 6 — Optional: Runtime Integration (riir-ai)

**Deferred to a future riir-ai plan** (after this GOAT gate passes). The runtime integration connects VFD to:
- CLR L1 evidence gate (P307) — VFD as a calibrated confidence input.
- Sleep-time anticipator (P334) — high-VFD queries prioritized for offline resolution.
- cgsp_runtime curiosity (P299) — `curiosity_t = α · forecast_error + β · VFD_t`.
- Per-NPC abstention / delegation / escalation logic (Plan 385 follow-up).

This is pillar-level work for riir-ai (connects R170 Super-GOAT to P8 Reasoning Pack to P334 Sleep-Time). Not in this katgpt-rs plan.

---

## File Layout (target)

```
crates/katgpt-core/src/velocity_field_disagreement.rs   ~600-800 lines (target < 2048)
├── pub struct VfdScore                                  ~10 lines
├── pub struct VfdScratch<const M, const D>              ~30 lines
├── pub fn vfd_score_into                                ~120 lines (the core algorithm)
├── pub struct VfdVarianceSignal                         ~20 lines
├── impl QgfVarianceSignal for VfdVarianceSignal         ~15 lines
├── helpers (κ_s computation, batched ODE integration)   ~80 lines
└── tests                                                ~200 lines
    ├── test_vfd_approximates_known_kl_2d_gaussian       (G1)
    ├── test_vfd_m2_sufficiency_smoke                    (T1.6)
    ├── test_vfd_score_in_range_0_1_after_normalization  (defensive)
    ├── test_vfd_zero_disagreement_when_members_identical (sanity)
    ├── test_vfd_kappa_s_linear_schedule_matches_paper   (κ_s = s/(1−s) exact)
    └── test_vfd_kappa_s_trig_schedule_diverges_correctly (shape check)
```

---

## Constraints check

| Constraint | Status |
|---|---|
| Modelless / inference-time | ✅ VFD is a pure read of frozen velocity fields + ODE integration. No backprop, no weight mutation. The M members are pre-trained offline (riir-train's job, once) and frozen. |
| Latent-to-latent preferred | ✅ Operates entirely on velocity-field outputs (latent vectors). The score is a scalar summary. Never crosses to tokens. |
| Use sigmoid not softmax | ✅ The `VfdVarianceSignal::normalized_disagreement` uses sigmoid normalization to `[0,1]`. No softmax anywhere. (Note: the raw VFD score is unbounded — it's a sum of weighted L2 norms. Sigmoid is for the normalized readout only.) |
| Freeze/thaw over fine-tuning | ✅ The M ensemble members are frozen snapshot artifacts. The conformal threshold is a frozen config. No runtime weight mutation. |
| 5-repo discipline | ✅ Open primitive → katgpt-rs (this plan). Runtime integration → riir-ai (deferred, future plan). Chain commitment → riir-chain (deferred). SAVE → riir-train (out of scope for VFD; SAVE is a separate training method). |
| Raw scalars at sync boundary | ✅ The VFD score is a local latent scalar (not synced). The conformal threshold per target (one f32) crosses sync as one LatCal fixed-point scalar (deferred to Phase 5). |
| Zero-alloc hot path | ✅ All ops via caller-provided `VfdScratch<M, D>` with const-generic stack arrays. `vfd_score_into` takes `&mut VfdScratch`. |
| CPU/SIMD first | ✅ Inner loops are L2 norms + linear combinations — `simd::simd_dot` candidates. |
| File size < 2048 lines | ✅ Target ~600-800 lines. |
| `Uuid::now_v7()` if Uuid needed | N/A — no Uuids (member IDs are field indices). |
| UQ-bearing → report the floor | ✅ Phase 2 G2 — VFD-calibrated intervals vs `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` (Plan 340, m=1). Mandatory per Issue 010. |

---

## Validation

- [ ] `cargo test -p katgpt-core --features velocity_field_disagreement --lib` passes (≥6 tests).
- [ ] `cargo check --features velocity_field_disagreement` (single feature) passes — 0 warnings.
- [ ] `cargo check --all-features` (combo) passes — combo-regression check per AGENTS.md.
- [ ] `cargo clippy -p katgpt-core --features velocity_field_disagreement -- -D warnings` clean.
- [ ] Phase 2 G2 verdict recorded in `.benchmarks/432_vfd_uq_floor.md`.
- [ ] Phase 3 GOAT gate verdict recorded in `.benchmarks/432_vfd_goat.md`.
- [ ] **If promoted to default:** update `Cargo.toml` `default = [...]`; re-run `cargo test -p katgpt-core --lib velocity_field_disagreement` (default features) passes.

---

## Honest Risk Notes

1. **The G2 UQ floor gate is the make-or-break.** VFD is UQ-bearing (claims a calibrated epistemic-UQ scalar). Per Issue 010, it MUST beat `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` on CRPS / coverage / Winkler. If VFD cannot beat the floor on at least one corpus, the UQ claim is dropped (Phase 3 T3.3) — the primitive still ships as a non-UQ disagreement score for CLR gating / runtime failure detection, but no calibrated-UQ claim. **Plan for both outcomes.**

2. **VFD's cost is M × N_s × B × (one velocity-field eval) per inference.** For M=2, N_s=10, B=5, eval=200ns → 20µs per inference. At 10K NPCs × 20Hz tick = 200K inferences/sec → 4 sec/sec of compute. **VFD is NOT a per-tick per-NPC primitive.** It is computed (a) off the hot path (sleep-time), (b) per-NPC every `T_vfd` ticks (e.g., T_vfd=10 → 2Hz per NPC), or (c) only on flagged low-confidence NPCs. The Phase 6 runtime integration (riir-ai) owns this scheduling; the katgpt-rs primitive just exposes the API.

3. **The κ_s grid convention is critical.** Never evaluate VFD at `s = 1` exactly (`κ_s` diverges). Always use `s_ℓ = ℓ · δ_s` for `ℓ ∈ {0, …, N_s − 1}`. The Plan 432 implementation must enforce this (defensive assertion in `vfd_score_into`).

4. **The "single shared trajectory" misimplementation is the #1 bug risk.** VFD requires per-member trajectories (`x^{(i)}` integrated under member i's velocity field, then member j evaluated at those states). A naïve implementation that integrates ONE shared trajectory and evaluates both members at it produces Action-L2 (a different, weaker score). Plan 432 T1.2 must implement per-member trajectories explicitly; T1.5 G1 test catches this bug (Action-L2 does NOT approximate the analytic KL on the 2D Gaussian toy).

5. **The 2-member sufficiency is empirically demonstrated for VLAs, not for our substrates.** The G1 test uses M=2 on the 2D Gaussian toy; the G2 UQ floor uses M=2. If M=2 turns out insufficient for our velocity-field ensemble class, the API accepts arbitrary M — bumping to M=3 is a one-line change. Document the M=2 default as "empirically validated for VLAs; validated for our substrates on the G1/G2 toys".

6. **VFD does NOT subsume BoMSampler.** BoM is for autoregressive token-sample disagreement; VFD is for flow-matching velocity-field disagreement. Different model classes. Do NOT position VFD as a BoM replacement — it is a sibling UQ probe for a different model class.

7. **The proof of Theorem 4.1 assumes "velocity fields decay sufficiently fast at infinity".** For finite-width neural networks on bounded latent spaces, this holds empirically but is not formally proven. The VFD score is an UPPER BOUND on epistemic uncertainty (eq. 4c via Jensen), not the exact value. Document this in the docstring.

8. **The QGF integration is a thin bridge but must respect `QgfVarianceSignal`'s `[0,1]` semantics.** VFD's raw score is unbounded. The sigmoid normalization `σ(VFD · τ)` is a HEURISTIC mapping, not a probability. Document this clearly — callers feeding VFD into `confidence_from_disagreement` must understand they're getting a heuristic confidence, not a calibrated one. (Calibrated UQ comes from the conformal threshold, Phase 2 G2 — separate from the QGF bridge.)

9. **SAVE is explicitly out of scope.** A reviewer reading Research 420 + Plan 432 might expect SAVE (the active-fine-tuning half of the paper) to be addressed. It is NOT — SAVE is gradient descent (4,000 steps × 15 rounds × replay ratio 0.5), a training method that goes to `riir-train/.research/`. VFD is the UQ estimator, separable from SAVE. The paper even uses VFD for runtime failure detection (§6.4) WITHOUT any fine-tuning — that is the use case we ship.

10. **The DEC cross-check (R296) is independent, not redundant.** `codifferential(b̂) ≈ 0` checks the COMBINED drift's mass conservation; VFD measures MEMBER-vs-MEMBER disagreement. They catch different failure modes. A combined drift can be mass-conserving while members disagree wildly (and vice versa). Document as complementary sanity signals, not substitutes.

---

## TL;DR

Phase 1 ships `VfdScore` + `VfdScratch<M, D>` + `vfd_score_into` + `VfdVarianceSignal` (closes the QGF "ensemble KL" open item) behind `velocity_field_disagreement` feature. Phase 2 runs the GOAT gate — especially G2 (UQ floor per Issue 010) which is the make-or-break. Phase 3 promotes to default-on if G2 passes (closes Plan 376 Phase 6 G7 deferred gate — the velocity-field ensemble Super-GOAT becomes UQ-bearing). Phases 4–6 (heterogeneous-d, LatCal commitment, riir-ai runtime integration) deferred. SAVE (the paper's active-fine-tuning half) → riir-train, out of scope.
