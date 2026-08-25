# Plan 578: Kinematic Rollout Primitive (`katgpt-core::kinematics`)

**Date:** 2026-08-25
**Research:** [katgpt-rs/.research/506_LDR_Kinematic_Integration_Closed_Form_Rollout.md](../.research/506_LDR_Kinematic_Integration_Closed_Form_Rollout.md)
**Source paper:** [arXiv:2608.09926](https://arxiv.org/abs/2608.09926) — Li et al., Latent Dynamics Reasoning (Adobe/UCSD 2026)
**Target:** `crates/katgpt-core/src/kinematics/` (new module) + Cargo feature `kinematic_rollout` (opt-in until GOAT)
**Cross-repo:** riir-ai Research 345 (consumer guide), riir-ai Issue 757 (falsifiable PoC — no consumer wiring before it passes)
**Status:** Active — Phase 0

---

## Goal

Ship the modelless core of LDR as a generic operator family: measured low-order dynamics (finite-difference state) + closed-form kinematic rollout + looming time-to-contact + regime predicates + residual-event surprise + an extrapolation-horizon admission bound. Zero deps, zero heap, per-channel SIMD-able. Exact (bit-identical, any horizon) on degree ≤ 3 motion — the provable ID-OOD gap ≡ 0 strengthening of the paper's empirical 20×. GOAT: G1 exactness certificates + determinism, G2 ns-cost, G3 default-off until pass, G4 alloc-free, UQ floor for the bound. Promote to default only on PASS (pure math with no default-path risk — expected outcome per the KARC precedent).

## Phase 1 — Core rollout math

### Tasks

- [ ] **T1.1** `kinematics/mod.rs` — `KinState { pos: [f32; D], vel: [f32; D], acc: [f32; D], tick: u32, n_obs: u8 }` POD + `observe_into` (ring-fed FD init; central 3-point stencil; ladder n∈{1,2,3,4} → order {0,1,2,3}; NaN + zero-Δt screens refuse).
- [ ] **T1.2** `kinematic_extrapolate_into(state, k, sched, out)` — Newton forward-difference closed form with precomputed coefficient lattice `[f32; K_MAX]`; `Sched::{ZeroJerk, ConstJerk, ClampedCorrection, GeometricDrag}` (drag with closed forms + terminal `vel_inf`); bit-identical to a reference step-by-step chain (same op order), cross-checked in tests.
- [ ] **T1.3** Unit fixtures: uniform / parabola / const-jerk analytic trajectories — prediction error exactly 0 at k ∈ {1, 10, 100, 1000}; Δt-rescale invariance (Δt vs Δt/2 agree at matched wall-times).
- [ ] **T1.4** G2 bench: single-target extrapolate < 10 ns (d=4, k=100); G4 counting-allocator 0 steady-state allocs.

## Phase 2 — Perception operators

### Tasks

- [ ] **T2.1** `time_to_contact(extent_ring) -> f32` — τ = σ/σ̇, log-extent variant, σ̇≈0 → ∞ guard; fixture: planted contact time recovered within 1 tick.
- [ ] **T2.2** `regime_predicates(state, extent_state) -> Regime` — {Uniform, Parabolic(g), Impulse, Looming, Drag} closed-form predicates + sigmoid-gated hysteresis; fixture: 100% classification on interleaved 5-regime streams (the paper's joint-training failure mode solved modellessly — falsifiable A/B vs a single mixed predictor).
- [ ] **T2.3** `residual_event` — prediction-residual z-score + sigmoid gate; CUSUM sustained-drift variant; impulse-vs-force discriminator (`|Δv|/Δt ≫ running |a|`) with wall-inference + restitution `e = |v_after|/|v_before|` on detection; fixtures: planted bounce detected at the exact tick, 0 alarms on 10⁵ clean ticks.
- [ ] **T2.4** `closest_approach(p1, v1, p2, v2)` — t*, miss distance, intercept quadratic, head-on elastic resolve; t*-ascending ordering law fixture (sorted t* = ground-truth contact order).
- [ ] **T2.5** `extrapolation_horizon(state, thr) -> (k*, conf)` — error-propagation bound B(k) + admission gate. **UQ-bearing:** bench vs the conformal-naive floor (`ConformalIntervalCalibrator<SeasonalNaiveForecaster>`, Plan 340 harness) on CRPS/coverage/Winkler over noisy-init fixtures; if it cannot beat the floor, ship rank-only (k* ordering, no coverage claim) — document which.

## Phase 3 — GOAT gate + benchmark doc

### Tasks

- [ ] **T3.1** `.benchmarks/578_kinematic_rollout_goat.md` — G1 (multi-seed determinism + all exactness fixtures), G2 (ns table incl. 1000-target batch), G3 (default build bit-identical, feature off), G4 (alloc counts), UQ-floor verdict row.
- [ ] **T3.2** PhyWorld-style deterministic fixture generator (paper's exact ID/OOD ranges: v∈[1,4]→[0.05,6], r∈[0.7,1.4]→[0.6,2], |ṙ|∈[0,0.03]→[0.05,0.09], T=31) emitting (trajectory, regime tag, event ticks, ID/OOD label); record the in-family ID-OOD gap ≡ 0 table vs the paper's published numbers.
- [ ] **T3.3** Promotion decision: on PASS promote `kinematic_rollout` to default (pure math, no dep surface — the KARC precedent); on any FAIL stay opt-in with the failing gate documented.

## Phase 4 — Optional hardening (defer-marked)

### Tasks

- [ ] **T4.1** `[-]` Lean exactness theorem in `.proofs/KatgptProof/` (deg ≤ 3 family exactness, rational-arithmetic version) — the FV-moat strengthening; defer until Phase 3 PASS.
- [ ] **T4.2** `[-]` `soft_argmax_statistic` (μ, σ geometric summary) — defer until a consumer lands (riir-ai P3: crowd cohesion σ, threat-field mass).
- [ ] **T4.3** `[-]` Consumer wiring (predictive `GenericSpatialBelief`, τ-gated flee) — riir-ai territory, gated on Issue 757 PoC; NOT in this plan.
