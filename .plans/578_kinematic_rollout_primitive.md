# Plan 578: Kinematic Rollout Primitive (`katgpt-core::kinematics`)

**Date:** 2026-08-25
**Research:** [katgpt-rs/.research/506_LDR_Kinematic_Integration_Closed_Form_Rollout.md](../.research/506_LDR_Kinematic_Integration_Closed_Form_Rollout.md)
**Source paper:** [arXiv:2608.09926](https://arxiv.org/abs/2608.09926) — Li et al., Latent Dynamics Reasoning (Adobe/UCSD 2026)
**Target:** `crates/katgpt-core/src/kinematics/` (new module) + Cargo feature `kinematic_rollout` (opt-in until GOAT)
**Cross-repo:** riir-ai Research 345 (consumer guide), riir-ai Issue 757 (falsifiable PoC — no consumer wiring before it passes)
**Status:** COMPLETE — promoted to DEFAULT-ON 2026-08-26 (Bench 680). See the status section at the end.

---

## Goal

Ship the modelless core of LDR as a generic operator family: measured low-order dynamics (finite-difference state) + closed-form kinematic rollout + looming time-to-contact + regime predicates + residual-event surprise + an extrapolation-horizon admission bound. Zero deps, zero heap, per-channel SIMD-able. Exact (bit-identical, any horizon) on degree ≤ 3 motion — the provable ID-OOD gap ≡ 0 strengthening of the paper's empirical 20×. GOAT: G1 exactness certificates + determinism, G2 ns-cost, G3 default-off until pass, G4 alloc-free, UQ floor for the bound. Promote to default only on PASS (pure math with no default-path risk — expected outcome per the KARC precedent).

## Phase 1 — Core rollout math

### Tasks

- [x] **T1.1** `kinematics/mod.rs` — `KinState { pos: [f32; D], vel: [f32; D], acc: [f32; D], tick: u32, n_obs: u8 }` POD + `observe_into` (ring-fed FD init; central 3-point stencil; ladder n∈{1,2,3,4} → order {0,1,2,3}; NaN + zero-Δt screens refuse).
- [x] **T1.2** `kinematic_extrapolate_into(state, k, sched, out)` — Newton forward-difference closed form with precomputed coefficient lattice `[f32; K_MAX]`; `Sched::{ZeroJerk, ConstJerk, ClampedCorrection, GeometricDrag}` (drag with closed forms + terminal `vel_inf`); bit-identical to a reference step-by-step chain (same op order), cross-checked in tests.
- [x] **T1.3** Unit fixtures: uniform / parabola / const-jerk analytic trajectories — prediction error exactly 0 at k ∈ {1, 10, 100, 1000}; Δt-rescale invariance (Δt vs Δt/2 agree at matched wall-times).
- [x] **T1.4** G2 bench: single-target extrapolate < 10 ns (d=4, k=100); G4 counting-allocator 0 steady-state allocs.

## Phase 2 — Perception operators

### Tasks

- [x] **T2.1** `time_to_contact(extent_ring) -> f32` — τ = σ/σ̇, log-extent variant, σ̇≈0 → ∞ guard; fixture: planted contact time recovered within 1 tick.
- [x] **T2.2** `regime_predicates(state, extent_state) -> Regime` — {Uniform, Parabolic(g), Impulse, Looming, Drag} closed-form predicates + sigmoid-gated hysteresis; fixture: 100% classification on interleaved 5-regime streams (the paper's joint-training failure mode solved modellessly — falsifiable A/B vs a single mixed predictor).
- [x] **T2.3** `residual_event` — prediction-residual z-score + sigmoid gate; CUSUM sustained-drift variant; impulse-vs-force discriminator (`|Δv|/Δt ≫ running |a|`) with wall-inference + restitution `e = |v_after|/|v_before|` on detection; fixtures: planted bounce detected at the exact tick, 0 alarms on 10⁵ clean ticks.
- [x] **T2.4** `closest_approach(p1, v1, p2, v2)` — t*, miss distance, intercept quadratic, head-on elastic resolve; t*-ascending ordering law fixture (sorted t* = ground-truth contact order).
- [x] **T2.5** `extrapolation_horizon(state, thr) -> (k*, conf)` — error-propagation bound B(k) + admission gate. **UQ-bearing:** bench vs the conformal-naive floor (`ConformalIntervalCalibrator<SeasonalNaiveForecaster>`, Plan 340 harness) on CRPS/coverage/Winkler over noisy-init fixtures; if it cannot beat the floor, ship rank-only (k* ordering, no coverage claim) — document which.

## Phase 3 — GOAT gate + benchmark doc

### Tasks

- [x] **T3.1** GOAT bench doc — **Bench 680** (renumbered from the plan's literal `578_...` filename per `.benchmarks/.highwater`; renumbering noted inside the doc): G1 (multi-seed determinism + all exactness fixtures), G2 (ns table incl. 1000-target batch), G3 (default build bit-identical, feature off), G4 (alloc counts), UQ-floor verdict row.
- [x] **T3.2** PhyWorld-style deterministic fixture generator (paper's exact ID/OOD ranges: v∈[1,4]→[0.05,6], r∈[0.7,1.4]→[0.6,2], |ṙ|∈[0,0.03]→[0.05,0.09], T=31) emitting (trajectory, regime tag, event ticks, ID/OOD label); the in-family ID-OOD gap ≡ 0 table vs the paper's published numbers.
- [x] **T3.3** Promotion decision: **PASS → promoted to `default`** (pure math, no dep surface — the KARC precedent; recorded in Bench 680 + the Cargo.toml Phase 28 comment).

## Phase 4 — Optional hardening (defer-marked)

### Tasks

- [-] **T4.1** Lean exactness theorem in `.proofs/KatgptProof/` (deg ≤ 3 family exactness, rational-arithmetic version) — the FV-moat strengthening; defer until Phase 3 PASS.
- [-] **T4.2** `soft_argmax_statistic` (μ, σ geometric summary) — defer until a consumer lands (riir-ai P3: crowd cohesion σ, threat-field mass).
- [-] **T4.3** Consumer wiring (predictive `GenericSpatialBelief`, τ-gated flee) — riir-ai territory, gated on Issue 757 PoC; NOT in this plan.

---

## Status — COMPLETE (2026-08-26, Bench 680)

**GOAT G1–G4 ALL PASS → `kinematic_rollout` DEFAULT-ON** (KARC precedent). UQ floor: **RANK-ONLY** (the policy's legitimate fallback). Full record: [`.benchmarks/680_kinematic_rollout_goat.md`](../.benchmarks/680_kinematic_rollout_goat.md).

### Measured numbers

| Gate | Result |
|---|---|
| G1 exactness | uniform/parabola error **exactly 0.0** at k ∈ {1,10,100,1000}; const-jerk 0 at {1,10,100}; closed ≡ chain **bit-identical** on the exactness family; random-walk divergence band 1.71e-5; drag bit-identity above `DRAG_ACC_FLOOR`, rel < 1e-6 below |
| G1 determinism | 6 seeds × ID/OOD pipeline hashes bit-equal across runs |
| G1 f32 boundary | cubic k=1000 asserts rel < 1e-6 — `C(1002,3) = 167,167,000` needs 25 mantissa bits (the honest limit, documented; deg ≤ 2 exact through the full lattice) |
| G2 | single-target extrapolate **4.64 ns** (d=4, k=100; < 10 ns gate); batch 4.63 ns/target; drag arm 9.77 ns (powf); TTC 1.42 ns; horizon 20.9 ns; closest approach 2.70 ns; regime classify 35.5 ns; monitor 15.7 ns |
| G3 | pre-promotion default lib = **1917/0/7** (identical with the feature off); post-promotion **1951/0/7** (+34 module tests) |
| G4 | **0** steady-state allocs (10k iterations, every operator); **0** per-tick in the fixture pipeline |
| UQ floor | parabola **BeatsFloor** (CRPS ratio 0.10, Winkler 0.08, coverage 0.86 vs floor's 0.17 collapse); uniform 1.08× LOSE, white-noise 1.08× LOSE at h=1 — the √2·σ shared-floor analysis. **RANK-ONLY shipped** (k* ordering, no coverage claim) |
| T3.2 gap table | **all five regimes exactly 0.0 both arms** (ID + OOD, k ∈ {1,8,31}, 6 seeds × 2 range sets) — gap ≡ 0 by construction vs the paper's empirical ~23.9× |
| Classification | 100% on interleaved 5-regime streams (5 seeds × ID/OOD); planted bounce at the exact tick, restitution exact; 0 alarms on 10⁵ clean uniform ticks (+ 5k parabola ticks — the parabola's own 24-bit exactness ceiling) |

### Design deviations from the plan text (all documented in the module docs + Bench 680)

1. **Newton *backward* form, anchored at the latest observation** — the plan's
   "Newton forward-difference" phrasing describes the same Gregory-Newton
   polynomial; the backward differences use only past data at the anchor.
   The state's `vel` is the mean velocity over the last step (exact under
   the binomial form); `central_velocity` provides the O(Δt²) unbiased
   instantaneous estimate.
2. **`Sched::Measured` added** (a 5th variant) — the ladder's n=4 → order 3
   output needs a consumer to complete the exactness story at order 3.
3. **The drag closed form's chain convention** (decay-before-apply for future
   steps) is pinned by hand-verification at k=1 (G=ρ) and k=2 (G=2ρ+ρ²);
   `terminal_velocity` carries the ρ factor (Research 506's quote assumed
   the undecayed-first-step convention).
4. **The k=1000 cubic row** asserts the 2⁻²⁴ band, not exactly 0 (see the
   f32 boundary above).
5. **`time_to_contact` takes the last two extents** rather than a ring type
   (the ring's last two samples ARE the statistic; a slice API would be
   ceremony).

### En-route bug finds (the reason the fixtures exist)

- Two `ResidualMonitor` warmup designs failed before the running-average +
  CUSUM-reset fix (the ladder-fill transient reads as sustained drift).
- The regime classifier's hysteresis originally let a sticky low-priority
  regime block a higher-priority switch forever (fast parabolic motion looks
  locally straight); priority-supersedes-immediately is the fix.
- The force-scale EMA needs winsorization (both the raw and the
  hard-exclusion variants fail in opposite directions — measured).
- The floor adapter needed three estimator fixes found by the bench itself:
  σ̂ from the full-ladder residual (order-screen death spiral), the
  predictive `+1` (new-observation noise), and EMA-smoothed order screens.

**Consumer unblock:** riir-ai Issue 757 (anticipatory-belief PoC) is now
unblocked — `katgpt_core::kinematics` is live and DEFAULT-ON.
