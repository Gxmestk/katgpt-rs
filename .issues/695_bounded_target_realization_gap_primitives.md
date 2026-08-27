# Issue 695: `bounded_target` + `realization_gap` open primitives (DiffusionOPSD extraction)

**Status:** Open — POC/proof task (filed from [Research 432](../.research/432_DiffusionOPSD_Bounded_Target_OPSD.md) / riir-train; cross-ref [Plan 360](../riir-train/.plans/360_opsd_bounded_target_gdsd_l4.md))

Distilled from arXiv:2608.24646 (DiffusionOPSD). Two composable modelless primitives for `katgpt-core`; SPSA itself is textbook (Spall 1987) — the value is the **shipped composition** (constructor + scorer-vitality canary + realization triage), verified absent from the corpus by grep 2026-08-27 (closest cousin `contrastive_scope` scores in/out-of-scope relevance, not a local-improvement direction).

## P1 — `bounded_target` module (`katgpt-core`, opt-in feature `bounded_target`)

- [ ] **T1** `spsa_direction(q: impl Fn(&[f32])->f32, x: &[f32], delta: f32, seed: u64) -> Option<UnitDir>` — ĝ = [Q(x+δΔ)−Q(x−δΔ)]/(2δ)·Δ, Rademacher Δ from BLAKE3(seed); pure-sign fallback form; `Option` = flagged indeterminate when |ΔQ| ≤ noise floor (2σ guard from the paper's ε→0 limit).
- [ ] **T2** `bounded_pair(x, d̂, ε) -> (t_plus, t_minus)` with ‖t±−x‖₂ = ε exact by construction; `eps_ladder(q, x, d̂, ε)` — 5-eval line search over {ε/4, ε/2, ε, 2ε, 4ε}, negative-control flag when Q is monotone (no interior optimum).
- [ ] **T3** `BoundedCorrection { dir: UnitDir, eps: f32 }` newtype — no constructor can produce ‖Δ‖ > ε (type-level contract, mirrors latent→raw clamp discipline).
- [ ] **T4** Scorer-vitality canary: `contrast(q, x, d̂, ε, g_min)` — fires when |Q(x+εd̂)−Q(x−εd̂)| < 2ε·g_min·(1−tol) on a known-g₁ fixture (dead-scorer detector for self-evolve loops).
- [ ] **T5** GOAT: G1 determinism + norm-bound bit-exact + sign-consistency on seeded concave Q + indeterminate on flat Q (negative control); G2 direction math ≤ ~100 ns @ d=16 (scorer caller-owned); G4 zero-alloc, fixed `[f32; 64]` cap.

## P2 — `realization_gap` module (`katgpt-core`, same feature or rider)

- [ ] **T6** `fixpoint_position(v0, t, eta, k) -> Vec-free closed form` — v_k = t + (1−η)^k·(v₀−t); budget law "k_needed > 3/η ⇒ re-anchor, don't iterate".
- [ ] **T7** `realization_ratio(promised: f32, realized: f32, k, eta, eps) -> Rho` + `triage(rho) -> FittingStarved | TargetStarved | OnModel` — ρ̂(k,η,ε) = (1−(1−η)^k)(1−cε²), c calibrated offline.
- [ ] **T8** GOAT: G1 ρ deterministic on frozen fixtures + calibration |ρ−ρ̂| ≤ tol; G2 O(1); G4 counters only. **UQ note:** as a point diagnostic it is not UQ-bearing; promoting ρ̂ to an interval predictor triggers the conformal-naive floor rule (Report-the-Floor, `.benchmarks/010`) — gate at that promotion, not before.

## Consumers (named, not filed here)

- riir-clippy score-bench: promised-vs-realized axis (target-axis score vs post-bounded-fixpoint healing) + `EvolveRecorder` fields — file in riir-clippy `.issues/` at adoption time.
- riir-train Plan 360 T1.4: ad-hoc ρ logging swaps for this primitive in Phase 3.
- riir-ai self-adaptive loops: per-cycle ρ on direction-vector updates within the (1−α)·D_max drift cap.

## Honest scope

Not Super-GOAT (Research 432 §5): diagnostic + constructor, not a new capability class; SPSA is prior art as math. GOAT-tier if the score-bench axis demonstrates a caught-and-triaged regression the realized-only axis misses.
