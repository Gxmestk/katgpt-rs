# Plan 575: Risk-Controlled Exit Primitive (Dual-Threshold + UCB Calibration)

**Date:** 2026-08-20
**Research:** [katgpt-rs/.research/494_Conformal_Thinking_Dual_Threshold_Risk_Control_Exit.md](../.research/494_Conformal_Thinking_Dual_Threshold_Risk_Control_Exit.md) (+ riir-ai Research 339 guide)
**Source paper:** [arXiv:2602.03814](https://arxiv.org/abs/2602.03814) — "Conformal Thinking: Risk Control for Reasoning on a Compute Budget", Wang et al., ICML 2026.
**Target:** `katgpt-rs/crates/katgpt-core/src/risk_control_exit.rs` (new module) + Cargo feature `risk_control_exit`
**Status:** Active — Phase 1 pending

---

## Goal

Ship `RiskControlledExit` — a generic, modelless dual-threshold compute-exit primitive: (1) upper threshold (stop-when-confident), (2) parametric lower threshold = squeezed-sigmoid confidence schedule `λ−(t) = σ(c(ωt − sB), l, u)` (stop-when-not-progressing), (3) an offline UCB/Hoeffding calibrator (`Risk̂ + √(log(1/δ)/2n) ≤ ε`) that turns a labeled validation set into guaranteed thresholds, (4) efficiency-loss selection among feasible (signal, threshold) pairs, (5) the App. C p_i ≥ p_c disarm tripwire. Consumers: MCTS termination, loop/speculative halting (Plan 304 fusion, Bebop Issue 023), riir-ai per-NPC deliberation (Research 339), riir-clippy fixpoint exit. **This is a UQ-bearing primitive — the exit-floor rule applies** (G2 must beat fixed-budget AND single-threshold floors at matched realized risk).

## Phase 1 — Core primitive (leaf-clean, modelless)

### Tasks

- [ ] **T1.1** `risk_control_exit.rs`: `DualExitPolicy { lambda_plus, c, s, l, u }` + `fn exit(&self, s_tilde: f32, omega_t: u32, budget: u32) -> ExitVerdict {Continue, Commit, Abandon}` — 2 comparisons + 1 squeezed sigmoid (`σ(z,l,u) = (u−l)/(1+e^−z) + l`), zero-alloc, `#[inline]`.
- [ ] **T1.2** Shape helpers: `linear (s=0.5, cB≪1) / exponential (s>1) / log (s<0) / constant (c→0)` presets + doc table (paper Eq. 12–13).
- [ ] **T1.3** Loss module: `fp_loss` (Eq. 8), `fn_farsighted_loss` (Eq. 9 — the future-correctness sum), `regret_loss` (Eq. 10), `past_wrongness` (Eq. 11); all bounded [0,1], pure fns over outcome slices.
- [ ] **T1.4** UCB calibrator: `calibrate(traj: &[(signal, outcome)], epsilon, delta, grid) -> CalibratedPolicy` — empirical risk + `sqrt(ln(1/delta)/(2n))` correction; `delta_over_grid` variant (δ/|G| multiple-comparison); refuse non-monotone risk-vs-hyperparam spans (monotonicity is ASSUMED by the paper — we verify empirically and fail loud).
- [ ] **T1.5** Two-step decoupled selection (`λ+` at ε+, then `{c,s,l}` at ε− conditioned) + efficiency-loss argmin among feasible pairs (Algorithm 1).
- [ ] **T1.6** Runtime tripwire: `PiGePcMonitor` — rolling count of lower-exit wrongs vs rights; disarm lower threshold when p_i < p_c (App. C); upper-only mode remains guaranteed.
- [ ] **T1.7** Unit tests: sigmoid schedule shapes pinned; exit verdict boundaries (λ+ > λ− invariant, mutual exclusivity); Hoeffding bound numerics; monotonicity refusal; tripwire disarm path.

## Phase 2 — GOAT gate (feature flag + bench + floors)

### Tasks

- [ ] **T2.1** Feature flag `risk_control_exit = []` (opt-in); module gated; lib test surface.
- [ ] **T2.2** G1 risk-hold bench: synthetic confidence trajectories with known ground truth (solvable/unsolvable mixture); assert realized exit-risk ≤ ε across 40 validation/test resplits — AND show naive (no-correction) calibration violating the target on some resplits (reproduce paper Fig. 4 shape; if naive never violates at these n, record honestly and shrink n).
- [ ] **T2.3** G2 exit-floor bench (the floor rule): crowd-composition sweep 3:1 / 1:1 / 1:3 trivial:stuck — dual-exit vs (a) fixed-budget floor vs (b) single-threshold floor; plot correctness-vs-compute (paper Fig. 6 protocol); dual must win or tie each floor at matched risk and win overall (expect: matches paper — upper-only ≈ dual at 3:1, dual dominates at 1:1/1:3). FAIL → do not promote; record as refuted axis.
- [ ] **T2.4** G4 alloc gate: per-call exit check alloc-free (counting allocator), calibration scratch reused.
- [ ] **T2.5** Benchmark doc `.benchmarks/NNN_risk_control_exit_goat.md` with verdicts + honest caveats (monotonicity spans found, decoupling, shift sensitivity).
- [ ] **T2.6** GOAT review: if G1+G2+G4 pass modellessly → promote `risk_control_exit` to default; else stay opt-in with documented reason.

## Phase 3 — First in-tree consumers (each its own gate)

### Tasks

- [ ] **T3.1** MCTS: optional dual-exit termination alongside budget exhaustion (confident-line stop; hopeless stop); labels from self-play outcomes; bench vs fixed-budget search at matched move quality.
- [ ] **T3.2** Plan 304 fusion: `GainCostLoopHalter` accepts an optional calibrated dual bound (τ fallback preserved); regression-test parity at τ-equivalent settings.
- [ ] **T3.3** Bebop Issue 023 re-gate: adaptive-γ via UCB-calibrated schedule (replaces the unproven regression); closes the open issue if G1 holds.
- [ ] **T3.4** riir-ai consumer handoff: `SwarmDeliberationSystem` wiring per Research 339 P0 (lands in riir-ai behind its own feature; not this repo).

## Non-goals

- The paper's trained MLP probe signal (modelless substitute: sigmoid projection onto direction vectors — consumer-side choice, not primitive).
- Online/instance-wise threshold adaptation (paper's own future work).
- Any softmax anywhere (house rule).
