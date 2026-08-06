# Issue 573: PoC — Beckmann Divergence Constraint on CCE LP (close the MFG dynamics gap)

**Date:** 2026-08-06
**Research:** [`katgpt-rs/.research/468_Beckmann_Transport_Divergence_Constraint_CCE_MFG_Dynamics.md`](../.research/468_Beckmann_Transport_Divergence_Constraint_CCE_MFG_Dynamics.md)
**Source paper:** [arXiv:2608.01692](https://arxiv.org/abs/2608.01692) — Beckmann Transport Models (Lee et al., May 2026)
**Blocks:** Promoting the CCE moderator's "MFG dynamics" limitation from documented-gap to closed
**Status:** Open

## Context

The shipped LP-CCE Moderator (Plan 295) has a documented limitation (`.docs/04_calibration/cce_moderator.md` §Limitations #2):

> "No dynamics. The LP treats the state distribution as free. MFG dynamics (occupation-measure flow constraints) are a Plan 325 follow-up."

riir-ai Plan 325 (the CCE runtime) shipped COMPLETE (2026-06-22) **without** closing this gap. The symptom: on RPS (zero-sum), the LP exploits the free state distribution to find a "CCE" that beats the zero-sum baseline (`.benchmarks/029_cce_moderator_goat.md` §G1 RPS) — a trivial artifact that shouldn't happen with honest dynamics.

Research 468 (BTM, arXiv:2608.01692) identifies that BTM's divergence equation `∇·(νb) = μ₀ − μ₁` (the Beckmann OT flux constraint) is the missing "occupation-measure flow constraint". The DEC `codifferential` (`crates/katgpt-dec/src/`) already ships the operator.

## The PoC question

**Does adding a Beckmann divergence feasibility constraint to the CCE LP close the RPS trivial-CCE artifact?**

The constraint: candidate `ρ` must be transport-feasible from initial `μ₀` — i.e., there must exist an edge flow `j = νb` such that `codifferential(j) = μ₀ − ρ` on the discretized state-space `CellComplex`.

## Tasks

- [ ] **T1** Build a minimal `CellComplex` over the CCE state-action space (RPS: 3 states × 3 actions = 9 cells, or a 3×3 grid).
- [ ] **T2** For the RPS game, compute the current (unconstrained) CCE via `CceLp::solve` — verify the trivial-CCE artifact (γ₀ < 0 for player 1's cost = exploits free state distribution).
- [ ] **T3** Add a linear feasibility constraint row: `codifferential(j) = μ₀ − ρ` for some edge-flow variable `j`. This makes `j` an auxiliary LP variable, with the divergence equation as a constraint.
- [ ] **T4** Re-solve the constrained CCE LP. Verify: does the constraint eliminate the trivial-CCE artifact (γ₀ ≈ 0 for the zero-sum RPS case)?
- [ ] **T5** Honest negative control: does the constraint over-restrict the feasible set on chicken / BoS (where a real Pareto-dominant CCE exists)? If the constraint makes these infeasible or degenerate, the Beckmann formulation is too restrictive and the transition-kernel form (Campi et al.) is needed instead.
- [ ] **T6** Record verdict in Research 468 §PoC Addendum:
  - If T4 PASS + T5 PASS → Gain confirmed; open a plan for `BeckmannFeasibleCce` behind a feature flag.
  - If T4 FAIL → Beckmann formulation doesn't close the artifact; the gap needs the discrete transition-kernel form. BTM's value is the theoretical lens (DEC/Stokes vocabulary for MFG dynamics), not the specific formulation.
  - If T4 PASS + T5 FAIL → Beckmann closes the artifact but over-restricts the feasible set; investigate whether a relaxed formulation (partial transport, soft penalty) preserves the fix without the restriction.

## What this PoC does NOT test

- Quality parity with the full MFG system (HJB backward equation). The Beckmann constraint is only the forward-equation (occupation-measure flow) half. The full MFG Nash equilibrium also needs the value-function (backward) half. This PoC tests whether the forward constraint alone closes the free-state-distribution artifact — a necessary but not sufficient condition for honest MFG dynamics.
- Crowd-scale latency. The constraint adds rows to the LP; the BFS-enumeration solver (`N·A + |D| ≤ ~25`) may need a real simplex for larger state spaces. This is a Plan-stage concern, not a PoC concern.

## Reference

- Research 468 §2.6 (honest uncertainty: Beckmann vs transition-kernel vs MFG continuity equation)
- `.docs/04_calibration/cce_moderator.md` §Limitations #2 (the documented gap)
- `.benchmarks/029_cce_moderator_goat.md` §G1 RPS (the trivial-CCE artifact)
- `crates/katgpt-dec/src/operators.rs` — `codifferential` (= divergence `δ`)
- `crates/katgpt-dec/src/stokes_calculus.rs` — `belief_mass_divergence` (the `δ(j) ≈ 0` steady-state special case)
