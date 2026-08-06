# Issue 575: PoC — RPS Trivial-CCE Artifact Closure (options 1 + 3)

**Date:** 2026-08-06
**Prior PoCs:** [Issue 573](573_beckmann_divergence_constraint_cce_poc.md) (Beckmann FAIL),
[Issue 574](574_transition_kernel_constraint_cce_poc.md) (transition-kernel PASS on action-dependent MDPs)
**Research:** [`katgpt-rs/.research/468_Beckmann_Transport_Divergence_Constraint_CCE_MFG_Dynamics.md`](../.research/468_Beckmann_Transport_Divergence_Constraint_CCE_MFG_Dynamics.md) §7
**Blocks:** Closing the CCE moderator's RPS artifact for zero-sum games (`.docs/04_calibration/cce_moderator.md` §Limitations)

**Status:** RESOLVED — Option 1 FAIL, Option 3 PASS (verdict reached 2026-08-06)

## PoC Verdict: Option 1 FAIL, Option 3 PASS — RPS artifact CLOSED by 2-player CCE

The RPS trivial-CCE artifact (free-state-distribution exploitation on zero-sum games)
is **closed by option 3** (2-player CCE via `HeterogeneousPayoff<9, 9>`). Option 1
(richer deviation class) is **analytically + empirically proven insufficient** — the
artifact is a fixed point of best-response play.

### Results (6/6 tests PASS)

| Test | Result | Detail |
|---|---|---|
| T1 (artifact reproduced) | PASS | γ₀ = −1.0, ρ(R,P)=1.0 with constant deviations |
| T2 (option 1 — full 27-deviation class) | **FAIL** | max ER = 0.000000 over all 27 deviations. Artifact is an unconquerable CCE for ANY deviation class. |
| T3+T4 (2-player RPS constructed) | PASS | N=9, A=9 joint-action model, zero-sum verified |
| T5 (artifact rejected) | **PASS** | P2 ER = 2.0 > 0 (P2 deviates Rock → Scissors, cost 1.0 → −1.0). Artifact is NOT a 2-player CCE. |
| T6 (Nash accepted) | PASS | Nash (uniform honest joint): γ₁=γ₂=0, no profitable deviation |
| T7 (verdict) | PASS | RPS artifact CLOSED by option 3 |

### Why option 1 FAILS (the fixed-point proof)

The artifact ρ(state=Rock, action=Paper) = 1 has γ(ρ) = −1.0. Any deviation κ:
- γ_dev(ρ, κ) = μ(Rock) · Σ_a κ(Rock)[a] · cost(Rock, a).
- Best case: κ(Rock) = Paper → γ_dev = −1.0, ER = 0 (ties).
- Any other κ(Rock): γ_dev > −1.0, ER < 0 (worse).

**Max ER = 0 over ALL deviation classes.** The artifact IS the best-response strategy —
no deviation in any class can beat it. This was confirmed empirically with all 27
deterministic state-dependent deviations.

### Why option 3 PASSES (player 2's profitable deviation)

The N=3 single-player model only constrains P1. Player 2 is assumed honest. But in
the artifact, P2 is stuck playing Rock — P2 deviates to Scissors (beats Paper) for
profit. Modeling BOTH players' constraints via `HeterogeneousPayoff<9, 9>` (joint
recommendation state, joint play action) catches this: P2's ER = 2.0 > 0, violating
the 2-player CCE condition. The Nash equilibrium (uniform) is the valid CCE.

### What this enables

The existing `is_heterogeneous_cce` substrate can **verify** the closure (reject the
artifact, accept Nash). The BFS solver cannot **solve** the 2-player CCE LP at NA=81
(variable count exceeds the ~25-variable limit) — production use on larger games
needs a real simplex solver (cce_moderator §Limitations #3, pre-existing).

For the CCE moderator runtime (riir-ai Plan 325): zero-sum games should use the
2-player CCE path (`is_heterogeneous_cce` for verification), not the single-player
`solve` (which produces the trivial artifact). The single-player path is correct for
games with a genuine single decision-maker (MDPs, Issue 574).

### PoC artifact

[`tests/rps_artifact_closure_poc.rs`](../tests/rps_artifact_closure_poc.rs) — 6 tests,
gated on `cce_moderator`. Part A (option 1 FAIL) + Part B (option 3 PASS). Kept as
a reproducible artifact completing the Research 468 §7 candidate triage.

## The remaining gap

Issue 574 closed the free-state-distribution artifact for games with **action-dependent
transitions** (MDPs). Issue 573 proved the transition-kernel constraint **reduces to the
marginal constraint** for state-independent transitions (like RPS), which doesn't close
the artifact (γ₀ = −1.0 persists).

Research 468 §7 listed three candidates that WOULD close the RPS artifact:

1. **Richer deviation class** — state-dependent deviations
2. ~~**Transition-kernel constraint**~~ — **DONE** (Plan 569), closes artifact for
   action-dependent MDPs but NOT for RPS (Issue 574 T6)
3. **Honest-mediator / both players' deviations** — model player 2's constraints too

This PoC tests candidates **1** and **3**.

## Analytical pre-analysis (informs the PoC design)

### Why option 1 (richer deviation class) is predicted to FAIL

The RPS artifact on N=3 (state = opponent's action) is `ρ(state=Rock, action=Paper) = 1`,
giving `γ₀ = cost(Rock, Paper) = −R[Paper][Rock] = −1.0`.

The CCE condition: for all κ ∈ D, `γ(ρ) ≤ γ_dev(ρ, κ)`.
- `γ(ρ) = −1.0` (following = play Paper when opponent plays Rock = win).
- Any deviation κ: `γ_dev(ρ, κ) = μ(Rock) · Σ_a κ(Rock)[a] · cost(Rock, a)`.
  - `κ(Rock) = Paper`: γ_dev = −1.0, ER = −1 − (−1) = **0** (ties).
  - `κ(Rock) = Rock`: γ_dev = 0.0, ER = −1 − 0 = −1 (worse).
  - `κ(Rock) = Scissors`: γ_dev = 1.0, ER = −1 − 1 = −2 (worse).

**Max ER = 0 over ALL deviation classes.** The artifact IS a best-response — no
deviation in ANY class can make ER > 0. This is a fixed point of best-response play.

Even with the marginal constraint ν = uniform (from the transition-kernel reduction),
the artifact persists as `ρ(s, BR(s)) = 1/3` for each s, with `γ₀ = −1.0` and every
state individually playing its best response.

### Why option 3 (2-player CCE) is predicted to PASS

The fundamental issue: the N=3 model only constrains **player 1's** deviations. Player
2 is assumed "honest" (always plays the state). But in the artifact, player 2 is stuck
playing Rock — player 2 would deviate to Scissors (beats Paper) for profit.

Modeling RPS as a true 2-player game with **N=9** (state = joint recommendation
`(s₁, s₂)`) and **A=9** (action = joint play `(a₁, a₂)`):
- Player 1's cost: `cost₁((s₁,s₂), (a₁,a₂)) = −R[a₁][a₂]`.
- Player 2's cost: `cost₂((s₁,s₂), (a₁,a₂)) = R[a₁][a₂]` (zero-sum).
- P1 deviations: replace a₁ keeping a₂ = s₂ (honest P2).
- P2 deviations: replace a₂ keeping a₁ = s₁ (honest P1).

Under the artifact `ρ((P,R), (P,R)) = 1`:
- P2's cost: `γ₂ = R[P][R] = 1` (loses).
- P2 deviating to Scissors: `cost₂((P,R), (P,S)) = R[P][S] = −1` (wins).
- `ER₂ = 1 − (−1) = 2 > 0`. **VIOLATED.** The artifact is NOT a 2-player CCE.

## PoC design

### Part A: Option 1 — richer deviation class (N=3, A=3)

- T1: Reproduce the artifact (γ₀ = −1.0, ρ(R,P) = 1).
- T2: Add ALL 27 deterministic state-dependent deviations. Re-solve.
- Expected: γ₀ = −1.0 persists (analytical fixed-point proof confirmed empirically).

### Part B: Option 3 — 2-player CCE (N=9, A=9)

- T3: Construct the 2-player RPS game via `HeterogeneousPayoff<9, 9>`.
- T4: Construct P1 + P2 deviation classes (3 each = 6 total, joint-action kernels).
- T5: Verify the artifact ρ is **REJECTED** by `is_heterogeneous_cce` (ER₂ > 0).
- T6: Verify the Nash equilibrium ρ (uniform honest joint) is **ACCEPTED**.

### Verdict + documentation

- T7: Record verdict in Research 468 §9 (PoC Addendum 3).

## Tasks

- [x] **T1** (Part A) Reproduce the N=3 RPS artifact with constant deviations.
      - γ₀ = −1.000000, ρ(Rock=0, Paper=1) = 1.0.
- [x] **T2** (Part A) Verify the artifact ρ is a valid CCE under all 27
      deterministic state-dependent deviations (no re-solve needed — direct ER check).
      - max ER = 0.000000 over all 27 deviations (dev #9 = best-response ties).
      - Option 1 FAIL: the artifact is an unconquerable CCE for ANY deviation class.
- [x] **T3** (Part B) Construct the N=9, A=9 2-player RPS `HeterogeneousPayoff`.
- [x] **T4** (Part B) Construct P1 + P2 joint-action deviation classes (3 each).
- [x] **T5** (Part B) Verify artifact ρ is rejected by `is_heterogeneous_cce`.
      - P2 best ER = 2.0 > 0 (deviates Rock → Scissors: cost 1.0 → −1.0).
- [x] **T6** (Part B) Verify Nash ρ is accepted.
      - Nash (uniform honest joint): γ₁ = γ₂ = 0, is_heterogeneous_cce PASS.
- [x] **T7** Record verdict in Research 468 §9 (PoC Addendum 3).

## What this PoC does NOT test

- **Solving the 2-player CCE LP at N=9, A=9.** NA=81 variables exceeds the BFS solver's
  ~25-variable limit (cce_moderator §Limitations #3). The PoC VERIFIES closure
  (via `is_heterogeneous_cce`) but does not SOLVE for the optimal CCE. Production use
  on larger games needs a real simplex solver — a separate concern.
- **General-sum 2-player games.** This PoC uses RPS (zero-sum). General-sum 2-player
  CCE closure would need the same joint-action model with asymmetric cost tensors.

## Reference

- Research 468 §7 (the three candidates)
- Research 468 §8 (Issue 574 PoC Addendum — transition-kernel PASS)
- [Issue 573](573_beckmann_divergence_constraint_cce_poc.md) — Beckmann PoC, T4 FAIL
- [Issue 574](574_transition_kernel_constraint_cce_poc.md) — transition-kernel PoC, T4 PASS
- `.docs/04_calibration/cce_moderator.md` §Limitations
- `crates/katgpt-core/src/cce/heterogeneous.rs` — `HeterogeneousPayoff` trait
- `crates/katgpt-core/src/cce/lp.rs` — `is_heterogeneous_cce`, `solve_heterogeneous`
