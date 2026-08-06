# Issue 574: PoC — Transition-Kernel Constraint on CCE LP (close the MFG dynamics gap)

**Date:** 2026-08-06
**Prior PoC:** [Issue 573](573_beckmann_divergence_constraint_cce_poc.md) — Beckmann divergence constraint, T4 FAIL
**Research:** [`katgpt-rs/.research/468_Beckmann_Transport_Divergence_Constraint_CCE_MFG_Dynamics.md`](../.research/468_Beckmann_Transport_Divergence_Constraint_CCE_MFG_Dynamics.md) §7
**Source paper:** [arXiv:2606.20062](https://arxiv.org/abs/2606.20062) — Campi, Cannerozzi, Tzoumans — Optimal CCEs in MFGs via LP + No-Regret Learning
**Blocks:** Promoting the CCE moderator's "MFG dynamics" limitation (`.docs/04_calibration/cce_moderator.md` §Limitations #2) from documented-gap to closed
**Status:** RESOLVED — T4 PASS (verdict reached 2026-08-06)

## PoC Verdict: T4 PASS — transition-kernel constraint CLOSES the artifact on games with action-dependent transitions

The transition-kernel (balance equation) constraint definitively closes the
free-state-distribution artifact on a real MDP with action-dependent transitions.
This is the first formulation to succeed where Issue 573's Beckmann divergence
constraint failed.

### Results (6/6 tests PASS)

### The closing result

| Variant | γ₀ | Artifact? |
|---|---|---|
| Unconstrained CCE (free state dist) | **0.000000** | **Yes** — all mass on (HIGH, WAIT), below true optimum |
| True MDP optimum (policy iteration) | **0.833333** (5/6) | N/A — the honest baseline |
| Constrained CCE (balance equation) | **0.833333** | **No** — matches true optimum exactly (gap = 0.0) |

The balance equation forces the occupation measure to be a stationary
distribution of the transition kernel `P(s'|s,a)`. The CCE can no longer freely
choose the state distribution — it must be self-consistent with the MDP
dynamics. This closes the artifact **completely** (residual gap = 0.000000).

### Why this works (and Beckmann didn't)

The Beckmann constraint (Issue 573) operates on the state marginal `ν(s)`,
restricting which state distributions are transport-reachable. But on a
connected graph, ALL distributions are mutually transport-reachable → the
constraint is vacuous.

The transition-kernel constraint is different: it restricts which `(state,
action)` distributions are **self-consistent** under the dynamics. You can't
put all mass on `(HIGH, WAIT)` because the transitions from `(HIGH, WAIT)`
only keep you in HIGH with probability 0.5 — the stationary distribution
requires mass on LOW too.

### RPS reduction (T6)

For RPS (state-independent transitions `P(s'|s,a) = 1/3`), the balance
equation reduces to `ν = uniform` — identical to Issue 573 T4a (already tested,
γ₀ = -1.0 persists). The transition-kernel constraint CANNOT close the RPS
artifact because RPS has no real dynamics. RPS needs the richer-deviation-class
fix (option 1 from Research 468 §7) — a separate PoC.

### What this enables

Per T7 verdict branch: **PLAN WARRANTED for `TransitionKernelCce`** as a CCE
LP variant behind a `transition_kernel` feature flag. The constraint adds
`N-1` balance-equation rows to the CCE LP (one per state, one redundant with
normalization). For the BFS-enumeration solver, the complexity increase is
`C(N·A + |D|, 1 + |D| + N-1)` vs `C(N·A + |D|, 1 + |D|)` — still exact for
`N·A + |D| ≤ ~25`.

### PoC artifact

[`tests/transition_kernel_cce_poc.rs`](../tests/transition_kernel_cce_poc.rs) —
6 tests, gated on `cce_moderator`. Kept as a reproducible positive-result
artifact (the mirror image of Issue 573's negative-result PoC).

## Context

Issue 573 (Beckmann divergence constraint) produced a definitive **T4 FAIL**: the
Beckmann formulation does NOT close the RPS trivial-CCE artifact. Research 468 §7
identified three candidates that WOULD close it:

1. Richer deviation class (state-dependent best-response deviations)
2. **Transition-kernel constraint** (Campi et al.) — `ν(s') = Σ_{s,a} ρ(s,a)·P(s'|s,a)` — this PoC
3. Honest-mediator constraint (both players' deviations)

This PoC tests candidate 2.

## The PoC question

**Does adding the stationary transition-kernel balance equation to the CCE LP close the free-state-distribution artifact on a real MDP?**

The constraint (standard MDP stationarity / Campi et al. MFG consistency):

```
ν(s') = Σ_s Σ_a ρ(s,a) · P(s'|s,a)     for each s'
```

where `ν(s') = Σ_a ρ(s',a')` is the state marginal. This says: the state
distribution must be a stationary distribution of the transition kernel P induced
by the occupation measure ρ.

Equivalently (balance equation, one redundant row per normalization):

```
ν(s) = Σ_{s',a'} ρ(s',a') · P(s|s',a')     for each s
```

## Critical nuance from Issue 573

For RPS modeled as MFG (state = opponent's action, action = player's action),
the transition kernel is **state/action-independent**: `P(s'|s,a) = 1/3` for all
s,a,s' (opponent plays uniformly at random, independent of history). In this
regime, the balance equation reduces to `ν(s') = 1/3` for all s' — exactly the
marginal constraint that Issue 573 T4a already tested (γ₀ = -1.0 persists).

**Therefore: testing the transition-kernel constraint on RPS would reproduce
Issue 573 T4a's failure.** The PoC MUST test on a game with **action-dependent
transitions** to be meaningful.

## PoC design

### MDP game (2 states, 2 actions, action-dependent transitions)

States: `LOW (0)`, `HIGH (1)`
Actions: `WAIT (0)`, `INVEST (1)`

Transitions:
- `P(HIGH | LOW,  WAIT)  = 0.1`  (waiting in a downturn → usually stay low)
- `P(HIGH | LOW,  INVEST) = 0.6` (investing in a downturn → likely recover)
- `P(HIGH | HIGH, WAIT)  = 0.5`  (waiting in a boom → maintain)
- `P(HIGH | HIGH, INVEST) = 0.2` (overinvesting in a boom → crash)

Costs:
- `cost(LOW,  WAIT)  = 1.0` (low productivity + idle = cost)
- `cost(LOW,  INVEST) = 3.0` (expensive investment in downturn)
- `cost(HIGH, WAIT)  = 0.0` (boom + idle = free profit — the exploit target)
- `cost(HIGH, INVEST) = 2.0` (unnecessary investment)

### Expected artifact

**Unconstrained CCE** (free state distribution): concentrates all mass on
`(HIGH, WAIT)` → `γ₀ = 0`. This is below the true MDP optimum (exploiting the
free state distribution to "always be in the boom state").

**True MDP optimum** (average-cost policy iteration): always WAIT. Stationary
distribution `ν(LOW) = 5/6, ν(HIGH) = 1/6`. Average cost `γ₀ = 5/6 ≈ 0.833`.

**Constrained CCE** (with balance equation): should recover the honest optimum
`γ₀ ≈ 5/6`.

### Analytical verification (closed-form)

Unconstrained BFS solution: `ρ(HIGH, WAIT) = 1`, all others 0. `γ₀ = 0`.
CCE constraints satisfied (following = 0, deviating to INVEST costs 2).
This IS the artifact — γ₀ = 0 < 5/6.

Constrained BFS solution: `ρ(LOW, WAIT) = 5/6, ρ(HIGH, WAIT) = 1/6`, all others 0.
Balance check: `ν(HIGH) = 1/6`. Inflow = `0.1(5/6) + 0.5(1/6) = 1/12 + 5/12 = 6/12 = 1/2`... 

Wait — that doesn't balance. Let me recheck. `ν(HIGH) = 1/6`, but inflow =
`(5/6)(0.1) + (1/6)(0.5) = 0.5/6 + 0.5/6 = 1/6`. ✓ (Self-consistent.)

`γ₀ = (5/6)(1) + (1/6)(0) = 5/6 ≈ 0.833`. Matches true MDP optimum.

## Tasks

- [x] **T1** Build the MDP game struct implementing `PayoffTensor<2, 2>` with the
      transition kernel `P(s'|s,a)` defined as a local `[[[f64; 2]; 2]; 2]` array.
      Constant deviation class: {always WAIT, always INVEST} (2 deviations).
- [x] **T2** Unconstrained CCE: verify the artifact `γ₀ ≈ 0` (all mass on
      `(HIGH, WAIT)`). Cross-check with shipped `CceLp::solve`.
      - γ₀ = 0.000000, ρ(HIGH,WAIT) = 1.0. Cross-checked with shipped CceLp::solve.
- [x] **T3** Compute the true MDP optimum independently via policy iteration
      (4 deterministic policies to enumerate for N=2,A=2 — trivially exact).
      Record the honest baseline `γ₀ = 5/6`.
      - Optimal policy = always WAIT, γ₀ = 5/6 ≈ 0.833.
- [x] **T4** Add `N-1 = 1` balance-equation constraint row to the LP:
      `0.1·ρ(LOW,WAIT) + 0.6·ρ(LOW,INVEST) + 0.5·ρ(HIGH,WAIT) + 0.2·ρ(HIGH,INVEST)
       - ρ(HIGH,WAIT) - ρ(HIGH,INVEST) = 0`.
      Re-solve the constrained CCE LP. Verify `γ₀ ≈ 5/6` (artifact closed —
      matches the honest MDP optimum from T3).
      - **PASS**: γ₀ = 0.833333 (residual gap to true optimum = 0.000000).
- [x] **T5** Verify the constrained ρ is still a valid CCE (ER ≤ 0 for both
      deviations) — the constraint does NOT over-restrict.
      - PASS: ER = 0.0 for both deviations. Shipped `is_cce` check passes.
- [x] **T6** Analytical note: for RPS (state-independent transitions), the
      balance equation reduces to `ν = uniform` — identical to Issue 573 T4a
      (already tested, γ₀ = -1.0 persists). Document this reduction explicitly
      so the verdict is complete across both game classes.
      - Verified numerically: ν = [1/3, 1/3, 1/3] for any valid ρ under uniform transitions.
- [x] **T7** Record verdict in Research 468 §8 (PoC Addendum 2):
  - **T4 PASS** → transition-kernel constraint closes the artifact on games with
    action-dependent transitions. Plan warranted for `TransitionKernelCce` as a
    CCE LP variant behind a `transition_kernel` feature flag. The RPS case needs
    the richer-deviation-class fix (option 1) — a separate PoC.

## What this PoC does NOT test

- **RPS artifact closure.** The transition-kernel constraint reduces to the
  marginal constraint (Issue 573 T4a) for state-independent transitions. RPS
  needs the richer-deviation-class fix (option 1), which is a separate PoC.
- **Multi-player MFG CCE.** This PoC uses a single-agent MDP (the CCE is trivially
  satisfied). The multi-player case (where the CCE is non-trivial + the balance
  constraint couples multiple players' occupation measures) is a plan-stage concern.
- **Crowd-scale latency.** The constraint adds O(N) rows to the LP; the BFS
  solver may need a real simplex for larger state spaces. Plan-stage concern.

## Reference

- Research 468 §7 (Issue 573 PoC Addendum — what would close the artifact)
- Research 468 §2.6 (honest uncertainty: Beckmann vs transition-kernel vs MFG continuity)
- `.docs/04_calibration/cce_moderator.md` §Limitations #2 (the documented gap)
- [Issue 573](573_beckmann_divergence_constraint_cce_poc.md) — Beckmann PoC, T4 FAIL
- `crates/katgpt-core/src/cce/lp.rs` — shipped `CceLp::solve` (no transition kernel)
