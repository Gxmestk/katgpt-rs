# Issue 585 — MOP defend-wrong PoC (Super-GOAT obligation, Research 478)

> **Source:** [Research 478](../.research/478_MOP_Maximum_Occupancy_Principle.md) — Maximum Occupancy Principle (Ramírez-Ruiz et al. 2024 Nature Comms, [s41467-024-49711-1](https://www.nature.com/articles/s41467-024-49711-1)).
> **Type:** POC / proof / optimization task
> **Filed:** 2026-08-14
> **Status:** Open
> **Priority:** High — gates §3.3 Super-GOAT mandatory outputs #3 (open primitive plan) + #4 (private runtime guide)
> **Scope:** This issue tracks the **defend-wrong PoC obligation** from research skill §3.6. The Super-GOAT verdict in Research 478 is architectural-only on the quality axis until this PoC runs.

## Context

Research 478 declared the Maximum Occupancy Principle (MOP) paper as **Super-GOAT** (4/4 novelty gates PASS). Per research skill §3.6, any quality-parity verdict ("MOP produces paper-Fig-2-class emergent behavior in our civ/game domain") requires a head-to-head PoC on a controlled toy — architectural reasoning is NOT sufficient.

The math is correct (paper Theorem 3 + Supplement §C proves convergence of the value-iteration map). What is **unproven** is whether MOP, when wired into our game-zone-level MDP (zone KG states + civ action enum), produces qualitatively the same emergent behavior the paper shows (survival instinct, dancing, hide-and-seek, behavioral variability post-convergence).

## PoC design (three arms, head-to-head)

**Where it lives:** `riir-ai/crates/riir-poc/src/mop_poc.rs` (the existing defend-wrong R&D crate). Build with `CARGO_TARGET_DIR=/tmp/mop_poc` and clean up after.

**Arena:** civ 4-room gridworld from paper Fig. 2a (we have the analogous civ arena already shipped via `riir-games-civ`). Plus the prey-predator arena from paper Fig. 3 for the hide-and-seek emergence gate.

| Arm | What | Tests |
|---|---|---|
| **A — Random walk baseline** | Uniform-random policy over `A(s)` (paper's RW agent) | "Dies before exploring" reference |
| **B — Shipped runtime analog** | CGSP with `r_synth = (1 - solve_rate)·guide_score` (Plan 274's reward, the current stack SOTA) | "Lingers at one food source" reference (paper Fig. 2b middle panel) |
| **C — MOP (this PoC)** | Local `MopSolver` impl over the civ 4-room transition kernel + prey-predator transition kernel | The distilled paper mechanism |

## Gates (must defend on all four)

- [ ] **G1 — Survival instinct (no reward function):** Arm C average lifetime ≥ 0.5 × Arm B lifetime, *without any extrinsic reward function*. The paper proves absorbing states have `V^π(s+) = 0` regardless of policy (Eq. 3) — Arm C should avoid death *emergently*.
- [ ] **G2 — Physical-space coverage:** Arm C visits ≥ 80% of gridworld locations in 5×10⁴ steps; ≥ 20pp above Arm B. Paper Fig. 2d: MOP ~100%, R-agent ~30-50%, RW dies early.
- [ ] **G3 — Behavioral variability post-convergence:** After value iteration converges, average `H(π*(·|s)) ≥ 0.5·ln(|A(s)|)` for Arm C; Arm B collapses to ≤ 0.1·ln(|A(s)|). Paper §Discussion: MOP's optimal policy is non-deterministic by construction.
- [ ] **G4 — Hide-and-seek emergence:** In prey-predator arena, Arm C's clockwise-rotation ratio ∈ [0.4, 0.6] (paper Fig. 3c, MOP does both directions); Arm B collapses to ≥ 0.85 (paper Fig. 3c, R-agent prefers one direction).

## Verdict rules (from Research 478 §8)

- **All 4 PASS** → Super-GOAT confirmed. Mandatory outputs ship: open primitive plan + private runtime guide + consumer plan.
- **3/4 PASS, 1 refuted on quality** → Super-GOAT stays on confirmed axes; the refuted axis becomes a tracked follow-up issue. Most likely refutation: G3 (if CGSP's existing bandit already enforces stochasticity, the delta over the shipped baseline is small — that refutes the *delta*, not the *mechanism*).
- **2/4 PASS, 2 refuted** → downgrade to **GOAT**. The open primitive still ships (it's correct math) but the runtime wiring story weakens. The plan narrows to "ship the math primitive + bench; defer runtime wiring".
- **≤1/4 PASS** → revise the verdict in Research 478. Record raw numbers honestly; do NOT silently revise the note.

## Substrate inventory (what to consume, NOT duplicate)

| Substrate | Where | Role in PoC |
|---|---|---|
| `cgsp::traits::{CuriosityConjecturer, QualityGuide, Solver, HintDeltaBandit}` | `katgpt-core/src/cgsp/` | Arm B baseline loop |
| `OccupationMeasure<N, A>` | `katgpt-core/src/cce/types.rs` | Type for storing policies (NOT the MOP itself — different math) |
| `entropy_nats` | `katgpt-core/src/cgsp/types.rs` | Entropy computation for `H(A|s)` and `H(S'|s,a)` |
| `log_sum_exp` patterns | Throughout katgpt-core | The partition function `Z(s)` in MOP Eq. 6 |

## Implementation outline (~300 LOC)

```rust
// riir-ai/crates/riir-poc/src/mop_poc.rs

struct FourRoomArena { /* paper Fig. 2a layout */ }
impl FourRoomArena {
    fn transition_kernel(&self) -> ([N, A, N] f32) { /* p(s'|s,a) */ }
    fn action_mask(&self) -> ([N, A] u8) { /* w(s,a) availability */ }
    fn absorbing_mask(&self) -> ([N] bool) { /* s+ absorbing states */ }
}

struct MopSolver<const N: usize, const A: usize> {
    p: [[N, A, N] f32],
    w: [[N, A] u8],
    alpha: f32, beta: f32, gamma: f32,
    z: [f32; N],  // current iterate (Eq. 7)
}

impl MopSolver {
    fn iterate(&mut self) -> f32 { /* one Eq. 7 step; returns sup-norm delta */ }
    fn solve(&mut self, tol: f32, max_iter: u32) { /* loop until converged */ }
    fn v_star(&self) -> [f32; N] { /* V*(s_i) = α/γ · ln z_i */ }
    fn pi_star(&self, s: usize) -> [f32; A] { /* π*(a|s) from Eq. 6 */ }
}

// Three arms:
fn arm_a_random_walk(arena) -> Vec<usize> { /* sample uniformly from A(s) */ }
fn arm_b_cgsp_baseline(arena) -> Vec<usize> { /* run CgspLoop with paper reward */ }
fn arm_c_mop(arena) -> Vec<usize> { /* MopSolver::solve + sample from π* */ }

#[test]
fn g1_mop_survives_without_reward() { /* gate G1 */ }
#[test]
fn g2_mop_covers_arena() { /* gate G2 */ }
#[test]
fn g3_mop_policy_stays_stochastic() { /* gate G3 */ }
#[test]
fn g4_mop_hide_and_seek_bidirectional() { /* gate G4 */ }
```

## Notes

- The civ 4-room arena already ships (it's the canonical CGSP testbed). Reuse, don't duplicate.
- The prey-predator arena needs to be constructed for this PoC — ~50 LOC, mirrors paper Fig. 3a.
- `CARGO_TARGET_DIR=/tmp/mop_poc` to avoid contention with sibling agents.
- The PoC stays in `riir-poc/` as a permanent regression check regardless of verdict (per research skill §3.6 "PoC defends OR refutes... PoC stays as permanent regression check").

## Related

- [Research 478 — MOP Super-GOAT verdict](../.research/478_MOP_Maximum_Occupancy_Principle.md)
- [Research 240 — CGSP (closest curiosity cousin)](../.research/240_SGS_Curiosity_Guided_Self_Play.md)
- [Research 423 — FORE (closest occupancy cousin)](../.research/423_*)  <!-- TODO: confirm exact filename -->
- [Plan 274 — CGSP GOAT gate](../.plans/274_curiosity_guided_self_play.md)
- [Plan 438 — FORE primitive](../.plans/438_*)  <!-- TODO: confirm exact filename -->
- [riir-ai Research 041 — Curiosity Pulse (closest immediate-uncertainty cousin)](../../riir-ai/.research/041_Curiosity_Pulse_Entropy_Driven_Information_Gathering.md)
