# Issue 585 — MOP defend-wrong PoC (Super-GOAT obligation, Research 478)

> **Source:** [Research 478](../.research/478_MOP_Maximum_Occupancy_Principle.md) — Maximum Occupancy Principle (Ramírez-Ruiz et al. 2024 Nature Comms, [s41467-024-49711-1](https://www.nature.com/articles/s41467-024-49711-1)).
> **Type:** POC / proof / optimization task
> **Filed:** 2026-08-14
> **Status:** PoC RUN 2026-08-14 — **3/4 gates PASS, G4 refuted (marginal)** → Super-GOAT stays on confirmed axes; G4 axis tracked as [Issue 653](653_mop_g4_bidirectionality_followup.md). Full record: [riir-ai Bench 675](../../riir-ai/.benchmarks/675_mop_defend_wrong_poc.md). Stays OPEN — still gates §3.3 outputs #3 (open primitive plan) + #4 (private runtime guide), both now unblocked by the PoC verdict.
> **Priority:** High — gates §3.3 Super-GOAT mandatory outputs #3 (open primitive plan) + #4 (private runtime guide)
> **Scope:** This issue tracks the **defend-wrong PoC obligation** from research skill §3.6. The Super-GOAT verdict in Research 478 is architectural-only on the quality axis until this PoC runs. **→ PoC obligation DISCHARGED (Bench 675); the remaining scope is the plan/guide outputs.**

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

- [x] **G1 — Survival instinct (no reward function): PASS.** Arm C = 10 000/10 000 mean/median lifetime (all 32 seeds hit the 10k cap), identical to reward-driven Arm B; RW reference 66 median. Note: the C ≥ 0.5×B clause is near-unfalsifiable once both arms cap — the load-bearing contrast is C vs RW. **Honest finding:** RW median 66 (not the paper's <50) — our trap placement is less path-critical than Fig 2a; recorded as arena-fidelity note in Bench 675, not a gate revision.
- [x] **G2 — Physical-space coverage: PASS.** Arm C = **1.0000** (8 seeds × 50k steps, min=max=1.0) vs Arm B = 0.1922 (orbits one food source — the paper's R-agent "lingers" behavior) vs RW = 0.2705. Both clauses clear with large margin (≥80% ✓, Δ=80.8pp ≥ 20pp ✓).
- [x] **G3 — Behavioral variability post-convergence: PASS.** Arm C avg H(π\*) = **1.0280** ≥ 0.5·ln(3.104)=0.5664; Arm B = 0.0445 ≤ 0.1133. MOP's optimal policy stays stochastic by construction, exactly as the paper claims.
- [ ] **G4 — Hide-and-seek emergence: REFUTED (marginal, 1.9pp).** Arm C pooled CW ratio = **0.6117** ∉ [0.4, 0.6] at paper defaults (α=1, β=0, γ=0.95); Arm B collapsed to 0.94 as predicted. Per-seed strongly bimodal (mean 0.556, sd 0.270, min 0.067/max 0.903) — episodes are directional but the policy itself does not collapse. γ-diagnostic monotone: in-band at γ ≤ 0.90 (0.4006 / 0.4489), out-of-band at γ ≥ 0.95. Part of the tilt is attributable to the spec's fixed CW antipode tie-break. **Tracked as [Issue 653](653_mop_g4_bidirectionality_followup.md).**

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

## PoC result (2026-08-14, Bench 675)

**Implemented:** `riir-ai/crates/riir-poc/src/mop_poc.rs` (~1500 LOC) — `FourRoomArena` (N=82, A=4), `PreyPredatorRing` (N=17, A=3), `MopSolver<N,A>` (Eq. 7 in log-space LSE form, absorbing pinning, exact V(s+)=0), three arms (RW / Plan-274-reward Q-learning / MOP π* sampling), 7 unit tests + 4 gate tests + γ-diagnostic. Clippy clean; 231/231 lib tests pass (G4 follows the honest-negative-result pattern — asserts the correctness invariants that hold, prints the recorded FAIL verdict, omits the bidirectionality assertion).

**Verdict per the rules above: 3/4 PASS, 1 refuted on quality → Super-GOAT stays on the confirmed axes.** The math primitive is correct (all solver invariants pass: V(s+)=0 exactly, π* normalized, Theorem-3 init-invariance at 0.0 max diff, convergence in 290/86 iters) and two of the three headline behaviors (coverage 1.0 vs 0.19; post-convergence entropy 1.03 vs 0.04) reproduce decisively. The refuted axis (G4 bidirectionality) is γ-regime-sensitive and partly arena-tie-break-attributable — see Issue 653.

**Deviations from this issue's spec (recorded in Bench 675):**
1. Arm B consumes the Plan 274 **reward formula** per-cell rather than the `cgsp` direction-pool trait wiring — the trait suite is puzzle-domain-shaped, not MDP-policy-shaped; forcing it would be a category error, not substrate consumption.
2. No existing four-room substrate was found (the issue's "already ships" claim was aspirational — grep `FourRoom|GridWorld|Rooms` across all repos returned only riir-train's unrelated ReMax envs); the arena is built inline per the issue's own outline.
3. Door gaps placed adjacent to center so the mandated (4,4) start is walkable.
4. π\* normalization uses the explicit log-sum-exp (the fixed-point normalizer is `z^{1/γ}`, not the `z^{-1}` in Research 478 §2.1's pseudocode line — corrected in the implementation, noted in the file header).

**Remaining scope (this issue stays open):** §3.3 outputs #3 (`.plans/` open primitive plan — now carries the 3/4 verdict + the G4 caveat) and #4 (riir-ai private runtime guide). Per Research 478 next-steps: guide before plan.

## Related

- [Research 478 — MOP Super-GOAT verdict](../.research/478_MOP_Maximum_Occupancy_Principle.md)
- [Research 240 — CGSP (closest curiosity cousin)](../.research/240_SGS_Curiosity_Guided_Self_Play.md)
- [Research 423 — FORE (closest occupancy cousin)](../.research/423_*)  <!-- TODO: confirm exact filename -->
- [Plan 274 — CGSP GOAT gate](../.plans/274_curiosity_guided_self_play.md)
- [Plan 438 — FORE primitive](../.plans/438_*)  <!-- TODO: confirm exact filename -->
- [riir-ai Research 041 — Curiosity Pulse (closest immediate-uncertainty cousin)](../../riir-ai/.research/041_Curiosity_Pulse_Entropy_Driven_Information_Gathering.md)
