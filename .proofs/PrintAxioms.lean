/-
! Axiom inventory printer.

Run with: `lake env lean PrintAxioms.lean` (or, as an assertion rather than a
printout, `scripts/proof_gate.sh`).

Prints the axiom dependencies of every shipped theorem. The axiom budget is
`{propext, Classical.choice, Quot.sound}` — the standard Lean foundation, and
the same one Mathlib itself stands on. No `sorry`, no `sorryAx`, no
`Lean.ofReduceBool` (which `native_decide` would introduce, trading the kernel
for the compiler). If this printer ever shows a theorem depending on anything
outside that budget, the proof is invalid.

`KatgptProof` depends on Mathlib (`Real.exp` analysis for sigmoid strict
monotonicity, `Filter.Tendsto` for the asymptotics), so `Classical.choice` and
`Quot.sound` are expected here — unlike the Mathlib-free
`riir-neuron-db/.proofs`, where most theorems come out axiom-free.

Uses `open` + unqualified names because fully-qualified `#print axioms` through
the root import does not reliably resolve transitively-imported theorem names
in Lean 4 (a known quirk); the unqualified form via `open <Namespace>` works.

Adding a theorem? Add its `#print axioms` line here, then bump
`EXPECTED_THEOREMS` in `scripts/proof_gate.sh`. The count is pinned there on
purpose: deriving it from this file would be self-referential, so deleting a
line would shrink both the expectation and the result together and the check
could never fire.
-/

import KatgptProof.Bridge.Basic
import KatgptProof.Bridge.RankingPreserved
import KatgptProof.Hope.Basic
import KatgptProof.Hope.SpecTests
import KatgptProof.Ssmax.Basic
import KatgptProof.Ssmax.DilutionBound
import KatgptProof.Ssmax.Asymptotic

open KatgptProof.Bridge
open KatgptProof.Hope
open KatgptProof.Ssmax

-- Action-bridge ranking preservation (the sigmoid strict-monotonicity chain).
#print axioms action_bridge_ranking_preserved
#print axioms action_bridge_ranking_preserved'
#print axioms action_bridge_argmax_preserved

-- HOPE spec instances (normal CDF at 0, ReLU self-kernel positivity).
#print axioms normalCdf_zero
#print axioms reluSelfKernel_pos_gamma_zero

-- Ssmax core: the alphaGold denominator is positive and alphaGold is bounded.
#print axioms alphaGold_denom_pos
#print axioms alphaGold_bounded
#print axioms strictMono_rpow_of_gt_one

-- Ssmax dilution bound: strict monotonicity in the context length `c`.
#print axioms alphaGold_strictMono_in_c
#print axioms alphaGold_lt_of_c_lt
#print axioms ssmax_dominates_base

-- Ssmax asymptotics: leakage → 0 and alphaGold → 1 as `c → ∞`.
#print axioms alphaGold_eq
#print axioms leakage_nonneg
#print axioms leakage_le_inv
#print axioms eventually_c_ge_two
#print axioms tendsto_leakage_zero
#print axioms tendsto_alphaGold_one
