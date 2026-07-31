/-
! Spec self-tests on concrete instances — the "spec tested on vectors" pattern.

Distilled from SymCrypt `feature/verifiedcrypto` §4 ("Running the Lean spec on
test vectors") — see `.research/425_*.md` and `.plans/441_*.md`. SymCrypt runs
each Lean `Spec/` module on standard test vectors (CAVP/ACVP) to catch spec
transcription errors. We do the same here for our self-authored specs: the
`dot` and `sigmoid` definitions are tested on known-good values so a typo in
`Basic.lean` is caught at `lake build` time, not by a downstream consumer.

Why this matters: the ranking-preservation theorems in `RankingPreserved.lean`
prove the spec against itself. The Rust spec-match test
(`bridge_spec_match.rs`) tests Rust against the spec. **Neither** catches a
spec authoring error — if the Lean `dot` definition has a sign typo, the proof
still type-checks (proving the wrong property) and the Rust test still passes
(Rust likely has the same typo, written by the same author). Only concrete
instances with independently-known answers close this gap.
-/

import Mathlib.Analysis.SpecialFunctions.Sigmoid
import KatgptProof.Bridge.Basic

namespace KatgptProof.Bridge

open Real

/-! ## Dot product on concrete vectors

The mathematical dot product is universally known. These instances check that
our Lean `dot` definition computes the expected values — catching sign errors,
accumulation-order bugs, or index-range mistakes in the `∑ i, q i * d i`
definition.
-/

/-- Self-dot-product of a unit axis vector is 1. -/
example : dot (![1, 0] : Fin 2 → ℝ) (![1, 0]) = 1 := by
  simp [dot]

/-- Orthogonal vectors have zero dot product. -/
example : dot (![1, 1] : Fin 2 → ℝ) (![1, -1]) = 0 := by
  simp [dot]

/-- Non-trivial 2D dot product: (2,3)·(4,5) = 8+15 = 23. `simp` unfolds
    the sum and evaluates `Matrix.cons_val`; `norm_num` closes the arithmetic. -/
example : dot (![2, 3] : Fin 2 → ℝ) (![4, 5]) = 23 := by
  simp [dot]; norm_num

/-! ## Sigmoid on concrete inputs

`Real.sigmoid 0 = 1/2` is the defining property of the sigmoid origin. If a
future refactor swaps `sigmoid` for a different activation (e.g. tanh, which
gives 0 at the origin), these tests break — signaling that the ranking-
preservation theorem may need re-proof.
-/

/-- Sigmoid at the origin is exactly 1/2. Mathlib states this as
    `sigmoid 0 = 2⁻¹`; `norm_num` closes `2⁻¹ = 1/2`. -/
example : Real.sigmoid 0 = (1/2 : ℝ) := by
  rw [Real.sigmoid_zero]; norm_num

/-- Sigmoid maps positive inputs to (1/2, 1). Uses strict monotonicity:
    `0 < 1 ⟹ sigmoid 0 < sigmoid 1`, then `sigmoid 0 = 1/2`. -/
example : (1/2 : ℝ) < Real.sigmoid 1 := by
  have h : Real.sigmoid 0 < Real.sigmoid 1 :=
    Real.sigmoid_lt (by norm_num : (0:ℝ) < 1)
  rw [Real.sigmoid_zero] at h
  linarith

/-- Sigmoid maps negative inputs to (0, 1/2). Same approach: `-1 < 0` and
    `sigmoid 0 = 1/2`. -/
example : Real.sigmoid (-1) < (1/2 : ℝ) := by
  have h : Real.sigmoid (-1 : ℝ) < Real.sigmoid 0 :=
    Real.sigmoid_lt (by norm_num : (-1 : ℝ) < 0)
  rw [Real.sigmoid_zero] at h
  linarith

end KatgptProof.Bridge
