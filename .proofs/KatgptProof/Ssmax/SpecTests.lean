/-
! Spec self-tests for `alphaGold` on concrete dilution-curve instances.

Distilled from SymCrypt `feature/verifiedcrypto` §4 ("Running the Lean spec on
test vectors") — see `.research/425_*.md` and `.plans/441_*.md`.

The `alphaGold N c = 1 / (1 + (N − 1) · N^(−c))` formula in `Basic.lean` is
transcribed from arXiv:2607.01538 §2. A sign typo (`N^c` instead of `N^(-c)`)
would propagate silently:

- The monotonicity theorems in `DilutionBound.lean` would still type-check
  (they prove monotonicity of whatever formula is transcribed).
- The Rust spec-match test (`ssmax_spec_match.rs`) would still pass (the Rust
  doc comment in `ssmax.rs` likely has the same typo, written by the same
  author referencing the same paper).

The `alphaGold 2 1 = 2/3` instance catches this because it requires
`2^(-1) = 1/2` (correct sign); with the wrong sign `N^c`, the value becomes
`1 / (1 + 1 · 2^1) = 1/3 ≠ 2/3`.
-/

import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.Complex.Exponential
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import KatgptProof.Ssmax.Basic

namespace KatgptProof.Ssmax

open Real

/-! ## Helper: concrete `rpow` values

For `c = 0`: `N^(-0) = N^0 = 1` (since `−0 = 0` for `ℝ`).
For `c = 1`: `N^(-1) = 1/N` (via `exp(−log N) = (exp(log N))⁻¹ = N⁻¹`).
-/

private lemma rpow_neg_one_eq_inv {N : ℝ} (hN : 0 < N) :
    N ^ (-(1 : ℝ)) = N⁻¹ := by
  rw [Real.rpow_def_of_pos hN]
  rw [show Real.log N * (-(1:ℝ)) = -(Real.log N) by ring]
  rw [Real.exp_neg, Real.exp_log hN]

/-! ## Concrete dilution-curve values

These check that the `alphaGold` definition matches the paper's published
dilution curves at representative `(N, c)` points. All values are
independently computable from the formula `1 / (1 + (N−1) · N^(−c))` by hand.
-/

/-- Trivial bound: with no sharpening (c=0), gold mass is uniformly 1/N.
    For N=2: `alphaGold = 1 / (1 + 1 · 2^0) = 1/2`. -/
example : alphaGold 2 0 = (1/2 : ℝ) := by
  have h : (2 : ℝ) ^ (-(0 : ℝ)) = 1 := by
    have h0 : (-(0:ℝ)) = 0 := neg_zero
    rw [h0, Real.rpow_zero]
  rw [alphaGold, h]; ring

/-- The sign-check instance: `alphaGold 2 1 = 1 / (1 + 1 · 2^(-1)) = 1 / (3/2) = 2/3`.
    Requires `2^(-1) = 1/2` — the correct sign in `N^(-c)`. A `N^c` typo yields
    `1 / (1 + 2) = 1/3` and this example fails. -/
example : alphaGold 2 1 = (2/3 : ℝ) := by
  have h : (2 : ℝ) ^ (-(1 : ℝ)) = (1/2 : ℝ) := by
    rw [rpow_neg_one_eq_inv (by norm_num : (0:ℝ) < 2)]
    norm_num
  rw [alphaGold, h]; ring

/-- Dilution at scale: with no sharpening, N=10 gives 1/10. -/
example : alphaGold 10 0 = (1/10 : ℝ) := by
  have h : (10 : ℝ) ^ (-(0 : ℝ)) = 1 := by
    have h0 : (-(0:ℝ)) = 0 := neg_zero
    rw [h0, Real.rpow_zero]
  rw [alphaGold, h]; ring

/-- The paper's motivating regime: N=10, c=1.
    `alphaGold = 1 / (1 + 9 · 10^(-1)) = 1 / (1 + 9/10) = 1 / (19/10) = 10/19`. -/
example : alphaGold 10 1 = (10/19 : ℝ) := by
  have h : (10 : ℝ) ^ (-(1 : ℝ)) = (1/10 : ℝ) := by
    rw [rpow_neg_one_eq_inv (by norm_num : (0:ℝ) < 10)]
    norm_num
  rw [alphaGold, h]; ring

/-- Large-corpus dilution collapse: N=100, c=0 gives 1/100. -/
example : alphaGold 100 0 = (1/100 : ℝ) := by
  have h : (100 : ℝ) ^ (-(0 : ℝ)) = 1 := by
    have h0 : (-(0:ℝ)) = 0 := neg_zero
    rw [h0, Real.rpow_zero]
  rw [alphaGold, h]; ring

end KatgptProof.Ssmax
