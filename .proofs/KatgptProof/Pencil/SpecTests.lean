/-
! Spec self-tests for the spectral pencil package (Issue 678 P2).

Concrete instances with independently-known answers, closing the gap that
neither the Lean proofs (spec vs itself) nor the Rust spec-match tests
(Rust vs spec) cover: a spec TRANSCRIPTION error.

* `sym` packing: a hand-computed 2×2 Frobenius identity and inner-product
  identity. Storage `!![1, 2√2; 2√2, 3]` carries `!![1, 2; 2, 3]`;
  `‖A‖_F² = 18 = 1² + 3² + (2√2)²` by hand. A packing typo (`1/2` scale
  instead of `1/√2`) makes the off-diagonal contribute `2·(2√2/2)² = 4`
  and the identity fails — these tests catch it.

* T2 Weyl: `A = diag(1,−1)`, `B = diag(1,0)` — the eigenvalue displacement
  `|λ₁(A) − λ₁(B)| = 1` and `‖A−B‖₂ ≤ 1`, so the Lipschitz bound is exactly
  attained (tight, not slack).

* T3 Loewner: `B − A = diag(1,0) ⪰ 0` forces `λ₀(diag(0,0)) = 0 ≤
  1 = λ₀(diag(1,0))` — monotonicity pinned on the school-level ground
  truth (a diagonal's spectrum is its diagonal).

* T4 eigengap: the Fin-2 ladder `ladderDn 0 = diag(0,−1)` has unit gap
  (`0 − (−1) = 1`, independently: the diagonal itself); the perturbed
  pencil `diag(0,−1) + diag(1/4,0) + ½·1` is still diagonal
  (`diag(3/4,−1/2)`), so its exact gap `5/4` is hand-computable and the
  theorem's `≥ 1/2` is verified against it.
-/

import KatgptProof.Pencil.Sym
import KatgptProof.Pencil.Weyl
import KatgptProof.Pencil.Loewner
import KatgptProof.Pencil.Eigengap

namespace KatgptProof.Pencil

open Matrix
open scoped Matrix.Norms.L2Operator

noncomputable section

/-- The storage square for `A = !![1,2;2,3]`: offs pre-multiplied by √2. -/
private def v22 : Matrix (Fin 2) (Fin 2) ℝ :=
  !![1, 2 * Real.sqrt 2; 2 * Real.sqrt 2, 3]

private theorem v22_isSymm : v22.IsSymm := by
  show v22ᵀ = v22
  apply Matrix.ext
  intro i j
  simp only [Matrix.transpose_apply, v22]
  match i, j with
  | 0, 0 => rfl
  | 0, 1 => rfl
  | 1, 0 => rfl
  | 1, 1 => rfl

/-- Frobenius identity on the hand instance (squared form): both sides
are 18 by direct computation. -/
example : frobSq (symMat v22) = 18 := by
  have h : frobSq (symMat v22) = packedNormSq v22 := sym_isometry_norm_sq v22_isSymm
  rw [h]
  unfold packedNormSq
  rw [Fin.sum_univ_two]
  have hfilt : (Finset.univ.filter (fun p : Fin 2 × Fin 2 => p.1 < p.2))
      = ({(0, 1)} : Finset (Fin 2 × Fin 2)) := by decide
  rw [hfilt, Finset.sum_singleton]
  have hsq : Real.sqrt 2 * Real.sqrt 2 = 2 := by
    have := Real.sq_sqrt (by norm_num : (0:ℝ) ≤ 2)
    nlinarith [this]
  simp only [Fin.isValue, v22, Matrix.cons_val_zero, Matrix.cons_val_one,
    Matrix.head_cons, Matrix.head_cons, dotProduct, Pi.add_apply, Pi.mul_apply]
  norm_num
  nlinarith [hsq]

/-- Inner-product identity on the hand instance (u = v). -/
example : frobDot (symMat v22) (symMat v22) = packedDot v22 v22 :=
  sym_isometry_dot v22_isSymm v22_isSymm

/-! ## Concrete eigenvalue pins (T2–T4)

All instances are Fin 2 so that every literal evaluates by `simp` +
`norm_num`. The independently-known ground truth throughout: a diagonal
matrix's spectrum is its diagonal (`eigval_diagonal_antitone` pins the
antitone-sorted array to the decreasing diagonal entries). -/

/-- Antitone for a two-point decreasing function. -/
private theorem antitone_pair2 {f : Fin 2 → ℝ} (h : f 1 ≤ f 0) : Antitone f := by
  intro a b hab
  have hb : (a : ℕ) ≤ (b : ℕ) := Fin.le_def.1 hab
  fin_cases a
  · fin_cases b
    · exact le_refl _
    · exact h
  · fin_cases b
    · exfalso
      exact absurd hab (by decide)
    · exact le_refl _

private theorem antitone_litA : Antitone (![1, -1] : Fin 2 → ℝ) := by
  refine antitone_pair2 ?_
  norm_num

private theorem antitone_litB : Antitone (![1, 0] : Fin 2 → ℝ) := by
  refine antitone_pair2 ?_
  norm_num

private theorem antitone_litC : Antitone (![3 / 4, -1 / 2] : Fin 2 → ℝ) := by
  refine antitone_pair2 ?_
  norm_num

/-- **T2 pin**: the eigenvalues of the hand pair are the diagonals. -/
example : eigval (diagonal_isHermitian (![1, -1] : Fin 2 → ℝ)) 1 = -1 :=
  eigval_diagonal_antitone antitone_litA 1

example : eigval (diagonal_isHermitian (![1, 0] : Fin 2 → ℝ)) 1 = 0 :=
  eigval_diagonal_antitone antitone_litB 1

/-- **T2 tightness**: `|λ₁(A) − λ₁(B)| = 1` and `‖A−B‖₂ ≤ 1` — the Weyl
Lipschitz bound is exactly attained on this instance. -/
example : |eigval (diagonal_isHermitian (![1, -1] : Fin 2 → ℝ)) 1
    - eigval (diagonal_isHermitian (![1, 0] : Fin 2 → ℝ)) 1| = 1 := by
  rw [eigval_diagonal_antitone antitone_litA 1,
    eigval_diagonal_antitone antitone_litB 1]
  norm_num

example : |eigval (diagonal_isHermitian (![1, -1] : Fin 2 → ℝ)) 1
    - eigval (diagonal_isHermitian (![1, 0] : Fin 2 → ℝ)) 1|
    ≤ ‖(Matrix.diagonal (![1, -1] : Fin 2 → ℝ))
      - (Matrix.diagonal (![1, 0] : Fin 2 → ℝ))‖ :=
  weyl_lipschitz (diagonal_isHermitian _) (diagonal_isHermitian _) 1

example : ‖(Matrix.diagonal (![1, -1] : Fin 2 → ℝ))
    - (Matrix.diagonal (![1, 0] : Fin 2 → ℝ))‖ ≤ 1 := by
  rw [Matrix.diagonal_sub]
  exact opNorm_diagonal_le _ 1 (by norm_num) (by
    intro j
    fin_cases j <;> simp <;> norm_num)

/-- **T3 instance**: `B − A = diag(1,0) ⪰ 0` gives `λ₀(A) ≤ λ₀(B)`;
both endpoints pinned by hand (`0 ≤ 1`). -/
example : eigval (diagonal_isHermitian (![0, 0] : Fin 2 → ℝ)) 0
    ≤ eigval (diagonal_isHermitian (![1, 0] : Fin 2 → ℝ)) 0 := by
  have hBA : (Matrix.diagonal (![1, 0] : Fin 2 → ℝ)
      - Matrix.diagonal (![0, 0] : Fin 2 → ℝ)).PosSemidef := by
    have hEq : Matrix.diagonal (![1, 0] : Fin 2 → ℝ)
        - Matrix.diagonal (![0, 0] : Fin 2 → ℝ)
        = Matrix.diagonal (![1, 0] : Fin 2 → ℝ) := by
      rw [Matrix.diagonal_sub]
      congr 1
      funext j
      fin_cases j <;> simp
    rw [hEq]
    exact Matrix.PosSemidef.diagonal (Pi.le_def.2 fun j => by
      fin_cases j <;> simp)
  exact loewner_mono (diagonal_isHermitian _) (diagonal_isHermitian _) hBA 0

/-- **T4 pin (independent route)**: the Fin-2 ladder is `![0, −1]` as a
function, so the school-level ground truth gives the pins directly. -/
example : ladderDn (0 : Fin 2) = (![0, -1] : Fin 2 → ℝ) := by
  funext j
  fin_cases j <;> simp [ladderDn]

example : eigval (ladderDn_isHermitian (0 : Fin 2)) 0
    - eigval (ladderDn_isHermitian (0 : Fin 2)) 1 = 1 :=
  ladder_unit_gap (0 : Fin 2) (by norm_num)

/-- **T4 exact perturbed gap**: the perturbed Fin-2 ladder pencil equals
`diag(3/4, −1/2)` (still diagonal, still antitone), so its exact gap is
`3/4 − (−1/2) = 5/4` — hand-computable, and `≥ 1/2` as the theorem
claims. -/
example : (Matrix.diagonal (ladderDn (0 : Fin 2))
    + Matrix.diagonal (![1 / 4, 0] : Fin 2 → ℝ)
    + (1 / 2 : ℝ) • (1 : Matrix (Fin 2) (Fin 2) ℝ))
    = Matrix.diagonal (![3 / 4, -1 / 2] : Fin 2 → ℝ) := by
  have hs1 : ((1 / 2 : ℝ) • (1 : Matrix (Fin 2) (Fin 2) ℝ))
      = Matrix.diagonal (![1 / 2, 1 / 2] : Fin 2 → ℝ) := by
    apply Matrix.ext
    intro a b
    fin_cases a <;> fin_cases b <;> simp
  rw [Matrix.diagonal_add, hs1, Matrix.diagonal_add]
  congr 1
  funext j
  fin_cases j <;> simp [ladderDn] <;> norm_num

example : eigval (diagonal_isHermitian (![3 / 4, -1 / 2] : Fin 2 → ℝ)) 0
    - eigval (diagonal_isHermitian (![3 / 4, -1 / 2] : Fin 2 → ℝ)) 1
    = 5 / 4 := by
  rw [eigval_diagonal_antitone antitone_litC 0,
    eigval_diagonal_antitone antitone_litC 1]
  norm_num

/-- **T4 bound**: the same instance satisfies the theorem's `≥ 1/2` (the
exact gap above is `5/4`, comfortably above the bound). -/
example : eigval (ladder_perturbed_isHermitian (0 : Fin 2)
    (diagonal_isHermitian (![1 / 4, 0] : Fin 2 → ℝ)) (1 / 2 : ℝ)) 0
    - eigval (ladder_perturbed_isHermitian (0 : Fin 2)
    (diagonal_isHermitian (![1 / 4, 0] : Fin 2 → ℝ)) (1 / 2 : ℝ)) 1
    ≥ 1 / 2 :=
  eigengap_ladder_ge_half (1 / 2 : ℝ) (0 : Fin 2) (by norm_num)
    (diagonal_isHermitian (![1 / 4, 0] : Fin 2 → ℝ))
    (opNorm_diagonal_le _ (1 / 4) (by norm_num) (by
      intro j
      fin_cases j <;> simp <;> norm_num))

end

end KatgptProof.Pencil
