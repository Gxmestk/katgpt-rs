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

The Weyl/Loewner/eigengap theorems' concrete-instance tests depend on
pinning the eigenvalue ARRAY of a concrete matrix (the same "concrete
eigenvalue pinning for diagonal matrices" machinery as the eigengap
ladder — deferred, tracked in Issue 678).
-/

import KatgptProof.Pencil.Sym

namespace KatgptProof.Pencil

open Matrix

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

end

end KatgptProof.Pencil
