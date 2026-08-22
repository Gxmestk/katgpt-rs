/-
! Spec for the spectral pencil `sym` isometric packing (Issue 678 T1;
! Research 495 §3.1 P2; paper arXiv:2608.08003 §7.1).

The Rust substrate (`katgpt-core/src/spectral_pencil/sym.rs`, Issue 676 T1)
represents a symmetric `D×D` matrix by its upper-triangle parameter vector,
stored as a **mirrored full square** (`data[i][j] = data[j][i]`), with
off-diagonal entries pre-multiplied by `√2`. The actual matrix applies the
`1/√2` off scale on the way out (`to_full`).

This module proves the packing is an **isometry**:

```text
‖sym(v)‖_F == ‖v‖₂          ⟨sym(u), sym(v)⟩_F == ⟨u, v⟩
```

where the right-hand sides are the compact parameter-vector norm / dot
(diagonal + upper triangle — exactly the index sets the Rust
`frobenius_norm` and `frobenius_dot` loops run). This is what makes every
attribution / genome-similarity query a plain SIMD dot on packed vectors,
and `‖A‖₂ ≤ ‖A‖_F = ‖v‖₂` free.

## Rust reference

```rust
pub fn to_full(&self) -> [[f32; D]; D] {
    // out[i][i] = data[i][i]; out[i][j] = data[i][j] * RCP_SQRT_2
}
pub fn frobenius_norm(&self) -> f32 {
    // Σ_i data[i][i]² + Σ_{i<j} data[i][j]²   (f64 accumulation)
}
```

The Lean model is the idealised `ℝ` contract (the Rust doc's "1–2 ulp"
float honesty note does not affect the mathematical statement).
-/

import Mathlib

namespace KatgptProof.Pencil

open Matrix

variable {D : ℕ}

/-! ## The packing

`symMat v` is the matrix the representation carries: diagonal entries
as stored, off-diagonals scaled by `(√2)⁻¹`. The hypothesis `v.IsSymm`
(`vᵀ = v`) is the mirror invariant of the Rust `SymPacked.data` square.
-/


/-- Entrywise mirror access for a symmetric parameter square. -/
theorem symm_apply {v : Matrix (Fin D) (Fin D) ℝ} (hv : v.IsSymm) (i j : Fin D) :
    v i j = v j i := by
  have h : v i j = vᵀ i j := by rw [hv]
  rw [h, Matrix.transpose_apply]

/-- The actual symmetric matrix carried by a packed representation:
off-diagonal storage entries are scaled by `(√2)⁻¹` (Rust `to_full`). -/
noncomputable def symMat (v : Matrix (Fin D) (Fin D) ℝ) : Matrix (Fin D) (Fin D) ℝ :=
  Matrix.of fun i j => if i = j then v i j else (Real.sqrt 2)⁻¹ * v i j

/-- Frobenius norm squared: `∑ i, ∑ j, A i j * A i j`. -/
def frobSq (A : Matrix (Fin D) (Fin D) ℝ) : ℝ := ∑ i, ∑ j, A i j * A i j

/-- Compact parameter-vector norm squared: diagonal + upper triangle —
the exact index set of the Rust `frobenius_norm` loop. -/
def packedNormSq (v : Matrix (Fin D) (Fin D) ℝ) : ℝ :=
  ∑ i, v i i * v i i
    + ∑ p ∈ Finset.univ.filter (fun p : Fin D × Fin D => p.1 < p.2),
        v p.1 p.2 * v p.1 p.2

/-- Frobenius inner product `∑ i, ∑ j, A i j * B i j`. -/
def frobDot (A B : Matrix (Fin D) (Fin D) ℝ) : ℝ := ∑ i, ∑ j, A i j * B i j

/-- Compact parameter-vector dot: diagonal + upper triangle — the Rust
`frobenius_dot` loop. -/
def packedDot (u v : Matrix (Fin D) (Fin D) ℝ) : ℝ :=
  ∑ i, u i i * v i i
    + ∑ p ∈ Finset.univ.filter (fun p : Fin D × Fin D => p.1 < p.2),
        u p.1 p.2 * v p.1 p.2

/-! ## The mirror pairing identity

For a mirrored (symmetric) parameter square, each strict-upper-triangle
entry appears twice in the off-diagonal double sum — the `(i,j)`/`(j,i)`
mirror pair contributes equally. This is the arithmetic heart of the
isometry: it is what cancels the `2` in `2 · ((√2)⁻¹)² = 1`.
-/


/-- **Mirror pairing (squares).** Off-diagonal double sum = twice the
strict upper triangle, for a symmetric parameter square. -/
theorem offdiag_sq_eq_two_upper {v : Matrix (Fin D) (Fin D) ℝ} (hv : v.IsSymm) :
    ∑ p ∈ Finset.univ.filter (fun p : Fin D × Fin D => p.1 ≠ p.2),
        v p.1 p.2 * v p.1 p.2
      = 2 * ∑ p ∈ Finset.univ.filter (fun p : Fin D × Fin D => p.1 < p.2),
          v p.1 p.2 * v p.1 p.2 := by
  classical
  have hsplit : (Finset.univ.filter (fun p : Fin D × Fin D => p.1 ≠ p.2))
      = (Finset.univ.filter (fun p : Fin D × Fin D => p.1 < p.2))
        ∪ (Finset.univ.filter (fun p : Fin D × Fin D => p.2 < p.1)) := by
    ext p
    simp only [Finset.mem_union, Finset.mem_filter, Finset.mem_univ, true_and]
    constructor
    · intro hne
      rcases lt_trichotomy p.1 p.2 with h | h | h
      · exact Or.inl h
      · exact absurd h hne
      · exact Or.inr h
    · rintro (h | h)
      · exact ne_of_lt h
      · exact ne_of_gt h
  have hdisj : Disjoint (Finset.univ.filter (fun p : Fin D × Fin D => p.1 < p.2))
      (Finset.univ.filter (fun p : Fin D × Fin D => p.2 < p.1)) := by
    rw [Finset.disjoint_iff_ne]
    rintro a ha b hb habs
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ha hb
    subst habs
    exact absurd (lt_trans ha hb) (lt_irrefl a.1)
  have hlo_eq_up :
      ∑ p ∈ (Finset.univ.filter (fun p : Fin D × Fin D => p.2 < p.1)),
          v p.1 p.2 * v p.1 p.2
        = ∑ p ∈ (Finset.univ.filter (fun p : Fin D × Fin D => p.1 < p.2)),
            v p.1 p.2 * v p.1 p.2 := by
    refine Finset.sum_nbij' (fun p => (p.2, p.1)) (fun p => (p.2, p.1)) ?_ ?_ ?_ ?_ ?_
    · intro p hp
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hp ⊢
      exact hp
    · intro p hp
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hp ⊢
      exact hp
    · intro p _; rfl
    · intro p _; rfl
    · intro p hp
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hp
      rw [symm_apply hv p.2 p.1]
  rw [hsplit, Finset.sum_union hdisj, hlo_eq_up]
  ring

/-- **Mirror pairing (bilinear form).** Off-diagonal double dot = twice the
strict-upper-triangle dot, for two symmetric parameter squares. -/
theorem offdiag_dot_eq_two_upper {u v : Matrix (Fin D) (Fin D) ℝ}
    (hu : u.IsSymm) (hv : v.IsSymm) :
    ∑ p ∈ Finset.univ.filter (fun p : Fin D × Fin D => p.1 ≠ p.2),
        u p.1 p.2 * v p.1 p.2
      = 2 * ∑ p ∈ Finset.univ.filter (fun p : Fin D × Fin D => p.1 < p.2),
          u p.1 p.2 * v p.1 p.2 := by
  classical
  have hsplit : (Finset.univ.filter (fun p : Fin D × Fin D => p.1 ≠ p.2))
      = (Finset.univ.filter (fun p : Fin D × Fin D => p.1 < p.2))
        ∪ (Finset.univ.filter (fun p : Fin D × Fin D => p.2 < p.1)) := by
    ext p
    simp only [Finset.mem_union, Finset.mem_filter, Finset.mem_univ, true_and]
    constructor
    · intro hne
      rcases lt_trichotomy p.1 p.2 with h | h | h
      · exact Or.inl h
      · exact absurd h hne
      · exact Or.inr h
    · rintro (h | h)
      · exact ne_of_lt h
      · exact ne_of_gt h
  have hdisj : Disjoint (Finset.univ.filter (fun p : Fin D × Fin D => p.1 < p.2))
      (Finset.univ.filter (fun p : Fin D × Fin D => p.2 < p.1)) := by
    rw [Finset.disjoint_iff_ne]
    rintro a ha b hb habs
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ha hb
    subst habs
    exact absurd (lt_trans ha hb) (lt_irrefl a.1)
  have hlo_eq_up :
      ∑ p ∈ (Finset.univ.filter (fun p : Fin D × Fin D => p.2 < p.1)),
          u p.1 p.2 * v p.1 p.2
        = ∑ p ∈ (Finset.univ.filter (fun p : Fin D × Fin D => p.1 < p.2)),
            u p.1 p.2 * v p.1 p.2 := by
    refine Finset.sum_nbij' (fun p => (p.2, p.1)) (fun p => (p.2, p.1)) ?_ ?_ ?_ ?_ ?_
    · intro p hp
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hp ⊢
      exact hp
    · intro p hp
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hp ⊢
      exact hp
    · intro p _; rfl
    · intro p _; rfl
    · intro p hp
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hp
      rw [symm_apply hu p.2 p.1, symm_apply hv p.2 p.1]
  rw [hsplit, Finset.sum_union hdisj, hlo_eq_up]
  ring

/-! ## The isometry theorems (T1)
-/


/-- `((√2)⁻¹)² = 1/2` — the constant that cancels the mirror factor 2. -/
theorem inv_sqrt_two_sq : ((Real.sqrt 2 : ℝ)⁻¹) ^ 2 = 1 / 2 := by
  have h2 : (Real.sqrt 2 : ℝ) ^ 2 = 2 := by simp
  rw [inv_pow, h2]
  norm_num

/-- The off-diagonal entries of `symMat v` are `(√2)⁻¹` times the stored
entries. -/
theorem symMat_off {v : Matrix (Fin D) (Fin D) ℝ} {i j : Fin D} (h : i ≠ j) :
    symMat v i j = (Real.sqrt 2 : ℝ)⁻¹ * v i j := by
  simp only [symMat, Matrix.of_apply, if_neg h]

/-- **T1a — sym isometry (norm).** The Frobenius norm squared of the
carried matrix equals the compact parameter-vector norm squared. -/
theorem sym_isometry_norm_sq {v : Matrix (Fin D) (Fin D) ℝ} (hv : v.IsSymm) :
    frobSq (symMat v) = packedNormSq v := by
  classical
  -- double sum → sum over pairs, split by the diagonal predicate
  have h1 : frobSq (symMat v)
      = ∑ p ∈ (Finset.univ.filter (fun p : Fin D × Fin D => p.1 = p.2)),
          symMat v p.1 p.2 * symMat v p.1 p.2
        + ∑ p ∈ (Finset.univ.filter (fun p : Fin D × Fin D => ¬ (p.1 = p.2))),
            symMat v p.1 p.2 * symMat v p.1 p.2 := by
    rw [frobSq, ← Fintype.sum_prod_type
      (fun p : Fin D × Fin D => symMat v p.1 p.2 * symMat v p.1 p.2),
      ← Finset.sum_filter_add_sum_filter_not
        Finset.univ (fun p : Fin D × Fin D => p.1 = p.2)
        (fun p => symMat v p.1 p.2 * symMat v p.1 p.2)]
  -- diagonal part: reindex `(i,i) ↦ i`
  have hdiag : ∑ p ∈ (Finset.univ.filter (fun p : Fin D × Fin D => p.1 = p.2)),
        symMat v p.1 p.2 * symMat v p.1 p.2 = ∑ i, v i i * v i i := by
    refine Finset.sum_nbij' (fun p => p.1) (fun i => (i, i)) ?_ ?_ ?_ ?_ ?_
    · intro p hp
      simp only [Finset.mem_univ]
    · intro i _
      simp only [Finset.mem_filter, Finset.mem_univ]
      exact ⟨trivial, trivial⟩
    · intro p hp
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hp
      exact Prod.ext rfl hp
    · intro i _; rfl
    · intro p hp
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hp
      simp only [symMat, Matrix.of_apply, if_pos hp]
      rw [hp]
  -- off-diagonal part: per-term constant, then `mul_sum` to pull it out
  have hoff' : ∑ p ∈ (Finset.univ.filter (fun p : Fin D × Fin D => ¬ (p.1 = p.2))),
        symMat v p.1 p.2 * symMat v p.1 p.2
      = ((Real.sqrt 2 : ℝ)⁻¹) ^ 2 * ∑ p ∈ (Finset.univ.filter
          (fun p : Fin D × Fin D => ¬ (p.1 = p.2))), v p.1 p.2 * v p.1 p.2 := by
    have hterm : ∀ p ∈ (Finset.univ.filter (fun p : Fin D × Fin D => ¬ (p.1 = p.2))),
        symMat v p.1 p.2 * symMat v p.1 p.2
          = ((Real.sqrt 2 : ℝ)⁻¹) ^ 2 * (v p.1 p.2 * v p.1 p.2) := by
      intro p hp
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hp
      have hne : p.1 ≠ p.2 := fun h => hp h
      rw [symMat_off hne]
      ring
    rw [Finset.sum_congr rfl hterm, Finset.mul_sum]
  rw [h1, hdiag, hoff', inv_sqrt_two_sq]
  -- `Ne` filter ↔ `¬ =` filter for the pairing lemma
  have hpair : (Finset.univ.filter (fun p : Fin D × Fin D => ¬ (p.1 = p.2)))
      = Finset.univ.filter (fun p : Fin D × Fin D => p.1 ≠ p.2) := by
    rfl
  rw [hpair, offdiag_sq_eq_two_upper hv]
  simp only [packedNormSq]
  ring

/-- **T1b — sym isometry (inner product).** The Frobenius inner product of
two carried matrices equals the compact parameter-vector dot. -/
theorem sym_isometry_dot {u v : Matrix (Fin D) (Fin D) ℝ}
    (hu : u.IsSymm) (hv : v.IsSymm) :
    frobDot (symMat u) (symMat v) = packedDot u v := by
  classical
  have h1 : frobDot (symMat u) (symMat v)
      = ∑ p ∈ (Finset.univ.filter (fun p : Fin D × Fin D => p.1 = p.2)),
          symMat u p.1 p.2 * symMat v p.1 p.2
        + ∑ p ∈ (Finset.univ.filter (fun p : Fin D × Fin D => ¬ (p.1 = p.2))),
            symMat u p.1 p.2 * symMat v p.1 p.2 := by
    rw [frobDot, ← Fintype.sum_prod_type
      (fun p : Fin D × Fin D => symMat u p.1 p.2 * symMat v p.1 p.2),
      ← Finset.sum_filter_add_sum_filter_not
        Finset.univ (fun p : Fin D × Fin D => p.1 = p.2)
        (fun p => symMat u p.1 p.2 * symMat v p.1 p.2)]
  have hdiag : ∑ p ∈ (Finset.univ.filter (fun p : Fin D × Fin D => p.1 = p.2)),
        symMat u p.1 p.2 * symMat v p.1 p.2 = ∑ i, u i i * v i i := by
    refine Finset.sum_nbij' (fun p => p.1) (fun i => (i, i)) ?_ ?_ ?_ ?_ ?_
    · intro p hp
      simp only [Finset.mem_univ]
    · intro i _
      simp only [Finset.mem_filter, Finset.mem_univ]
      exact ⟨trivial, trivial⟩
    · intro p hp
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hp
      exact Prod.ext rfl hp
    · intro i _; rfl
    · intro p hp
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hp
      simp only [symMat, Matrix.of_apply, if_pos hp]
      rw [hp]
  have hoff' : ∑ p ∈ (Finset.univ.filter (fun p : Fin D × Fin D => ¬ (p.1 = p.2))),
        symMat u p.1 p.2 * symMat v p.1 p.2
      = ((Real.sqrt 2 : ℝ)⁻¹) ^ 2 * ∑ p ∈ (Finset.univ.filter
          (fun p : Fin D × Fin D => ¬ (p.1 = p.2))), u p.1 p.2 * v p.1 p.2 := by
    have hterm : ∀ p ∈ (Finset.univ.filter (fun p : Fin D × Fin D => ¬ (p.1 = p.2))),
        symMat u p.1 p.2 * symMat v p.1 p.2
          = ((Real.sqrt 2 : ℝ)⁻¹) ^ 2 * (u p.1 p.2 * v p.1 p.2) := by
      intro p hp
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hp
      have hne : p.1 ≠ p.2 := fun h => hp h
      rw [symMat_off hne, symMat_off hne]
      ring
    rw [Finset.sum_congr rfl hterm, Finset.mul_sum]
  rw [h1, hdiag, hoff', inv_sqrt_two_sq]
  have hpair : (Finset.univ.filter (fun p : Fin D × Fin D => ¬ (p.1 = p.2)))
      = Finset.univ.filter (fun p : Fin D × Fin D => p.1 ≠ p.2) := by
    rfl
  rw [hpair, offdiag_dot_eq_two_upper hu hv]
  simp only [packedDot]
  ring

/-- Nonnegativity of the packed norm square (sums of squares). -/
theorem packedNormSq_nonneg (v : Matrix (Fin D) (Fin D) ℝ) : 0 ≤ packedNormSq v := by
  unfold packedNormSq
  have h1 : 0 ≤ ∑ i, v i i * v i i :=
    Finset.sum_nonneg fun i _ => mul_self_nonneg (v i i)
  have h2 : 0 ≤ ∑ p ∈ Finset.univ.filter (fun p : Fin D × Fin D => p.1 < p.2),
      v p.1 p.2 * v p.1 p.2 :=
    Finset.sum_nonneg fun p _ => mul_self_nonneg (v p.1 p.2)
  exact add_nonneg h1 h2

/-- Nonnegativity of the Frobenius square. -/
theorem frobSq_nonneg (A : Matrix (Fin D) (Fin D) ℝ) : 0 ≤ frobSq A := by
  unfold frobSq
  refine Finset.sum_nonneg fun i _ => Finset.sum_nonneg fun j _ => ?_
  exact mul_self_nonneg (A i j)

/-- **T1a (sqrt form).** `‖symMat v‖_F = ‖v‖_packed` — the isometry as an
equality of (real, nonnegative) square roots. -/
theorem sym_isometry_norm {v : Matrix (Fin D) (Fin D) ℝ} (hv : v.IsSymm) :
    Real.sqrt (frobSq (symMat v)) = Real.sqrt (packedNormSq v) := by
  rw [sym_isometry_norm_sq hv]

end KatgptProof.Pencil
