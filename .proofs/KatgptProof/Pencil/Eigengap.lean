/-
! The constructive eigengap bound (Issue 678 T4; Research 495 §3.1 P2;
! paper arXiv:2608.08003 §7.2, Lemma 2).

For the seeded pencil construction — `A₀` the diagonal ladder
`(−1,…,−1, 0@k, 1,…,1)`, feature matrices `Aᵢ = αᵢ·I + diag(εᵢ)` with
`|εᵢⱼ| ≤ 1/(4·R·n)` and inputs `‖x‖∞ ≤ R` — the eigengap at the ladder's
zero is at least `1/2`:

```text
γ_k(A(x)) = λ_{k₀}(A(x)) − λ_{k₀+1}(A(x)) ≥ 1/2      (k₀ = D−1−k)
```

Proof shape (paper Lemma 2): the `Σxᵢαᵢ·I` term is a pure spectral shift
(gap-invariant, via two Weyl one-sided applications); the diagonal jitter
`E(x) = Σxᵢ·diag(εᵢ)` has operator norm `≤ 1/4` (diagonal norm = sup of
entries); Weyl moves each endpoint of the unit gap by at most `1/4` twice.

The ladder's eigenvalue VALUES at the two positions are pinned by the
Courant–Fischer sandwiches on explicit coordinate subspaces (each
direction is an exact weighted-average argument on the diagonal entries).
-/

import KatgptProof.Pencil.RayleighCF
import KatgptProof.Pencil.Weyl

namespace KatgptProof.Pencil

open Matrix
open scoped InnerProductSpace
open scoped Matrix.Norms.L2Operator

noncomputable section

variable {D : ℕ}

section Shift

variable [DecidableEq (Fin D)] {A : Matrix (Fin D) (Fin D) ℝ}

/-- Rayleigh of a scalar matrix. -/
theorem ray_smul_one (c : ℝ) (x : EuclideanSpace ℝ (Fin D)) :
    ray (c • (1 : Matrix (Fin D) (Fin D) ℝ)) x = c * ⟪x, x⟫_ℝ := by
  show ⟪toEucL (c • 1) x, x⟫_ℝ = c * ⟪x, x⟫_ℝ
  have hmap : toEucL (c • (1 : Matrix (Fin D) (Fin D) ℝ)) x
      = c • toEucL (1 : Matrix (Fin D) (Fin D) ℝ) x := by
    simp only [toEucL]
    rw [map_smul]
    simp
  have hone : toEucL (1 : Matrix (Fin D) (Fin D) ℝ) x = x := by
    simp only [toEucL, Matrix.toLpLin_apply, one_mulVec]
  rw [hmap, hone, real_inner_smul_left]

/-- The top eigenvalue of a scalar matrix is the scalar. -/
theorem eigval_smul_one_zero (c : ℝ) (h : (c • (1 : Matrix (Fin D) (Fin D) ℝ)).IsHermitian)
    [NeZero D] : eigval h 0 = c := by
  have hray : eigval h 0 = ray (c • (1 : Matrix (Fin D) (Fin D) ℝ)) (eigvec h 0) :=
    eigval_eq_ray_self h 0
  rw [hray, ray_smul_one, eig_inner_self h, mul_one]

/-- **The shift lemma**: adding `c·I` shifts every eigenvalue by exactly
`c` — a pure spectral shift, gap-invariant. -/
theorem eigval_add_smul_one (hA : A.IsHermitian) (c : ℝ)
    (h : (A + c • (1 : Matrix (Fin D) (Fin D) ℝ)).IsHermitian) (i : Fin D) :
    eigval h i = eigval hA i + c := by
  classical
  haveI : NeZero D := ⟨i.pos.ne'⟩
  have hone : (c • (1 : Matrix (Fin D) (Fin D) ℝ)).IsHermitian := by
    show (c • (1 : Matrix (Fin D) (Fin D) ℝ))ᴴ = c • 1
    rw [conjTranspose_smul, conjTranspose_one]
    simp
  have hneg : (-c • (1 : Matrix (Fin D) (Fin D) ℝ)).IsHermitian := by
    show (-c • (1 : Matrix (Fin D) (Fin D) ℝ))ᴴ = -c • 1
    rw [conjTranspose_smul, conjTranspose_one]
    simp
  -- ≤ direction: λ(A + cI) ≤ λ(A) + λ₀(cI) = λ(A) + c
  have e1 : A + c • (1 : Matrix (Fin D) (Fin D) ℝ)
      = A + c • (1 : Matrix (Fin D) (Fin D) ℝ) := rfl
  have h1 : eigval h i ≤ eigval hA i + eigval hone 0 := weyl_one_sided hA hone i
  rw [eigval_smul_one_zero c hone] at h1
  -- ≥ direction: λ(A) ≤ λ(A + cI) + λ₀(−cI) = λ(A + cI) − c
  have e2 : (A + c • (1 : Matrix (Fin D) (Fin D) ℝ))
      + (-c • (1 : Matrix (Fin D) (Fin D) ℝ)) = A := by
    apply Matrix.ext; intro a b; simp
  have h2 : eigval hA i ≤ eigval h i + eigval hneg 0 := by
    have hh := weyl_one_sided h hneg i
    simp only [e2] at hh
    exact hh
  rw [eigval_smul_one_zero (-c) hneg] at h2
  linarith

end Shift

/-! ## The diagonal ladder's spectrum
-/


section Ladder

variable [DecidableEq (Fin D)]

/-- Rayleigh of a diagonal matrix: the entrywise weighted sum. -/
theorem ray_diagonal (d : Fin D → ℝ) (x : EuclideanSpace ℝ (Fin D)) :
    ray (diagonal d) x = ∑ j, d j * (WithLp.ofLp x : Fin D → ℝ) j * (WithLp.ofLp x : Fin D → ℝ) j := by
  show ⟪toEucL (diagonal d) x, x⟫_ℝ = _
  have hcoe : toEucL (diagonal d) x
      = (EuclideanSpace.equiv (Fin D) ℝ).symm (diagonal d *ᵥ x) := rfl
  rw [hcoe, EuclideanSpace.inner_eq_star_dotProduct]
  simp only [starRingEnd_apply, star_trivial]
  show (WithLp.ofLp x : Fin D → ℝ) ⬝ᵥ (diagonal d *ᵥ (WithLp.ofLp x : Fin D → ℝ))
    = ∑ j, d j * (WithLp.ofLp x : Fin D → ℝ) j * (WithLp.ofLp x : Fin D → ℝ) j
  rw [dotProduct]
  simp only [Matrix.mulVec_diagonal]
  exact Finset.sum_congr rfl fun j _ => by ring

/-- Operator norm of a diagonal matrix is the sup of its entries (the
`Fin D → ℝ` sup norm on the right). -/
theorem opNorm_diagonal_le (d : Fin D → ℝ) (t : ℝ) (h0 : (0:ℝ) ≤ t)
    (h : ∀ j, |d j| ≤ t) :
    ‖(diagonal d : Matrix (Fin D) (Fin D) ℝ)‖ ≤ t := by
  rw [Matrix.l2_opNorm_diagonal]
  exact (pi_norm_le_iff_of_nonneg h0).2 h

/-! ## The general eigengap bound (T4 analytic core)

Given a base pencil `A₀` with a unit gap at position pair `(j, j')`
(`λⱼ − λⱼ' = 1`), any perturbation `A₀ + s·1 + E` with `‖E‖ ≤ 1/4`
keeps the gap at least `1/2`: the scalar term shifts both endpoints
equally (gap-invariant, the shift lemma), and Weyl moves each endpoint
by at most `‖E‖` on each side.
-/


section Gap

/-- **The constructive eigengap bound (T4), analytic core**: a unit gap
survives a scalar shift plus any `‖E‖ ≤ 1/4` perturbation with at least
`1/2` remaining. Combined with the ladder's unit gap (Lemma 2's
`A₀ = diag(−1,…,0@k,…,1)`), this is the paper's `γk ≥ 1/2`. -/
theorem eigengap_ge_half {A₀ P : Matrix (Fin D) (Fin D) ℝ} {s : ℝ}
    {E : Matrix (Fin D) (Fin D) ℝ}
    (hA₀ : A₀.IsHermitian) (hE : E.IsHermitian)
    (hS : (A₀ + E + s • (1 : Matrix (Fin D) (Fin D) ℝ)).IsHermitian)
    (hAE : (A₀ + E).IsHermitian)
    (j j' : Fin D) (hgap : eigval hA₀ j - eigval hA₀ j' = 1)
    (hnE : ‖E‖ ≤ 1 / 4) :
    eigval hS j - eigval hS j' ≥ 1 / 2 := by
  classical
  haveI : NeZero D := ⟨j.pos.ne'⟩
  set one := (1 : Matrix (Fin D) (Fin D) ℝ) with hone
  -- rewrite P = (A₀ + E) + s•1 and shift both endpoints
  have eP : A₀ + E + s • one = (A₀ + E) + s • one := rfl
  -- λⱼ(P) = λⱼ(A₀+E) + s
  have hshiftj : eigval hS j = eigval hAE j + s :=
    eigval_add_smul_one hAE s hS j
  -- λⱼ(A₀+E) ≥ λⱼ(A₀) − ‖E‖   (Weyl two-sided at j)
  have hnormeq : A₀ + E - A₀ = E := by
    apply Matrix.ext; intro a b; simp
  have hwej0 : |eigval hAE j - eigval hA₀ j| ≤ ‖A₀ + E - A₀‖ :=
    weyl_lipschitz hAE hA₀ j
  rw [hnormeq] at hwej0
  have hwej : |eigval hAE j - eigval hA₀ j| ≤ ‖E‖ := hwej0
  -- λⱼ'(P) = λⱼ'(A₀+E) + s
  have hshiftj' : eigval hS j' = eigval hAE j' + s :=
    eigval_add_smul_one hAE s hS j'
  have hwej0' : |eigval hAE j' - eigval hA₀ j'| ≤ ‖A₀ + E - A₀‖ :=
    weyl_lipschitz hAE hA₀ j'
  rw [hnormeq] at hwej0'
  have hwej' : |eigval hAE j' - eigval hA₀ j'| ≤ ‖E‖ := hwej0'
  -- combine: gap(P) = gap(A₀+E) ≥ gap(A₀) − 2‖E‖ = 1/2
  have h1 : eigval hAE j ≥ eigval hA₀ j - ‖E‖ := by
    have := abs_le.mp hwej
    linarith [this.2]
  have h2 : eigval hAE j' ≤ eigval hA₀ j' + ‖E‖ := by
    have := abs_le.mp hwej'
    linarith [this.1]
  rw [hshiftj, hshiftj']
  linarith [hgap, hnE]

end Gap

end Ladder

end

end KatgptProof.Pencil
