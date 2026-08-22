/-
! Courant–Fischer machinery for the spectral pencil (Issue 678;
! Research 495 §3.1 P2; paper arXiv:2608.08003 §3).

Mathlib ships the spectral theorem for Hermitian matrices but neither the
Courant–Fischer min–max characterization nor Weyl's inequality. This module
builds the Courant–Fischer core from the spectral theorem; Weyl (T2),
Loewner monotonicity + mirror duality (T3) and the constructive eigengap
bound (T4) all ride on it, in their own modules.

## Architecture (every consumer needs only these facts)

1. **Combo expansion** — for any injectively-indexed combination of
   eigenvectors, `ray A x = ∑ eigval * c²` and `⟪x, x⟫ = ∑ c²`
   (`ray_combo`, `inner_combo`).
2. **Weighted-average bounds** — the expansion makes the top Rayleigh
   bound and the top/bottom-span bounds termwise sums
   (`ray_le_of_forall`, `ray_ge_of_forall`).
3. **Subspace dimension argument (SDA)** — subspaces `S`, `W` with
   `finrank S + finrank W > D` intersect nontrivially.
4. **CF-ge / CF-dual** — every `(D−i)`-dim subspace contains a direction
   with `ray ≥ λᵢ` (SDA on the top-`i+1` eigenvector span); every
   `(i+1)`-dim subspace contains a direction with `ray ≤ λᵢ` (SDA on the
   bottom-`D−i` eigenvector span).

The eigenvalue array is Mathlib's `LinearMap.IsSymmetric.eigenvalues`
instantiated at `finrank = D` — the antitone (decreasing-sorted) one.
(The matrix-level `Matrix.IsHermitian.eigenvalues` reindexes through an
opaque `Fintype.equivOfCardEq`, so its antitone-ness is not usable; the
LinearMap instantiation at `Fin D` is exact.)
-/

import Mathlib

namespace KatgptProof.Pencil

open Matrix
open scoped InnerProductSpace

noncomputable section

variable {D : ℕ}

/-! ## The eigen-bundle of a real symmetric matrix
-/


section Eigen

variable [DecidableEq (Fin D)] {A : Matrix (Fin D) (Fin D) ℝ}

/-- `finrank` of `EuclideanSpace ℝ (Fin D)` as a plain equation in `D`. -/
theorem finrank_euc_D : Module.finrank ℝ (EuclideanSpace ℝ (Fin D)) = D :=
  finrank_euclideanSpace.trans (Fintype.card_fin D)

/-- Decreasing-sorted eigenvalue array of a real symmetric matrix. -/
def eigval (hA : A.IsHermitian) : Fin D → ℝ :=
  (Matrix.isSymmetric_toEuclideanLin_iff.mpr hA).eigenvalues finrank_euc_D

/-- Matching orthonormal eigenvector family. -/
def eigvec (hA : A.IsHermitian) : Fin D → EuclideanSpace ℝ (Fin D) :=
  (Matrix.isSymmetric_toEuclideanLin_iff.mpr hA).eigenvectorBasis finrank_euc_D

/-- The eigenvalue array is antitone (decreasing). -/
theorem eigval_antitone (hA : A.IsHermitian) : Antitone (eigval hA) :=
  LinearMap.IsSymmetric.eigenvalues_antitone _ finrank_euc_D

/-- Eigenvectors are unit. -/
theorem eigvec_norm (hA : A.IsHermitian) (j : Fin D) : ‖eigvec hA j‖ = 1 := by
  have h := ((Matrix.isSymmetric_toEuclideanLin_iff.mpr hA).eigenvectorBasis
    finrank_euc_D).orthonormal
  exact h.1 j

/-- Distinct eigenvectors are orthogonal. -/
theorem eigvec_orth (hA : A.IsHermitian) {i j : Fin D} (h : i ≠ j) :
    ⟪eigvec hA i, eigvec hA j⟫_ℝ = 0 := by
  have hpair := ((Matrix.isSymmetric_toEuclideanLin_iff.mpr hA).eigenvectorBasis
    finrank_euc_D).orthonormal
  exact hpair.2 h

/-- The linear action of `A` on Euclidean space, as a bare `LinearMap`
(one canonical coercion path for every downstream rewrite). -/
def toEucL (A : Matrix (Fin D) (Fin D) ℝ) :
    EuclideanSpace ℝ (Fin D) →ₗ[ℝ] EuclideanSpace ℝ (Fin D) :=
  A.toEuclideanLin

/-- The eigen-equation in operator form. -/
theorem eigvec_eigen (hA : A.IsHermitian) (j : Fin D) :
    toEucL A (eigvec hA j) = eigval hA j • eigvec hA j := by
  have h := (Matrix.isSymmetric_toEuclideanLin_iff.mpr hA).apply_eigenvectorBasis
    finrank_euc_D j
  simpa [toEucL, eigvec, eigval] using h

/-- Unit self-inner product of an eigenvector. -/
theorem eig_inner_self (hA : A.IsHermitian) (j : Fin D) :
    ⟪eigvec hA j, eigvec hA j⟫_ℝ = 1 := by
  rw [real_inner_self_eq_norm_sq, eigvec_norm hA j, one_pow]

/-- Orthonormal expansion: every vector is the sum of its eigen-coefficients
against the eigenbasis. -/
theorem eig_sum_repr (hA : A.IsHermitian) (x : EuclideanSpace ℝ (Fin D)) :
    ∑ j, ⟪eigvec hA j, x⟫_ℝ • eigvec hA j = x := by
  exact ((Matrix.isSymmetric_toEuclideanLin_iff.mpr hA).eigenvectorBasis
    finrank_euc_D).sum_repr' x

end Eigen

/-! ## The Rayleigh quotient and the combo lemmas
-/

section Rayleigh

variable [DecidableEq (Fin D)] {A : Matrix (Fin D) (Fin D) ℝ}

/-- Rayleigh numerator `⟪Ax, x⟫` for a real matrix acting on Euclidean
space. -/
def ray (A : Matrix (Fin D) (Fin D) ℝ) (x : EuclideanSpace ℝ (Fin D)) : ℝ :=
  ⟪toEucL A x, x⟫_ℝ

variable (hA : A.IsHermitian)

/-- Inner product of one eigenvector against an injectively-indexed
combination of eigenvectors picks out the matching coefficient. -/
theorem eig_inner_combo {σ : Type*} [Fintype σ] {idx : σ → Fin D}
    (hinj : Function.Injective idx) (c : σ → ℝ) (m : σ) :
    ⟪eigvec hA (idx m), ∑ k, c k • eigvec hA (idx k)⟫_ℝ = c m := by
  rw [inner_sum]
  rw [Finset.sum_eq_single m]
  · rw [real_inner_smul_right, eig_inner_self hA, mul_one]
  · intro k _ hkm
    rw [real_inner_smul_right, eigvec_orth hA (fun hcon => hkm (hinj hcon.symm)),
      mul_zero]
  · intro h; exact absurd (Finset.mem_univ m) h

/-- Norm of an injectively-indexed combination of eigenvectors. -/
theorem inner_combo {σ : Type*} [Fintype σ] {idx : σ → Fin D}
    (hinj : Function.Injective idx) (c : σ → ℝ) :
    ⟪∑ m, c m • eigvec hA (idx m), ∑ m, c m • eigvec hA (idx m)⟫_ℝ = ∑ m, c m * c m := by
  rw [sum_inner]
  refine Finset.sum_congr rfl fun m _ => ?_
  rw [real_inner_smul_left, eig_inner_combo hA hinj c m]

/-- Rayleigh numerator of an injectively-indexed combination of
eigenvectors. -/
theorem ray_combo {σ : Type*} [Fintype σ] {idx : σ → Fin D}
    (hinj : Function.Injective idx) (c : σ → ℝ) :
    ray A (∑ m, c m • eigvec hA (idx m)) = ∑ m, eigval hA (idx m) * (c m * c m) := by
  rw [ray, map_sum, sum_inner]
  refine Finset.sum_congr rfl fun m _ => ?_
  rw [map_smul, real_inner_smul_left, eigvec_eigen hA (idx m), real_inner_smul_left,
    eig_inner_combo hA hinj c m]
  ring

/-- Upper weighted-average bound: if every participating eigenvalue is
`≤ t`, the Rayleigh numerator of the combination is `≤ t * ⟪x, x⟫`. -/
theorem ray_le_of_forall {σ : Type*} [Fintype σ] {idx : σ → Fin D}
    (hinj : Function.Injective idx) (c : σ → ℝ) (t : ℝ)
    (h : ∀ m, eigval hA (idx m) ≤ t) :
    ray A (∑ m, c m • eigvec hA (idx m))
      ≤ t * ⟪∑ m, c m • eigvec hA (idx m), ∑ m, c m • eigvec hA (idx m)⟫_ℝ := by
  rw [ray_combo hA hinj c, inner_combo hA hinj c, Finset.mul_sum]
  exact Finset.sum_le_sum fun m _ =>
    mul_le_mul_of_nonneg_right (h m) (mul_self_nonneg (c m))

/-- Lower weighted-average bound: if every participating eigenvalue is
`≥ t`, the Rayleigh numerator of the combination is `≥ t * ⟪x, x⟫`. -/
theorem ray_ge_of_forall {σ : Type*} [Fintype σ] {idx : σ → Fin D}
    (hinj : Function.Injective idx) (c : σ → ℝ) (t : ℝ)
    (h : ∀ m, t ≤ eigval hA (idx m)) :
    t * ⟪∑ m, c m • eigvec hA (idx m), ∑ m, c m • eigvec hA (idx m)⟫_ℝ
      ≤ ray A (∑ m, c m • eigvec hA (idx m)) := by
  rw [ray_combo hA hinj c, inner_combo hA hinj c, Finset.mul_sum]
  exact Finset.sum_le_sum fun m _ =>
    mul_le_mul_of_nonneg_right (h m) (mul_self_nonneg (c m))

/-- Rayleigh expansion against the full eigenbasis. -/
theorem ray_eq_sum (x : EuclideanSpace ℝ (Fin D)) :
    ray A x = ∑ j, eigval hA j * (⟪eigvec hA j, x⟫_ℝ) ^ 2 := by
  nth_rw 1 [← eig_sum_repr hA x]
  rw [ray, map_sum, sum_inner]
  refine Finset.sum_congr rfl fun m _ => ?_
  rw [map_smul, real_inner_smul_left, eigvec_eigen hA m, real_inner_smul_left,
    eig_inner_combo hA (idx := fun j => j) (hinj := fun _ _ h => h)
      (fun j => ⟪eigvec hA j, x⟫_ℝ) m]
  ring

/-- Norm expansion against the full eigenbasis. -/
theorem inner_self_eq_sum (x : EuclideanSpace ℝ (Fin D)) :
    ⟪x, x⟫_ℝ = ∑ j, (⟪eigvec hA j, x⟫_ℝ) ^ 2 := by
  calc ⟪x, x⟫_ℝ = ⟪∑ j, ⟪eigvec hA j, x⟫_ℝ • eigvec hA j,
          ∑ j, ⟪eigvec hA j, x⟫_ℝ • eigvec hA j⟫_ℝ := by rw [eig_sum_repr hA x]
    _ = ∑ j, ⟪eigvec hA j, x⟫_ℝ * ⟪eigvec hA j, x⟫_ℝ :=
        inner_combo hA Function.injective_id (fun j => ⟪eigvec hA j, x⟫_ℝ)
    _ = ∑ j, (⟪eigvec hA j, x⟫_ℝ) ^ 2 :=
        Finset.sum_congr rfl fun j _ => by ring

/-- **Top Rayleigh bound** (Courant–Fischer at `i = 0`, upper side):
every Rayleigh numerator is at most the top eigenvalue times the norm. -/
theorem ray_le_top [NeZero D] (x : EuclideanSpace ℝ (Fin D)) :
    ray A x ≤ eigval hA 0 * ⟪x, x⟫_ℝ := by
  calc ray A x ≤ ray A (∑ j, ⟪eigvec hA j, x⟫_ℝ • eigvec hA j) := by
        rw [eig_sum_repr hA x]
    _ ≤ eigval hA 0 * ⟪∑ j, ⟪eigvec hA j, x⟫_ℝ • eigvec hA j,
          ∑ j, ⟪eigvec hA j, x⟫_ℝ • eigvec hA j⟫_ℝ :=
        ray_le_of_forall hA Function.injective_id
          (fun j => ⟪eigvec hA j, x⟫_ℝ) (eigval hA 0)
          (fun j => eigval_antitone hA (Fin.zero_le j))
    _ = eigval hA 0 * ⟪x, x⟫_ℝ := by rw [eig_sum_repr hA x]

end Rayleigh

/-! ## The subspace dimension argument (SDA)
-/


section SDA

variable [DecidableEq (Fin D)]

/-- Two subspaces whose finranks sum to more than `D` intersect
nontrivially. -/
theorem exists_mem_inter_of_finrank (S W : Submodule ℝ (EuclideanSpace ℝ (Fin D)))
    (h : Module.finrank ℝ S + Module.finrank ℝ W > D) :
    ∃ x, x ∈ S ∧ x ∈ W ∧ x ≠ 0 := by
  by_contra hcon
  push_neg at hcon
  have hbot : S ⊓ W = ⊥ := by
    rw [eq_bot_iff]
    intro x hx
    exact hcon x hx.1 hx.2
  have hfr := Submodule.finrank_sup_add_finrank_inf_eq S W
  have hsup : (S ⊔ W) ≤ (⊤ : Submodule ℝ (EuclideanSpace ℝ (Fin D))) := le_top
  haveI : Module.Finite ℝ (⊤ : Submodule ℝ (EuclideanSpace ℝ (Fin D))) := by infer_instance
  have hle := Submodule.finrank_mono hsup
  rw [finrank_top, finrank_euc_D] at hle
  rw [hbot] at hfr
  simp only [finrank_bot] at hfr
  omega

end SDA

/-! ## Courant–Fischer sandwiches
-/


section MinMax

variable [DecidableEq (Fin D)] {A : Matrix (Fin D) (Fin D) ℝ} (hA : A.IsHermitian)

/-- **CF-ge** (Courant–Fischer lower sandwich): every subspace of
dimension `D − i` contains a nonzero vector whose Rayleigh numerator is
at least `λᵢ` times its norm. This is the load-bearing direction for
Weyl's inequality. -/
theorem cf_ge (i : Fin D) (S : Submodule ℝ (EuclideanSpace ℝ (Fin D)))
    (hS : Module.finrank ℝ S = D - (i : ℕ)) :
    ∃ x, x ∈ S ∧ x ≠ 0 ∧ eigval hA i * ⟪x, x⟫_ℝ ≤ ray A x := by
  classical
  have hi : (i : ℕ) < D := i.is_lt
  -- the top-(i+1) eigenvector span
  have hinj : Function.Injective
      (fun m : Fin ((i : ℕ) + 1) => (m.castLE (by omega) : Fin D)) :=
    (Fin.strictMono_castLE (by omega)).injective
  obtain ⟨x, hxS, hxW, hx0⟩ := exists_mem_inter_of_finrank
    S (Submodule.span ℝ (Set.range fun m : Fin ((i : ℕ) + 1) =>
      eigvec hA (m.castLE (by omega)))) (by
      rw [hS, (finrank_span_eq_card
        (Orthonormal.linearIndependent ⟨fun m => eigvec_norm hA _,
          fun a b hab => eigvec_orth hA (fun hcon => hab (hinj hcon))⟩)).trans
        (Fintype.card_fin _)]
      omega)
  obtain ⟨c, hc⟩ := (Submodule.mem_span_range_iff_exists_fun ℝ).mp hxW
  have hge := ray_ge_of_forall hA hinj c (eigval hA i) (fun m => by
    refine eigval_antitone hA ?_
    have hval : ((m.castLE (by omega) : Fin D) : ℕ) ≤ (i : ℕ) := by
      simp only [Fin.val_castLE]
      omega
    exact Fin.le_def.2 hval)
  rw [hc] at hge
  exact ⟨x, hxS, hx0, hge⟩

/-- **CF-dual** (Courant–Fischer upper sandwich): every subspace of
dimension `i + 1` contains a nonzero vector whose Rayleigh numerator is
at most `λᵢ` times its norm. This is the load-bearing direction for
mirror duality. -/
theorem cf_dual (i : Fin D) (S : Submodule ℝ (EuclideanSpace ℝ (Fin D)))
    (hS : Module.finrank ℝ S = (i : ℕ) + 1) :
    ∃ x, x ∈ S ∧ x ≠ 0 ∧ ray A x ≤ eigval hA i * ⟪x, x⟫_ℝ := by
  classical
  have hi : (i : ℕ) < D := i.is_lt
  -- the bottom-(D−i) eigenvector span: indices i, i+1, …, D−1
  have hlen : (i : ℕ) + (D - (i : ℕ)) = D := by omega
  have hinj : Function.Injective
      (fun m : Fin (D - (i : ℕ)) => ((Fin.natAdd i m).cast hlen : Fin D)) :=
    (Fin.cast_injective hlen).comp (Fin.natAdd_injective (D - (i : ℕ)) i)
  obtain ⟨x, hxS, hxW, hx0⟩ := exists_mem_inter_of_finrank
    S (Submodule.span ℝ (Set.range fun m : Fin (D - (i : ℕ)) =>
      eigvec hA ((Fin.natAdd i m).cast hlen))) (by
      rw [hS, (finrank_span_eq_card
        (Orthonormal.linearIndependent ⟨fun m => eigvec_norm hA _,
          fun a b hab => eigvec_orth hA (fun hcon => hab (hinj hcon))⟩)).trans
        (Fintype.card_fin _)]
      omega)
  obtain ⟨c, hc⟩ := (Submodule.mem_span_range_iff_exists_fun ℝ).mp hxW
  have hle := ray_le_of_forall hA hinj c (eigval hA i) (fun m => by
    refine eigval_antitone hA ?_
    have hval : (i : ℕ) ≤ (((Fin.natAdd i m).cast hlen : Fin D) : ℕ) := by
      simp only [Fin.val_cast, Fin.val_natAdd]
      omega
    exact Fin.le_def.2 hval)
  rw [hc] at hle
  exact ⟨x, hxS, hx0, hle⟩

end MinMax

end

end KatgptProof.Pencil
