/-
! Weyl's inequality for the spectral pencil (Issue 678 T2; Research 495
! §3.1 P2; paper arXiv:2608.08003 §3, Stewart–Sun).

`|λᵢ(A) − λᵢ(B)| ≤ ‖A − B‖₂` — eigenvalues are 1-Lipschitz in the
spectral norm. This is the load-bearing theorem behind the paper's global
feature-influence bound `|f(x+δ) − f(x)| ≤ Σ|δᵢ|·‖Aᵢ‖₂` (Cor. 1) and the
Lipschitz tamper check.

Mathlib has neither Weyl nor Courant–Fischer; this rides entirely on the
`RayleighCF` core (the one-sided proof is CF-ge applied to `A + E` on the
bottom eigenvector span of `A`, plus the top Rayleigh bound for `E`).
-/

import KatgptProof.Pencil.RayleighCF

namespace KatgptProof.Pencil

open Matrix
open scoped InnerProductSpace
open scoped Matrix.Norms.L2Operator

noncomputable section

variable {D : ℕ}

section Weyl

variable [DecidableEq (Fin D)] {A E : Matrix (Fin D) (Fin D) ℝ}

/-- Rayleigh is additive in the matrix. -/
theorem ray_add (A B : Matrix (Fin D) (Fin D) ℝ) (x : EuclideanSpace ℝ (Fin D)) :
    ray (A + B) x = ray A x + ray B x := by
  rw [ray, ray, ray]
  have h : toEucL (A + B) x = toEucL A x + toEucL B x := by
    simp only [toEucL]
    rw [map_add]
    simp
  rw [h, inner_add_left]

/-- An eigenvalue is the Rayleigh numerator of its eigenvector. -/
theorem eigval_eq_ray_self (hA : A.IsHermitian) (j : Fin D) :
    eigval hA j = ray A (eigvec hA j) := by
  rw [ray, eigvec_eigen hA j, real_inner_smul_left, eig_inner_self hA, mul_one]

/-- **Weyl one-sided**: adding a Hermitian perturbation moves the `i`-th
eigenvalue up by at most the top eigenvalue of the perturbation. -/
theorem weyl_one_sided [NeZero D] (hA : A.IsHermitian) (hE : E.IsHermitian)
    (i : Fin D) :
    eigval (hA.add hE) i ≤ eigval hA i + eigval hE 0 := by
  classical
  have hAE : (A + E).IsHermitian := hA.add hE
  -- the bottom-(D−i) eigenvector span of A
  have hlen : (i : ℕ) + (D - (i : ℕ)) = D := by omega
  have hinj : Function.Injective
      (fun m : Fin (D - (i : ℕ)) => ((Fin.natAdd i m).cast hlen : Fin D)) :=
    (Fin.cast_injective hlen).comp (Fin.natAdd_injective (D - (i : ℕ)) i)
  obtain ⟨x, hxS, hx0, hle⟩ := cf_ge (hA := hAE) i
    (Submodule.span ℝ (Set.range fun m : Fin (D - (i : ℕ)) =>
      eigvec hA ((Fin.natAdd i m).cast hlen))) (by
      rw [finrank_span_eq_card
        (Orthonormal.linearIndependent ⟨fun m => eigvec_norm hA _,
          fun a b hab => eigvec_orth hA (fun hcon => hab (hinj hcon))⟩),
        Fintype.card_fin _])
  obtain ⟨c, hc⟩ := (Submodule.mem_span_range_iff_exists_fun ℝ).mp hxS
  -- bound ray A x above on the bottom span
  have hbot : ray A x ≤ eigval hA i * ⟪x, x⟫_ℝ := by
    have := ray_le_of_forall hA hinj c (eigval hA i) (fun m => by
      refine eigval_antitone hA ?_
      have hval : (i : ℕ) ≤ (((Fin.natAdd i m).cast hlen : Fin D) : ℕ) := by
        simp only [Fin.val_cast, Fin.val_natAdd]
        omega
      exact Fin.le_def.2 hval)
    rw [hc] at this
    exact this
  -- top Rayleigh bound for E
  have htop : ray E x ≤ eigval hE 0 * ⟪x, x⟫_ℝ := ray_le_top hE x
  -- combine
  have hsplit : eigval hAE i * ⟪x, x⟫_ℝ ≤ ray A x + ray E x := by
    rw [← ray_add]
    exact hle
  have hn : (0:ℝ) < ⟪x, x⟫_ℝ := by
    have h1 : 0 ≤ ⟪x, x⟫_ℝ := real_inner_self_nonneg (x := x)
    have h2 : ⟪x, x⟫_ℝ ≠ 0 := fun h => hx0 (inner_self_eq_zero.mp h)
    exact lt_of_le_of_ne h1 h2.symm
  nlinarith [hbot, htop, hn]

/-- The top eigenvalue is bounded by the L2 operator norm. -/
theorem eigval_zero_le_opNorm (hE : E.IsHermitian) [NeZero D] :
    eigval hE 0 ≤ ‖E‖ := by
  have hray : eigval hE 0 = ray E (eigvec hE 0) := eigval_eq_ray_self hE 0
  set v := eigvec hE 0 with hv
  have hv1 : ‖v‖ = 1 := eigvec_norm hE 0
  have habs : eigval hE 0 ≤ |⟪toEucL E v, v⟫_ℝ| := by
    rw [hray, ray]; exact le_abs_self _
  have hcs : |⟪toEucL E v, v⟫_ℝ| ≤ ‖toEucL E v‖ := by
    have h := abs_real_inner_le_norm (toEucL E v) v
    rwa [hv1, mul_one] at h
  have hop : ‖toEucL E v‖ ≤ ‖E‖ := by
    have h1 := Matrix.l2_opNorm_mulVec E v
    have h2 : toEucL E v = (EuclideanSpace.equiv (Fin D) ℝ).symm (E *ᵥ v) := rfl
    rw [h2]
    rw [hv1, mul_one] at h1
    exact h1
  exact habs.trans (hcs.trans hop)

/-- **Weyl's inequality (T2)**: eigenvalues are 1-Lipschitz in the
spectral norm. -/
theorem weyl_lipschitz {B : Matrix (Fin D) (Fin D) ℝ} (hA : A.IsHermitian)
    (hB : B.IsHermitian) (i : Fin D) :
    |eigval hA i - eigval hB i| ≤ ‖A - B‖ := by
  classical
  haveI : NeZero D := ⟨i.pos.ne'⟩
  have hsub1 : (B - A).IsHermitian := hB.sub hA
  have hsub2 : (A - B).IsHermitian := hA.sub hB
  have eBA : A + (B - A) = B := by
    apply Matrix.ext; intro a b; simp
  have eAB : B + (A - B) = A := by
    apply Matrix.ext; intro a b; simp
  -- one-sided applications (transport through the matrix identities)
  have h1 : eigval hB i ≤ eigval hA i + eigval hsub1 0 := by
    have h := weyl_one_sided hA hsub1 i
    simp only [eBA] at h
    exact h
  have h2 : eigval hA i ≤ eigval hB i + eigval hsub2 0 := by
    have h := weyl_one_sided hB hsub2 i
    simp only [eAB] at h
    exact h
  -- top eigenvalues of the differences are ≤ the norms
  have hE1 : eigval hsub1 0 ≤ ‖B - A‖ := eigval_zero_le_opNorm hsub1
  have hE2 : eigval hsub2 0 ≤ ‖A - B‖ := eigval_zero_le_opNorm hsub2
  -- |x - y| ≤ c from both one-sided bounds
  rcases le_or_gt (eigval hA i) (eigval hB i) with hle | hlt
  · rw [abs_of_nonpos (by linarith)]
    have hnorm : ‖B - A‖ = ‖A - B‖ := norm_sub_rev _ _
    rw [← hnorm]
    linarith
  · rw [abs_of_nonneg (by linarith)]
    linarith

end Weyl

end

end KatgptProof.Pencil
