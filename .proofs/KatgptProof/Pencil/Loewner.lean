/-
! Loewner monotonicity + mirror duality for the spectral pencil
! (Issue 678 T3; Research 495 §3.1 P2; paper arXiv:2608.08003 §4.7).

* **Loewner monotonicity** — `B − A ⪰ 0 ⇒ λᵢ(A) ≤ λᵢ(B)` for every `i`:
  the shape DSL's soundness core (a PSD feature matrix makes the gate
  non-decreasing in that feature, by construction rather than tuning).
* **Mirror duality** — `λⱼ(−A) = −λᵢ(A)` whenever `j = D−1−i`: k=1
  concave ↔ k=D convex for free (the paper's §7 "mirror" — negating the
  pencil flips the eigenvalue ladder end-to-end).

Loewner rides on Weyl one-sided; mirror duality rides on the CF
sandwiches directly.
-/

import KatgptProof.Pencil.RayleighCF
import KatgptProof.Pencil.Weyl

namespace KatgptProof.Pencil

open Matrix
open scoped InnerProductSpace

noncomputable section

variable {D : ℕ}

section Loewner

variable [DecidableEq (Fin D)] {A B : Matrix (Fin D) (Fin D) ℝ}

/-- Rayleigh is odd in the matrix. -/
theorem ray_neg (A : Matrix (Fin D) (Fin D) ℝ) (x : EuclideanSpace ℝ (Fin D)) :
    ray (-A) x = -ray A x := by
  show ⟪toEucL (-A) x, x⟫_ℝ = -⟪toEucL A x, x⟫_ℝ
  have hmap : toEucL (-A) x = -(toEucL A x) := by
    simp only [toEucL]
    rw [map_neg]
    simp
  rw [hmap, inner_neg_left]

/-- The Rayleigh numerator of a PSD matrix is nonnegative. -/
theorem ray_nonneg_of_posSemidef {E : Matrix (Fin D) (Fin D) ℝ}
    (hE : E.PosSemidef) (x : EuclideanSpace ℝ (Fin D)) : 0 ≤ ray E x := by
  have h0 : (0:ℝ) ≤ (WithLp.ofLp x : Fin D → ℝ) ⬝ᵥ (E *ᵥ (WithLp.ofLp x)) :=
    hE.dotProduct_mulVec_nonneg _
  show 0 ≤ ⟪toEucL E x, x⟫_ℝ
  have hcoe : toEucL E x = (EuclideanSpace.equiv (Fin D) ℝ).symm (E *ᵥ x) := rfl
  rw [hcoe, EuclideanSpace.inner_eq_star_dotProduct]
  simp only [star_trivial, dotProduct_comm]
  exact h0

/-- **Loewner monotonicity (T3a)**: if `B − A` is PSD then every
eigenvalue of `A` is at most the matching eigenvalue of `B`. -/
theorem loewner_mono (hA : A.IsHermitian) (hB : B.IsHermitian)
    (hBA : (B - A).PosSemidef) (i : Fin D) :
    eigval hA i ≤ eigval hB i := by
  classical
  haveI : NeZero D := ⟨i.pos.ne'⟩
  have hsub : (A - B).IsHermitian := hA.sub hB
  -- λᵢ(A) ≤ λᵢ(B) + λ₀(A−B)  (Weyl one-sided through B + (A−B) = A)
  have e : B + (A - B) = A := by
    apply Matrix.ext; intro a b; simp
  have h2 : eigval hA i ≤ eigval hB i + eigval hsub 0 := by
    have h := weyl_one_sided hB hsub i
    simp only [e] at h
    exact h
  -- λ₀(A−B) ≤ 0 because ray (A−B) = −ray (B−A) ≤ 0
  have h0le : eigval hsub 0 ≤ 0 := by
    have hray : eigval hsub 0 = ray (A - B) (eigvec hsub 0) :=
      eigval_eq_ray_self hsub 0
    have hneg : ∀ y : EuclideanSpace ℝ (Fin D),
        ray (A - B) y = -ray (B - A) y := by
      intro y
      have heq : (A - B) = -(B - A) := by
        apply Matrix.ext; intro a b; simp
      rw [heq, ray_neg]
    rw [hray, hneg]
    have hpos : 0 ≤ ray (B - A) (eigvec hsub 0) := ray_nonneg_of_posSemidef hBA _
    linarith
  linarith

/-- **Mirror duality (T3b)**: negating the matrix reverses the eigenvalue
ladder: `λⱼ(−A) = −λᵢ(A)` whenever `j = D−1−i`. -/
theorem mirror_dual (hA : A.IsHermitian) (i : Fin D) :
    eigval hA.neg (⟨D - 1 - (i : ℕ), by omega⟩ : Fin D) + eigval hA i = 0 := by
  classical
  set j : Fin D := ⟨D - 1 - (i : ℕ), by omega⟩ with hj
  -- span of the TOP (i+1) eigenvectors of A
  have hinj1 : Function.Injective
      (fun m : Fin ((i : ℕ) + 1) => (m.castLE (by omega) : Fin D)) :=
    (Fin.strictMono_castLE (by omega)).injective
  -- span of the BOTTOM (D−i) eigenvectors of A
  have hlen : (i : ℕ) + (D - (i : ℕ)) = D := by omega
  have hinj2 : Function.Injective
      (fun m : Fin (D - (i : ℕ)) => ((Fin.natAdd i m).cast hlen : Fin D)) :=
    (Fin.cast_injective hlen).comp (Fin.natAdd_injective (D - (i : ℕ)) i)
  -- direction ≤:  λⱼ(−A) ≤ −λᵢ(A)
  -- cf_ge on −A at index j with S* = top-(i+1) span of A (dim D−j = i+1)
  have hdimS : Module.finrank ℝ
      (Submodule.span ℝ (Set.range fun m : Fin ((i : ℕ) + 1) =>
        eigvec hA (m.castLE (by omega)))) = D - (j : ℕ) := by
    rw [finrank_span_eq_card
      (Orthonormal.linearIndependent ⟨fun m => eigvec_norm hA _,
        fun a b hab => eigvec_orth hA (fun hcon => hab (hinj1 hcon))⟩),
      Fintype.card_fin _]
    simp only [Fintype.card_fin, hj]
    omega
  obtain ⟨x, hxS, hx0, hle⟩ := cf_ge (hA := hA.neg) j
    (Submodule.span ℝ (Set.range fun m : Fin ((i : ℕ) + 1) =>
      eigvec hA (m.castLE (by omega)))) hdimS
  obtain ⟨c, hc⟩ := (Submodule.mem_span_range_iff_exists_fun ℝ).mp hxS
  have htop : (eigval hA i : ℝ) * ⟪x, x⟫_ℝ ≤ ray A x := by
    have := ray_ge_of_forall hA hinj1 c (eigval hA i) (fun m => by
      refine eigval_antitone hA ?_
      have hval : ((m.castLE (by omega) : Fin D) : ℕ) ≤ (i : ℕ) := by
        simp only [Fin.val_castLE]; omega
      exact Fin.le_def.2 hval)
    rw [hc] at this
    exact this
  have hn : (0:ℝ) < ⟪x, x⟫_ℝ := by
    have h1 : 0 ≤ ⟪x, x⟫_ℝ := real_inner_self_nonneg (x := x)
    have h2 : ⟪x, x⟫_ℝ ≠ 0 := fun h => hx0 (inner_self_eq_zero.mp h)
    exact lt_of_le_of_ne h1 h2.symm
  have hle' : eigval hA.neg j ≤ -eigval hA i := by
    have h1 : eigval hA.neg j * ⟪x, x⟫_ℝ ≤ ray (-A) x := hle
    rw [ray_neg] at h1
    nlinarith [hn, htop]
  -- direction ≥:  λⱼ(−A) ≥ −λᵢ(A)
  -- cf_dual on −A at index j with S = bottom-(D−i) span of A (dim j+1)
  have hdimS2 : Module.finrank ℝ
      (Submodule.span ℝ (Set.range fun m : Fin (D - (i : ℕ)) =>
        eigvec hA ((Fin.natAdd i m).cast hlen))) = (j : ℕ) + 1 := by
    rw [finrank_span_eq_card
      (Orthonormal.linearIndependent ⟨fun m => eigvec_norm hA _,
        fun a b hab => eigvec_orth hA (fun hcon => hab (hinj2 hcon))⟩),
      Fintype.card_fin _]
    simp only [Fintype.card_fin, hj]
    omega
  obtain ⟨y, hyS, hy0, hle2⟩ := cf_dual (hA := hA.neg) j
    (Submodule.span ℝ (Set.range fun m : Fin (D - (i : ℕ)) =>
      eigvec hA ((Fin.natAdd i m).cast hlen))) hdimS2
  obtain ⟨c2, hc2⟩ := (Submodule.mem_span_range_iff_exists_fun ℝ).mp hyS
  have hbot : ray A y ≤ eigval hA i * ⟪y, y⟫_ℝ := by
    have := ray_le_of_forall hA hinj2 c2 (eigval hA i) (fun m => by
      refine eigval_antitone hA ?_
      have hval : (i : ℕ) ≤ (((Fin.natAdd i m).cast hlen : Fin D) : ℕ) := by
        simp only [Fin.val_cast, Fin.val_natAdd]; omega
      exact Fin.le_def.2 hval)
    rw [hc2] at this
    exact this
  have hn2 : (0:ℝ) < ⟪y, y⟫_ℝ := by
    have h1 : 0 ≤ ⟪y, y⟫_ℝ := real_inner_self_nonneg (x := y)
    have h2 : ⟪y, y⟫_ℝ ≠ 0 := fun h => hy0 (inner_self_eq_zero.mp h)
    exact lt_of_le_of_ne h1 h2.symm
  have hge' : -eigval hA i ≤ eigval hA.neg j := by
    have h2 : ray (-A) y ≤ eigval hA.neg j * ⟪y, y⟫_ℝ := hle2
    rw [ray_neg] at h2
    nlinarith [hn2, hbot]
  linarith [hle', hge']

end Loewner

end

end KatgptProof.Pencil
