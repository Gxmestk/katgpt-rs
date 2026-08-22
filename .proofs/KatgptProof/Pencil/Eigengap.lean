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

/-! ## Standard coordinate singles — the eigenvector family of a diagonal

A diagonal matrix acts on each standard single by scaling (`eucSingle_eigen`),
so the singles play for `diagonal d` exactly the role `eigvec` plays for a
general symmetric matrix — and the same combo machinery goes through. This
is the substrate for pinning the eigenvalue array of a diagonal matrix
(`eigval_diagonal_antitone`), which the ladder-value pinning rides on.
-/

section Singles

variable [DecidableEq (Fin D)]

/-- The standard basis single `e_j` in Euclidean space — an exact
eigenvector of every real diagonal matrix. -/
def eucSingle (j : Fin D) : EuclideanSpace ℝ (Fin D) := EuclideanSpace.single j 1

/-- Singles are unit. -/
theorem eucSingle_norm (j : Fin D) : ‖eucSingle j‖ = 1 :=
  EuclideanSpace.orthonormal_single.1 j

/-- Distinct singles are orthogonal. -/
theorem eucSingle_orth {i j : Fin D} (h : i ≠ j) : ⟪eucSingle i, eucSingle j⟫_ℝ = 0 :=
  EuclideanSpace.orthonormal_single.2 h

/-- Unit self-inner product of a single. -/
theorem eucSingle_inner_self (j : Fin D) : ⟪eucSingle j, eucSingle j⟫_ℝ = 1 := by
  rw [real_inner_self_eq_norm_sq, eucSingle_norm j, one_pow]

/-- Inner product of one single against an injectively-indexed combination
of singles picks out the matching coefficient. -/
theorem eucSingle_inner_combo {σ : Type*} [Fintype σ] {idx : σ → Fin D}
    (hinj : Function.Injective idx) (c : σ → ℝ) (m : σ) :
    ⟪eucSingle (idx m), ∑ k, c k • eucSingle (idx k)⟫_ℝ = c m := by
  rw [inner_sum]
  rw [Finset.sum_eq_single m]
  · rw [real_inner_smul_right, eucSingle_inner_self, mul_one]
  · intro k _ hkm
    rw [real_inner_smul_right, eucSingle_orth (fun hcon => hkm (hinj hcon.symm)),
      mul_zero]
  · intro h; exact absurd (Finset.mem_univ m) h

/-- Norm of an injectively-indexed combination of singles. -/
theorem inner_combo_eucSingle {σ : Type*} [Fintype σ] {idx : σ → Fin D}
    (hinj : Function.Injective idx) (c : σ → ℝ) :
    ⟪∑ m, c m • eucSingle (idx m), ∑ m, c m • eucSingle (idx m)⟫_ℝ = ∑ m, c m * c m := by
  rw [sum_inner]
  refine Finset.sum_congr rfl fun m _ => ?_
  rw [real_inner_smul_left, eucSingle_inner_combo hinj c m]

/-- Every diagonal matrix acts on the singles by scaling — the singles are
exact eigenvectors with the diagonal entries as eigenvalues. This is the
diagonal-matrix analogue of `eigvec_eigen`, and the combo lemmas below
mirror the eigenvector ones verbatim. -/
theorem eucSingle_eigen (d : Fin D → ℝ) (j : Fin D) :
    toEucL (Matrix.diagonal d) (eucSingle j) = d j • eucSingle j := by
  refine PiLp.ext fun l => ?_
  simp only [toEucL, Matrix.toLpLin_apply, WithLp.ofLp_toLp, WithLp.ofLp_smul,
    eucSingle, PiLp.single_apply, Matrix.mulVec_diagonal, Pi.smul_apply, smul_eq_mul]
  by_cases h : l = j
  · subst h; simp
  · rw [if_neg h]; ring

/-- Rayleigh numerator of an injectively-indexed combination of singles
under a diagonal: the entrywise weighted sum. -/
theorem ray_combo_diagonal (d : Fin D → ℝ) {σ : Type*} [Fintype σ] {idx : σ → Fin D}
    (hinj : Function.Injective idx) (c : σ → ℝ) :
    ray (Matrix.diagonal d) (∑ m, c m • eucSingle (idx m))
      = ∑ m, d (idx m) * (c m * c m) := by
  rw [ray, map_sum, sum_inner]
  refine Finset.sum_congr rfl fun m _ => ?_
  rw [map_smul, real_inner_smul_left, eucSingle_eigen d (idx m), real_inner_smul_left,
    eucSingle_inner_combo hinj c m]
  ring

end Singles

/-! ## The diagonal ladder's spectrum
-/


section Ladder

variable [DecidableEq (Fin D)]

/-- A real diagonal matrix is Hermitian. -/
theorem diagonal_isHermitian (d : Fin D → ℝ) :
    (Matrix.diagonal d).IsHermitian := by
  show (Matrix.diagonal d)ᴴ = Matrix.diagonal d
  apply Matrix.ext
  intro a b
  simp only [Matrix.conjTranspose_apply, Matrix.diagonal_apply, star_trivial]
  by_cases h : a = b
  · subst h; simp
  · rw [if_neg (fun hc => h hc.symm), if_neg h]

/-- **Eigenvalues of an antitone diagonal**: the antitone-sorted eigenvalue
array of a decreasing diagonal matrix is the diagonal itself. This is the
concrete-eigenvalue-pinning substrate — a diagonal's spectrum is its
diagonal (independently-known ground truth), and the sort is the identity
exactly when the diagonal is already decreasing. -/
theorem eigval_diagonal_antitone {d : Fin D → ℝ} (hd : Antitone d) (j : Fin D) :
    eigval (diagonal_isHermitian d) j = d j := by
  classical
  set hdiag := diagonal_isHermitian d with hh
  -- ── upper: eigval j ≤ d j, via cf_ge on the bottom-(D−j) coordinate span ──
  have hi : (j : ℕ) < D := j.is_lt
  have hlen : (j : ℕ) + (D - (j : ℕ)) = D := by omega
  have hinjU : Function.Injective
      (fun m : Fin (D - (j : ℕ)) => ((Fin.natAdd j m).cast hlen : Fin D)) :=
    (Fin.cast_injective hlen).comp (Fin.natAdd_injective (D - (j : ℕ)) j)
  have hornU : Orthonormal ℝ
      (fun m : Fin (D - (j : ℕ)) => eucSingle ((Fin.natAdd j m).cast hlen : Fin D)) :=
    EuclideanSpace.orthonormal_single.comp _ hinjU
  obtain ⟨x, hxS, hx0, hleU⟩ := cf_ge (hA := hdiag) j
    (Submodule.span ℝ (Set.range fun m : Fin (D - (j : ℕ)) =>
      eucSingle ((Fin.natAdd j m).cast hlen : Fin D))) (by
      rw [finrank_span_eq_card (Orthonormal.linearIndependent hornU),
        Fintype.card_fin _])
  obtain ⟨cU, hcU⟩ := (Submodule.mem_span_range_iff_exists_fun ℝ).mp hxS
  have hposU : (0:ℝ) < ⟪x, x⟫_ℝ := by
    have h1 : 0 ≤ ⟪x, x⟫_ℝ := real_inner_self_nonneg (x := x)
    have h2 : ⟪x, x⟫_ℝ ≠ 0 := fun h => hx0 (inner_self_eq_zero.mp h)
    exact lt_of_le_of_ne h1 h2.symm
  have hleU' : ray (Matrix.diagonal d) x ≤ d j * ⟪x, x⟫_ℝ := by
    have hray := ray_combo_diagonal d hinjU cU
    have hin := inner_combo_eucSingle hinjU cU
    rw [← hcU]
    rw [hray, hin, Finset.mul_sum]
    refine Finset.sum_le_sum fun m _ => ?_
    refine mul_le_mul_of_nonneg_right ?_ (mul_self_nonneg (cU m))
    refine hd ?_
    have hval : (j : ℕ) ≤ (((Fin.natAdd j m).cast hlen : Fin D) : ℕ) := by
      simp only [Fin.val_cast, Fin.val_natAdd]; omega
    exact Fin.le_def.2 hval
  have hup : eigval hdiag j ≤ d j := by
    have := hleU.trans hleU'
    nlinarith [hposU]
  -- ── lower: d j ≤ eigval j, via cf_dual on the top-(j+1) coordinate span ──
  have hinjL : Function.Injective
      (fun m : Fin ((j : ℕ) + 1) => (m.castLE (by omega) : Fin D)) :=
    (Fin.strictMono_castLE (by omega)).injective
  have hornL : Orthonormal ℝ
      (fun m : Fin ((j : ℕ) + 1) => eucSingle (m.castLE (by omega) : Fin D)) :=
    EuclideanSpace.orthonormal_single.comp _ hinjL
  obtain ⟨y, hyS, hy0, hleL⟩ := cf_dual (hA := hdiag) j
    (Submodule.span ℝ (Set.range fun m : Fin ((j : ℕ) + 1) =>
      eucSingle (m.castLE (by omega) : Fin D))) (by
      rw [finrank_span_eq_card (Orthonormal.linearIndependent hornL),
        Fintype.card_fin _])
  obtain ⟨cL, hcL⟩ := (Submodule.mem_span_range_iff_exists_fun ℝ).mp hyS
  have hposL : (0:ℝ) < ⟪y, y⟫_ℝ := by
    have h1 : 0 ≤ ⟪y, y⟫_ℝ := real_inner_self_nonneg (x := y)
    have h2 : ⟪y, y⟫_ℝ ≠ 0 := fun h => hy0 (inner_self_eq_zero.mp h)
    exact lt_of_le_of_ne h1 h2.symm
  have hgeL' : d j * ⟪y, y⟫_ℝ ≤ ray (Matrix.diagonal d) y := by
    have hray := ray_combo_diagonal d hinjL cL
    have hin := inner_combo_eucSingle hinjL cL
    rw [← hcL]
    rw [hray, hin, Finset.mul_sum]
    refine Finset.sum_le_sum fun m _ => ?_
    refine mul_le_mul_of_nonneg_right ?_ (mul_self_nonneg (cL m))
    refine hd ?_
    have hval : ((m.castLE (by omega) : Fin D) : ℕ) ≤ (j : ℕ) := by
      simp only [Fin.val_castLE]; omega
    exact Fin.le_def.2 hval
  have hdown : d j ≤ eigval hdiag j := by
    have := hgeL'.trans hleL
    nlinarith [hposL]
  exact le_antisymm hup hdown

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
theorem eigengap_ge_half {A₀ : Matrix (Fin D) (Fin D) ℝ} {s : ℝ}
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

/-! ## The ladder itself and the T4 final assembly
-/

section LadderGap

/-- The decreasing ladder: `1` below `k`, `0` at `k`, `-1` above `k` —
the paper's `A₀ = diag(−1,…,0@k,…,1)` read in decreasing order. For the
Rust substrate's increasing diagonal (`init.rs`: `-1` at `i < k, 0@k, 1
at i > k`) the spectra agree with the zero at antitone index `k₀ = D−1−k`.
-/
def ladderDn (k : Fin D) : Fin D → ℝ :=
  fun j => if (j : ℕ) < (k : ℕ) then 1 else if (j : ℕ) = (k : ℕ) then 0 else -1

/-- The decreasing ladder is antitone. -/
theorem ladderDn_antitone (k : Fin D) : Antitone (ladderDn k) := by
  intro a b hab
  have hab' : (a : ℕ) ≤ (b : ℕ) := Fin.le_def.1 hab
  have hle1 : ∀ j : Fin D, ladderDn k j ≤ 1 := by
    intro j; unfold ladderDn; split_ifs <;> norm_num
  have hle0 : ∀ j : Fin D, (k : ℕ) ≤ (j : ℕ) → ladderDn k j ≤ 0 := by
    intro j hj
    unfold ladderDn
    rw [if_neg (by omega : ¬(j : ℕ) < (k : ℕ))]
    split_ifs <;> norm_num
  have heq1 : ∀ j : Fin D, (k : ℕ) < (j : ℕ) → ladderDn k j = -1 := by
    intro j hj
    unfold ladderDn
    rw [if_neg (by omega : ¬(j : ℕ) < (k : ℕ)),
      if_neg (by omega : ¬(j : ℕ) = (k : ℕ))]
  rcases lt_trichotomy (a : ℕ) (k : ℕ) with ha | ha | ha
  · have hfa : ladderDn k a = 1 := by
      unfold ladderDn; rw [if_pos ha]
    rw [hfa]; exact hle1 b
  · have hfa : ladderDn k a = 0 := by
      unfold ladderDn; rw [if_neg (by omega : ¬(a : ℕ) < (k : ℕ)), if_pos ha]
    rw [hfa]; exact hle0 b (by omega)
  · have hfa : ladderDn k a = -1 := by
      unfold ladderDn
      rw [if_neg (by omega : ¬(a : ℕ) < (k : ℕ)), if_neg (by omega : ¬(a : ℕ) = (k : ℕ))]
    rw [hfa, heq1 b (by omega)]

/-- Hermitianity of the ladder matrix. -/
theorem ladderDn_isHermitian (k : Fin D) :
    (Matrix.diagonal (ladderDn k)).IsHermitian := diagonal_isHermitian _

/-- **Ladder-value pin (upper)**: the ladder's antitone eigenvalue at the
zero position `k` is exactly `0`. -/
theorem eigval_ladder_zero (k : Fin D) :
    eigval (ladderDn_isHermitian k) k = 0 := by
  rw [eigval_diagonal_antitone (ladderDn_antitone k) k]
  simp [ladderDn]

/-- **Ladder-value pin (lower)**: the antitone eigenvalue just below the
zero is exactly `−1`. -/
theorem eigval_ladder_next (k : Fin D) (hk : (k : ℕ) + 1 < D) :
    eigval (ladderDn_isHermitian k) ⟨(k : ℕ) + 1, hk⟩ = -1 := by
  rw [eigval_diagonal_antitone (ladderDn_antitone k) ⟨(k : ℕ) + 1, hk⟩]
  have hne : ¬((⟨(k : ℕ) + 1, hk⟩ : Fin D) : ℕ) = (k : ℕ) := by
    simp only [Fin.val_mk]
    omega
  unfold ladderDn
  rw [if_neg (by simp only [Fin.val_mk]; omega), if_neg hne]

/-- **The ladder's unit gap**: `λk − λk₊₁ = 1` exactly — the input gap
Lemma 2's perturbation argument erodes to `1/2`. -/
theorem ladder_unit_gap (k : Fin D) (hk : (k : ℕ) + 1 < D) :
    eigval (ladderDn_isHermitian k) k
      - eigval (ladderDn_isHermitian k) ⟨(k : ℕ) + 1, hk⟩ = 1 := by
  rw [eigval_ladder_zero k, eigval_ladder_next k hk]
  norm_num

/-- Hermitianity of the perturbed ladder pencil `A₀ + E + s·1`. -/
theorem ladder_perturbed_isHermitian (k : Fin D) {E : Matrix (Fin D) (Fin D) ℝ}
    (hE : E.IsHermitian) (s : ℝ) :
    (Matrix.diagonal (ladderDn k) + E + s • (1 : Matrix (Fin D) (Fin D) ℝ)).IsHermitian := by
  have hs : (s • (1 : Matrix (Fin D) (Fin D) ℝ)).IsHermitian := by
    show (s • (1 : Matrix (Fin D) (Fin D) ℝ))ᴴ = s • 1
    rw [conjTranspose_smul, conjTranspose_one]
    simp
  exact ((ladderDn_isHermitian k).add hE).add hs

/-- **The constructive eigengap bound (T4), final assembly**: the paper's
Lemma 2 for the ladder — the seeded pencil `A(x) = ladder + E + s·1` keeps
at least half the ladder's unit gap under any Hermitian perturbation of
spectral norm ≤ 1/4 (the scalar feature shift `s` erodes nothing; the
diagonal jitter erodes at most 2·‖E‖). -/
theorem eigengap_ladder_ge_half {E : Matrix (Fin D) (Fin D) ℝ} (s : ℝ)
    (k : Fin D) (hk : (k : ℕ) + 1 < D)
    (hE : E.IsHermitian) (hnE : ‖E‖ ≤ 1 / 4) :
    eigval (ladder_perturbed_isHermitian k hE s) k
      - eigval (ladder_perturbed_isHermitian k hE s) ⟨(k : ℕ) + 1, hk⟩ ≥ 1 / 2 := by
  exact eigengap_ge_half (ladderDn_isHermitian k) hE
    (ladder_perturbed_isHermitian k hE s) ((ladderDn_isHermitian k).add hE) k
    ⟨(k : ℕ) + 1, hk⟩ (ladder_unit_gap k hk) hnE

end LadderGap

end Ladder

end

end KatgptProof.Pencil
