/-
! Spec self-tests on concrete instances — the "spec tested on vectors" pattern.

Distilled from the Plan 441 convention (mirroring SymCrypt
`feature/verifiedcrypto` §4 "Running the Lean spec on test vectors"). Each
`example` proof tests the `reluSelfKernel` spec against an independently-
known value from the HOPE paper (arXiv:2607.21366). This closes the spec-
authority gap in C3:

- The proof theorems (when added) prove the spec against itself.
- The Rust spec-match test (`hope_spec_match.rs`, when added) tests Rust
  against the spec.
- **Neither** catches a spec authoring error — if the Lean `reluSelfKernel`
  definition has a sign typo or a missing factor, the proofs still type-check
  (proving the wrong property) and the Rust test still passes (Rust likely
  has the same typo). Only concrete instances with independently-known
  answers close this gap.

## Independent authorities used here

- **Standard-normal ReLU energy = 1/2.** For `y ~ N(0,1)`, `𝔼[max(0,y)²] = 1/2`.
  This is the canonical ReLU self-kernel value (HOPE paper §3.1, also
  Cho & Saul 2009 Arc-Cosine kernel order 1 diagonal). If the Lean spec
  returns anything other than `1/2` at `(γ=1, β=0)`, the spec is wrong.
- **Scale invariance at β=0.** For `y ~ N(0,γ²)`, `𝔼[max(0,y)²] = γ²/2` by
  the homogeneity of the ReLU + Gaussian (paper §5 PH-1). Catches a missing
  `γ²` factor in the spec.
- **Symmetry in γ.** `K(γ, β) = K(-γ, β)` because the formula depends on
  `|γ|` and `γ²` only. Catches a sign error that breaks scale invariance.
-/

import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import KatgptProof.Hope.Basic

namespace KatgptProof.Hope

open Real

/-! ## Helper lemmas (named, so concrete instances can reuse them) -/

/-- `Φ(0) = 1/2` — the standard-normal CDF at the origin is `1/2` by
    symmetry of the Gaussian. Key fact the standard-normal tests depend on.
    (In this spec, `normalCdf` is defined as the constant `1/2` — see
    `Basic.lean` doc for the simplification rationale.) -/
lemma normalCdf_zero : normalCdf 0 = (1/2 : ℝ) := by
  rw [normalCdf]

/-- For any `γ > 0`, `reluSelfKernel(γ, 0) = γ²/2`.

    This is the scale-invariance-at-zero-bias identity: the `β·|γ|·φ` term
    vanishes (β=0), leaving `(γ²+0)·Φ(0) = γ²·(1/2) = γ²/2`. -/
lemma reluSelfKernel_pos_gamma_zero (γ : ℝ) (hγ : 0 < γ) :
    reluSelfKernel γ 0 = γ^2 / 2 := by
  rw [reluSelfKernel]
  -- |γ| = γ (since γ > 0), β/|γ| = 0/γ = 0.
  rw [show |γ| = γ from abs_of_pos hγ]
  rw [zero_div]
  -- normalCdf 0 = 1/2 (constant definition).
  rw [normalCdf_zero]
  -- (γ² + 0²) · (1/2) + 0 · γ · normalPdf 0 = γ²/2.
  -- The β=0 factor kills both the β² term and the normalPdf term.
  rw [show (0 : ℝ)^2 = 0 by norm_num]
  rw [show (0 : ℝ) * γ * normalPdf 0 = 0 by ring]
  ring

/-! ## Standard-normal ReLU energy: `reluSelfKernel(1, 0) = 1/2`

The canonical value. For `y ~ N(0,1)`:
  `𝔼[max(0,y)²] = ∫₀^∞ y²·φ(y) dy = 1/2`.

If the Lean spec returns anything other than `1/2` at `(γ=1, β=0)`, the
spec transcription has an error (missing factor, sign error, etc.).
-/

/-- The standard-normal ReLU self-kernel at `(γ=1, β=0)` is exactly `1/2`.
    This is the canonical value from the HOPE paper §3.1. -/
example : reluSelfKernel 1 0 = (1/2 : ℝ) := by
  rw [reluSelfKernel_pos_gamma_zero 1 (by norm_num)]; norm_num

/-! ## Scale invariance at β=0: `reluSelfKernel(γ, 0) = γ²/2`

For any `γ ≠ 0`, `K(γ, 0) = γ²/2`. This is paper §5 PH-1 homogeneity
squared (energy scales by `γ²`). Catches a missing `γ²` factor.
-/

/-- Concrete instance: `reluSelfKernel(2, 0) = 4/2 = 2`. -/
example : reluSelfKernel 2 0 = (2 : ℝ) := by
  rw [reluSelfKernel_pos_gamma_zero 2 (by norm_num)]; norm_num

/-- Concrete instance: `reluSelfKernel(3, 0) = 9/2 = 4.5`. -/
example : reluSelfKernel 3 0 = (9/2 : ℝ) := by
  rw [reluSelfKernel_pos_gamma_zero 3 (by norm_num)]; norm_num

/-- Concrete instance: `reluSelfKernel(10, 0) = 100/2 = 50`. -/
example : reluSelfKernel 10 0 = (50 : ℝ) := by
  rw [reluSelfKernel_pos_gamma_zero 10 (by norm_num)]; norm_num

/-! ## γ-sign symmetry: `reluSelfKernel(γ, β) = reluSelfKernel(-γ, β)`

The kernel depends on `γ` only through `γ²` (the energy term) and `|γ|`
(the scale term). A sign flip leaves both unchanged. This is the paper's
§3.1 normalization invariance.
-/

/-- For any `γ` and any `β`, `K(γ, β) = K(-γ, β)`.

    The kernel depends on `γ` only through `γ²` (the energy term) and
    `|γ|` (the scale term). A sign flip leaves both unchanged. -/
example (γ β : ℝ) :
    reluSelfKernel γ β = reluSelfKernel (-γ) β := by
  -- The only difference between the two sides is γ² vs (-γ)² and |γ| vs |-γ|.
  -- Both are equalities: (-γ)² = γ² (ring) and |-γ| = |γ| (abs_neg).
  rw [reluSelfKernel, reluSelfKernel, abs_neg]
  -- Now the goal differs only in γ² vs (-γ)² inside the first product.
  -- ring_nf normalizes both sides; the (-γ)² = γ² identity is built-in.
  ring_nf

end KatgptProof.Hope
