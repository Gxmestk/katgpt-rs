# Issue 678: Lean 4 package for the spectral pencil — sym-isometry, Weyl Lipschitz, Loewner monotonicity, constructive eigengap bound (Research 495)

> Source: [Research 495](../.research/495_Spectral_Neuron_Affine_Pencil_Shape_Gates.md) §3.1 P2 / §8 (arXiv:2608.08003). Blocked on Issue 676 T1–T4 (the Rust substrate must exist before the paired `spec_match` tests can be written) — **UNBLOCKED 2026-08-22** (676 complete, Bench 671). All four theorems are static matrix algebra over ℝ — fits the FV doctrine (static invariants the runtime depends on; no dynamic properties). Public math, no game/chain/shard IP → `KatgptProof` instance.

## Status (2026-08-22)

**Core theorems T1–T3 COMPLETE + T4 analytic core complete; ladder-value pinning deferred.** Gate green at 35 audited theorems (17→35); negative tests 4/4; Rust spec_match 5/5.

Mathlib ships the spectral theorem for Hermitian matrices (`LinearMap.IsSymmetric.eigenvalues`, antitone) but **neither Courant–Fischer nor Weyl** — both were built here from the spectral theorem (the CF sandwiches `cf_ge`/`cf_dual` are the load-bearing core; everything else composes).

## Theorems

- [x] T1 **sym-isometry**: `‖sym(v)‖_F = ‖v‖₂` for the 1/√2 off-diagonal packing, and inner-product preservation `⟨sym(u), sym(v)⟩_F = ⟨u, v⟩`. (`Pencil/Sym.lean` — the mirror-pairing lemma is the arithmetic heart: it cancels the 2 in `2·((√2)⁻¹)² = 1`.)
- [x] T2 **Weyl 1-Lipschitz**: `|λk(A) − λk(B)| ≤ ‖A − B‖₂`. (`Pencil/Weyl.lean` — one-sided via CF-ge on the bottom eigenvector span + the top Rayleigh bound; two-sided by applying both directions; `eigval_zero_le_opNorm` via Cauchy–Schwarz + Mathlib's scoped `L2Operator` norm.)
- [x] T3 **Loewner monotonicity**: `A ⪯ B ⇒ λk(A) ≤ λk(B)` (`Pencil/Loewner.lean::loewner_mono`, via Weyl one-sided + PSD Rayleigh nonneg from `PosSemidef.dotProduct_mulVec_nonneg`) + mirror duality `λk(−A) = −λ_{d−k−1}(A)` (`mirror_dual`, via both CF sandwiches on the reflected matrix).
- [-] T4 **The constructive eigengap bound**: **analytic core COMPLETE** (`Pencil/Eigengap.lean::eigengap_ge_half` — a unit gap survives a scalar shift + `‖E‖ ≤ 1/4` with `≥ 1/2`; shift lemma `eigval_add_smul_one` = gap-invariance of `c·I`, both Weyl directions composed) — **ladder-value pinning deferred**: the remaining piece is `λ_{k₀}(ladder) = 0 ∧ λ_{k₀+1}(ladder) = −1` for `A₀ = diag(−1,…,0@k,…,1)`, which needs concrete eigenvalue pinning of a given matrix (CF on coordinate-subspace spans — the standard-basis singles are eigenvectors of a diagonal). Same machinery as the deferred concrete-instance spec tests below. **The paper's Lemma 2 argument is otherwise fully covered** (shift ✓, Weyl-endpoint-move ✓, diagonal opNorm ≤ sup|entries| ✓ via `opNorm_diagonal_le`).

## Protocol (house FV doctrine)

- [x] P1 Eigenvalues via Mathlib's `LinearMap.IsSymmetric.eigenvalues` instantiated at `finrank = D` (`Fin D`-indexed, antitone). NOTE: the matrix-level `Matrix.IsHermitian.eigenvalues` reindexes through an opaque `Fintype.equivOfCardEq` — its antitone-ness is NOT usable; the LinearMap instantiation at `Fin D` is exact. Mathlib-required instance (RiirAiProof pattern).
- [-] P2 Paired `SpecTests.lean` — T1 hand instances COMPLETE (`!![1, 2√2; 2√2, 3]`, Frobenius square 18 — catches any packing typo). T2–T4 concrete instances deferred with the ladder pinning (same "concrete eigenvalue array of a diagonal matrix" dependency).
- [x] P3 Paired Rust `spec_match` tests: `crates/katgpt-core/tests/pencil_spec_match.rs` (5 tests, 5 pass) — T1 norm/dot isometry vs independent full-matrix computation (128 seeded sweeps each + the hand instance), T2 Weyl Lipschitz sampled over 64 seeded matrix pairs (exact Jacobi spectra + spectral norms), T4 shift gap-invariance (32 trials, every eigenvalue shifts by exactly c, gaps identical).
- [x] P4 Axiom budget `{propext, Classical.choice, Quot.sound}` only (35/35 within budget); `EXPECTED_THEOREMS` 17→35; negative tests `scripts/proof_negative_test.sh` — 4 perturbations (1/√2→1/2 packing typo, the hand-instance value 18→17, dropped mirror ×2, Weyl ≤→<), all 4 caught (build fails as required).
- [x] P5 Zero `sorry` — gate Layer 2 (build output + source scan) green.

## Non-goals

No dynamic/runtime properties (training convergence, behavioral claims). No interior-k universality (existence theorem, not code). KatgptProof only — the riir-ai/riir-chain/riir-neuron-db instances are unaffected unless a private composition invariant emerges from Issue 736.

## Remaining (deferred)

- Ladder-value pinning (`λ_{k₀}(ladder) = 0 ∧ λ_{k₀+1} = −1`) — CF on coordinate spans with standard-basis singles as the eigenvector family of a diagonal matrix; completes T4's final assembly `γ_k(A(x)) ≥ 1/2` from `eigengap_ge_half`. ~150 lines, mirrors the proven `cf_ge`/`cf_dual` structure.
- T2–T4 concrete-instance SpecTests (same dependency).
