# Issue 678: Lean 4 package for the spectral pencil — sym-isometry, Weyl Lipschitz, Loewner monotonicity, constructive eigengap bound (Research 495)

> Source: [Research 495](../.research/495_Spectral_Neuron_Affine_Pencil_Shape_Gates.md) §3.1 P2 / §8 (arXiv:2608.08003). **Blocked on Issue 676** T1–T4 (the Rust substrate must exist before the paired `spec_match` tests can be written). All four theorems are static matrix algebra over ℝ — fits the FV doctrine (static invariants the runtime depends on; no dynamic properties). Public math, no game/chain/shard IP → `KatgptProof` instance.

## Theorems

- [ ] T1 **sym-isometry**: `‖sym(v)‖_F = ‖v‖₂` for the 1/√2 off-diagonal packing, and inner-product preservation `⟨sym(u), sym(v)⟩_F = ⟨u, v⟩`. One-liner once the packing is defined; load-bearing for every packed-dot consumer (attribution, genome similarity).
- [ ] T2 **Weyl 1-Lipschitz**: `|λk(A) − λk(B)| ≤ ‖A − B‖₂` via the Courant–Fischer min–max characterization. The load-bearing theorem behind the global feature-influence bound `|f(x+δ) − f(x)| ≤ Σ|δᵢ|·‖Aᵢ‖₂` and the Lipschitz tamper check.
- [ ] T3 **Loewner monotonicity**: `A ⪯ B ⇒ λk(A) ≤ λk(B)` — the shape DSL's soundness core (Aᵢ ⪰ 0 ⇒ monotone in xᵢ), plus mirror duality `λk(−A) = −λ_{d−k+1}(A)`.
- [ ] T4 **The constructive eigengap bound** (the strongest candidate — a *constructive* conditioning guarantee, machine-checked): for `A0 = Qᵀdiag(−1,…,−1, 0@k, 1,…,1)Q` and `Ai = αᵢI + diag(εᵢ)` with `|εᵢⱼ| ≤ 1/(4Rn)` and `‖x‖∞ ≤ R`, the eigengap `γk(A(x)) ≥ 1/2`. Proof shape: αᵢI is a pure spectral shift (gap-invariant); `‖Σxᵢ·diag(εᵢ)‖₂ ≤ 1/4`; Weyl moves each endpoint of the unit gap ≤ 1/4. (Independently re-derived by the Research 495 No-GD panel — matches paper Lemma 2.)

## Protocol (house FV doctrine)

- [ ] P1 Eigenvalues definable via Courant–Fischer characterization in Mathlib (or core `ιn`-style); keep the instance Mathlib-free if the characterizations are decidable in core (mirror the RiirChainProof pattern), else Mathlib-required (RiirAiProof pattern) — decide at implementation.
- [ ] P2 Paired `SpecTests.lean` — concrete instances with independently-known values (e.g. the 2×2 sym-packing norm identity by hand; a 3×3 ladder pencil's gap at x=0 computed exactly).
- [ ] P3 Paired Rust `spec_match` tests against Issue 676's kernel (bit-level agreement on finite samples).
- [ ] P4 `#print axioms` budget `{propext, Classical.choice, Quot.sound}` only; bump `EXPECTED_THEOREMS`; add perturbations to the negative-test script (each theorem's spec test must FAIL under a deliberate spec error — e.g. a 1/√2 → 1/2 packing typo, or ε-bound 1/(4Rn) → 1/(2Rn) which breaks the gap).
- [ ] P5 Zero `sorry` — `lake build` warns but exits 0; the gate script is the enforcement.

## Non-goals

No dynamic/runtime properties (training convergence, behavioral claims). No interior-k universality (existence theorem, not code). KatgptProof only — the riir-ai/riir-chain/riir-neuron-db instances are unaffected unless a private composition invariant emerges from Issue 736.
