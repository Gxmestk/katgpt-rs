# Issue 676: `spectral_pencil` open primitive — affine matrix pencil scalar gates (Research 495)

> Source: [Research 495](../.research/495_Spectral_Neuron_Affine_Pencil_Shape_Gates.md) — arXiv:2608.08003 "The Spectral Neuron" (Shtoff, TII 2026). Public-tier generic math (symmetric eigenvalues), no game/chain/shard IP. Sibling issues: riir-ai 736 (consumer PoC), riir-train 472 (training recipes).

## Why

The stack has rich extremal-eigenvalue machinery (`cp_hopfield::hermitian_top_eigenvector`, `beta_fitter`, `zone_manifold` deflation, `hla_eigenbasis`, DEC `hodge`) but ships NOTHING for: interior-k single-eigenvalue evaluation of a per-input affine pencil `f(x) = λk(A0 + ΣᵢxᵢAᵢ)`, shape-by-construction gates (PSD ⇒ monotone; k=d ⇒ convex), closed-form per-feature attribution (vᵀAᵢv), derived per-feature Lipschitz bounds (‖Aᵢ‖₂), or a seeded construction with a proven eigengap guarantee (γk ≥ ½). Verified zero prior art in-workspace (dual-vocab grep, Research 495 §2/§4). Consumer demand: riir-ai 736 (personality gates + Super-GOAT re-gate), CommittedFieldBlend certificate upgrade, hero attribution ledger integration.

## Scope (katgpt-core, opt-in feature `spectral_pencil = []`)

- [ ] T1 `sym` isometric packing: `#[repr(C)]` fixed-size packed layout, off-diag ×1/√2; `‖sym(v)‖_F == ‖v‖₂` and `⟨sym(u),sym(v)⟩_F == ⟨u,v⟩` property tests (1-ulp).
- [ ] T2 Dense small-d single-eigenvalue kernel (d ≤ 32): pinned cyclic Jacobi or fixed-iteration bisection, deterministic per-binary; caller-owned scratch, zero allocs.
- [ ] T3 Tridiagonal pencil family + Sturm-count bisection: any single eigenvalue ≈ 50·d ops; **exact integer count of eigenvalues below a threshold in O(d)** (LDLᵀ pivot-sign count, pinned zero-pivot convention); Sturm count == full-solve count on 10⁶ random tridiags.
- [ ] T4 Seeded init constructor: `A0 = Qᵀdiag(−1,…,0@k,…,1)Q` (Q from BLAKE3-seeded pinned QR-of-Gaussian, sign-fix convention), `Aᵢ = αᵢI + diag(εᵢ)`; property test γk ≥ ½ on ‖x‖∞ ≤ 5 box; non-commutativity certificate `‖[A0,Aᵢ]‖_F > 0`; PSD-diagonal variant via squareplus.
- [ ] T5 Bounds: per-feature global bound ‖Aᵢ‖₂ (power-iteration estimate OK; exact for rank-one); linear growth envelope; Loewner ordering laws + mirror duality `λk(−A) = −λ_{d−k+1}(A)` property tests.
- [ ] T6 Hellmann–Feynman attribution: `∂f/∂xᵢ = vᵀAᵢv` (packed-dot implementation); subdifferential interval `‖VᵀAᵢV‖₂` at repeated eigenvalues; attribution-vs-finite-difference gate (test-only FD, 10⁵ random simple-eigenvalue probes); attribution never exceeds ‖Aᵢ‖₂.
- [ ] T7 Shape DSL constructors: per-feature PSD/NSD (diag-squareplus + rank-one `βᵢdᵢdᵢᵀ` fast path over BLAKE3 direction vectors), k index selection; monotone sweeps + midpoint-convexity property tests; rank-one path == dense path bitwise.
- [ ] T8 Canonical gauge (orthogonal invariance): conjugate-by-A0-eigendecomposition canonicalization; f-invariance under random conjugations; canonical-bytes stability test (re-canonicalize → identical bytes) — prerequisite for BLAKE3 commitment in consumers.
- [ ] T9 Invertible monotone warp: `g(x) = λk(A + xB)`, `B = I + Σβᵢdᵢdᵢᵀ ≻ 0`; closed-form inverse `g⁻¹(z) = λ_{d−k+1}(B^{−1/2}(zI−A)B^{−1/2})`; round-trip property test 10⁵ random constructions.
- [ ] T10 Bench + GOAT gate (G1–G4 per Research 495 §8): ns/eval at d∈{8,16,32} dense vs tridiag; 10k NPC × 20 Hz headroom arithmetic printed; zero steady-state allocs (counting allocator); feature stays opt-in.
- [ ] T11 Doc: module `.md` with the cost table (8K vs 800 FLOPs), the γk ≥ ½ lemma statement + proof sketch, the "no library QR on committed paths" determinism policy.
- [ ] T12 UQ follow-through: eigengap-confidence (`σ(−γk/τ)`-class readouts) and monotone box→interval (`A(lo)⪯A(x)⪯A(hi)` ⇒ 2-solve certified interval) ship ONLY with a conformal-floor benchmark vs `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` (floor at `katgpt-core/src/conformal/floor_harness.rs`) — beat it or scope-limit per the KARC precedent.

## Non-goals

- No training (riir-train 472). No game semantics (riir-ai 736). No chain commitment (follow-up once a Glacial-rate consumer exists; integer Sturm predicates are the preferred quorum class — via riir-dapps if ever). No quality-vs-trained-claims (§3.6 untouched).
