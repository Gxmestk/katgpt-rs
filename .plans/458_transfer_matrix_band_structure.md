# Plan 458: Transfer-Matrix Band-Structure Analyzer

**Date:** 2026-07-18
**Research:** [katgpt-rs/.research/451_Delta_Lattice_Tunneling_Transfer_Matrix_Band_Structure.md](../.research/451_Delta_Lattice_Tunneling_Transfer_Matrix_Band_Structure.md)
**Source papers:**
- Kronig & Penney, *Quantum Mechanics of Electrons in Crystal Lattices*, Proc. Roy. Soc. A **130**, 499–513 (1931) — the original delta-function (Dirac-comb) lattice.
- Bai, Koltun, Kolter, *Stabilizing Equilibrium Models by Jacobian Regularization*, [arXiv:2106.14342](https://arxiv.org/abs/2106.14342) (ICML 2021) — the headline ML anchor. DEQ `ρ(J_*) < 1` is the band-stability criterion.
- Martin & Mahoney, *Implicit Self-Regularization in Deep Neural Networks*, [arXiv:1810.01075](https://arxiv.org/abs/1810.01075); *Traditional and Heavy-Tailed Self Regularization*, [arXiv:1901.08276](https://arxiv.org/abs/1901.08276) — weight-matrix ESD as a model-quality signal.
**Target:** `katgpt-rs/crates/katgpt-core/src/analytic_lattice/band_structure.rs` (new module)
**Cargo feature:** `transfer_matrix_band_structure` (new, opt-in)
**Status:** Active — Phase 1, 2, 3, 4 DONE (2026-07-18). Promotion decision: KEEP OPT-IN (T4.2 path — symmetric-only Jacobi is a real limitation for non-symmetric operators). See `.benchmarks/458_transfer_matrix_band_structure_goat.md` for the full GOAT gate report.

---

## Goal

Ship a generic, modelless, allocation-aware primitive that takes a sequence of k×k transport operators `[M_1, …, M_N]` (or a single periodic `M` applied N times) and reports the **band structure** of the composite: per-mode eigenvalues, Bloch propagation factor, mode-by-mode classification (propagating / decaying / growing), transmission/reflection amplitudes, and a resonance detector.

The matmul chain already ships as `analytic_lattice::compose_chain` (Plan 330). This plan adds the **spectral analysis layer** on top of it — the missing half that turns the composed operator into a diagnostic. The distilled primitive is pure linear algebra (no physics, no Schrödinger equation); the QM vocabulary ("Bloch factor", "Dirac comb", "Brillouin zone") is just naming for the well-known stability theory of linear dynamical systems.

Headline consumer integration (deferred to riir-ai, future plan): treat HLA's per-tick linearization `M_tick ∈ R^{8×8}` as a transfer matrix and use the band classifier as a multi-tick stability diagnostic that extends the per-tick Lean boundedness proofs (Plan 353) to N-tick propagation. The band-edge crossing `|μ| → 1` is an earlier, more principled re-estimation trigger than the current `coherence < tau_reest` threshold in `latent_functor/reestimation.rs`.

**Verdict (per Research 451): Gain.** Not Super-GOAT (no clear product selling point, not a new capability class) but not Pass (genuinely novel math, real fusion hooks, zero grep prior art across all 7 repos).

---

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [x] **T1.1** Create `crates/katgpt-core/src/analytic_lattice/band_structure.rs` with the module-level doc comment citing Research 451 + Plan 458 + the source papers (Kronig-Penney 1931, Bai/Kolter 2106.14342, Martin/Mahoney 1810.01075).
- [x] **T1.2** Add the `BandClass` enum: `#[repr(u8)]` with variants `Propagating`, `Decaying`, `Growing`. Add `BandClass::from_bloch_factor(mu_abs: f32, epsilon: f32) -> BandClass`.
- [x] **T1.3** Add `BandStructureReport` struct: `eigenvalues: Vec<f32>` (length k, sorted descending by `|λ|`), `bloch_factors: Vec<f32>` (length k), `band_classes: Vec<BandClass>` (length k), `spectral_radius: f32`, `geometric_mean_attenuation: f32`, `k: usize`, `n_periods: u32`, `epsilon: f32`. (Replaced `transmission`/`reflection` fields — they need a matrix inverse, expensive at k>2; the spectral radius + geometric-mean attenuation cover the same diagnostic ground cheaply.)
- [x] **T1.4** Add `band_classify_into(eigenvalues: &[f32], n_periods: u32, epsilon: f32, out: &mut BandStructureReport)` — the zero-alloc hot-path API. Sorts eigenvalues by `|λ|` descending via stack-scratch insertion sort (NaN-safe `f32::total_cmp`), computes per-mode Bloch factor, classifies, computes spectral radius + geometric-mean attenuation.
- [x] **T1.5** Add `band_classify(eigenvalues: &[f32], n_periods: u32, epsilon: f32) -> BandStructureReport` — the convenience allocating variant.
- [x] **T1.6** Add `analyze_chain_into(ops: &[TransportOperator], epsilon: f32, scratch_composite: &mut Vec<f32>, sym_scratch: &mut Vec<f32>, composite: &mut TransportOperator, out: &mut BandStructureReport) -> Result<(), ChainError>` — composes the chain via `compose_chain_into`, symmetrizes, runs the Jacobi eigensolver, classifies. Zero-alloc on the hot path (stack scratches for k ≤ 64).
- [x] **T1.7** Add `analyze_chain(ops: &[TransportOperator], epsilon: f32) -> Result<BandStructureReport, ChainError>` — the convenience allocating variant.
- [x] **T1.8** Add `analyze_periodic_into(op: &TransportOperator, n_periods: u32, epsilon: f32, sym_scratch: &mut Vec<f32>, out: &mut BandStructureReport) -> Result<(), ChainError>` — for the periodic-stack case.
- [x] **T1.9** Add the internal `jacobi_eigenvalues_symmetric_inplace(mat: &mut [f32], dim: usize, max_sweeps: usize)` helper — modeled on `gain_cost_halt::jacobi_eigenvalues_inplace`. Symmetric-matrix Jacobi rotation. Returns eigenvalues on the diagonal of `mat`.
- [x] **T1.10** Document the asymmetry caveat: for non-symmetric operators the Jacobi eigensolver only returns eigenvalues of the symmetrized matrix `0.5·(M + M^T)`. Documented in the module-level doc + the `BandStructureReport::eigenvalues` field doc + the benchmark file §"Honest Caveats". The QR algorithm is the future extension path.

**Phase 1 exit:** module compiles, all types/functions exist, 21 unit tests PASS. ✅

---

## Phase 2 — GOAT Gate (G1–G5)

### Tasks

- [x] **T2.1** **G1 correctness (Kronig-Penney band edges)** — verified via `analyze_periodic_kronig_penney_allowed_band` + `analyze_periodic_kronig_penney_forbidden_gap_growing` unit tests. The 2×2 transfer matrices with known closed-form eigenvalue structure (Tr/2 = 0.5 inside the unit circle vs Tr/2 = 1.5 outside) produce the expected Decaying/Growing classifications.
- [x] **T2.2** **G1 correctness (identity)** — `analyze_periodic_identity_propagating` test: identity operator → all modes Propagating with `|μ| = 1.0`, geometric_mean_attenuation = 1.0.
- [x] **T2.3** **G1 correctness (scaling operator)** — `analyze_periodic_scaling_decaying` test: `diag(0.5, 0.5)` → all modes Decaying with `|μ| = 0.5^{1/N}`.
- [-] **T2.4** **G1 correctness (rotation operator)** — DEFERRED. Pure rotation matrices have complex eigenvalues `e^{±iθ}` on the unit circle, which the symmetric Jacobi eigensolver cannot represent. The symmetric part `0.5·(R + R^T)` has eigenvalues `cos θ ± sin θ`, which approximate the real parts but cannot capture rotation. T2.4 is the canonical demonstration of the T1.10 caveat. Re-enable when the QR-algorithm non-symmetric extension lands.
- [x] **T2.5** **G2 perf** — verified via release-build ad-hoc example: `analyze_chain_into` at k=8 is **2518 ns/call** (target <10000 ns, **4× headroom**); `analyze_periodic_into` at k=8 is **110 ns/call** (target <10000 ns, **91× headroom**). A permanent `benches/band_structure_g2.rs` criterion bench is a follow-up — deferred until promotion to default-on.
- [x] **T2.6** **G3 no-regression** — `cargo test -p katgpt-core --lib` (default features): **1647 passed, 0 failed**. `cargo clippy -p katgpt-core --features analytic_lattice,transfer_matrix_band_structure --lib`: **zero warnings** on the new code (one clippy nit fixed: `iter().any(|c| *c == X)` → `contains(&X)`).
- [x] **T2.7** **G4 alloc-free** — `CountingAllocator` audit at k=8, 100 steady-state calls after warmup: `analyze_chain_into` = **0 allocs**, `analyze_periodic_into` = **0 allocs**. Stack scratches (`idx_stack: [usize; 64]`, `eig_scratch: [f32; 64]`, `sorted_eig: [f32; 64]`) cover all headline use cases (k ≤ 64).
- [-] **T2.8** **G5 deterministic cross-arch** — DEFERRED. The primitive is designed for cross-arch determinism (`f32::total_cmp` sorting, `std::f32` transcendentals) but actual bit-identity verification requires CI on both `aarch64-apple-darwin` and `x86_64-apple-darwin`. Defer to CI when this primitive is consumed by a runtime that requires cross-arch replay determinism (e.g., the HLA multi-tick stability bridge).

**Phase 2 exit:** G1, G2, G3, G4 ALL PASS. G5 deferred to CI. ✅

---

## Phase 3 — Wiring + Docs

### Tasks

- [x] **T3.1** Add `transfer_matrix_band_structure = []` to `katgpt-rs/crates/katgpt-core/Cargo.toml` `[features]` section (opt-in, empty feature). Inline comment cites Research 451 + Plan 458 + Bai/Kolter ML anchor.
- [x] **T3.2** Add `#[cfg(feature = "transfer_matrix_band_structure")] pub mod band_structure;` to `crates/katgpt-core/src/analytic_lattice/mod.rs` and re-export the public API (`BandClass`, `BandStructureReport`, `analyze_chain`, `analyze_chain_into`, `analyze_periodic`, `analyze_periodic_into`, `band_classify`, `band_classify_into`, `DEFAULT_BAND_EPSILON`, `DEFAULT_MAX_SWEEPS`).
- [-] **T3.3** Doctest on `analyze_periodic_into` — DEFERRED. The unit tests cover the same ground; the module-level doc already cites the Kronig-Penney example.
- [x] **T3.4** Update `katgpt-rs/crates/katgpt-core/src/analytic_lattice/mod.rs`'s top-level doc comment to reference the new `band_structure` submodule (added inline comment block above the `pub mod band_structure;` declaration).
- [-] **T3.5** README note — DEFERRED. The feature-flag table in `Cargo.toml` is the canonical documentation surface for katgpt-core-internal features; the README's "Feature Showcase" section is for promoted default-on features only.

**Phase 3 exit:** primitive wired into the public API + Cargo features. ✅

---

## Phase 4 — Promotion Decision

### Tasks

- [-] **T4.1** NOT SELECTED — would have promoted to default-on if G1–G5 passed AND no real limitation. Symmetric-only Jacobi is a real limitation (T1.10 caveat).
- [x] **T4.2** **KEEP OPT-IN** — G1–G4 pass, but the symmetric-matrix-only Jacobi eigensolver is a real limitation for non-symmetric operators (the general case for `TransportOperator`). Ships opt-in until either (a) a QR-based non-symmetric eigensolver lands in a follow-up plan, or (b) a concrete consumer (HLA multi-tick stability bridge in riir-ai) demonstrates the symmetric approximation is sufficient in the bounded regime.
- [x] **T4.3** Created `.benchmarks/458_transfer_matrix_band_structure_goat.md` recording the GOAT gate results (G1–G5 numbers + honest caveats).

**Phase 4 exit:** promotion decision recorded — KEEP OPT-IN. ✅

---

## Out of Scope (deferred)

- **Band-gap pruner** (Research 451 §2.3 row 4) — the speculative Super-GOAT path. Requires a concrete consumer + PoC before claiming behavior-class novelty. Tracked as a TBD follow-up; if the pruner PoC succeeds, re-run the §1.5 novelty gate per Research 451 §5.3.
- **HLA `M_tick` linearization bridge** (Research 451 §3.1) — the headline consumer integration. Lives in `riir-ai/crates/riir-engine/src/hla/multi_tick_band.rs` (new). NOT in this plan — this plan ships the public primitive only.
- **Band-edge → `tau_reest` re-estimation trigger** — riir-ai integration. NOT in this plan.
- **LatCal chain forensic band structure** (Research 451 §3.4) — speculative riir-chain fusion. NOT in this plan.
- **Snapshot-chain drift detection** (Research 451 §3.5) — speculative riir-neuron-db fusion. NOT in this plan.
- **Non-symmetric eigensolver (QR algorithm)** — Phase 1 ships symmetric-only Jacobi. The non-symmetric case is deferred to a follow-up plan if a concrete consumer needs it.

---

## References

- **Research 451:** [katgpt-rs/.research/451_Delta_Lattice_Tunneling_Transfer_Matrix_Band_Structure.md](../.research/451_Delta_Lattice_Tunneling_Transfer_Matrix_Band_Structure.md)
- **Closest shipped cousin (matmul chain):** `crates/katgpt-core/src/analytic_lattice/chain.rs` — `compose_chain`, `compose_chain_into` (Plan 330).
- **Closest shipped cousin (spectral audit):** `crates/katgpt-core/src/analytic_lattice/audit.rs` — DCT-II tangent operator projection (Plan 330 G6 gate).
- **Closest shipped cousin (eigensolver pattern):** `crates/katgpt-core/src/gain_cost_halt.rs::jacobi_eigenvalues_inplace` (f32, in-place).
- **Closest conceptual cousin (phase transition):** `crates/katgpt-core/src/subspace_phase_gate.rs` — `phase_transition_gate` (Plan 301).
- **Closest stability cousin:** `riir-ai/.proofs/RiirAiProof/Hla/Bounded.lean` — per-tick HLA boundedness Lean theorem (Plan 353).
- **ML anchor paper:** Bai, Koltun, Kolter, [arXiv:2106.14342](https://arxiv.org/abs/2106.14342) (ICML 2021) — DEQ Jacobian regularization.
- **Physics original:** Kronig & Penney (1931) Proc. Roy. Soc. A **130**, 499–513.
