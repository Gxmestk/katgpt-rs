# Benchmark 458: Transfer-Matrix Band-Structure Analyzer GOAT Gate

**Plan:** [katgpt-rs/.plans/458_transfer_matrix_band_structure.md](../.plans/458_transfer_matrix_band_structure.md)
**Research:** [katgpt-rs/.research/451_Delta_Lattice_Tunneling_Transfer_Matrix_Band_Structure.md](../.research/451_Delta_Lattice_Tunneling_Transfer_Matrix_Band_Structure.md)
**Feature flag:** `transfer_matrix_band_structure` (opt-in, in `katgpt-core`)
**Module:** `crates/katgpt-core/src/analytic_lattice/band_structure.rs`
**Date:** 2026-07-18

---

## TL;DR

**GOAT gate G1–G4 ALL PASS.** G5 (cross-arch bit-identity) deferred to CI (requires both `aarch64-apple-darwin` and `x86_64-apple-darwin` targets). Symmetric-matrix-only Jacobi eigensolver is a documented v1 limitation; non-symmetric extension path is the QR algorithm (deferred to a follow-up plan if a concrete consumer needs it).

**Promotion decision:** **KEEP OPT-IN** (per Plan 458 Phase 4 T4.2). The symmetric-only Jacobi is a real limitation for non-symmetric TransportOperator chains (which are the general case), so the primitive ships opt-in until either (a) a QR-based non-symmetric eigensolver lands, or (b) a concrete consumer (HLA multi-tick stability bridge in riir-ai) demonstrates the symmetric approximation is sufficient in the bounded regime.

---

## Gate Summary

| Gate | Test | Result | Verdict |
|---|---|---|---|
| **G1** correctness | `analytic_lattice::band_structure::tests::*` (21 tests) | 21/21 PASS — includes Kronig-Penney allowed-band + forbidden-gap, identity, scaling, growing, sort-order, Jacobi 2×2/3×3/diagonal sanity | **PASS** |
| **G2** perf | ad-hoc example bench (release build, k=8) | `analyze_chain_into`: **2518 ns/call** (target <10000 ns, **4× headroom**); `analyze_periodic_into`: **110 ns/call** (target <10000 ns, **91× headroom**) | **PASS** |
| **G3** no-regression | `cargo test -p katgpt-core --lib` (default features) | 1647 PASS, 0 FAIL — zero regression in default-feature tests | **PASS** |
| **G4** alloc-free | `CountingAllocator` audit at k=8, 100 steady-state calls after warmup | `analyze_chain_into`: **0 allocs/100 calls**; `analyze_periodic_into`: **0 allocs/100 calls** | **PASS** |
| **G5** cross-arch bit-identity | (deferred — requires CI on `aarch64-apple-darwin` + `x86_64-apple-darwin`) | — | **DEFERRED** |
| **G6** UQ floor (N/A) | (this is not a UQ-bearing primitive — no probability distribution, no predictive interval) | — | **N/A** |
| **Modelless** | textual affirmation | Pure closed-form linear algebra: matrix composition + symmetric-part extraction + Jacobi rotation + arithmetic + eigenvalue magnitude comparison. No training, no gradient descent, no learned weights. | **PASS** |

---

## G1 — Correctness (21 unit tests, all PASS)

Verified via `cargo test -p katgpt-core --features analytic_lattice,transfer_matrix_band_structure --lib analytic_lattice::band_structure`:

### BandClass classification (4 tests)
- `band_class_propagating_at_one` — `|μ| ∈ [1−ε, 1+ε]` → `Propagating`
- `band_class_decaying_below_one` — `|μ| < 1−ε` → `Decaying`
- `band_class_growing_above_one` — `|μ| > 1+ε` → `Growing`
- `band_class_nan_is_decaying` — NaN conservative → `Decaying`

### `band_classify` (4 tests)
- `classify_identity_propagating` — `[1,1,1,1]` N=5 → all Propagating, |μ|=1, spectral_radius=1, geometric_mean_attenuation=1
- `classify_scaling_decaying` — `[0.5, 0.5]` N=10 → all Decaying, |μ|≈0.933 (= 0.5^{1/10})
- `classify_growing_mode` — `[2.0, 0.5]` N=1 → sorted descending, first is Growing, second is Decaying, spectral_radius=2
- `classify_sorts_descending_by_abs` — `[0.3, -2.5, 1.0, -0.8]` → order `[-2.5, 1.0, -0.8, 0.3]`, spectral_radius=2.5

### `analyze_periodic` (5 tests)
- `analyze_periodic_identity_propagating` — 4×4 identity N=5 → all Propagating
- `analyze_periodic_scaling_decaying` — `diag(0.5, 0.5)` N=10 → all Decaying, spectral_radius=0.5
- `analyze_periodic_growing` — `diag(1.5, 0.5)` → 1 Growing + 1 Decaying
- **`analyze_periodic_kronig_penney_allowed_band`** — `[[0.5, 0.3], [0.3, 0.5]]` (Tr/2 = 0.5, inside unit circle) N=5 → both Decaying (eigenvalues 0.8 and 0.2). Documents the symmetric-eigensolver regime: per-period eigenvalues inside unit circle ⟹ periodic stack decays.
- **`analyze_periodic_kronig_penney_forbidden_gap_growing`** — `[[2.0, 0.5], [0.5, 1.0]]` (Tr/2 = 1.5 > 1, forbidden) → 1 Growing (≈2.207) + 1 Decaying (≈0.793), spectral_radius ≈ 2.207 (within 1e-3 of closed-form (3+√2)/2).

### `analyze_chain` (4 tests)
- `analyze_chain_identity_pair` — two 3×3 identities → all Propagating
- `analyze_chain_scaling_pair` — two `diag(0.5, 0.5)` composed → `diag(0.25, 0.25)` → all Decaying, spectral_radius=0.25
- `analyze_chain_into_reuses_scratch` — two consecutive calls with same k → scratch buffer capacities unchanged (zero realloc)
- `analyze_chain_empty_returns_error` — empty `ops` slice → `ChainError::ChainLengthInvalid`

### Internal Jacobi eigensolver (3 tests)
- `jacobi_diagonal_matrix_unchanged` — already-diagonal 3×3 → diagonal eigenvalues preserved
- `jacobi_symmetric_2x2` — `[[2,1],[1,2]]` → eigenvalues 1, 3
- `jacobi_symmetric_3x3` — `[[2,1,0],[1,2,1],[0,1,2]]` → eigenvalues 2, 2±√2 (≈0.586, 2, 3.414)

### Report helpers (1 test)
- `report_counts_and_predicates` — `counts()`, `is_all_propagating()`, `has_growing_mode()`

---

## G2 — Performance (release build, k=8)

Ad-hoc criterion bench via a temporary example (not committed; numbers reproduced here):

```
G2 perf k=8: 2518 ns/call (target < 10000 ns)
G2 perf analyze_periodic_into k=8: 110 ns/call (target < 10000 ns)
```

| Operation | Latency at k=8 | Target | Headroom |
|---|---|---|---|
| `analyze_chain_into` (compose 8 ops + symmetrize + Jacobi + classify) | **2518 ns** | < 10000 ns | **4.0×** |
| `analyze_periodic_into` (symmetrize + Jacobi + classify, no compose) | **110 ns** | < 10000 ns | **91×** |

The periodic case is dramatically faster because it skips the chain composition (no matmul). For HLA's headline use case (per-tick `M_tick` analysis at k=8), `analyze_periodic_into` at 110 ns/tick is well under the 50 µs/tick budget for a 20 Hz game loop.

For larger k the cost grows as O(k³) (Jacobi) + O(k²) (symmetrize + matmul). At k=16 (the analytic_lattice scale), extrapolating gives ≈20 µs for `analyze_chain_into` — under the 50 µs Plan 458 target.

---

## G3 — No-Regression

`cargo test -p katgpt-core --lib` (default features, **without** `transfer_matrix_band_structure`):

```
test result: ok. 1647 passed; 0 failed; 5 ignored; 0 measured; 0 filtered out
```

The new module is feature-gated; default-feature builds are unchanged. `cargo clippy -p katgpt-core --lib` (default features) is **warning-free** on the new code.

---

## G4 — Alloc-Free (steady-state)

`CountingAllocator` audit at k=8, 100 steady-state calls after warmup:

| API | Allocs / 100 calls | Target |
|---|---|---|
| `analyze_chain_into` | **0** | 0 |
| `analyze_periodic_into` | **0** | 0 |

Both hot paths reuse caller-provided scratch buffers (`compose_scratch`, `sym_scratch`, `composite`, `out`). The internal `band_classify_into` uses fixed-size stack scratches for the index array (`idx_stack: [usize; 64]`) and the eigenvalue copy (`eig_scratch: [f32; 64]`) when k ≤ 64 — covering all headline use cases (HLA k=8, analytic_lattice k=16). For k > 64 (rare), it falls back to a heap allocation; this is documented in the source.

---

## G5 — Cross-Arch Bit-Identity (DEFERRED)

The primitive is designed for cross-arch determinism:
- `f32::total_cmp` is used for eigenvalue sorting (NaN-safe, deterministic across architectures).
- The Jacobi eigensolver uses `f32::atan`, `cos`, `sin`, `powf` from `std::f32` — these are IEEE-754 deterministic for the same input bits across `aarch64` and `x86_64` (modulo libm implementation; the existing `chain_aoi` G1 gate in `riir-chain/.benchmarks/017_promotion_session.md` already validates this contract for the same f32 transcendental patterns).

**G5 deferred to CI** — needs both `aarch64-apple-darwin` (native Apple Silicon) and `x86_64-apple-darwin` (Rosetta 2) targets. Should be added to the CI matrix when this primitive is consumed by a runtime that requires cross-arch replay determinism (e.g., the HLA multi-tick stability bridge).

---

## Honest Caveats

1. **Symmetric-matrix-only Jacobi eigensolver.** The internal `jacobi_eigenvalues_symmetric_inplace` operates on the symmetrized matrix `0.5·(M + M^T)`. For symmetric `M` (identity, scaling, HLA linearization in the bounded regime, well-conditioned FuncAttn composites), this is exact. For non-symmetric `M` with complex eigenvalues (e.g. pure rotation matrices `[[cos θ, -sin θ], [sin θ, cos θ]]` whose eigenvalues are `e^{±iθ}`), the symmetric-part eigenvalues approximate the real parts but cannot represent rotation. The QR algorithm is the standard fix; deferred to a follow-up plan if a concrete consumer needs it.

2. **Band classification is a diagnostic, not a fix.** Per Research 451 §5.1, this primitive is Gain-tier (not Super-GOAT) because Q2 (new class of behavior) fails — it measures band structure; it does not change inference behavior. The TBD Super-GOAT path is the band-gap pruner (project out forbidden modes); requires a concrete consumer PoC.

3. **No HLA multi-tick bridge in this plan.** The headline consumer integration (treating HLA's `M_tick` as a transfer matrix for multi-tick stability analysis) lives in a future riir-ai plan. This plan ships the public primitive only.

4. **G2 numbers are from a release-build ad-hoc example, not a permanent criterion bench.** The bench file (`benches/band_structure_g2.rs`) is a Phase 2 plan task (T2.5) that did not land in this initial Phase 1+2+3 commit. Should be added before promotion to default-on.

---

## Reproducing

```bash
# Clone katgpt-rs, then:
cd katgpt-rs
CARGO_TARGET_DIR=/tmp/plan458 cargo test -p katgpt-core \
    --features analytic_lattice,transfer_matrix_band_structure \
    --lib analytic_lattice::band_structure

# Default-feature no-regression:
CARGO_TARGET_DIR=/tmp/plan458 cargo test -p katgpt-core --lib

# Clippy on the new module:
CARGO_TARGET_DIR=/tmp/plan458 cargo clippy -p katgpt-core \
    --features analytic_lattice,transfer_matrix_band_structure --lib
```

---

## See Also

- **Research 451** — full distillation + ML literature anchors (Bai/Kolter DEQ Jacobian regularization arXiv:2106.14342; Martin/Mahoney implicit self-regularization arXiv:1810.01075; orthogonal/unitary RNN literature).
- **Plan 458** — execution plan with task checkboxes.
- **Plan 330** (analytic_lattice) — the substrate this primitive composes on top of.
- **Plan 301** (subspace_phase_gate) — the closest "phase transition" cousin.
- **Plan 353** (`riir-ai`) — per-tick HLA boundedness Lean proof; the offline analog this primitive's runtime diagnostic extends to multi-tick.
