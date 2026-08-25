# Plan 579: Orthogonal Factorization Primitives (Issue 687)

> **Source:** Issue 687 (`.issues/687_orthogonal_factorization_primitives.md`) —
> Research 504 (arXiv:2608.20065 "Orthogonal JEPA") Path 0 modelless extraction.
> **Repo/crate:** katgpt-rs / `katgpt-core` (open primitives).
> **Feature:** `orthogonal_factorization = ["spectral_pencil"]` (opt-in;
> promotion requires GOAT + a consumer — the no-default-consumer rule).
> **Filed:** 2026-08-25

## Substrate check (substrate-first skill, 2026-08-25)

- Searched for (vocabulary-translated, 3+ variants each):
  - Gram-Schmidt / orthonormalize: `gram_schmidt`, `orthonormal`, `orthogonaliz`, `qr`, `householder`
  - Running variance: `welford`, `running_variance`, `online_variance`
  - Parseval / Hadamard: `parseval`, `hadamard`, `walsh`
  - Spectral norm / conditioning: `spectral_pencil`, `lambda_max`, `spectral_norm`, `norm_jacobi`, `norm_power_iter`
- Found:
  - Gram-Schmidt ships ONLY as test/bench fixtures (`cross_resolution.rs::tests::random_orthonormal`,
    `benches/bench_425` `gram_schmidt_basis`, `benches/bench_417` `random_orthonormal`,
    `katgpt-attn/funcattn_compose/spectral_pre_rotate.rs` tests) and one specialized
    k≤4 partial-correlation GS in `katgpt-band/band_conditioner.rs`. No production
    general primitive with reorthogonalization + defect score — the issue's own
    framing ("the shape to productionize").
  - Welford: multiple module-LOCAL single-stream copies (`karc/regime_gate.rs::imp::WelfordVariance`
    private, `katgpt-attn/chiaroscuro/regime.rs::WelfordVariance` public-but-downstream-crate,
    `hint_regret` three parallel accumulators, `curator.rs` inline). No shared core
    substrate; none is the per-coordinate (K·r parallel-stream) shape T2 needs.
  - Hadamard: `katgpt-kv/src/kvarn/hadamard.rs::hadamard_transform_inplace` — downstream
    crate (kv depends on core); core cannot consume. T3's optional `hadamard_factorize`
    builds the ±1/√d basis rows (a different artifact: a basis, not an in-place tile
    rotation) and stays in core per the issue.
  - Spectral norm: `spectral_pencil::bounds::{norm_jacobi_exact, norm_power_iter}` +
    `spectral_pencil::{DenseScratch, jacobi_eigen}` — EXISTS, exact, pinned-Jacobi.
  - Parseval: documented conceptually in `cross_resolution.rs` ("is exact (Parseval)")
    — no runtime check primitive anywhere.
- Decision: **build new** module `orthogonal_factorization` (the issue is the filed
  gap), **consuming** `spectral_pencil::{DenseScratch, jacobi_eigen}` for T4's
  head norms (substrate-first: consume, don't duplicate). The per-coordinate Welford
  is a new multi-stream shape; the existing scalar copies are noted as future
  consolidation candidates (out of scope here — they are private/module-local and
  downstream).
- Architectural rules checked: latent-space diagnostic/certificate primitives —
  semantic domain, local consumption, never on a sync surface (compliant with the
  Latent vs Raw rules); zero-alloc + gateable (bridge-pattern compliant); no
  softmax anywhere (hinge is `max(0, γ−σ)`, aggregates are means); modelless
  (closed-form linear algebra only).

## Feature skeleton refinement (documented deviation)

The issue sketches `orthogonal_factorization = []`. T4's conditioning certificate
is specified "via `spectral_pencil` (one constant matrix)" — consuming that
substrate (instead of duplicating a Jacobi eigensolver) requires the feature dep:

```toml
orthogonal_factorization = ["spectral_pencil"]
```

`spectral_pencil` itself implies `hebbian_kernel_memory` + `karc_forecaster`
(pre-existing chain, needed for its own compilation under no-default-features).
Opt-in cost only; verified with `--no-default-features --features
orthogonal_factorization` in G3.

## Tasks

- [x] T1 `orthonormalize_into<const D>(vectors, out, defect)` — twice-reorthogonalized modified
  Gram–Schmidt, in-place in `out`, zero scratch/alloc, f64 lane-pattern accumulators,
  fixed ascending iteration order; `defect` = the INPUT set's L_orth
  (`Σ_k(‖b_k‖²−1)² + Σ_{i<j}(b_i·b_j)²`, unit-block form of the paper's
  `Σ‖B_k^TB_k−I‖²_F + Σ_{i<j}‖B_i^TB_j‖²_F`) — the one-shot redundancy audit
  that fires on planted collisions; degenerate rows (residual ≤ 1e-6·‖original‖)
  zeroed + fire the defect. Plus `orthogonality_defect(vectors)` standalone
  (audit any set without orthogonalizing). DONE — 5 unit tests incl. closed-form
  exactness (9.0 / 1.0 / 0.5625) + rank-spill + duplicate zeroing.
- [x] T2 `FactorActivityScratch` (K·r parallel Welford streams, f64, Box once at
  construction) + `gamma_schedule(γ_min, c, n) = max(γ_min, c/√n)` (defaults
  `GAMMA_FAC_MIN = 0.25`, `GAMMA_SCHED_C = 1.0` ≈ √2× the Gaussian σ̂ sampling
  noise `σ/√(2n)` at unit σ) + `factor_activity_hinge(scratch, γ, per_coord)`
  → mean `(1/Kr) Σ max(0, γ−σ̂_{k,j})` + per-coordinate attribution + worst (k,j).
  n<2 ⇒ no claim (hinge 0, documented). DONE — matches two-pass reference within
  1e-5; dead channel fires bit-exact == γ.
- [x] T3 `parseval_energy_check(z, basis, coeffs)` (‖z‖² vs Σ_k(B_k·z)², relative
  residual, `PARSEVAL_TOL_REL`), `recompose_into` (Σ c_k·b_k), `kept_energy`
  (exact truncation certificate: dropped = total − kept by Parseval identity),
  `hadamard_factorize` (d=2ⁿ Walsh rows × 1/√d — exact/dyadic for d=4^m incl.
  d=64; integer-core cross-platform bit-identity). DONE — d=64 anchors EXACT
  (residual 0.0, recompose bit-exact, Gram == I); duplicate/incomplete bases
  caught; truncation identity pinned.
- [x] T4 `head_conditioning(heads, norms_out, scratch)` — per-head ‖W_k‖₂ =
  √λ_max(W_kᵀW_k) via `spectral_pencil::jacobi_eigen` on the f64-accumulated
  Gram (construction-time, zero alloc, caller buffers) → `ConditioningCert
  {sigma_max, worst_head, per_step_bound}` + `rollout_bound(per_step, T) =
  per_step^T`. Orthonormal B ⇒ κ(B)=1 by construction (paper's conditioning
  caveat void; heads remain certifiable) — documented in module docs. DONE —
  diag/rank-1/orthonormal-exact unit pins (diag → 4.0, rank-1 → ‖u‖‖v‖,
  Hadamard d=64 → EXACTLY 1.0).
- [x] T5 GOAT gate `benches/bench_687_orthogonal_factorization_goat.rs`
  (harness=false, repo bench convention) — ALL PASS (Bench 676): G1 determinism
  ×3 + all three dyadic anchors exactly true; G2 GS **4881 ns** < 5000 @ d=64/K=14
  (under load 4.3–6.6) + hinge **21 ns/sample** (Kr=64) / 8 ns (drive shape);
  G4 **0 allocs** (module lib test, TrackingAllocator); G8a planted pair —
  healthy defect 5.2e-15 vs planted 0.9957 (1.9e14×), GS output |cos| 1.38e-8,
  survivor unit-norm; G8b dead channel — hinge == γ BIT-EXACT at (3,5),
  worst_flat 29, healthy coords exactly 0.
- [x] T6 Bench doc `.benchmarks/676_orthogonal_factorization_goat.md` + verdict
  table + promote/demote decision recorded (**stays opt-in** — the
  no-default-consumer rule; consumer wiring is the issue's own Non-goal);
  G3 no-regression sweep (default 1913/0/7i, feature-on 1989/0/8i,
  `--no-default-features --features orthogonal_factorization` compiles, default
  clippy `-D warnings` clean); `.benchmarks/.highwater` → 676; Issue 687 marked
  resolved + removed per the noise rule.

## Non-goals (per Issue 687)

- Learned/data-adaptive bases + trained heads → riir-train Plan 351.
- Consumer wiring (riir-ai affect orthogonalization A/B — gameplay owner call;
  riir-neuron-db blend leakage gate) → separate issues AFTER this GOAT passes.
