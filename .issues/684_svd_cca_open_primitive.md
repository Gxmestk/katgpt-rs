# Issue 684: `svd_cca` open primitive — SVD-denoised CCA subspace similarity for katgpt-core

> **Research:** [katgpt-rs/.research/501_SVCCA_SVD_CCA_Subspace_Similarity.md](../.research/501_SVCCA_SVD_CCA_Subspace_Similarity.md)
> **Source paper:** [arXiv:1706.05806](https://arxiv.org/abs/1706.05806) — SVCCA (Raghu et al., NIPS 2017)
> **Training-side counterpart:** riir-train [Plan 349](../../riir-train/.plans/349_svcca_freeze_training_recipes.md)
> **Date:** 2026-08-24
> **Repo:** katgpt-rs — `crates/katgpt-core/src/data_probe/cca.rs` (new module, opt-in feature `svd_cca`)

## Why

The stack cannot answer *"are these two representation snapshots the same function, up to invertible linear re-mixing?"* BLAKE3/Merkle prove same **bytes**; `cka_linear` is orthogonal-invariance-only (single scalar, no denoise, no spectrum); PEIRA learns its alignment (~500 iterations). The SVCCA operator is closed-form over **already-shipped** linalg: `thin_svd_into` (`subspace_phase_gate.rs`), `ns_inv_sqrt_psd` (`newton_schulz.rs`), `symmetric_eig` (`linalg/`), numerical-rank η=0.99 (`phase_gate.rs` consumer precedent). ~200 LOC composition. Consumers waiting on it: riir-neuron-db `can_freeze` v2 (comparative convergence gate), riir-ai hot-swap semantic-equivalence gate, cross-NPC belief alignment (KG triples), riir-train dist_guard cross-time monitor (Plan 349 T1).

## Scope

```rust
pub struct CcaReport { pub rho: [f32; MAX_K], pub mean_rho: f32,
                       pub kx: usize, pub ky: usize, pub degenerate: bool }
pub struct CcaScratch { /* caller-owned, zero-alloc steady state */ }
pub fn svcca_into(x: &[f32], y: &[f32], dx: usize, dy: usize, n: usize,
                  var_keep: f32, ridge: f32, s: &mut CcaScratch) -> CcaReport;
```

Pipeline: column-center → thin SVD → keep numerical_rank(η=0.99) → ridge above noise floor (Batch-54 rule, coupled to damping cap, `!is_finite() ||` guard) → `ns_inv_sqrt_psd` whiten → M = Wx·Cxy·Cy⁻¹·Cxyᵀ·Wx → `symmetric_eig` → ρᵢ = √clamp(λᵢ,0,1), ρ̄ = mean. `kx == 0` → `degenerate` (the collapse signal, not an error path). Fixed iteration counts everywhere → bit-stable. Not UQ-bearing — the conformal-floor rule does not bind.

## Tasks

- [ ] **T1** Module `data_probe/cca.rs` + `CcaScratch` + `svcca_into` behind feature `svd_cca = []` (opt-in); d ∈ {8,16,32,64}, n ∈ {16..256}; sample-space (n×n) solve variant noted for d > 256 (Jacobi ceiling), not implemented.
- [ ] **T2** G1 correctness: synthetic recovery (Y = AX + ε with known ρ profile, assert recovered); invariance bit-stability `svcca(X,Y) == svcca(X, ΠY) == svcca(X, cY) == svcca(X, AY)` for well-conditioned invertible A; degenerate/rank-deficient inputs → no NaN, `degenerate` flag; SVD-before-CCA pathology fixture (50-aligned+150-noise vs 50-aligned+150-useful must NOT both read as identical — naive CCA does).
- [ ] **T3** G2 perf bench: p50 < 25 µs @ 32×32, n=128, release-only gate.
- [ ] **T4** G4 alloc: counting-allocator zero steady-state across repeat calls (scratch reuse).
- [ ] **T5** G3 no-regression: default features untouched; `cargo clippy` clean; doc example.
- [ ] **T6** Promotion decision after G1–G4: opt-in by default (no default consumer yet — the no-default-consumer rule; promote when the first consumer's GOAT passes).
- [ ] **T7** Consumer follow-ups (file in their repos, NOT here): ndb `can_freeze` v2 (`repr_converged` via ρ̄ pre/post consolidation, all sleep paths audited per the can_freeze lesson); riir-ai `LoRAHotSwap` equivalence gate + belief-alignment PoC (`riir-poc`, three competitors: paper-analog vs frozen-baseline vs shipped scalar-cosine analog); riir-train Plan 349 T1 consumes the primitive.
- [-] **T8** DFT block-diagonal variant (Theorem 1) — deferred until an equivariant-field consumer materializes (katgpt-dec cochains / heightfield LOD).

## GOAT gate

G1 invariance+recovery+pathology; G2 < 25 µs p50 (32×32, n=128, release); G3 default-features bit-untouched; G4 zero steady-state allocs. Dual-track note: modelless primitive here; the freeze-training *regime* lives in riir-train Plan 349.
