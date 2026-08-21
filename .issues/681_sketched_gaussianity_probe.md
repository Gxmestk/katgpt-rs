# Issue 681: Sketched Gaussianity Probe — multi-direction projection-normality for embedding populations

> **Source:** Research 498 (LeVLJEPA arXiv:2607.00784 — SIGReg distilled from training loss to inference-time diagnostic)
> **Kind:** new primitive (POC → feature-gated module)
> **Repo:** katgpt-rs — `katgpt-core/src/data_probe/gaussianity.rs`, feature `gaussianity_probe` (opt-in until GOAT)

## Problem

Every shipped representation-health metric is **second-moment only**:
- `data_probe/geometry.rs::effective_rank` — entropy of covariance eigenvalues
- `avg_cosine_similarity` — pairwise normalized dots
- `within_class_effective_rank` — same spectral statistic on class residuals
- `riir-neuron-db/spectral_flatness.rs` — Wiener entropy (single vector)

A population can be **full-rank and pass all of these while its marginals are non-Gaussian**:
- bimodal mixture `½N(−μe,σI)+½N(+μe,σI)`: covariance σ²I+μ²eeᵀ → near-full erank, "healthy"; projection onto e is two-point-separated → any normality test rejects. (The "shard population that is two disjoint styles glued together" failure — exactly what a consolidation pipeline should catch before freezing.)
- heavy tails / outlier contamination (5% @ 10σ): outliers *inflate* eigenvalues — erank passes.
- discrete/quantized marginals (snap values): possible full-rank covariance, non-Gaussian in every direction.

Meanwhile a shipped assumption depends on exactly this unchecked property: `katgpt-band/src/band_conditioner.rs` L31-33 — Fisher-z "requires approximate Gaussianity of residuals" — no runtime guard.

Univariate KS-vs-Gaussian already ships (`katgpt-spectral/spectral.rs:837` `ks_d_statistic`, Plan 224/252/568 consumers) but only on **weights** and only **single-coordinate**. The new primitive is the Cramér-Wold sketch for **d-dimensional embedding populations**: project onto |A| fixed random directions, test each 1D marginal, aggregate.

## Design sketch

```rust
// data_probe/gaussianity.rs
pub struct GaussianityReport {
    pub score: f32,          // sigmoid-bounded aggregate ∈ [0,1] (band_conditioner precedent); 1 = Gaussian
    pub worst_direction: usize,
    pub per_direction: [f32; A],  // fixed A = 16
}

/// Project `states` (n × d, caller-owned flat buffer) onto BLAKE3-seeded
/// Rademacher directions (deterministic — the spectral.rs L820 generation
/// pattern), run a 1D normality statistic per direction, aggregate.
pub fn sketched_gaussianity(states: &[f32], d: usize, scratch: &mut [f32]) -> GaussianityReport;
```

- Directions: BLAKE3-derived Rademacher table (±1 entries — exact in f32, deterministic construction house pattern).
- Per-direction statistic: KS-vs-fitted-Gaussian on the B projected values (or Epps-Pulley — settle by G2 bench, EP is O(B²) vs KS O(B log B); the paper uses EP, our shipped precedent is KS).
- **Leaf constraint:** `katgpt-core` must not depend on `katgpt-spectral` (the rrq_quant scalar-inversion rule). Implement the 1D statistic self-contained in `data_probe`; the cross-crate agreement test lives in `katgpt-spectral` (which CAN see core): same 1D sample → core's per-direction score ≡ `ks_d_statistic` within fp tolerance.
- Zero-alloc: caller-owned scratch (sort buffer), the `ks_d_statistic` single-pass-scratch pattern.

## GOAT gate

- **G1 (falsifiable core):**
  - seeded isotropic Gaussian → score above threshold, bit-identical across 3 runs
  - bimodal mixture fixture → ≥ k of 16 directions reject
  - 5%-outlier Gaussian → reject
  - discrete/lattice fixture → reject
  - **non-redundancy pin**: `effective_rank` computed on the SAME fixtures (i)-(iv) must be HIGH (pins the blind-spot claim against the shipped metric — the `p415_g2_nonredundancy_vs_global` pattern applied to shape-vs-rank)
  - cross-crate agreement vs `katgpt_spectral::ks_d_statistic` on 1D projections
- **G2:** latency vs `effective_rank` at d=64 (erank pays O(d³) Jacobi sweep; the probe is O(B·d·|A| + |A|·B log B) — no eigensolve, should win; pin the ratio; audit-cadence amortization precedent `CachedSinkClassification` Plan 287)
- **G3:** default build untouched (feature-gated, like `sink_aware_attn` gates geometry)
- **G4:** zero steady-state allocs (CountingAllocator test)

## Consumers (why now)

1. `band_conditioner` Fisher-z precondition guard (turn the documented assumption into a checked precondition — advisory field, no gate-semantics change)
2. riir-ai #742 edge_lora hidden-space monitor (consumes this or erank)
3. riir-neuron-db freeze-gate advisory (`FreezeGateReport` additive field — the bimodal-two-styles-before-freeze case)

## Tasks

- [ ] T1: implement `data_probe/gaussianity.rs` behind `gaussianity_probe` feature
- [ ] T2: G1 fixtures + non-redundancy pin + cross-crate agreement test
- [ ] T3: G2 latency bench vs `effective_rank`
- [ ] T4: G4 alloc test; clippy clean both feature states
- [ ] T5: GOAT verdict + promote/demote decision (opt-in until a consumer promotes)
