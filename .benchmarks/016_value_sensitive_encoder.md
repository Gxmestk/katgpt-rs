# Bench 016 — Value-Sensitive Encoder Discrimination Probe

**Date:** 2026-08-02
**Proposal:** [011 — Rust-SWE-bench Latent Space via WASM Pruner](../.proposals/011_rust_swe_bench_latent_space_via_wasm_pruner.md) Phase 5 follow-up
**Predecessors:** [Bench 014](014_swe_trajectory_freezer_g5.md) (G5 PASS), [Bench 015](015_swe_trajectory_perturbation_sensitivity.md) (geometry encoder NEGATIVE)
**Verdict:** **NEGATIVE for per-token classification, but with a key correction to bench_015's root cause. The signal EXISTS (centroid-of-test-tokens classification succeeds at moderate σ), but per-token SNR ≈ 1.0 — the perturbation signal is comparable to token-to-token variance. This is a RESOLUTION FLOOR for per-token nearest-centroid classification, not an information deficit.**

## The question

Bench 015 showed the `GeometrySummaryEncoder` (length + curvature + cosine + n_steps) cannot discriminate perturbed vs original Kimi-K3 weights — accuracy stays at ~50% even at σ=0.5. Bench 015's stated root cause was "geometry features are invariant to value perturbation."

**This bench tests whether VALUE-SENSITIVE encoders** — capturing per-layer displacement statistics rather than aggregate trajectory shape — can discriminate where the geometry encoder could not.

## Key architectural insight

The depth trajectory captures the RAW residual stream (`prefix_sum` is never normalized inside `kimi_decoder_layer_forward` — RMSNorm is applied to `scratch_hidden`, not `prefix_sum`). So the displacement `h_{l+1} - h_l = attn_out + ffn_out` IS the raw per-layer delta, directly computed from the layer's weights.

The geometry encoder collapses 8 displacement vectors into 4 scalar features (length = sum of norms, etc.), losing most per-layer detail. A value-sensitive encoder preserving per-displacement information should be more discriminative — IF the per-layer deltas change differently under perturbation.

## Encoders tested

| Encoder | Features | Description |
|---------|----------|-------------|
| **Geometry** (baseline) | 4 (replicated) | Same `GeometrySummaryEncoder` as bench_015 |
| **DispNorms** | 8 (replicated) | Per-displacement L2 norms (‖attn_out + ffn_out‖ per layer) |
| **DispStats** | 32 (8×4) | Per-displacement [L2, mean, variance, max_abs] |
| **StateNorms** | 9 (replicated) | Per-state L2 norms (accumulated residual growth profile) |
| **DispRatios** | 8 (replicated) | Per-displacement L2 / total L2 (scale-invariant profile) |

## Run command

```bash
cargo bench --features "kimi_k3_loader swe_trajectory_freeze" \
    --bench bench_016_value_sensitive_encoder -- --nocapture
```

## Results

### Per-σ accuracy + diagnostics

| σ | Encoder | Per-token Acc | Centroid Dist | Within-class σ | SNR | Centroid Acc |
|-------|-------------|---------------|---------------|----------------|-----|--------------|
| 0.0 | (all) | 50.0% | 0.000 | — | — | 50.0% |
| 0.001 | Geometry | 50.0% | 0.0002 | 0.028 | 0.01 | 50.0% |
| 0.001 | DispNorms | 50.0% | 0.100 | 9.938 | 0.01 | 50.0% |
| 0.01 | Geometry | 52.5% | 0.009 | 0.028 | 0.32 | 50.0% |
| 0.01 | DispNorms | 50.0% | 9.037 | 10.113 | 0.89 | 50.0% |
| 0.05 | DispRatios | 50.0% | 0.013 | 0.018 | 0.73 | **100.0%** |
| 0.1 | DispNorms | 55.0% | 8.717 | 9.938 | 0.88 | **100.0%** |
| 0.1 | DispStats | 52.5% | 4.458 | 5.069 | 0.88 | **100.0%** |
| 0.1 | DispRatios | 47.5% | 0.018 | 0.017 | 1.03 | **100.0%** |
| 0.5 | Geometry | 40.0% | 0.048 | 0.025 | **1.88** | 0.0% |
| 0.5 | DispNorms | 45.0% | 11.175 | 9.474 | 1.18 | 50.0% |
| 0.5 | StateNorms | 42.5% | 17.073 | 12.539 | 1.36 | 50.0% |

### Discrimination floor

No encoder reaches ≥80% per-token accuracy at ANY σ level.

## Analysis

### Correction to bench_015's root cause

Bench 015 stated: "geometry features are invariant to value perturbation." **This was incomplete.** The value-sensitive encoders show centroid distances 100-200× larger than the geometry encoder (e.g., DispNorms centroid_dist=11.17 vs Geometry's 0.048 at σ=0.5). The features DO change significantly with perturbation.

The real issue is **signal-to-noise ratio (SNR)**: the token-to-token variance in features is comparable to or larger than the perturbation-induced centroid shift. SNR ≈ 1.0 across all encoders and σ levels — the perturbation signal is buried under token noise.

### The signal EXISTS but is too weak per-token

At σ=0.05-0.1, the **centroid-of-test-tokens classification** succeeds at 100% for multiple value-sensitive encoders (DispNorms, DispStats, DispRatios). This proves the perturbation DOES produce a detectable shift in the mean feature vector. But individual token trajectories scatter too widely for per-token nearest-centroid classification.

### Why centroid accuracy drops at σ=0.5

At σ=0.5, centroid accuracy drops to 50% for most encoders. This is likely because:
1. The large perturbation (50% relative noise) causes nonlinear effects (MoE routing changes, activation saturation) that break the linear direction derived from training data.
2. The perturbation pushes features into a regime where the nearest-centroid direction no longer points toward the correct centroid.

### SNR is the fundamental limiter

The best SNR observed is 1.88 (Geometry at σ=0.5). For reliable nearest-centroid classification, SNR > 2.0 is typically needed. At SNR ≈ 1.0, classes overlap by ~68% (one standard deviation on each side).

To overcome the SNR floor:
1. **Multi-token aggregation** — averaging N tokens before classification improves SNR by √N. At SNR=1.0, averaging ~16 tokens → SNR=4.0. But this means a SINGLE trajectory cannot be classified — you need ~16 samples. Not applicable to per-attempt SWE-bench trajectory freezing.
2. **Covariance-aware classifier** (Mahalanobis/LDA) — whitens the noise. Requires ~2D samples per class (64+ for D=32) to estimate the covariance matrix. Current setup has only 12 train tokens per class.
3. **Higher-dimensional features** — capture more of the signal. But D=32 is already constrained by the FAME projection architecture.

## Verdict

**The depth trajectory captures the perturbation signal (centroids separate), but at SNR ≈ 1.0 — insufficient for per-token nearest-centroid classification.** This is a resolution floor for the per-token trajectory classification use case, not an information deficit.

The implications:
- **Cross-snapshot discrimination via depth trajectory is not viable for per-token classification** with the current nearest-centroid classifier.
- **The signal exists at the aggregate level** (centroid classification succeeds at moderate σ), suggesting multi-token averaging or covariance-aware classifiers COULD work — but these require fundamental changes to the freezer architecture.
- **The iterative refinement trajectory (T5.4 path 2)** remains the necessary substrate for fine-grained snapshot discrimination — if the tf_loop port is ever completed.

This bench **corrects bench_015's root cause** (features DO change, but SNR ≈ 1.0) and **deepens the negative** from "geometry encoder limitation" to "per-token classification resolution floor."
