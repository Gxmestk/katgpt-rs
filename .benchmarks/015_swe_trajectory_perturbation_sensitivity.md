# Bench 015 — SweTrajectoryFreezer Perturbation Sensitivity (Cross-Snapshot Proxy)

**Date:** 2026-08-02
**Proposal:** [011 — Rust-SWE-bench Latent Space via WASM Pruner](../.proposals/011_rust_swe_bench_latent_space_via_wasm_pruner.md) Phase 5 follow-up
**Predecessor:** [Bench 014 — T5.6 G5 gate (cross-model discrimination)](014_swe_trajectory_freezer_g5.md)
**Verdict:** **NEGATIVE RESULT — depth trajectory geometry is a STRUCTURAL fingerprint, not a VALUE fingerprint. Cross-snapshot discrimination via additive perturbation is not achievable.**

## The question

Bench 014 proved the `SweTrajectoryFreezer` discriminates real Kimi-K3 from RANDOM weights at 100% accuracy — the EXTREME case (completely different weight distributions). The open question: **how robust is this signal to SUBTLE weight changes?** At what perturbation magnitude does discrimination emerge?

This is a **cross-snapshot proxy**: two real model checkpoints from different training steps differ by structured gradient drift. We don't have a second real checkpoint, so we perturb the real weights by additive relative noise at increasing σ and measure the discrimination accuracy curve.

## Method

- **Model A** = real `model.safetensors` (Kimi-K3-0.40B, loaded once).
- **Model B(σ)** = clone of Model A with every weight perturbed: `w' = w * (1 + σ * noise)` where `noise ~ Uniform(-0.5, +0.5)`.
- **Relative perturbation** preserves each weight tensor's magnitude distribution (unlike absolute noise, which would be dominated by large tensors like the embedding).
- For each σ ∈ {0.0, 0.001, 0.01, 0.05, 0.1, 0.5}: extract 32-token depth trajectories for both models (12 train + 20 test), fit directions from the train split, classify the held-out test split, record accuracy.

## Run command

```bash
cargo bench --features "kimi_k3_loader swe_trajectory_freeze" \
    --bench bench_015_swe_trajectory_perturbation_sensitivity -- --nocapture
```

Requires real `model.safetensors` at `data/kimi-k3-0.40b/`. No random-weight fallback — the experiment is meaningless on random weights (perturbing random weights produces more random weights — the extreme case already tested in Bench 014).

## Results

```
       sigma    accuracy  correct/total   centroid_dist     verdict
  ------------------------------------------------------------------
      0.0000        0.50      20/40              0.000000  OK (sanity)
      0.0010        0.50      20/40              0.000209   below 80%
      0.0100        0.52      21/40              0.008910   below 80%
      0.0500        0.50      20/40              0.018333   below 80%
      0.1000        0.52      21/40              0.025506   below 80%
      0.5000        0.40      16/40              0.047676   below 80%
```

**Accuracy stays at ~50% (coin flip) across ALL σ levels, even at σ=0.5 (50% relative noise).** No discrimination floor found.

## Analysis

### Why it fails: centroid separation is 10× too small

The diagnostic column `centroid_dist` shows the L2 distance between the two mode centroids in the D=32 summary space. Compare to Bench 014:

| Experiment | centroid_dist | accuracy |
|------------|--------------|----------|
| Bench 014 (real vs random) | ~0.54+ (first block alone) | 100% |
| Bench 015 σ=0.5 (perturbed) | 0.048 | 40% |

The centroid separation at σ=0.5 is **~10× smaller** than the real-vs-random separation. The freezer's nearest-centroid classifier needs sufficient separation to derive non-degenerate direction vectors; with centroids only 0.048 apart (in a space where features are normalized to [0,1]), the directions are dominated by noise.

### The root cause: geometry features are SHAPE-invariant

The `GeometrySummaryEncoder` captures four features:
1. `length_norm` — total trajectory length (how far the hidden state travels)
2. `curvature_norm` — mean turning angle (how sharply it turns)
3. `cosine_norm` — min adjacent cosine (smoothness)
4. `n_steps_norm` — number of steps (constant for depth trajectories)

These are all **SHAPE features** — they describe the trajectory's geometric form, not its specific values. Additive weight perturbation changes the VALUES of hidden states but preserves the SHAPE of the trajectory (same layers, same activation patterns, same relative magnitudes). The geometry features are **invariant** to value perturbation.

In contrast, real-vs-random produces a STRUCTURAL difference: random weights produce chaotic trajectories with much larger lengths (>1000 vs ~471 for real), different curvature distributions, etc. The geometry features ARE sensitive to structural change — that's why Bench 014's G5 passed at 100%.

### Implication for cross-snapshot discrimination

Cross-snapshot discrimination (two checkpoints from different training steps) involves structured gradient drift — changes in weight VALUES while preserving the overall STRUCTURE (same architecture, same layer types, same low-rank patterns). This is qualitatively similar to additive perturbation: it changes values without destroying structure.

**The negative result here suggests cross-snapshot discrimination via depth trajectory geometry is UNLIKELY to work.** The geometry features capture structural fingerprints, not value-level differences. Two checkpoints of the same architecture would produce similar-shaped depth trajectories regardless of weight value differences.

### What WOULD work (conjecture, not tested)

To discriminate value-level differences (cross-snapshot), the summary encoder would need features that are sensitive to weight VALUES, not just trajectory SHAPE:

- **Mean activation magnitude per layer** — captures the scale of hidden states, which shifts with weight drift.
- **Variance/spectrum of hidden states** — captures the distribution shape, which is sensitive to weight perturbation.
- **Specific neuron activation patterns** — captures fine-grained weight-dependent behavior.

This is a DIFFERENT encoder than the geometry encoder. The geometry encoder is designed for failure-mode discrimination (T5.1: oscillation vs committed-wrong vs converged-correct produce different SHAPES). A value-sensitive encoder would be a separate substrate.

Alternatively, the **iterative refinement trajectory** (T5.4 path 2 — hidden states across generated tokens, not across layers) might capture value-level differences that depth trajectories miss. This requires either porting `tf_loop` to Kimi-K3 or a different trajectory extraction strategy (blocked on architecture compatibility).

## What this does NOT invalidate

- **T5.6 G5 PASS stands.** The freezer DOES discriminate real vs random (structural difference). This is valid + useful for model-family identification.
- **T5.1 failure-mode discrimination stands.** Synthetic failure modes (oscillation, committed-wrong, converged-correct) produce different SHAPES — the geometry encoder captures these. This is the substrate's primary use case.
- **The modelless Layer 4 path is validated** for structural/failure-mode discrimination. The limitation is specifically for fine-grained value-level discrimination (cross-snapshot within the same architecture).

## Substrate change

Added `#[derive(Clone)]` to `KimiK3ModelWeights` (`src/kimi_k3/loader.rs`). All inner types (`KimiDecoderLayerWeights`, `MlaWeights`, `KdaWeights`, `MoeWeights`, `AttnResWeights`, `SwiGluExpertWeights`) already derive `Clone`; the outer struct was missing it. This enables the bench to clone the loaded weights for perturbation. No runtime behavior change (Clone is only invoked when explicitly called).

## Honest caveats

1. **This is a proxy, not the real test.** Additive Gaussian noise is unstructured; training-step drift is structured (gradient updates concentrate in specific directions). The proxy may under-estimate discrimination (structured drift could produce larger centroid shifts than uniform noise at the same σ). Only real checkpoints can confirm.

2. **N=2 modes (binary classification).** The test is binary (original vs perturbed). Multi-class discrimination (multiple perturbation levels simultaneously) might behave differently.

3. **32 tokens may be too few.** More tokens would give more stable centroid estimates. But the centroid distances are SO small (0.048 at σ=0.5) that more tokens wouldn't close a 10× gap.

4. **The noise is relative (`w * (1 + σ * noise)`), not absolute.** Absolute noise might produce different results, but relative is the more physically meaningful perturbation (preserves weight magnitude distribution).

## Verdict

**NEGATIVE RESULT, as expected for a sensitivity probe.** The freezer discriminates STRUCTURE (real vs random = different architectures), not VALUES (perturbed vs original = same architecture, different values). Cross-snapshot discrimination via depth trajectory geometry is unlikely to work without a value-sensitive encoder.

This result is valuable: it documents the freezer's resolution floor honestly and motivates either (a) a value-sensitive encoder substrate, or (b) the iterative refinement trajectory (T5.4 path 2), or (c) Layer 4b (riir-train LoRA fallback) for fine-grained snapshot discrimination.

**No feature promotion.** `swe_trajectory_freeze` stays opt-in. The G5 gate (Bench 014) passed for structural discrimination; this probe shows the limitation for value-level discrimination.
