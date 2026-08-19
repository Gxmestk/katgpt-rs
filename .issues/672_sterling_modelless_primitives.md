# Issue 672: Sterling-derived modelless primitives — ReLU-gated suppression, exact-decomposition readout, lift-set steering targets

**Date:** 2026-08-19
**Research:** [katgpt-rs/.research/491_Sterling_Additive_Concept_Attribution_Steering.md](../.research/491_Sterling_Additive_Concept_Attribution_Steering.md)
**Source paper:** [arXiv:2608.07594](https://arxiv.org/abs/2608.07594) — "Scaling Inherently Interpretable Language Models" (Guide Labs, Steerling-8B) §6.2 (steering), §5.3 (additive bottleneck), §10.2.4 (lift sets)
**Target:** `crates/katgpt-core/src/` (sampling/logit-mask surface + steering surface) — feature `sterling_primitives` (opt-in until GOAT)

## Problem

Three closed-form mechanisms from Steerling-8B have no shipped analog (grep-verified, Research 491 §3):

1. **Naive logit subtraction promotes anti-aligned tokens.** Suppressing a concept by `ℓ_v − s·a_c[v]` boosts every token with `a_c[v] < 0` — unrelated anti-aligned tokens dominate generation (paper Fig. 19). We have activation-space erasure (MANCE, R409) but no output-space one-sided suppression.
2. **Attribution requires estimation everywhere.** Our attribution surfaces (causal_validation patching, step_attribution Δ) estimate effects via extra forward passes / replay A/B. When a consumer is a linear readout over additive components, the decomposition is **exact by construction** — we never exploit this.
3. **Steering targets are hand-picked.** `latent_field_steering` (Plan 309) directions are mined (MAG) but the *expression target* (which tokens/concepts a direction should raise) has no corpus-statistic basis.

## Proposal

Ship the three mechanisms as small opt-in katgpt-core primitives:

- **T1 `relu_gated_suppression`**: given concept alignment vector `a = W·e_c ∈ ℝ^|V|` and strength `s`, emit logit mask `−s·relu_pos(a)` (only positive alignments penalized; anti-aligned untouched). SIMD branch-free (`max(0,·)`). Consumer surface: sampling logit-mask + riir-ai action-suppression.
- **T2 `decomposed_readout`**: helper that, given a linear head `W_y` and additive components `(k̂, û, ε)`, returns per-component contributions + residual with the invariant `Σ parts + residual == fused` **bit-identical** (fixed summation order). Generic math, no game semantics — the riir-ai consumer is Issue 732.
- **T3 lift-set builder**: `lift(w, c) = P(w | chunks-tagged-c) / P(w)` over any tagged corpus; top-K lifted sets become (a) `latent_field_steering` expression targets, (b) candidate bias tables for `TernaryDraftModel`. Pure corpus statistic, zero training.
- **T4 (riders)**: promote the civ salience-gate noisy-OR (`1 − Π(1−k)`) into a core util with log1p-stable form; add the HSIC-style normalized cross-covariance `‖ΨᵀΦ‖²_F/(d²(M−1))` as a measure-only disentanglement gauge.

## Tasks

- [ ] **T1** `relu_gated_suppression` + falsifier test: naive subtraction arm asserts anti-aligned promotion (the bug), gated arm leaves them bit-unchanged
- [ ] **T2** `decomposed_readout` + exactness test (Σ + residual == fused, bit-identical, incl. empty-component degenerate cases)
- [ ] **T3** lift-set builder + monotonicity/boundary tests (all-0 → 0; any-1 → 1); wire one consumer demo (steering target or drafter bias table)
- [ ] **T4** noisy-OR core util (civ site delegates) + HSIC-metric gauge with constructed orthogonal/identical controls
- [ ] **T5** GOAT gates + `cargo check --all-features` + clippy clean; promote to default only if a consumer GOAT passes (riir-ai Issue 732 is the first candidate consumer)

## GOAT gates

- **G1 correctness**: T1 falsifier pair (naive vs gated); T2 bit-identity; T3/T4 boundary identities.
- **G2 perf**: T1/T2 vectorize — criterion vs scalar baseline; T2 ≤ 3× single fused GEMV.
- **G3 no-regression**: default feature set untouched; `--all-features` clean.
- **G4 alloc-free**: fixed scratch, zero steady-state allocation.

**Refs:** Research 491 (this paper's distillation) · R290/R397 (shipped steering + activation-norm calibration — T3's logit-space γ cousin noted, not duplicated) · R409 (erasure; T1 is its output-space complement).
