# Issue 571 — StateMagnitudeEncoder for value-level trajectory discrimination

**Date:** 2026-08-02  
**Status:** Open  
**Evidence:** [Bench 018](../.benchmarks/018_sequence_trajectory.md) — POSITIVE  
**Proposal:** [011](../.proposals/011_rust_swe_bench_latent_space_via_wasm_pruner.md) Phase 5 T5.6e  

## Problem

The `SweTrajectoryFreezer<N, D>` substrate (`crates/katgpt-core/src/swe_trajectory_freeze.rs`) is hardcoded to `GeometrySummaryEncoder`, which captures trajectory SHAPE features (length, curvature, cosine). Bench 015 proved these shape features are **perturbation-invariant** — they cannot discriminate value-level weight differences.

Bench 018 discovered that **state magnitude features** (aggregate L2 norm statistics of the final hidden states across a prompt's tokens) ARE highly discriminative:
- 100% per-prompt accuracy at σ≥0.1
- d_Mahalanobis = 14.526 at σ=0.5 (50× the geometry encoder's 0.285)
- Discrimination floor at σ≈0.03-0.05

But these features are not available in the substrate. The `freeze_attempt_into` method computes `from_states_into` (geometry) then encodes via `GeometrySummaryEncoder::encode_into(&geometry, ...)` — the raw states are discarded before encoding.

## Proposed change

1. **Add `StateMagnitudeEncoder`** to `swe_trajectory_freeze.rs`:
   - d=8 features: mean_norm, std_norm, max_norm, min_norm, initial_norm, final_norm, norm_ratio, mean_cosine
   - Computes from raw `&[&[f32]]` trajectory states (NOT from `LatentTrajectoryGeometry`)
   - Zero-training, modelless (pure aggregate statistics)

2. **Decouple encoder from geometry**: The current `freeze_attempt_into` hardcodes the geometry → encode pipeline. Options:
   - (A) Add a `freeze_attempt_value_into` method that uses `StateMagnitudeEncoder` instead of geometry
   - (B) Make the encoder a trait (`TrajectoryEncoder`) with two impls (Geometry, StateMagnitude) and generic over the freezer
   - (C) Add the state-magnitude features as additional dimensions in the existing encoder

   Option (A) is simplest and least invasive. Option (B) is cleanest but requires API change. Option (C) conflates two different signal types.

3. **Add sequence trajectory extraction guidance**: bench_018 uses `kimi_k3_forward_token_traced` in a loop WITHOUT `reset()` between tokens (growing KV cache). The current bench examples all use `reset()` per token. Document the sequence trajectory extraction pattern.

## GOAT gate

- **G1** (correctness): `StateMagnitudeEncoder` produces the same 8 features as bench_018's `encode_seq_state_stats` on identical input
- **G2** (perf): encoding < 1000 ns/call for D=1024, N=64 (matching GeometrySummaryEncoder's latency)
- **G3** (no-regression): existing GeometrySummaryEncoder tests + SweTrajectoryFreezer tests pass unchanged
- **G4** (alloc-free): zero allocations in the encoding path (compute in-place from scratch buffers)
- **G5** (value discrimination): bench_018-level accuracy (100% at σ≥0.1) when using the new encoder with the sequence trajectory

## Why this matters

This closes the gap between the G5 PASS (structural discrimination, bench_014) and the value-level discrimination failure (benches 015-017). The sequence trajectory + state magnitude encoder is the path to production-ready per-attempt freezing for SWE-bench: two model snapshots can be discriminated by their trajectory signatures.

## Substrate-first check

- `GeometrySummaryEncoder` exists but captures shape only
- No existing magnitude/norm-based encoder in `swe_trajectory_freeze.rs`
- `from_states_into` computes geometry but discards raw state magnitudes
- No conflict with existing substrate — this is a new encoder type, not a duplication

## Tasks

- [ ] T1: Implement `StateMagnitudeEncoder` (d=8, from raw states)
- [ ] T2: Add `freeze_attempt_value_into` or trait-based encoder dispatch
- [ ] T3: G1-G4 GOAT gate at substrate level
- [ ] T4: G5 value discrimination bench (port bench_018's SeqStateStats to substrate)
- [ ] T5: Document sequence trajectory extraction pattern (no-reset loop)
- [ ] T6: If GOAT passes → consider promotion to default (currently opt-in `swe_trajectory_freeze`)
