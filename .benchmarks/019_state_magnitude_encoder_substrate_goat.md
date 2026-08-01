# Bench 019 — StateMagnitudeEncoder Substrate GOAT

**Date:** 2026-08-02  
**Issue:** 571 (removed per noise-reduction rule — resolution captured in `.docs/09_feature_catalog/opt_in_features.md` §29)  
**Predecessor:** [Bench 018](018_sequence_trajectory.md) — sequence trajectory value discrimination (POSITIVE)  
**Result:** **G1-G5 ALL PASS.** Substrate-level `StateMagnitudeEncoder` ported from bench_018's `encode_seq_state_stats`. Promotion deferred (synthetic G5 + no consumer).

## Question

Bench 018 proved the sequence trajectory + state-magnitude features achieve 100% per-prompt accuracy at σ≥0.1 (d_Mahalanobis = 14.526). But the substrate (`SweTrajectoryFreezer`) was hardcoded to `GeometrySummaryEncoder` — no state-magnitude encoder existed, and `freeze_attempt_into` discarded raw states before encoding.

Can the substrate-level `StateMagnitudeEncoder` pass the GOAT gate (G1-G5) and close the gap between bench_018's finding and production-ready per-attempt freezing?

## What shipped

### `StateMagnitudeEncoder` (d=8, zero-alloc, single-pass)

Ported from bench_018's `encode_seq_state_stats`. Computes 8 aggregate statistics from the raw `&[&[f32]]` trajectory (NOT from `LatentTrajectoryGeometry`):

| Slot | Feature | Meaning |
|------|---------|---------|
| 0 | `mean_norm` | mean of per-step L2 norms |
| 1 | `std_norm` | std dev of per-step L2 norms |
| 2 | `max_norm` | max per-step L2 norm |
| 3 | `min_norm` | min per-step L2 norm |
| 4 | `initial_norm` | L2 norm of first hidden state |
| 5 | `final_norm` | L2 norm of last hidden state |
| 6 | `norm_ratio` | `final_norm / initial_norm` (0 if initial≈0) |
| 7 | `mean_cos` | mean cosine similarity between consecutive states |

**Algorithm:** Single-pass Welford's online algorithm for mean+variance, plus running min/max/sum + consecutive cosine (dot product + norms computed in the same dim loop as the current state's norm). This avoids the 3x recomputation of the naive three-pass approach (norms → mean → var → cosine).

### `FrozenValueAttempt<N, D>` + `freeze_attempt_value` / `freeze_attempt_value_into`

The value-level counterpart to `FrozenAttempt`. Commits to `(pi, summary)` via BLAKE3 — no geometry triple in the payload (the state-magnitude features ARE the payload). At production scale (N=3, D=32): 140 bytes.

### API surface

```rust
// Construct (parameterless — no scale tuning needed)
let encoder = StateMagnitudeEncoder::new();

// Encode (zero-alloc, writes 8 features into out[..8])
encoder.encode_into(&trajectory_refs, &mut summary);

// Freeze (full pipeline: encode → mean-center → FAME commit → BLAKE3 envelope)
let frozen = freezer.freeze_attempt_value(&trajectory_refs, &fields, version);
let gates = frozen.gates();
let argmax = frozen.argmax_archetype();
```

## GOAT gate results

| Gate | Status | Test | Detail |
|------|--------|------|--------|
| G1 (correctness) | ✅ PASS | `g1_state_magnitude_encoder_correctness` | Hand-computed expected values on a 3-state dim-2 trajectory: mean=5.0, std=√(50/3)≈4.082, max=10.0, min=0.0, initial=5.0, final=0.0, ratio=0.0, mean_cos=1.0. All match bit-identically. |
| G1 (edge cases) | ✅ PASS | `g1_state_magnitude_empty_and_single` | Empty trajectory → all zeros (no panic). Single state → mean/std/max/min/initial/final all equal, ratio=1.0, mean_cos=0.0 (no pairs). |
| G1b (determinism) | ✅ PASS | `g1b_freeze_attempt_value_deterministic` | Two freezes of the same trajectory produce bit-identical envelopes (commitment + merkle_root + pi + summary). |
| G2 (perf) | ✅ PASS | `g2_state_magnitude_encoder_under_100us` (release-only) | Value encoder: **51.8µs** vs geometry pipeline: **100.7µs** at D=1024, N=64. Value is **0.52x geometry** (faster). Under 100µs ceiling. |
| G3 (no-regression) | ✅ PASS | `g3_geometry_path_unaffected_by_value_addition` | Geometry G3 cross-mode discrimination still ≥80%. Full suite: 1851 lib tests pass (was 1845 before this change). |
| G4 (tamper-evidence) | ✅ PASS | `g4_value_envelope_tamper_evidence` | Header verification clean; tampered merkle_root + commitment both fail verification. Payload length = N*4 + D*4 = 140 bytes (no geometry triple). |
| G5 (value discrimination) | ✅ PASS | `g5_value_discrimination_synthetic_scale_shift` | 3 classes × 7 trajectories × 32 tokens × 16 dims. 100% accuracy (6/6 test trajectories correctly classified). |

### G2 perf detail

```
G2 perf: value=51854ns, geometry=100654ns, ratio=0.52x
```

The value encoder is **2x faster** than the geometry pipeline at the same scale because:
- Value: single-pass (Welford mean+var + cosine in one loop)
- Geometry: `from_states_into` does a separate displacement pass + `encode_into` on the geometry triple

The O(n·dim) = 64×1024 = 65K f32 muls dominate; both are memory-bandwidth-bound. The value encoder's single-pass advantage halved the wall-clock time.

### G5 synthetic test design

The G5 test uses synthetic trajectories that mimic weight perturbation:
- **3 classes**, each with distinct `(scale, variance_factor)` parameters
- **7 trajectories per class** (5 train, 2 test)
- Each trajectory: 32 tokens × 16 dims
- Two independent parameters prevent the centroids from being collinear (which would degenerate the nearest-centroid classifier to a 1D problem — the failure mode discovered during G5 debugging with single-parameter scale shifts)

The class-specific bias direction (fixed per class, varies across classes) + scale + variance factor produce feature vectors that span a multi-dimensional subspace, making the direction derivation non-degenerate.

## Full test run

```
cargo test -p katgpt-core --features swe_trajectory_freeze --lib
test result: ok. 1851 passed; 0 failed; 7 ignored; 0 measured; 0 measured out
```

(7 ignored = 6 pre-existing + 1 new release-only G2 perf test.)

```
cargo clippy -p katgpt-core --features swe_trajectory_freeze --lib --tests
# zero warnings, zero errors
```

## Promotion deferral (T6)

The GOAT gate passes, but promotion to default-on is **deferred**:

1. **G5 uses synthetic data.** The substrate-level G5 test uses synthetic scale+variance-shifted trajectories. Bench 018's 100% was on real Kimi-K3 weights with synthetic σ-perturbation. Real training drift may differ. Promotion should wait for a second checkpoint.

2. **No production consumer.** The primitive has no downstream caller yet. Per codebase pattern, promotion follows a consumer demonstrating the gain. The consumer here is the SWE-bench pruner (Proposal 011 Layer 4), not yet wired.

The primitive stays opt-in (`swe_trajectory_freeze`). Re-evaluate at the next SWE-bench integration milestone.

## What this closes

This closes the gap between:
- **Bench 014 G5 PASS** (structural discrimination — failure-mode classification via geometry, 100%)
- **Benches 015-017 NEGATIVE** (value discrimination via depth trajectory geometry — perturbation-invariant, Bayes-optimal ceiling ~54%)
- **Bench 018 POSITIVE** (value discrimination via sequence trajectory state magnitude — 100%)
- **Bench 019 (this)** — the substrate now supports both paths: `freeze_attempt` (geometry, structural) + `freeze_attempt_value` (state magnitude, value-level)

Layer 4 per-attempt freezing is now validated for BOTH structural AND value-level discrimination at the substrate level. The remaining gap is real-checkpoint validation (not a substrate issue — the substrate is ready).
