# Bench 014 — T5.6 SweTrajectoryFreezer G5: Cross-Model Discrimination

**Date:** 2026-08-02
**Proposal:** P011 Phase 5 T5.6
**Feature gates:** `kimi_k3_loader` + `swe_trajectory_freeze`
**Bench:** `benches/bench_014_swe_trajectory_freezer_g5.rs`

## Context

T5.5 shipped the `SweTrajectoryFreezer` with ALL 4 GOAT gates PASS on synthetic
data (Bench 013). T5.6 is the open G5 gate: **does trajectory geometry
discriminate across snapshots/models on real-model data?**

T5.4 PARTIAL (Bench 012) documented that the per-token DEPTH trajectory is
only 29% distinct across TOKENS (same model, different inputs) — depth
geometry is dominated by the LAYER WEIGHT STRUCTURE, which is the same across
tokens. **This finding is actually good news for G5:** if depth trajectories
are model-specific rather than input-specific, different MODELS should produce
measurably different depth trajectory geometries — exactly what G5 needs.

## Experiment design

Two "models" (snapshots) are loaded:
- **Model A** — real `model.safetensors` (Kimi-K3 0.40B, D=1024, 8 layers).
- **Model B** — random weights with seed=137 (same architecture, untrained).

For each of 32 diverse token IDs, both models produce a 9-state depth
trajectory (embed + 8 post-layer hidden states) via
`kimi_k3_forward_token_traced`. The trajectory geometry is computed + encoded
into a D=32 summary. The freezer is fit on 12 training tokens per model
(derive_directions_and_centroid) and tested on 20 held-out tokens per model.

**G5 PASS criterion:** classification accuracy ≥ 80% on the held-out split.

## Key substrate improvement landed alongside T5.6

### `from_states_into` — zero-allocation steady state

Added `from_states_into(states, disp_curr, disp_prev)` to
`latent_trajectory_geometry` — takes caller-managed scratch buffers. The
freezer's `freeze_attempt_into` variant achieves **0 allocs/call** (was 2
allocs/call with `from_states`). Verified by bench_013 G4:
- `freeze_attempt`: 2 allocs/call (documented substrate-inherited)
- `freeze_attempt_into`: **0 allocs/call** (true zero-alloc)

### Mean-centering fix — the critical bug found by T5.6

The initial T5.6 run FAILED at 50% accuracy. Root cause: the
`derive_directions` function computes `direction_k = normalize(centroid_k -
global_centroid)`, but the FAME `commit` projects the RAW summary (not
mean-centered). With non-centered summaries (all features in [0,1]), the
sigmoid gate's threshold at 0 doesn't align with the natural decision boundary
between clusters.

**Fix:** added `derive_directions_and_centroid` (outputs both directions AND
the global centroid) + `SweTrajectoryFreezer::fit` constructor (stores the
centroid) + mean-centering in `freeze_attempt_into` (subtracts the global
centroid before projection).

This is the mathematically correct nearest-centroid classifier: the summary
is mean-centered so `dot(centered_summary, direction_k)` measures deviation
from the global mean toward cluster k. The sigmoid at 0 is now the correct
decision boundary.

**Backward compatibility:** `new()` / `with_encoder()` default the centroid to
all-zeros (no centering), preserving bench_013's synthetic N=3 behavior.

## Results (REAL model.safetensors, D=1024, 8 layers)

### Training centroids

| Model | length_norm | curvature_norm | cosine_norm | n_steps_norm |
|-------|-------------|----------------|-------------|--------------|
| model_a (REAL) | 0.4711 | 0.4360 | -0.0308 | 0.8889 |
| model_b (random) | 1.0000 (clamped) | 0.5258 | -0.0295 | 0.8889 |

The random model's trajectory lengths exceed 1000.0 (the encoder's
`length_scale`), so they clamp to 1.0. The real model's lengths are ~471,
well within range. The curvature differs by ~0.09 rad. The cosine + n_steps
features are nearly identical across models.

### Mean-centered dot products (test split)

| Model | mean dot(centered_test, dir_0) |
|-------|-------------------------------|
| model_a (REAL) | +0.7514 |
| model_b (random) | -0.7580 |

Well-separated around 0 — the mean-centering fix works.

### G1: directions non-degenerate — ✅ PASS

N=2 produces antiparallel directions (cos = -1.0) by mathematical
necessity: 2 centroids define a line, the midpoint is on it, and the
directions from midpoint to each centroid are opposite. This is a valid
binary classifier axis, not a degeneracy. The directions are unit-norm.

### G5: cross-model discrimination — ✅ PASS

| metric | value |
|--------|-------|
| accuracy | **1.00 (40/40)** |
| target | ≥ 0.80 |

All 20 held-out tokens × 2 models classified correctly. Gate separation:
- model_a (REAL) → gate_a 0.64–0.73 (matching gate)
- model_b (random) → gate_b 0.68 (matching gate)

### G2: freeze_attempt latency — ✅ PASS

**per_call: 12327 ns** (target < 20000 ns).

At D=1024 (real model hidden dim, 8× the synthetic D=8), the geometry
computation is more expensive than the synthetic regime (4582 ns in bench_013).
~12µs is well within the 20µs budget for real-model scale.

## GOAT gates summary

| Gate | Claim | Verdict |
|------|-------|---------|
| G1 | directions non-degenerate (unit-norm + valid for N=2 binary axis) | ✅ PASS |
| G2 | freeze_attempt latency < 20µs (real-model D=1024) | ✅ PASS (12327 ns) |
| G5 | cross-model discrimination ≥ 80% accuracy | ✅ PASS (100%) |

## Interpretation

**T5.4 reframed, not refuted.** The T5.4 finding (depth trajectories are 29%
distinct across TOKENS) was initially read as "depth trajectories have weak
discriminative signal." T5.6 reveals the correct reading: **depth trajectories
are model-determined, not input-determined.** The weak cross-token signal
(29%) is the absence of input-dependence; the strong cross-model signal
(100%) is the presence of weight-dependence.

This is exactly what G5 needs: "different models produce measurably different
failure-trajectory shapes." The SweTrajectoryFreezer, with mean-centered
data-derived directions + FAME sigmoid blend, amplifies this model-specific
signal into a usable discrimination gate.

**Layer 4 modelless path validated for snapshot/model discrimination.** The
SweTrajectoryFreezer can tell real from random weights with 100% accuracy on
held-out depth trajectories. This doesn't prove it works for SWE-bench
failure-mode discrimination (that needs the agent loop — Layer 2/3, deferred),
but it proves the SIGNAL EXISTS and the PIPELINE WORKS on real-model data.

## What this validates + what it does NOT

**Validates:**
- The SweTrajectoryFreezer pipeline works end-to-end on real Kimi-K3 depth
  trajectories (D=1024, 8 layers, MLA/KDA/MoE/attn-res).
- Depth trajectory geometry IS discriminative across model snapshots.
- The mean-centering fix (derive_directions_and_centroid + fit constructor)
  is mathematically necessary for nearest-centroid classification.
- `freeze_attempt_into` achieves true zero-allocation steady state.

**Does NOT validate:**
- Cross-token discrimination within a single model (T5.4 showed this is weak
  at 29% — depth trajectories are model-determined, not input-determined).
- SWE-bench failure-mode discrimination (needs the agent loop — Layer 2/3).
- Discrimination between two REAL model checkpoints (e.g., Kimi-K3 at
  different training steps). The real-vs-random test is an extreme case;
  subtler differences (two real checkpoints) may or may not discriminate.
  This is a follow-up gated on having multiple real checkpoints.
- Cross-token generation trajectories (iterative refinement — T5.4 path 2).
  The current test uses depth trajectories only.

## Promotion

`swe_trajectory_freeze` stays **opt-in**. T5.6 G5 PASSES, but the proposal
explicitly states promotion requires a separate proposal + the full G5 suite
(cross-snapshot, cross-model, failure-mode discrimination on real SWE-bench
attempts). The current result validates the substrate works on real-model data;
production promotion needs the agent loop integration.

## Files

- `benches/bench_014_swe_trajectory_freezer_g5.rs` — this G5 gate.
- `crates/katgpt-core/src/swe_trajectory_freeze.rs` — `derive_directions_and_centroid`
  + `SweTrajectoryFreezer::fit` + `with_centroid` + mean-centering in
  `freeze_attempt_into`.
- `crates/katgpt-core/src/latent_trajectory_geometry.rs` — `from_states_into`
  (zero-alloc variant; `from_states` delegates to it).
- `benches/bench_013_swe_trajectory_freezer_goat.rs` — G4 updated to test both
  `freeze_attempt` (2 allocs) + `freeze_attempt_into` (0 allocs).
