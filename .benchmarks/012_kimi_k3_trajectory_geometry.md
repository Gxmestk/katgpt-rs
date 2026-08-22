# Bench 012 — Kimi-K3 Trajectory Geometry on Real Architecture (T5.4, Option 2)

**Date:** 2026-08-02
**Proposal:** P011 Phase 5 T5.4
**Feature gates:** `kimi_k3_loader` + `latent_trajectory_geometry`
**Bench:** `benches/bench_012_kimi_k3_trajectory_geometry.rs`

## Context

The T5.4 wiring investigation (prior session) found that neither `tf_loop` nor
`latent_trajectory_geometry` is wired into the Kimi-K3 forward path. `tf_loop`
is architecturally incompatible with Kimi-K3's hybrid MLA/KDA/MoE/attn-res layer
type (it operates on generic `TransformerLayer` — single Q/K/V/O + FFN).

**Option 2** (the recommended path): bypass tf_loop entirely. Extract trajectory
geometry directly from the production forward path by snapshotting `runtime.hidden`
after each of the 8 decoder layers per token.

## What shipped

1. **`kimi_k3_forward_token_traced`** (`src/kimi_k3/model.rs`) — traced variant
   of `kimi_k3_forward_token`. Same decoder path (embed → 8 layers → output
   attn-res → final norm) but snapshots `runtime.hidden` after embedding + after
   each layer into `traj_out` (9 states of `hidden_size` each). Skips the LM head
   (returns final normalized hidden state) — trajectory geometry operates on
   hidden states, not logits, and this avoids requiring the full [vocab × hidden]
   lm_head matrix. **Diagnostic only** — allocates per call.

2. **`bench_012_kimi_k3_trajectory_geometry.rs`** — runs the traced forward on
   both random weights (always runnable) and real `model.safetensors` (when
   present at `data/kimi-k3-0.40b/`). Tests three scenarios: per-token depth
   trajectory (no context), trajectory across sequence positions, and trajectory
   after a 16-token context prefix.

## Results (REAL model.safetensors, D=1024, 8 layers)

### Test 1: per-token depth trajectory (no KV context)

8 tokens, each processed in isolation. Trajectory = [embed → layer0 → ... → layer7]
(9 states, D=1024).

| token | n_steps | length | mean_curv (rad) | min_cos | finite |
|-------|---------|--------|-----------------|---------|--------|
| 1     | 8       | 437.17 | 1.3485          | -0.0305 | YES    |
| 2     | 8       | 440.33 | 1.4543          | +0.0037 | YES    |
| 5     | 8       | 490.96 | 1.3884          | +0.0336 | YES    |
| 10    | 8       | 465.36 | 1.3769          | +0.0052 | YES    |
| 42    | 8       | 564.70 | 1.3856          | -0.0297 | YES    |
| 100   | 8       | 575.68 | 1.3763          | -0.0154 | YES    |
| 200   | 8       | 422.14 | 1.4700          | +0.0067 | YES    |
| 500   | 8       | 500.07 | 1.4437          | -0.0071 | YES    |

Aggregate: length mean=487, std=54; curvature mean=1.4055, std=0.041.
Distinct pairs (curv diff > 0.1 OR length diff > 20%): **8/28 (29%)**.

### Test 2: trajectory across sequence positions (8-token prompt)

Geometry varies: length std=53, curvature std=0.091.

### Test 3: traced forward after 16-token context prefix

Geometry stays stable: length mean=470, std=50; curvature mean=1.367, std=0.025.

## GOAT gates

| Gate | Claim | Verdict |
|------|-------|---------|
| G1 | all geometry finite + in-range (D=1024) | ✅ PASS |
| G2 | non-degenerate (length > 0) | ✅ PASS |
| G3 | discriminative across tokens (>30% distinct) | ❌ FAIL (29%) |
| G4 | varies across sequence positions | ✅ PASS |

## Interpretation

**G1+G2+G4 PASS** confirms the substrate is numerically stable + non-degenerate
at production-model scale (D=1024). The curvature values (~1.4 rad, near π/2)
indicate the depth trajectory makes consistent orthogonal-ish turns through the
layers — geometrically meaningful, not random noise.

**G3 FAIL (29% vs 30% threshold)** is the load-bearing finding. The per-token
DEPTH trajectory (9 states: embed + 8 post-layer) is NOT strongly discriminative
across tokens, even with real weights. Length varies ±12% (422–576), curvature
varies ±5% (1.27–1.54), but most pairs don't cross the discrimination threshold.

**Why:** The depth trajectory is dominated by the LAYER WEIGHT STRUCTURE (which
is the same across tokens), not the input token. Each layer applies a fixed
transformation; the per-layer displacement is largely determined by the weight
matrices, not by which token triggered the forward pass. Different tokens produce
different starting points but follow similar geometric paths through the layer
stack.

**Contrast with the synthetic POC (T5.1, Bench 011):** the synthetic PoC tested
100-step ITERATIVE refinement trajectories (committed-wrong / oscillation / drift
/ stuck / converged-correct). Those are fundamentally different from 9-step depth
trajectories — iterative trajectories capture how the model's OUTPUT evolves
across repeated forward passes, which IS input-dependent and failure-mode-
dependent. Depth trajectories capture how the REPRESENTATION evolves through
layers within a single forward pass, which is largely architecture-determined.

## Design implication for Layer 4

The original P011 design assumed `tf_loop`'s iterative refinement as the
trajectory source. The T5.4 finding confirms this assumption was load-bearing:
the discriminative signal in Layer 4 likely lives in the **iterative refinement
trajectory** (repeated forward passes on evolving patch proposals), NOT in the
depth trajectory of a single forward pass.

Three paths forward:
1. **Port `tf_loop` to Kimi-K3** (Option 1 from the investigation) — largest
   scope, most faithful to the P011 design.
2. **Extract iterative trajectories from the existing generation loop** — a
   multi-token trajectory (hidden state at each generated token) would capture
   output evolution. This is a cross-token trajectory, not a depth trajectory.
3. **Accept the depth trajectory result** — if the goal is snapshot/model
   discrimination (T5.6), the 29% distinct rate may be sufficient when combined
   with geometry summary encoding (T5.3b's data-derived directions). The
   synthetic POC showed 100% accuracy with geometry-encoded summaries on
   structured trajectories; the real-model depth trajectory is less structured
   but not zero-signal.

## Promotion

`kimi_k3_forward_token_traced` stays in `model.rs` as a diagnostic tool (gated by
`kimi_k3_loader`). No feature promotion — it's a traced variant, not a production
primitive. `bench_012` stays as an opt-in bench (requires `kimi_k3_loader` +
`latent_trajectory_geometry`).
