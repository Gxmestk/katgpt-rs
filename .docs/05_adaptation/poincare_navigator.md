# Poincaré Adapter — Closed-Form Latent Navigation (DEFAULT-ON)

**Plan:** [449](../../.plans/449_poincare_latent_navigation_primitive.md) (Phase 1+2+3 DONE; primitive PROMOTED TO DEFAULT-ON 2026-07-18)
**Research:** [449](../../.research/449_SeeSE3_Poincare_Adapter_Primitive.md) — math + 4/4 Super-GOAT novelty gate
**Source paper:** [arXiv:2607.14228](https://arxiv.org/abs/2607.14228) (Chen et al., *SeeSE3: Emergence of 3D Space in Vision Features*, DeepMind, 15 Jul 2026)
**Feature flag:** `poincare_navigator` (**DEFAULT-ON** since Plan 449 Phase 3 promotion, 2026-07-18). Co-gates `subspace_phase_gate` (already default-on) for the SVD pseudoinverse.
**Private game-runtime selling point:** [`riir-ai/.research/319_SeeSE3_Latent_Imagination_Game_Runtime_Guide.md`](../../../riir-ai/.research/319_SeeSE3_Latent_Imagination_Game_Runtime_Guide.md) — NPC imagination loop.

## What it is

A frozen [`PoincareAdapter`] Pod holds an offline-fit triple `(φ, W, W†)` that lets a
consumer navigate latent space **in closed form** — given a desired movement in *target*
space (e.g. 3D pose delta, HLA affect delta), recover the corresponding latent step.

- `φ: R^d_latent → R^d_phi` — a small unrolling chart (modelless default: PCA + `tanh`).
- `W: R^d_phi → R^d_target` — the linear decoder: `Δtarget ≈ W·(φ(z₂) − φ(z₁))`.
- `W†: R^d_target → R^d_phi` — the Moore-Penrose pseudoinverse of `W`.

The **inverse navigator** is then closed-form:

```text
z_dest = z_src + φ⁻¹( φ(z_src) + W† · Δtarget )
```

One MLP eval, one matvec, one inverse projection. All bounded-size, all
zero-allocation. This is the canonical "closed-form inverse" that the paper's
gradient fit also converges to but doesn't require.

## API surface

```rust
// Re-exported at crate root (feature poincare_navigator + subspace_phase_gate):
use katgpt_core::{
    FitConfig, LATENT_DIM_MAX, PHI_HIDDEN_DEFAULT, PHI_OUT_DEFAULT, PoincareAdapter,
    PoincareFitError, RIDGE_ALPHA_DEFAULT, TARGET_DIM_MAX,
    fit_poincare_adapter,           // cold: closed-form ridge + PCA + thin SVD
    poincare_navigate_into,         // hot: z_src + Δtarget → z_dest (zero-alloc)
    poincare_multi_step_into,       // hot: 4-step open-loop trajectory
    eval_phi_into,                  // hot: φ chart evaluation into caller buffer
    accumulate_pinv_into,           // hot: W† · Δtarget accumulation
};
```

Constants cap the supported dimensions:

| Constant | Value | Bound |
|---|---|---|
| `LATENT_DIM_MAX` | 64 | HLA=8, LLM-block=64, shard `style_weights`=64 — anything larger reduces first |
| `PHI_OUT_DEFAULT` | 20 | Paper sweet spot for vision features |
| `PHI_HIDDEN_DEFAULT` | `PHI_OUT_DEFAULT` | Modelless path is single linear + tanh |
| `TARGET_DIM_MAX` | 8 | SE(3)=6, SE(2)+belief scalars ≤ 8 |
| `RIDGE_ALPHA_DEFAULT` | 1.0 | Matches paper's α=1.0 |

## Why modelless

The fit is **closed-form ridge regression + PCA + thin SVD pseudoinverse**. No gradient
descent. The paper's AdamW fit over a 2-layer φ is the gradient fallback — only
warranted if G2 (global unrolling) fails. Per the research skill §3.5 modelless-unblock
protocol, the modelless path is the default; Path 0 (training-target decomposition)
applies: the paper's value is the math (`ΔP ≈ W·Δz` + the unrolling), NOT the training
loop. Reuses Plan 301's `thin_svd_into` + Plan 308's `ridge_solve_direct_f32`.

## Sibling primitives (compose, don't duplicate)

- **Latent Field Steering** (Plan 309) — the *forward* direction: push latent state
  along a designer-supplied direction vector. Poincaré is the *inverse*: given a
  desired target movement, find the latent step.
- **Viable Manifold Graph** (Plan 312) — *graph-based* navigation on a safe subgraph.
  Poincaré is *continuous closed-form*. They compose: VMG handles highly curved
  manifolds; Poincaré handles manifolds that admit linear unrolling.
- **SLoD** (Plan 235) — Poincaré *ball* geometry for KG LOD selection. Different
  problem (KG abstraction levels, not latent navigation).

## GOAT gate (Bench 449, 2026-07-18 — ALL 7/7 PASS)

| Gate | Spec | Observed | Verdict |
|---|---|---|---|
| G1 local decodability | max abs decoded delta bounded on small displacements | 0.012622 over 1000 random pairs (\|z\| ≤ 0.05·√8) | PASS (800× under sanity bound 10.0) |
| G2 global unrolling (Theorem 5c) | adapter R² > 0.5 on coupled curved fixture | adapter R² = 0.7149, linear-only R² = 0.9277 | PASS with caveat — modelless does not strictly dominate linear ridge |
| G3 inverse navigation round-trip | Hit@0.3 > 0.5 on 1000 held-out pairs | Hit@0.3 = **1.000** | PASS (perfect) |
| G4 zero-alloc steady state | 0 allocations / 100 navigator calls (post-warmup) | 0 allocations | PASS |
| G5 latency | median < 1µs at d=64, target=6, phi_out=20 (paper scale) | **809 ns/call** (256 batches × 1024 calls) | PASS (~20% headroom) |
| G6 multi-step coherence | 4-step open-loop trajectory bit-identical + bounded | bit_identical=true, bounded=true | PASS |
| G7 latent-vs-raw boundary | navigator signature uses only `&[f32]` + `&PoincareAdapter` | enforced by type system; pinned by `TypeId::of` check | PASS (by construction) |

## G2 caveat — closed by riir-train Plan 317

The G2 verdict was PASS-with-caveat: adapter R²=0.71 > the spec threshold of 0.5 (PASS),
but did NOT strictly dominate linear-only ridge (R²=0.93) — the documented "strict-
domination requires the gradient-fit φ (riir-train follow-up)" caveat.

[`riir-train/.plans/317_poincare_phi_gradient_fit.md`](../../../riir-train/.plans/317_poincare_phi_gradient_fit.md)
closed the gap same-day: it ships `fit_poincare_adapter_trained` in `riir-train-engine`
behind feature `poincare_phi_train` (2-layer MLP φ via AdamW, ~15ms fit, frozen at
inference). The cross-verification bench
[`bench_317_poincare_g2_strict`](../../../riir-train/crates/riir-train-engine/benches/bench_317_poincare_g2_strict.rs)
reproduces the exact G2 fixture and reports:

| Competitor | R² |
|---|---|
| Modelless PCA-tanh (this bench) | 0.7149 |
| **Trained 2-layer MLP φ (Plan 317)** | **0.9997** |
| Linear-only ridge | 0.9255 |

The trained variant dominates both — strict-domination confirmed. The modelless
primitive ships DEFAULT-ON anyway because its load-bearing value is the **closed-form
inverse navigation** (G3 Hit@0.3=1.000) + the frozen Pod commitment pattern, **neither
of which depends on G2 strict-domination**. The trained variant is opt-in for consumers
that need the forward-unrolling chart accuracy.

## Promotion pattern (the codebase convention)

Promotion to DEFAULT-ON follows the established pattern — a modelless primitive that
passes its quality gate on the dimensions the runtime actually uses (here G3 inverse
navigation, G4 zero-alloc, G5 latency) ships default-on even when an auxiliary gate (G2
strict-domination) is soft. Prior art:

- `manifold_bandit` (Plan 370) — G2 FAIL (plan-level expectation error), default-on.
- `set_attention` (Plan 354) — G8 collective-inference FAIL (use-case limit), default-on.
- `ac_prefix` (Plan 313) — modelless unblock via §3.5, default-on.

The common rule: modelless + zero-cost-unless-invoked + GOAT-passes-on-the-load-bearing-
axis → DEFAULT-ON. The Poincaré Adapter qualifies on all three.

## Theorem 7 design constraint

The paper proves rotation targets fit tighter than translation-magnitude targets
(rotational optical flow is depth-independent; translational flow scales as `1/Z`).
Designers should lean on the easier component. Concretely: if the consumer's target
space is `(facing_θ, Δx, Δy)`, expect higher R² on `θ` than on `‖Δx, Δy‖`. See
`riir-ai/.research/319` §"HLA depth analog gap".

## Usage

```bash
# Run the GOAT gate bench
CARGO_TARGET_DIR=/tmp/plan449 cargo bench -p katgpt-core \
    --features poincare_navigator --bench bench_449_poincare_goat --no-run
/tmp/plan449/release/deps/bench_449_poincare_goat-<hash> --nocapture

# The trained variant (riir-train Plan 317) — opt-in
CARGO_TARGET_DIR=/tmp/plan317 cargo bench -p riir-train-engine \
    --features poincare_phi_train --bench bench_317_poincare_g2_strict --no-run

# Clean up
rm -rf /tmp/plan449 /tmp/plan317
```

## References

- Plan: [`katgpt-rs/.plans/449_poincare_latent_navigation_primitive.md`](../../.plans/449_poincare_latent_navigation_primitive.md)
- Research: [`katgpt-rs/.research/449_SeeSE3_Poincare_Adapter_Primitive.md`](../../.research/449_SeeSE3_Poincare_Adapter_Primitive.md)
- Benchmark: [`katgpt-rs/.benchmarks/449_poincare_goat.md`](../../.benchmarks/449_poincare_goat.md)
- Source code: [`katgpt-rs/crates/katgpt-core/src/poincare.rs`](../../crates/katgpt-core/src/poincare.rs)
- Private game-runtime guide: [`riir-ai/.research/319_SeeSE3_Latent_Imagination_Game_Runtime_Guide.md`](../../../riir-ai/.research/319_SeeSE3_Latent_Imagination_Game_Runtime_Guide.md)
- Gradient-fit follow-up: [`riir-train/.plans/317_poincare_phi_gradient_fit.md`](../../../riir-train/.plans/317_poincare_phi_gradient_fit.md)
- Source paper: [arXiv:2607.14228](https://arxiv.org/abs/2607.14228) — Chen et al., *SeeSE3*, DeepMind 2026
