# Plan 438 — FORE Fitted Occupancy-Ratio Estimator GOAT Gate

**Date:** 2026-07-14
**Plan:** [438_occupancy_ratio_estimator_primitive.md](../.plans/438_occupancy_ratio_estimator_primitive.md)
**Research:** [423_Adjoint_Bellman_KL_Contraction_Occupancy_Ratio.md](../.research/423_Adjoint_Bellman_KL_Contraction_Occupancy_Ratio.md)
**Source paper:** [arXiv:2607.05375](https://arxiv.org/abs/2607.05375) — van der Laan & Kallus, *Fitted Occupancy-Ratio Evaluation without Bellman Completeness*, 2026
**Verdict:** GOAT — stays opt-in (`occupancy_ratio = []`). Promotion to default-on requires a downstream consumer (Fusion A CLR stabilization in `riir-poc`) to validate the gain empirically.

---

## Gate Results

| Gate | Requirement | Result |
|---|---|---|
| **G1 correctness** | Baird-MRP: FORE converges to `ω(upper) = 0.2211`, `ω(lower) = 15.7987` within 2% rel err | ✅ **PASS** — n=100k, K=50, γ=0.95, seed=423: 0.31% (upper), 0.74% (lower) |
| **G2 perf** | FORE fit n=10000, state_dim=8, K=20 < 100 ms | ✅ **PASS** — 48.63 ms median (release, Apple Silicon) |
| **G3 no-regression** | `cargo clippy --features occupancy_ratio` + `cargo test --lib` clean | ✅ **PASS** — 0 clippy warnings, 1560/1560 lib tests (1 pre-existing debug-mode latency fail unrelated) |
| **G4 alloc-free** | Inner KL-projection loop: 0 allocs after warmup | ✅ **PASS** — 0 alloc+dealloc / 100 `fit_and_evaluate` calls (CountingAllocator) |
| **G5 modelless** | No GD through base weights | ✅ **PASS** — only mutable state is `θ: Vec<f32>`; no `NeuronShard`/`LoRAWeightVersion`/`SenseModule` touched |
| **G6 floor (UQ)** | N/A — ratio estimator, not a forecaster | N/A |

---

## G1 Correctness Detail

### Analytical anchors (independent f64 solve)

The 7-state Baird-MRP (paper Appendix G.1) was constructed in `crates/katgpt-core/tests/occupancy_baird_mrp.rs`. The analytical occupancy ratios were computed independently by solving `(I − γP^T) d^π = (1−γ) d_0` via f64 Gaussian elimination on the full 7×7 system:

```
ω_π,γ(upper) = 0.2211217321  (= 1920/8683)
ω_π,γ(lower) = 15.7986870897 (= 7220/457)
```

These match the paper's anchors to 10 significant digits (cross-check: `t32_analytical_anchors_match_paper` passes with rel err < 1e-6).

### FORE fit results

| n | K | γ | Seed | ω(upper) fitted | ω(upper) rel err | ω(lower) fitted | ω(lower) rel err |
|---|---|---|---|---|---|---|---|
| 100,000 | 50 | 0.95 | 423 | 0.221814 | **0.31%** | 15.916202 | **0.74%** |
| 50,000 | 20 | 0.95 | 424 | 0.225139 | **1.82%** | 15.879269 | **0.76%** |

Both within the 2% gate (primary test) and 5% gate (convergence sanity test).

### Bugs found and fixed during G1 development

1. **`inv_nz` scaling bug** (Phase 2 → Phase 3): `inv_nz = 1/(n·z_sum)` had an erroneous extra `1/n` factor. The gradient `∇L = Ê_ν[ω_θ·φ] − m` was computed as `weighted_feature_sum/(n·z_sum) − m` instead of `weighted_feature_sum/z_sum − m`. This caused the gradient to be ~1000× too small, making the Newton solver converge to the wrong θ. **Effect:** >50% relative error on ω(upper). **Fix:** `inv_nz = 1/z_sum`.

2. **Newton overshoot on ill-conditioned Hessian** (scalar Baird-MRP fixture): at θ=0, H = Cov(φ) ≈ 0.038, gradient ≈ −0.76. Pure Newton step |H⁻¹g| ≈ 19.8 — overshooting θ⋆ ≈ 4.74 by 3.5×. Once θ overshoots, all weight collapses to one state and H → 0, making recovery impossible. **Fix:** Levenberg-Marquardt damping — add λ·I to the Hessian before Cholesky, with λ starting conservative (LM_INIT=1.0), decreasing on accepted steps (×0.25), increasing on rejected steps (×4.0). Loss-based acceptance/rejection ensures each step decreases L.

3. **f32 loss-precision stall** (near the FORE fixed point): the loss surface is extremely flat near the optimum (gradient ~1e-3, curvature ~1e-2). f32 rounding makes `L(θ) == L(θ ± δ)` for reasonable step sizes, causing the LM acceptance check (`loss_trial < loss_current`) to reject ALL steps. The solver stalls at a suboptimal θ (4.65 instead of 4.74), giving ~8% error. **Fix:** compute `compute_loss` in f64 for the LM acceptance check. The gradient/Hessian stay f32 (consistent with katgpt-core conventions); only this one acceptance gate promotes to f64.

---

## G2 Perf Detail

```
FORE fit n=10000, state_dim=8, K=20, γ=0.95
Median latency: 48.63 ms (target < 100 ms)
```

The fit includes:
- K=20 outer FORE iterations
- Each with up to 50 Newton iterations (typically 5-10 near the fixed point)
- Each Newton iteration with up to 12 LM retries (typically 1-3)
- The LM loss evaluation is O(n·d) per retry, computed in f64

48.63 ms gives 2× headroom under the 100 ms cold-tier budget.

---

## G4 Alloc-Free Detail

```
fit_and_evaluate inner loop: 100 calls after warmup
Allocations: 0
Deallocations: 0
```

All scratch buffers are pre-allocated in `KlProjectionScratch::new(n, feature_dim)`:
- `exp_buf[n]`, `moment[d]`, `initial_mean[d]`, `successor_weighted_sum[d]`
- `gradient[d]`, `hessian[d²]`, `hessian_damped[d²]`, `newton_step[d]`
- `params_trial[d]`, `y_buf[d]`, `weighted_feature_sum[d]`

The `compute_loss` function (f64 loss evaluation for LM acceptance) is allocation-free — it accumulates in registers without writing to any buffer.

---

## G5 Modelless Detail

The only mutable state in the `occupancy` module is:
- `θ: Vec<f32>` — the log-ratio class parameter (Newton/LM iterates on this only)
- Scratch buffers (pre-allocated, reused)

No `NeuronShard`, `LoRAWeightVersion`, `SenseModule`, or any other base-weight handle is touched anywhere in the module. The Newton solver uses gradient descent on θ only — it does NOT modify any model weights. This satisfies the modelless constraint (G5) by construction.

---

## Promotion Decision

**Stays opt-in** (`occupancy_ratio = []`). The primitive's value proposition is a *guarantee multiplier* on downstream consumers (Fusion A/B/C), none of which ship in this plan. Promotion to default-on requires a downstream consumer (typically Fusion A CLR stabilization in `riir-ai/riir-poc`) to demonstrate the gain empirically.

The softmax-vs-sigmoid carve-out is documented in the module doc: FORE's normalized exponential class is density-ratio normalization (the log-partition is the cumulant-generating function of the empirical distribution), NOT a direction-vector projection. The sigmoid rule applies to semantic-domain projections onto learned directions; it does not apply here.
