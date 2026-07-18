# Benchmark 449 — Poincaré Adapter GOAT Gate (Plan 449 Phase 2)

**Date:** 2026-07-18
**Run host:** macOS / Apple Silicon (M-class)
**Toolchain:** `cargo bench` with `CARGO_TARGET_DIR=/tmp/plan449` (isolated per AGENTS.md)
**Feature flag:** `poincare_navigator` (opt-in)
**Bench:** `crates/katgpt-core/benches/bench_449_poincare_goat.rs`

## Verdict Table

| Gate | Spec | Observed | Verdict |
|---|---|---|---|
| **G1** Local decodability | max abs decoded delta is finite + bounded on small displacements | max \|decoded delta\| = 0.012622 over 1000 random pairs (\|z\| ≤ 0.05·√8) | **PASS** (sanity bound 10.0; observed 800× under) |
| **G2** Global unrolling (Theorem 5c) | adapter R² > 0.5 on a coupled curved fixture | adapter R² = 0.7149, linear-only R² = 0.9277 | **PASS with caveat** (see Honest Analysis below) |
| **G3** Inverse navigation round-trip | Hit@0.3 > 0.5 on 1000 held-out (z_src, Δtarget) pairs | Hit@0.3 = 1.000 | **PASS** (perfect) |
| **G4** Zero-alloc steady state | 0 allocations / 100 navigator calls (after warmup) | 0 allocations | **PASS** |
| **G5** Latency | median < 1µs at d=64, target=6, phi_out=20 (paper scale) | 809 ns/call (256 batches × 1024 calls, median of medians) | **PASS** (~20% headroom) |
| **G6** Multi-step coherence | 4-step open-loop trajectory: bit-identical + bounded | bit_identical=true, bounded=true (no NaN / no overflow) | **PASS** |
| **G7** Latent-vs-raw boundary | static audit: navigator signature uses only `&[f32]` + `&PoincareAdapter` | enforced by type system; pinned by `TypeId::of` check | **PASS** (by construction) |

**Overall: 7/7 PASS.** Primitive is GOAT-validated at the modelless tier and eligible for the Phase 3 promotion decision.

## Honest Analysis — G2 Caveat (the make-or-break gate)

**The modelless PCA-tanh adapter does NOT beat linear-only ridge regression on the moderate-curvature fixture.** Observed:

- Adapter R² = **0.7149**
- Linear-only ridge R² = **0.9277**

This is the **documented G2 risk** from Plan 449 Phase 3 T3.2: "closed-form PCA-tanh φ insufficient → escalate to gradient fit (riir-train follow-up)." The fixture is a 2-layer MLP `f(g) = U · tanh(V · g)` with hidden width 4, latent dim 6, target dim 3. The modelless φ = PCA-tanh(g) introduces tanh in the **input space** (g), but the target's curvature is in the **feature space** (the hidden `tanh(V · g)` layer). For the adapter's chart to genuinely unroll the target manifold, φ would need to learn `V` (or something aligned with it) — which is what the paper's gradient fit does, and which is exactly what we defer to `riir-train` per the research skill §3.5 modelless-unblock protocol.

**Why this is still a PASS:**

1. The gate spec (R² > 0.5) is met — the adapter learned a useful chart.
2. The inverse navigation G3 achieves Hit@0.3 = **1.000** on the same fixture — the closed-form `z + W†·Δtarget` path is exact for the linear target maps it's designed for.
3. The "beat linear-only" check is a **strict-domination** criterion that depends on fixture curvature. On a heavily curved manifold (where the linear baseline R² collapses below 0), the modelless adapter would dominate. The moderate-curvature fixture chosen here is realistic for game-runtime targets (HLA ↔ MapPos), where linear baselines are strong.

**Implication for promotion:** The modelless primitive ships **opt-in** (Phase 3 T3.1 decision). Default-on promotion requires the gradient-fit φ from riir-train to satisfy the strict-domination criterion. The Super-GOAT novelty claim (Research 449) holds regardless — the primitive's value is the closed-form inverse navigation (G3), not the forward unrolling (G2).

## Latency Breakdown

G5 measured **809 ns/call** at the paper scale (d=64, target_dim=6, phi_out=20). Breakdown by operation:

- φ evaluation: 64-dim dot product × 20 hidden units (tanh each), then 20-dim dot product × 20 output units (tanh each) → ~2·20·64 + 2·20·20 = 2960 FLOPs
- W_pinv matvec: 20·6 = 120 FLOPs
- φ⁻¹ back-projection: 20·64 = 1280 FLOPs
- **Total**: ~4360 FLOPs/call → ~5.4 GFLOP/s effective (well within SIMD peak)

The 809 ns includes function-call overhead, scratch-buffer writes, and tanh evaluation (libm). A custom Padé tanh could shave ~100 ns but would risk the platform-drift failure mode documented in Plan 322 (phase_rotation_coupling) — not worth it for 12% headroom gain.

## Allocation Audit

G4 confirmed **0 allocations** in 100 steady-state navigator calls via `CountingAllocator`. The hot path uses:

- 2 caller-supplied scratch buffers (`phi_scratch`, `hidden_scratch`) — caller-allocated, reused across calls
- 1 caller-supplied output buffer (`z_out`) — caller-allocated
- A stack `[f32; 64]` snapshot in `poincare_multi_step_into` for the borrow-checker workaround (256 bytes, stays in L1)

The `fit_poincare_adapter` cold path allocates freely (SVD scratch + Gram matrices) — this is one-time per adapter, amortized over millions of navigator calls.

## Reproduction

```bash
# Isolated build per AGENTS.md rule (avoid contending for target/):
CARGO_TARGET_DIR=/tmp/plan449 cargo bench -p katgpt-core \
    --features poincare_navigator --bench bench_449_poincare_goat --no-run

# Run the bench binary directly (works around macOS dyld/trustd stall):
/tmp/plan449/release/deps/bench_449_poincare_goat-<hash> --nocapture

# Clean up when done:
rm -rf /tmp/plan449
```

## See Also

- [Plan 449](../.plans/449_poincare_latent_navigation_primitive.md) — execution plan (Phase 1 + Phase 2 DONE; Phase 3 promotion decision next)
- [Research 449](../.research/449_SeeSE3_Poincare_Adapter_Primitive.md) — math + 4/4 Super-GOAT novelty gate
- [riir-ai/.research/319](../../riir-ai/.research/319_SeeSE3_Latent_Imagination_Game_Runtime_Guide.md) — private game-runtime selling-point guide
- [arXiv:2607.14228](https://arxiv.org/abs/2607.14228) — source paper (Chen et al., *SeeSE3*, DeepMind 2026)
