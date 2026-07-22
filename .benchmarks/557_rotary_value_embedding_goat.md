# Plan 557 Phase 2 — RoVE GOAT Gate

**Date:** 2026-07-22
**Bench:** `bench_557_rotary_value_embedding_goat.rs`
**Feature:** `rotary_value_embedding` (opt-in, implies `position_group_action`)
**Config:** Apple Silicon (M-series), release profile, single-thread.

## Verdict

| Gate | Target | Measured | Verdict |
|------|--------|----------|---------|
| **G1** correctness | pos=0 identity exact + round-trip < 1e-6 | worst 1.79e-7 | **PASS** ✓ |
| **G2** perf | RoVE overhead < 5% of V projection (n=1024, d=768) | 6.45% | **FAIL** ✗ |
| **G3** no-regression | opt-in + additive, default build unchanged | clean | **PASS** ✓ |
| **G4** alloc-free | 0 allocs / 1000 calls on batch hot path | 0 / 0 | **PASS** ✓ |
| **G5** FlashAttention compat | RoVE path ≡ attentive-convolution (rel err < 1e-4) | 2.69e-8 | **PASS** ✓ |

**Overall: 4/5 PASS, G2 honest FAIL.** Do NOT promote to default-on (two independent reasons: G2 FAIL + Phase 5 retrofit PoC not done).

## G1 — correctness (PASS)

The "bit-identical to RoPE-when-disabled" claim splits into two parts:

1. **pos=0 identity** (forward + inverse): angle=0 → cos=1.0, sin=0.0, both exact in IEEE 754. The rotation IS bit-identical to the identity at pos=0. Measured: 0.0 abs err (exact).

2. **Round-trip at nonzero pos** (rotate at `pos` then inverse at `pos`): recovers the input to f32 precision. The forward computes cos/sin at angle θ, the inverse recomputes at angle −θ. IEEE `cosf`/`sinf` are <1 ULP accurate but not bit-identical to algebraic negation (`cosf(-θ) ≠ cosf(θ)` in the last bit for some θ; `sinf(-θ) ≠ -sinf(θ)` ditto). This is the f32 floor — 1 ULP at magnitude 1 is ~1.2e-7. Measured worst: 1.79e-7 (across dims {8,16,32,64,128}, 20 random vectors each, positions {1,5,17,100,1023}). Budget: 1e-6 (≈8× headroom over the observed ULP). **PASS.**

The feature's surgical scope (the other half of "RoPE-when-disabled") is structural: the module is gated by `#[cfg(feature = "rotary_value_embedding")]` in `lib.rs`, so when the feature is off, no code path is touched. Verified by `cargo build -p katgpt-core --lib` (default features) — clean.

## G2 — perf (FAIL, honest)

**Measured:** RoVE overhead = 6.45% of V projection at n=1024, d=768.
- V projection: 33.2 ms/layer (1024 × `matmul` of [768,768]@[768])
- RoVE (rotate + inverse): 2.14 ms/layer
- Ratio: 0.0645 (6.45%)

**Target:** < 5%.

**Why the target was missed (honest root cause):**

The 5% target derives from a pure FLOP-ratio argument: RoVE adds O(nd) work on top of O(nd²). At d=768, the theoretical FLOP ratio is ~0.13% (well under 5%). The measured 6.45% is ~50× worse than the FLOP ratio predicts because of a constant-factor throughput gap:

| Operation | FLOPs (n=1024, d=768) | Throughput | Wall-clock |
|-----------|----------------------|------------|------------|
| V projection (`simd_matmul_rows`) | ~1.2 GFLOP | ~17 GFLOP/s (SIMD dot product) | 33.2 ms |
| RoVE rotation (scalar complex mul) | ~3.1 MFLOP | ~0.7 GFLOP/s (scalar + table lookup) | 2.14 ms |

The matmul baseline uses heavily SIMD-optimized `simd_matmul_rows` (NEON/AVX fused multiply-add dot products). The RoVE rotation is scalar complex-number arithmetic with per-pair cos/sin table lookups — no SIMD, no FMA fusion, data-dependent memory access. This ~24× throughput gap inflates the 0.13% FLOP ratio to 6.45% wall-clock.

**Unblock path:** SIMD RoVE kernel. The rotation `(x₀, x₁) → (x₀·cos − x₁·sin, x₀·sin + x₁·cos)` is embarrassingly parallel across dim/2 pairs and across n tokens — a natural target for NEON/AVX vectorization (4-wide f32 complex multiplies per SIMD instruction). Estimated post-SIMD throughput: ~10 GFLOP/s, which would bring the ratio to ~0.3% — well under 5%. This is Phase 3 hot-path wiring work (the SIMD kernel lives in the attention forward path, not in the primitive itself).

**Note:** the gate is NOT relaxed. The 5% target stands as the promotion bar. G2 is recorded as honest FAIL at the current scalar implementation level. The primitive is correct (G1, G5) and alloc-free (G4); the perf gate fails because the scalar implementation doesn't meet the SIMD-aware throughput target.

## G3 — no-regression (PASS)

The feature is opt-in (`rotary_value_embedding = ["position_group_action"]` in `Cargo.toml`). No existing module imports `rotary_value_embedding`. Turning it on cannot interact with any other feature.

Verified:
- `cargo build -p katgpt-core --lib` (default features) — clean.
- `cargo clippy --features rotary_value_embedding --bench bench_557_*` — zero warnings.
- `cargo test --features rotary_value_embedding --lib rotary_value_embedding` — 9/9 Phase 1 tests pass.
- Full CI feature matrix (`./scripts/ci_feature_guard.sh`) — verified externally (default + opt-in + --all-features).

## G4 — alloc-free (PASS)

`batch_rotate_values_into` + `batch_inverse_rotate_output_into` at n=1024, d=768:
- 0 allocations / 0 deallocations over 1000 calls (post-warmup).
- The batch primitives take caller-owned flat `[n*dim]` buffers and operate via per-token slice borrows — zero allocation in the loop by construction.

Verified via `CountingAllocator` (the `counting_allocator!()` macro from `tests/common/mod.rs`).

## G5 — FlashAttention output-equivalence (PASS)

The algebraic identity:

```
ỹ_i = R_{−i} · Σ_j A_ij · R_j · V_j      (RoVE path — what FlashAttention would do)
    = Σ_j A_ij · R_{j−i} · V_j            (attentive-convolution reference path)
```

**Method:** computed both paths on the same random fixture (n=16, d=32, random V + softmaxed attention weights) and compared element-wise.

- **Path A (RoVE):** rotate each V_j by R_j → standard attention aggregation → inverse-rotate output by R_{−i}.
- **Path B (reference):** apply R_{j−i} per (i,j) pair directly during aggregation.

**Result:** worst abs err = 1.19e-7, rel err = 2.69e-8 (vs ‖y_ref‖). Budget: 1e-4 (f32 accumulation-order tolerance — Path A accumulates over rotated V, Path B accumulates over per-pair rotated V; different summation orders produce ~1 ULP differences). **PASS** — the identity holds to f32 precision, confirming the RoVE path is output-equivalent to the attentive-convolution form and thus FlashAttention-compatible (rotations act on V pre-kernel and output post-kernel, never on the n×n score matrix).

## Promotion decision

**DEFERRED — do NOT promote to default-on.** Two independent blockers:

1. **G2 FAIL (6.45% vs 5%):** the scalar implementation doesn't meet the SIMD-aware throughput target. SIMD RoVE (Phase 3) is the unblock path.
2. **Phase 5 retrofit PoC not done:** the paper validates RoVE as a training-time choice. Inference-time retrofit onto RoPE-trained checkpoints is unvalidated. Phase 5 must train A (RoPE-only) vs B (RoVE retrofit) vs C (RoVE-trained) and show B > A before any promotion.

The feature stays opt-in at `rotary_value_embedding = ["position_group_action"]` until both blockers resolve.

## Raw output

```
╔══════════════════════════════════════════════════════════════════╗
║  Plan 557 Phase 2 — RoVE GOAT Gate                               ║
╚══════════════════════════════════════════════════════════════════╝

── G1 (correctness): bit-identical to RoPE-when-disabled ────────
   pos=0 identity (exact) + round-trip at nonzero pos (f32 floor)
   d=  8: max abs err = 1.192e-7 (budget 1e-6)
   d= 16: max abs err = 1.788e-7 (budget 1e-6)
   d= 32: max abs err = 1.192e-7 (budget 1e-6)
   d= 64: max abs err = 1.192e-7 (budget 1e-6)
   d=128: max abs err = 1.192e-7 (budget 1e-6)
   worst-overall abs err = 1.788e-7
   G1: PASS ✓

── G2 (perf): RoVE overhead < 5% of V projection ────────────────
   n=1024, d=768 (paper small-model config)
   V projection (n=1024, d=768): 33246219 ns/layer
   RoVE overhead (rotate+inv):  2142904 ns/layer
   ratio (rove/proj):           0.0645× (6.45%)
   target:                      ratio < 0.05 (5%)
   G2: FAIL ✗ (proj 33246219ns, rove 2142904ns, ratio 0.0645×)

── G3 (no-regression): feature is opt-in and additive ────────────
   (full G3 verified externally via ./scripts/ci_feature_guard.sh)
   G3: PASS ✓ (compiles when feature on; CI verifies off-case)

── G4 (alloc-free): 0 allocs / 1000 calls on batch hot path ─────
   batch_rotate + batch_inverse: allocs=0, deallocs=0 over 1000 calls
   G4: PASS ✓

── G5 (FlashAttention compat): RoVE path ≡ attentive-convolution ─
   (R_{-i} · Σ_j A_ij · R_j · V_j) = (Σ_j A_ij · R_{j-i} · V_j)
   n=16, d=32: worst abs err = 1.192e-7, rel err = 2.690e-8
   budget: rel err < 1e-4 (f32 accumulation-order tolerance)
   G5: PASS ✓ (rel err 2.690e-8)

──────────────────────────────────────────────────────────────────
❌ SOME GATES FAILED — see above. Do NOT promote to default-on.
```

## Run command

```bash
CARGO_TARGET_DIR=/tmp/plan557_p2 cargo bench -p katgpt-core \
  --features rotary_value_embedding \
  --bench bench_557_rotary_value_embedding_goat -- --nocapture
```
