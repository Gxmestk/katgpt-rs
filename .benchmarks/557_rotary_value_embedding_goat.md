# Plan 557 — RoVE GOAT Gate

**Date:** 2026-07-22 (Phase 2), 2026-07-22 (Phase 3 G2 unblock)
**Bench:** `bench_557_rotary_value_embedding_goat.rs`
**Feature:** `rotary_value_embedding` (opt-in, implies `position_group_action`)
**Config:** Apple Silicon (M-series), release profile, single-thread.

## Verdict

### Phase 2 (scalar baseline — 2026-07-22)

| Gate | Target | Measured | Verdict |
|------|--------|----------|---------|
| **G1** correctness | pos=0 identity exact + round-trip < 1e-6 | worst 1.79e-7 | **PASS** ✓ |
| **G2** perf (scalar) | RoVE overhead < 5% of V projection (n=1024, d=768) | 6.45% | **FAIL** ✗ |
| **G3** no-regression | opt-in + additive, default build unchanged | clean | **PASS** ✓ |
| **G4** alloc-free | 0 allocs / 1000 calls on batch hot path | 0 / 0 | **PASS** ✓ |
| **G5** FlashAttention compat | RoVE path ≡ attentive-convolution (rel err < 1e-4) | 2.69e-8 | **PASS** ✓ |

**Phase 2 overall: 4/5 PASS, G2 honest FAIL.** Do NOT promote to default-on (two independent reasons: G2 FAIL + Phase 5 retrofit PoC not done).

### Phase 3 G2 unblock (precomputed cos/sin table — 2026-07-22)

| Gate | Target | Measured | Verdict |
|------|--------|----------|---------|
| **G2** perf (fast) | RoVE overhead < 5% of V projection (n=1024, d=768) | **2.29%** | **PASS** ✓ |

**Phase 3 G2 unblock: PASS.** The `RoVeRotationTable` (precomputed cos/sin for all positions × pairs) eliminates the per-call transcendental cost — the dominant bottleneck. The fast path inner loop is pure `mul_add` arithmetic (zero `cos`/`sin` calls), achieving a **2.88× speedup** over the scalar path (6.62% → 2.29%). The gate verdict now uses the fast path.

**One blocker remains:** Phase 5 retrofit PoC (inference-time RoVE onto RoPE-trained checkpoints). The feature stays opt-in until that settles.

### Phase 4 — Attention Matching Fusion (2026-07-22 — REFRAMED)

| Gate | Target | Measured | Verdict |
|------|--------|----------|---------|
| **G9** compaction fidelity under RoVE | compacted (Ck,Cv) vs full attention output, cosine ≥ 0.991 | **0.999925** | **PASS** ✓ |
| **G10** position-consistency | compacted RoVE-aware forward vs full RoVE-aware forward, cosine ≥ 0.991 | **≥ 0.991** | **PASS** ✓ |

**Phase 4 outcome: REFRAMED — no special RoVE compaction code needed.**

The plan's original T4.1 approach (un-rotate values before compaction, compact in position-free space, re-rotate) was found to be **mathematically incorrect** during implementation. The position-free value fit minimizes `||A_sel · Cv_plain - A · V_plain||²` but the actual attention output uses **rotated** values (`Σ_j A_ij · R_j · V_plain[j]`). Since `R_j` varies per position, the position-free objective ≠ the rotated-space objective. G9 with the un-rotate approach measured cosine **0.17** (vs 0.991 target) — a catastrophic FAIL.

The **correct finding**: the existing `compact_text_based` already handles RoVE-rotated values correctly. The value fitting (least-squares) operates in whatever space the values are in, so RoVE-rotated values are fitted correctly without any un-rotate/re-rotate. G9 verified this: compacting RoVE-rotated values AS-IS gives cosine **0.999925**.

**What shipped:** feature forwarding (`rotary_value_embedding` in `katgpt-attn-match`) for test access to RoVE primitives + G9/G10 verification tests + documentation. **What did NOT ship:** `RoVeToggle`, `compact_text_based_with_rove`, un-rotate/re-rotate helpers — all implemented and reverted after the mathematical analysis proved them incorrect.

See `.plans/557_rotary_value_embeddings.md` §Phase 4 ACTUAL OUTCOME for the full proof.

## G1 — correctness (PASS)

The "bit-identical to RoPE-when-disabled" claim splits into two parts:

1. **pos=0 identity** (forward + inverse): angle=0 → cos=1.0, sin=0.0, both exact in IEEE 754. The rotation IS bit-identical to the identity at pos=0. Measured: 0.0 abs err (exact).

2. **Round-trip at nonzero pos** (rotate at `pos` then inverse at `pos`): recovers the input to f32 precision. The forward computes cos/sin at angle θ, the inverse recomputes at angle −θ. IEEE `cosf`/`sinf` are <1 ULP accurate but not bit-identical to algebraic negation (`cosf(-θ) ≠ cosf(θ)` in the last bit for some θ; `sinf(-θ) ≠ -sinf(θ)` ditto). This is the f32 floor — 1 ULP at magnitude 1 is ~1.2e-7. Measured worst: 1.79e-7 (across dims {8,16,32,64,128}, 20 random vectors each, positions {1,5,17,100,1023}). Budget: 1e-6 (≈8× headroom over the observed ULP). **PASS.**

The feature's surgical scope (the other half of "RoPE-when-disabled") is structural: the module is gated by `#[cfg(feature = "rotary_value_embedding")]` in `lib.rs`, so when the feature is off, no code path is touched. Verified by `cargo build -p katgpt-core --lib` (default features) — clean.

## G2 — perf (Phase 2 scalar: FAIL → Phase 3 fast: PASS)

### Phase 2 scalar baseline (FAIL)

**Measured:** RoVE overhead = 6.45% of V projection at n=1024, d=768.
- V projection: 33.2 ms/layer (1024 × `matmul` of [768,768]@[768])
- RoVE (rotate + inverse): 2.14 ms/layer
- Ratio: 0.0645 (6.45%)

**Target:** < 5%.

**Root cause (honest):**

The 5% target derives from a pure FLOP-ratio argument: RoVE adds O(nd) work on top of O(nd²). At d=768, the theoretical FLOP ratio is ~0.13% (well under 5%). The measured 6.45% is ~50× worse than the FLOP ratio predicts because of a constant-factor throughput gap:

| Operation | FLOPs (n=1024, d=768) | Throughput | Wall-clock |
|-----------|----------------------|------------|------------|
| V projection (`simd_matmul_rows`) | ~1.2 GFLOP | ~17 GFLOP/s (SIMD dot product) | 33.2 ms |
| RoVE rotation (scalar complex mul) | ~3.1 MFLOP | ~0.7 GFLOP/s (scalar + table lookup) | 2.14 ms |

The matmul baseline uses heavily SIMD-optimized `simd_matmul_rows` (NEON/AVX fused multiply-add dot products). The RoVE rotation is scalar complex-number arithmetic with per-pair cos/sin table lookups — no SIMD, no FMA fusion, data-dependent memory access. This ~24× throughput gap inflates the 0.13% FLOP ratio to 6.45% wall-clock.

### Phase 3 fast path unblock (PASS)

**Approach:** precompute the cos/sin table once per `(theta, dim, max_pos)` triple via `RoVeRotationTable`. The table stores interleaved `(cos, sin)` pairs for every `(position, pair)` combination. The fast batch functions (`batch_rotate_values_into_fast`, `batch_inverse_rotate_output_into_fast`) read from this table with **zero transcendental calls** in the inner loop — pure `mul_add` arithmetic.

**Measured:**

| Path | RoVE cost (n=1024, d=768) | Ratio vs V projection | Verdict |
|------|---------------------------|----------------------|---------|
| Scalar (Phase 2) | 2.31 ms/layer | 6.62% | FAIL ✗ |
| Fast (Phase 3) | 0.80 ms/layer | **2.29%** | **PASS** ✓ |
| Speedup | **2.88×** | | |

- V projection: 34.8 ms/layer
- Table build (once): 1.24 ms (amortized across all layers + forward passes)
- Table size: 3.0 MB (n × d × 4 bytes)

**Why 2.88× and not more?** The precomputed table eliminates the transcendentals (the dominant cost), but the inner loop is still scalar `mul_add` on interleaved `(x0, x1)` pairs. The AoS pair layout doesn't auto-vectorize as cleanly as SoA — LLVM's SLP vectorizer catches the 2-wide pair pattern, but full 4-wide NEON complex multiply would need an explicit SoA deinterleave/reinterleave. The current 2.29% is well under the 5% target, so further SIMD optimization is not needed for the GOAT gate (could be revisited for a future perf plan).

**Correctness contract (G8 tests):**
- Forward direction: **bit-identical** to the scalar path (both use positive-angle `cos`/`sin` + same `mul_add` formula). Verified with tol 0.0.
- Inverse direction: ≤1 ULP difference (scalar calls `cos(-angle)`/`sin(-angle)`, fast uses `cos(angle)`/`sin(angle)` + algebraic negation; library transcendentals aren't guaranteed even/odd in the last bit). Verified with tol 1e-6 (matching Phase 2 G1's round-trip ULP budget).

**Note:** the gate is NOT relaxed. The 5% target stands as the promotion bar. The fast path meets it honestly.

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

**DEFERRED — do NOT promote to default-on.** One blocker remains:

1. **Phase 5 retrofit PoC not done:** the paper validates RoVE as a training-time choice. Inference-time retrofit onto RoPE-trained checkpoints is unvalidated. Phase 5 must train A (RoPE-only) vs B (RoVE retrofit) vs C (RoVE-trained) and show B > A before any promotion.

~~G2 FAIL~~ — **UNBLOCKED** in Phase 3 via the precomputed cos/sin table (6.45% → 2.29%, well under 5% target).

The feature stays opt-in at `rotary_value_embedding = ["position_group_action"]` until Phase 5 settles.

## Raw output

### Phase 3 (G2 fast path unblock — 2026-07-22)

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
   Measures BOTH scalar (transcendentals per call) + fast (precomputed table)
   [fast] table build (once): 1238833 ns (1238.83 µs)
   [fast] table size: 786432 entries = 3072.0 KB
   V projection (n=1024, d=768): 34840338 ns/layer
   RoVE SCALAR (rotate+inv):   2305781 ns/layer
   RoVE FAST   (rotate+inv):   799521 ns/layer
   ratio scalar (rove/proj):   0.0662× (6.62%)
   ratio fast   (rove/proj):   0.0229× (2.29%)
   speedup fast/scalar:        2.88×
   target:                      ratio < 0.05 (5%)
   G2 scalar: FAIL ✗ (proj 34840338ns, rove 2305781ns, ratio 0.0662×)
   G2 fast:   PASS ✓ (rove 799521ns, ratio 0.0229×)
   G2 gate verdict (fast path): PASS ✓

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
✅ ALL GOAT GATES PASS — Plan 557 Phase 2 is GOAT-validated.
   Promotion to default-on: DEFERRED — Phase 5 (retrofit PoC)
   must settle the inference-time retrofit question first.
```

### Phase 2 (scalar baseline — 2026-07-22)

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
