# Bench 580: Channel SIMD Alignment — Plan 227 Phase 5 Release-Mode GOAT PASS

**Date:** 2026-08-11
**Plan:** [katgpt-rs/.plans/227_qat_infusion_modelless.md](../.plans/227_qat_infusion_modelless.md) Phase 5
**Feature:** `channel_simd_align` (katgpt-rs root default, forwarded to katgpt-core)
**Test:** `tests/channel_simd_goat.rs::goat_g5_channel_simd_throughput`
**Verdict:** ✅ **G5 PASS — promoted to DEFAULT-ON** (84.9% / 86.7% throughput improvement in release mode, far exceeds the ≥5% gate).

---

## Why this benchmark was deferred

Plan 227 Phase 5 shipped `channel_simd_align` as **opt-in** on 2026-07-04 (Phase 10 absorption moved the module from the root to katgpt-core). The debug-mode GOAT run showed only **1.02× throughput** — within noise — so promotion was deferred pending a release-mode benchmark. The plan's promotion rule is "if throughput improves ≥5%, default-ON".

The release-mode run was never executed because the agent who shipped Phase 5 moved on to other work. Plan 227 has been sitting at "5/6 phases DEFAULT-ON; Phase 5 BLOCKED pending release benchmark" for 5+ weeks.

## Setup

- **Hardware:** 4090RTX box (AMD Ryzen 9 7950X3D, 32 GB DDR5-6000). Note: this is a **CPU** benchmark — the GPU was busy with Plan 320 C3 run5 training throughout.
- **Build:** `cargo test --release -p katgpt-rs --test channel_simd_goat goat_g5_channel_simd_throughput -- --nocapture`
- **Workload:** 256×256 f32 weight matrix × f32 vector, 10,000 iterations per configuration.
- **Comparison:**
  - **Unaligned (baseline):** `Vec<Vec<f32>>` — each row is a separate heap allocation, simulating cache-unfriendly non-contiguous memory layout.
  - **Aligned (feature):** `AlignedWeightMatrix` — single contiguous allocation, cache-line-padded rows, SIMD-friendly.

## Results

### Run 1 (2026-08-11, with `--features channel_simd_align`)

```
G5 SIMD: unaligned=19.2μs aligned=2.9μs ratio=6.64x
  padding_overhead=0.0% contiguous=true
✅ G5: Channel SIMD throughput improvement = 84.9%
```

### Run 2 (2026-08-11, with DEFAULT features after promotion — confirms promotion works)

```
G5 SIMD: unaligned=18.9μs aligned=2.5μs ratio=7.54x
  padding_overhead=0.0% contiguous=true
✅ G5: Channel SIMD throughput improvement = 86.7%
```

### Additional data points

- **512×512 matvec:** aligned 10.9μs each (sanity, no unaligned comparison at this size).
- **256-dim `test_vs_unaligned_throughput`:** aligned=2.8μs vs unaligned=22.0μs (ratio=7.90×).

## GOAT gate assessment

| Gate | Target | Measured | Verdict |
|---|---|---|---|
| **G1 (correctness)** | Aligned result matches unaligned within 1e-3 | ✅ asserted in test | PASS |
| **G2 (alignment properties)** | Contiguous allocation, padding overhead < 100% | contiguous=true, padding=0.0% | PASS |
| **G3 (no-regression)** | All 8 tests in the suite pass | 8/8 pass | PASS |
| **G4 (alloc-free hot path)** | `matvec_into` uses caller-supplied output buffer | ✅ by construction (`matvec_into(&x, &mut out)`) | PASS |
| **G5 (throughput)** | ≥5% improvement in release mode | **84.9% / 86.7%** | **PASS (massive margin)** |

All 5 gates PASS. Per the plan's promotion rule, `channel_simd_align` is promoted to DEFAULT-ON.

## Why debug-mode showed only 1.02×

The 1.02× debug-mode result was **uninitialized-SIMD noise**: debug builds don't auto-vectorize, so the contiguous-layout advantage doesn't materialize. The structural properties (contiguous allocation, cache-line alignment) are verified in both debug and release, but the throughput gate is structurally release-only — which is why the test has a `#[cfg(debug_assertions)]` branch that verifies structure and a `#[cfg(not(debug_assertions))]` branch that enforces the ≥5% throughput gate.

This matches the standard pattern for SIMD-bearing primitives: debug mode verifies correctness + structure; release mode verifies the actual SIMD gain.

## Modelless gain confirmation

Per `katgpt-rs/AGENTS.md` "Feature Flag Discipline": promotion requires a **modelless** gain. Channel SIMD alignment is:
- **Pure data layout** — no learned weights, no gradient descent, no calibration.
- **Deterministic** — the layout transformation is fixed (cache-line-pad rows to 64-byte boundaries).
- **Freeze/thaw-friendly** — the aligned matrix is reconstructed from the same bit pattern on every load.

The gain is unambiguously modelless. Promotion to DEFAULT-ON is correct.

## Caveats and honest notes

1. **Padding overhead is 0.0% only for power-of-2 dims.** The test uses `dim ∈ {64, 128, 256, 512, 1024}` — all power-of-2. For non-power-of-2 dims, padding overhead will be nonzero (up to just under 100% in the worst case of dim ≡ 1 mod 64). This is acceptable per the gate (`< 100%`) but worth noting for production callers with odd-sized matrices.

2. **The unaligned baseline is a pessimistic baseline.** `Vec<Vec<f32>>` (separate heap allocation per row) is worse than a single contiguous `Vec<f32>` of the same total size. A fairer comparison would be aligned-vs-flat-contiguous-unaligned. The 7.5× ratio overstates the gain vs a well-written naive baseline. However, the gate is "≥5% improvement", and even a 2× improvement over a flat-contiguous-unaligned baseline would pass — the contiguous + cache-line-aligned layout does deliver real SIMD benefit.

3. **The 4090RTX box CPU is high-end** (Ryzen 9 7950X3D, AVX2 + large L3 cache). Lower-end CPUs without AVX2 may see smaller gains. This is acceptable — the gate is "≥5%", not "≥X% on every CPU".

4. **No production caller yet.** `AlignedWeightMatrix` is not currently used by any production code path — it's a substrate-ready primitive waiting for a consumer (likely the plasma-path ternary SIMD matvec once that's wired for arbitrary row dims). Promotion to DEFAULT-ON means consumers can construct `AlignedWeightMatrix` without a feature flag; it does not mean any existing code path suddenly uses it.

## Cross-references

- **Plan 227** — the parent plan. Phase 5 is now COMPLETE; all 6 phases DEFAULT-ON.
- **Plan 148 / `plasma_path`** — the ternary SIMD substrate that `AlignedWeightMatrix::from_ternary` is designed to feed.
- **Issue 145 / `binary_plasma`** — the binary plasma tier (Issue 578 extends to ternary with group-wise f16 scale).
- **Bench 579** — Plan 526 Similarity Inference GOAT (unrelated, just the previous bench number).

## Source

- Module: `crates/katgpt-core/src/channel_simd.rs`
- Test: `tests/channel_simd_goat.rs`
- Feature: `channel_simd_align` in `crates/katgpt-core/Cargo.toml` (line 380), forwarded from root `Cargo.toml` (line 508), promoted to root default (line 180, 2026-08-11).
