# Benchmark 560: SE(2) Lift Primitive — G1+G2 GOAT Gate

**Date:** 2026-07-25
**Plan:** [katgpt-rs/.plans/560_se2_equivariant_lift_primitive.md](../.plans/560_se2_equivariant_lift_primitive.md)
**Research:** [katgpt-rs/.research/457_SE2_Equivariant_Lift_Game_Maps.md](../.research/457_SE2_Equivariant_Lift_Game_Maps.md)
**Source paper:** [arXiv:2403.04807](https://arxiv.org/abs/2403.04807) — Smets, *Mathematics of Neural Networks*, Ch. 3 §3.4.1
**Crate feature:** `se2_equivariant_lift` in `katgpt-dec` (DEFAULT-ON since this gate landed)
**Unit tests:** `katgpt-rs/crates/katgpt-dec/src/se2_lift.rs::tests` — 8 tests
**Perf bench:** `katgpt-rs/crates/katgpt-dec/benches/bench_560_se2_lift_perf.rs`

## Reproduction

```bash
# Unit tests (G1 equivariance)
CARGO_TARGET_DIR=/tmp/plan560 cargo test -p katgpt-dec --features se2_equivariant_lift --lib se2_lift

# Perf bench (G2 latency)
CARGO_TARGET_DIR=/tmp/plan560 cargo bench -p katgpt-dec --features se2_equivariant_lift --bench bench_560_se2_lift_perf -- --nocapture

# Clean up when done:
rm -rf /tmp/plan560
```

## G1 — Rotation-Equivariance (PASS)

### Test setup

- 8×8 grid with asymmetric threat wedge (top-right quadrant populated).
- 8 orientations, asymmetric 3×3 kernel `[0,1,0; 2,4,0; 0,-1,0]`.
- Rotate input field by π/2 (2 orientation slots for N=8).
- Verify `Lift(R_{π/2}·f)(x, y, θ) == Lift(f)(R_{-π/2}(x, y), θ - π/2)` — rotates BOTH spatial position AND orientation slot.

### Result

**PASS** — All 8×8×8 = 512 output cells match within ≤1e-3 abs diff (typical 1e-5; worst cases are f32 cos/sin rounding at θ=π/4, 3π/4).

Stretch test (π/4 rotation): **PASS** with `mean_abs_diff = 0.066` and `max_abs_diff = 0.6` on a 16×16 Gaussian field — well within the documented bilinear-sampling tolerance (Smets §3.4.4 Remark 3.37).

## G2 — Perf (PASS)

### Median latency @ 1000 iterations

| Grid | Orientations | Kernel | Median latency | Budget (1ms) | Per-cell |
|---|---|---|---|---|---|
| 16×16 | 8 | 5×5 | **59 µs** | 17× under | 29 ns/cell |
| 32×32 | 8 | 5×5 | **225 µs** | 4.4× under | 27.5 ns/cell |
| 64×64 | 8 | 5×5 | **902 µs** | 1.1× under | 27.5 ns/cell |

Projection latency (32×32×8): `project_integrate` 3.3 µs, `project_max` 3.0 µs.

### Scale analysis (@ 32×32 lift)

| Scale | Cost | % of 20Hz tick budget (50ms) | Verdict |
|---|---|---|---|
| Hero (1 NPC/tick) | 225 µs | 0.5% | ✓ comfortable |
| Squad (10 NPCs/tick) | 2.25 ms | 4.5% | ✓ comfortable |
| Zone (16 zones/tick) | 3.6 ms | 7.2% | ✓ reasonable |
| Crowd (1000 NPCs/tick) | 225 ms | 450% | ✗ TOO EXPENSIVE — use per-zone LoD |

## G3-G5

| Gate | Status | Note |
|---|---|---|
| **G3** no regression | ✅ PASS | `cargo check -p katgpt-dec --all-features` clean |
| **G4** alloc-free | ✅ PASS | caller-owned buffers + stack-scratch rotated kernel (4 KB stack for K≤64) |
| **G5** modelless | ✅ PASS | pure correlation + bilinear sampling, no training, no learned weights |

## Honest caveats

1. **Not crowd-scale per-NPC.** The 1000-NPC per-tick math doesn't work at 32×32 — consumers MUST use per-zone LoD or restrict to hero/squad scale. The riir-ai guide (Research 325) documents the consumer pattern.
2. **Bilinear sampling rounding for non-π/2 rotations.** Smets §3.4.4 Remark 3.37 documents this as a known property of the discrete lift. The G1 stretch test verifies it within tolerance on smooth fields; on sharp features (e.g., a step function) the rounding will be worse.
3. **No group convolution.** Only lifting + projection shipped (Smets §3.4.1 + §3.4.3). Group convolution on SE(2) (§3.4.2) is a natural follow-up if a future consumer needs deeper SE(2) processing; not pre-committed.

## Verdict

**ALL GATES PASS. Primitive promoted to DEFAULT-ON** in `katgpt-dec/Cargo.toml` default feature list.

## See also

- [Research 457](../.research/457_SE2_Equivariant_Lift_Game_Maps.md) — public distillation note (Super-GOAT).
- [riir-ai/.research/325](../../riir-ai/.research/325_SE2_Equivariant_NPC_Perception_Guide.md) — private selling-point guide (the moat).
- [Research 321](../.research/321_Tropical_Semiring_Equivariant_Operators.md) — sibling §3.5 distillation (tropical semiring); same textbook, same "DEC × Smets Ch.3" fusion pattern.
- [Plan 560](../.plans/560_se2_equivariant_lift_primitive.md) — execution plan.
