# Plan 560: SE(2)-Equivariant Lift Primitive + G1+G2 GOAT Gate

**Date:** 2026-07-25
**Research:** [katgpt-rs/.research/457_SE2_Equivariant_Lift_Game_Maps.md](../.research/457_SE2_Equivariant_Lift_Game_Maps.md)
**Source paper:** [arXiv:2403.04807](https://arxiv.org/abs/2403.04807) — Smets, *Mathematics of Neural Networks*, Ch. 3 §3.4 (de-deferred from Research 321 §2.4)
**Target:** `katgpt-rs/crates/katgpt-dec/src/se2_lift.rs` (new module) + Cargo feature `se2_equivariant_lift`
**Status:** ✅ COMPLETE — Phase 1+2+3 ALL DONE (G1+G2 PASS, promoted to default-on). `se2_equivariant_lift` is DEFAULT-ON in `katgpt-dec/Cargo.toml`.

---

## Goal

Ship the modelless SE(2) lifting primitive (`se2_lift_into` + two projections) in `katgpt-dec` behind a feature flag, then run the G1 (rotation-equivariance bit-identical) + G2 (perf < 1ms @ 32×32) gate. If both pass + product selling point holds (it does — see Research 457 §3) → promote to default-on.

This de-defers the "SE(2)-equivariant game maps" item from Research 321 §2.4 per user request 2026-07-25.

## Phase 1 — Skeleton (CORE)

### Tasks

- [x] **T1.1** Add `se2_equivariant_lift = []` feature flag to `katgpt-dec/Cargo.toml` (opt-in, not in default).
- [x] **T1.2** Create `katgpt-dec/src/se2_lift.rs` module with the three primitives: `se2_lift_into`, `se2_project_integrate_into`, `se2_project_max_into`.
- [x] **T1.3** Wire module into `katgpt-dec/src/lib.rs` behind the feature flag.
- [x] **T1.4** Unit tests in `se2_lift.rs`: dim-zero noop, π/2 rotation equivariance bit-exact, projection shapes.
- [x] **T1.5** `cargo check -p katgpt-dec --features se2_equivariant_lift` clean.

## Phase 2 — G1 Equivariance + G2 Perf Gate (CORE)

### Tasks

- [x] **T2.1** Create `katgpt-dec/benches/bench_560_se2_lift_perf.rs` — G2 latency at 16×16, 32×32, 64×64 with 8 orientations, 5×5 kernel.
- [x] **T2.2** Add G1 STRETCH: 45° rotation equivariance up to bilinear-sampling tolerance.
- [x] **T2.3** (Merged with T2.1 — unit tests cover G1 equivariance in se2_lift.rs.)
- [x] **T2.4** Run both benches. Record results in `katgpt-rs/.benchmarks/560_se2_lift_goat.md`.
- [x] **T2.5** G1+G2 PASS → promote `se2_equivariant_lift` to default-on in `katgpt-dec/Cargo.toml`.

## Phase 3 — Doc Sync + riir-ai Guide (POSTERIOR)

### Tasks

- [x] **T3.1** Amend Research 457 with actual gate numbers.
- [x] **T3.2** Create `riir-ai/.research/325_SE2_Equivariant_NPC_Perception_Guide.md` (private Super-GOAT moat doc).
- [x] **T3.3** Update Research 321 §2.4 to note the SE(2) item is no longer deferred.
- [x] **T3.4** Update `katgpt-dec/Cargo.toml` feature comment block to reflect the new primitive.
- [x] **T3.5** Update `katgpt-rs/README.md` feature-flag table if/when promoted.

## Notes

- The primitive is **structurally simple** — pure correlation with N rotated kernels. The interesting part is the equivariance contract and the rotation kernel generation (bilinear sampling of the source kernel grid).
- Group convolution (§3.4.2) is **NOT shipped**. Lifting + projection is sufficient for the equivariance demo and any practical consumer (per-NPC perception, threat fields).
- Latency budget: 1ms target (well under the 50ms 20Hz tick). At 32×32 grid × 8 orientations × 5×5 kernel the inner product is ~205k FMAs — single-digit µs on Apple Silicon NEON.
- Use `CARGO_TARGET_DIR=/tmp/plan560` per AGENTS.md to isolate the build.
