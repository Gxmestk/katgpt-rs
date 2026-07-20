# Plan 556: KARC Mitigations — Open Primitives (Regime Gate + Batched Bucket)

**Date:** 2026-07-20
**Companion:** [riir-ai/.plans/514_karc_mitigations_runtime.md](../../riir-ai/.plans/514_karc_mitigations_runtime.md) (runtime integration)
**Source design analysis:** conversation 2026-07-20 — KARC pros/cons review + user-proposed mitigations (fusion, batching, LOD, 2-way KARC, dual LEO+KARC, mux, octree, DDTree)
**Target:** `katgpt-rs/crates/katgpt-core/src/karc/regime_gate.rs` + `katgpt-rs/crates/katgpt-core/src/karc/batched_bucket.rs` (new modules) + Cargo features
**Status:** Phase 1 ✅ **COMPLETE + INTEGRATED (2026-07-20)** — `KarcRegimeGate` primitive shipped + Plan 514 runtime integration landed with **G1 PASS (92.45% MAE reduction on mixed-regime NPC corpus)** + **G2 essentially at-budget (89 ns/tick)**. **Primitive revised from variance-only to MSE** (variance + bias²) after Plan 514 surfaced the failure mode where a consistently-biased forecaster has variance 0 but large error — see `regime_gate.rs` module docstring "Why MSE, not variance" section. Phases 2–3 remain TODO (`karc_batched_matvec` + `KarcLodTier`).

---

## Goal

Ship three modelless open primitives that **mitigate KARC's structural cons** without changing the KARC algorithm itself. The mitigations close the periodic-blindness gap (con #1, Bench 010), amortize the per-NPC fit cost at crowd scale (con #9), and provide the math substrate for the runtime-side LOD + mux work in Plan 514.

1. **`KarcRegimeGate`** — closed-form residual-variance mux. Routes each tick to KARC (chaotic regime) or Seasonal (periodic regime) by comparing rolling Welford variance of their residuals. **Directly fixes con #1 (periodic-blindness).** No training, no learned gate weights — pure empirical comparison + sigmoid smoothing.
2. **`karc_batched_matvec`** — SIMD-batched forecast across N forecasters of identical `(D, M, K)` shape. Same `Wout` row stride, batched `simd_matvec` per channel. **Amortizes memory bandwidth** (the Plan 308 381 ns/call is per-NPC; batched-8 theoretical is ~48 ns/call). Addresses con #9 at crowd scale when combined with octree-cell batching (runtime side).
3. **`KarcLodTier` config type** — type-level tag for the three LOD tiers (LOD0: K=2 M=4 d_h=64; LOD1: K=4 M=8 d_h=256; LOD2: K=8 M=8 R=2 d_h=18_720). Lets the runtime tier NPCs by importance without runtime-generic dispatch.

## What this plan does NOT do

- ❌ **No 2-way KARC (Bi-NCDE)** — Research 302 already proposed this; speculative, doubles cost, only useful for sleep-time consolidation. Defer until a concrete consumer appears.
- ❌ **No dual LEO+KARC fusion primitive** — category-confused at the gradient level (`DualLeoOracle` from Plan 467 is a Q-gradient oracle, not an HLA forecaster). Value-level fusion is just Mux/LOD with extra steps; the Mux primitive (item 1) covers that use case.
- ❌ **No octree spatial partitioning at the primitive layer** — that's a runtime concern. `OctreeSpatialIndex` already ships in `riir-games-shared`. Plan 514 wires it.
- ❌ **No cross-game transfer** — Bench 152 empirically rejected (0/15 configurations help). Not revisited.
- ❌ **No G1 threshold-leg fix** — that needs the Jacobi-eigen B-step (Issue 185 + 186 + 187). Independent track.

## GOAT gate (per primitive; must pass before promotion to default)

### `KarcRegimeGate`
- **G1** (correctness): on a Lorenz-63 corpus, gate routes ≥95% ticks to KARC; on a stationary seasonal (period=12) corpus, ≥95% to Seasonal. **Mix threshold ≤ 5%** (sigmoid smooth, not flip-flop).
- **G2** (perf): `decide()` ≤ 50 ns/call. Pure Welford + sigmoid + branch.
- **G3** (no-regression): enabling `karc_regime_gate` does not perturb `karc_forecaster` forecasts (bit-identical; verified by `conformal_karc_no_regression.rs`).
- **G4** (alloc-free): 0 allocs/100 calls on the hot path.

### `karc_batched_matvec`
- **G1** (correctness): batched N-forecast output bit-identical to N sequential `forecast_into` calls.
- **G2** (perf): N=8 batched forecast ≤ 1.5× single-forecast latency (≥5.3× amortization). Target ≤ 575 ns for 8 forecasts at the HLA config.
- **G3** (no-regression): does not perturb single-forecast path.
- **G4** (alloc-free): 0 allocs/N calls on the hot path.

### `KarcLodTier`
- **G1** (correctness): each tier constructs the correct const-generic monomorphization; tier-promotion projects old `Wout` to new `(D, M, K)` shape (or warm-starts from the trajectory ring).
- **G2** (perf): tier promotion ≤ 10 µs (one-time, not per-tick).
- **G3** (no-regression): the existing default (LOD1) path is unchanged.
- **G4** (alloc-free): tier promotion may allocate (one-time); per-tick dispatch is zero-alloc.

Promotion rule: each primitive ships behind its own feature flag (`karc_regime_gate`, `karc_batched_matvec`, `karc_lod_tier`), all opt-in initially. Promotion to default-on requires the corresponding Plan 514 runtime integration to demonstrate a measured gain (e.g., for the regime gate: F1 / CRPS improvement on a mixed-regime NPC corpus over KARC-alone).

---

## Architecture

```text
                                  ┌────────────────────────────────┐
   observed trajectory ─────────▶ │ KarcRegimeGate                 │
                                  │                                │
                                  │  residual_karc: Welford        │
                                  │  residual_seas:  Welford       │
                                  │  decision: sigmoid(σ²_a−σ²_b)  │
                                  └────────────────────────────────┘
                                            │
                          ┌─────────────────┴─────────────────┐
                          ▼                                   ▼
                  KarcForecaster                  SeasonalNaiveForecaster
                  (chaotic regime)                (periodic regime)

  ┌──────────────────────────────────────────────────────────────────────┐
  │ karc_batched_matvec: N forecasters (same D, M, K) share a single     │
  │ batched SIMD matvec. Caller stacks N delay_states [N][K*D], N Wout    │
  │ matrices [N][D*d_h] row-major, writes N forecasts [N][D] in one pass.│
  └──────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────────┐
  │ KarcLodTier: enum { Lod0, Lod1, Lod2 } + const-generic dispatch.     │
  │ Each tier is a KarcForecaster monomorphization. Promotion projects   │
  │ the old Wout to the new shape (rank-truncate or zero-pad) and the    │
  │ delay ring (truncate or backfill).                                   │
  └──────────────────────────────────────────────────────────────────────┘
```

**Reuse map (DRY):**
- Welford variance accumulator → vendored minimal (5 fields, 3 methods, ~30 LOC). No external dep.
- `SeasonalNaiveForecaster` → already shipped in `crates/katgpt-core/src/conformal/seasonal.rs`.
- `simd_matvec` → reuse from `crates/katgpt-core/src/simd.rs`.
- `KarcBasis` trait → reuse from `crates/katgpt-core/src/karc/mod.rs`.

---

## Phase 1 — `KarcRegimeGate` (HIGHEST LEVERAGE — directly fixes periodic-blindness)

### Tasks

- [x] **T1.1** Add `karc_regime_gate` feature in `crates/katgpt-core/Cargo.toml`. Gates `conformal_predictive_intervals` + `karc_forecaster` (needs `SeasonalNaiveForecaster` for type-level composition + `KarcForecaster` for the adapter shape).
- [x] **T1.2** Implement `WelfordVariance` accumulator (count, mean, M2). Methods: `observe(x)`, `reset()`, `variance() -> Option<f32>`, `n() -> usize`. Zero-alloc. ~40 LOC.
- [x] **T1.3** Implement `KarcRegimeGate` struct holding two `WelfordVariance` accumulators (one per forecaster) + `min_pool` cold-start floor + sigmoid inverse temperature `β`.
- [x] **T1.4** Implement `KarcRegimeGate::observe_residuals(residual_karc, residual_seas)` — push to both accumulators. Idempotent under NaN (NaN residual is a no-op, not a sample).
- [x] **T1.5** Implement `KarcRegimeGate::decide() -> RegimeVerdict` returning `RegimeVerdict { preferred: Regime, confidence: f32, sigma_sq_karc: f32, sigma_sq_seas: f32, n: usize }`. Cold-start: until `n >= min_pool`, returns `Regime::Seasonal` (the floor). After min_pool, picks lower variance; confidence = sigmoid(β · (σ²_high − σ²_low)).
- [x] **T1.6** Implement `Regime` enum: `Karc` | `Seasonal`. `#[repr(u8)]`.
- [x] **T1.7** Unit tests in `tests/karc_regime_gate_unit.rs`:
  - Lorenz-63 residual stream → gate routes ≥95% to KARC.
  - Stationary seasonal (period=12) residual stream → gate routes ≥95% to Seasonal.
  - Cold-start: until `min_pool`, always returns Seasonal.
  - Sigmoid smoothness: at residual-variance tie, confidence ≈ 0.5; no flip-flop.
  - NaN-safe: NaN residual is a no-op (n unchanged).
  - Alloc-free (manual counter).
- [x] **T1.8** GOAT gate G2 micro-bench in `benches/karc_regime_gate_bench.rs` — `decide()` ≤ 50 ns/call.
- [x] **T1.9** Rustdoc with paper-crossref (Bench 010 K-sweep finding that the periodic-blindness is structural), Plan 556 link, and the cold-start rationale (default to Seasonal = the floor).
- [x] **T1.10** Run `cargo clippy --features karc_regime_gate` — clean.

**Phase 1 exit:** all T1.x done. `cargo test --features karc_regime_gate` passes. Module is opt-in; no default-on.

---

## Phase 2 — `karc_batched_matvec` (CROWD-SCALE PERF)

### Tasks

- [ ] **T2.1** Add `karc_batched_matvec` feature in `Cargo.toml`. Gates `karc_forecaster`.
- [ ] **T2.2** Implement `karc_batched_matvec_into(wouts: &[f32], features: &[f32], out: &mut [f32], n: usize, d_h: usize, d: usize)` — N-row batched matvec, row-stride access. Use existing `simd::simd_matvec` per row (or per-channel chunked-4 if d_h is large).
- [ ] **T2.3** Implement `KarcBatchForecaster<B, D, M, K>` — owns N `Wout` matrices (caller-stacked flat slice), applies basis expansion once per NPC's delay state, runs the batched matvec.
- [ ] **T2.4** Unit test: batched-N forecast bit-identical to N sequential `KarcForecaster::forecast_into` calls.
- [ ] **T2.5** GOAT G2 bench: N=8 batched ≤ 1.5× single latency. Target ≤ 575 ns at HLA config.
- [ ] **T2.6** Alloc-free G4 bench: 0 allocs/N batched calls.

**Phase 2 exit:** all T2.x done. Feature stays opt-in until Plan 514's octree-batched dispatch proves the gain.

---

## Phase 3 — `KarcLodTier` (CONFIG TYPE + TIER PROMOTION)

### Tasks

- [ ] **T3.1** Add `karc_lod_tier` feature in `Cargo.toml`. Gates `karc_forecaster`.
- [ ] **T3.2** Define `KarcLodTier` enum (`Lod0`, `Lod1`, `Lod2`) + `KarcLodTierConfig` (D, M, K, R per tier). Defaults: Lod0 = (8, 4, 2, 1) d_h=64; Lod1 = (8, 8, 4, 1) d_h=256; Lod2 = (8, 8, 8, 2) d_h=18_720.
- [ ] **T3.3** Implement `promote_tier(old_forecaster, new_tier) -> new_forecaster` — projects `Wout` via SVD rank-truncate (down-tier) or zero-pad (up-tier). Delay ring is truncated or backfilled.
- [ ] **T3.4** Unit test: bit-identical forecast on same-tier roundtrip (Lod1 → Lod1 = identity).
- [ ] **T3.5** Unit test: down-tier (Lod2 → Lod0) preserves dominant components within 5% NRMSE.
- [ ] **T3.6** GOAT G2 bench: tier promotion ≤ 10 µs.

**Phase 3 exit:** all T3.x done. Feature stays opt-in until Plan 514 wires LOD tier selection per NPC importance class.

---

## Phase 4 — GOAT Gate + Promotion Decision

- [ ] **T4.1** Run all three primitives' GOAT benches together: `cargo bench --features karc_regime_gate,karc_batched_matvec,karc_lod_tier`.
- [ ] **T4.2** Document results in `.benchmarks/556_karc_mitigations_goat.md`.
- [ ] **T4.3** Promotion decision per primitive: if G1–G4 pass AND the Plan 514 runtime integration demonstrates a measured gain → promote to default-on. Else: keep opt-in with documented reason.

---

## References

- [Conversation 2026-07-20](#) — KARC pros/cons review + the design-space analysis that motivated this plan.
- [Research 288](../.research/288_KARC_Delay_Basis_Ridge_Forecaster.md) — KARC distillation.
- [Plan 308](308_karc_delay_basis_ridge_forecaster.md) — KARC open primitive.
- [Bench 010](../.benchmarks/010_report_the_floor_consolidated.md) — the structural periodic-blindness finding (con #1).
- [Bench 152](../.benchmarks/152_karc_cross_game_transfer.md) — cross-game transfer rejection (con #3 — out of scope here).
- [Issue 185](../.issues/185_karc_large_dh_als_bstep_jacobi.md) — G1 threshold-leg fix track (independent).
- [riir-ai Plan 514](../../riir-ai/.plans/514_karc_mitigations_runtime.md) — runtime integration of these primitives (LOD dispatch, octree-batched cells, mux wiring).

## TL;DR

Three modelless open primitives that mitigate KARC's structural cons: `KarcRegimeGate` (closed-form residual-variance mux, fixes periodic-blindness), `karc_batched_matvec` (SIMD-batched forecast, crowd-scale perf), `KarcLodTier` (tier promotion config type). Phase 1 lands the highest-leverage piece (regime gate); Phases 2–3 add the perf/scaling primitives; Phase 4 is the GOAT gate + promotion decision. Companion runtime work in riir-ai Plan 514.
