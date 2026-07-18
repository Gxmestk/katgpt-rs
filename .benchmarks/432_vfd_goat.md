# Benchmark 432 — VFD GOAT Gate Aggregate (Plan 432 Phase 3 T3.1)

**Date:** 2026-07-13
**Plan:** [katgpt-rs/.plans/432_vfd_velocity_field_disagreement_primitive.md](../.plans/432_vfd_velocity_field_disagreement_primitive.md) Phase 2 + Phase 3
**Primitive:** `velocity_field_disagreement` (VFD — Velocity-Field Disagreement epistemic UQ estimator)
**Source paper:** arXiv:2606.18043 — Römer et al., §4 (VFD estimator + Theorem 4.1)

---

## TL;DR

**GOAT gate: 4/5 PASS, 1/5 FAIL → no promotion to default-on.**

| Gate | Description | Result |
|---|---|---|
| G1 (mechanics) | Exact analytic VFD match for constant-disagreement fields | ✅ PASS |
| G2 (UQ floor) | VFD-calibrated intervals beat `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` m=1 | ❌ **FAIL** (for VFD's epistemic-UQ claim; λ\*=0 on both corpora) |
| G3 (no regression) | Single-feature + combo clippy clean; zero-alloc on score path | ✅ PASS |
| G4 (latency) | `vfd_score_into` ≤ 50µs at M=2 D=8 N_s=10 B=5 | ✅ PASS (10.43µs, 4.8× margin) |
| G5 (QGF integration) | `VfdVarianceSignal: QgfVarianceSignal` smoke test | ✅ PASS |

**Promotion decision:** `velocity_field_disagreement` stays **opt-in**. VFD ships as a **non-UQ disagreement score** — useful for CLR L1 gating, sleep-time prioritization, runtime failure detection (paper §6.4), but with **no calibrated-UQ claim**. The velocity-field ensemble (Plan 376) remains the UQ-bearing primitive; VFD does not upgrade it.

---

## Per-gate detail

### G1 (mechanics) — ✅ PASS

**Test:** `test_vfd_score_constant_disagreement_matches_analytic` + parameter sweep (`test_vfd_score_matches_analytic_various_params`) + trig schedule variant (`test_vfd_score_matches_analytic_trig_schedule`).

**What it checks:** For two linear fields `v^i(x) = μ_i − x`, the disagreement `‖v^i(x) − v^j(x)‖² = ‖μ_i − μ_j‖²` is constant (independent of x). The VFD score then has the exact analytic form:

```
VFD = ‖μ_i − μ_j‖² · (1/N_s) · Σ_{ℓ=0}^{N_s−1} κ_{s_ℓ}
```

This is deterministic (independent of the RNG). The test asserts this exact value across:
- Multiple `N_s ∈ {5, 10, 20, 50}`, `B ∈ {1, 5}`, D ∈ {2, 3}.
- Both `Schedule::Linear` (κ_s = s/(1−s)) and `Schedule::Trigonometric` (κ_s = (2/π)tan(πs/2)).
- Tolerance 1e-3 on the Linear schedule, 1e-2 on parametric sweeps (Monte Carlo variance from the SDE integration steps).

**Total Phase 1 tests:** 20/20 PASS (including κ_s monotonicity, VfdVarianceSignal range/monotonicity/NaN handling, per-member-trajectory bug catcher, panic-on-zero guards).

**This is stronger than the plan's loose KL=0.5 test.** The plan T1.5 expected VFD ≈ 0.5 (the analytic KL between N(μ1,I) and N(μ2,I)). But the toy fields `v^i(x) = μ_i − x` are NOT flow-matching marginal velocity fields (they lack the 1/(1−s) factor), so VFD ≠ KL for them. The exact analytic match for constant-disagreement fields is the correct G1 target.

### G2 (UQ floor) — ❌ FAIL (for VFD's epistemic-UQ claim)

**Full benchmark:** [`.benchmarks/432_vfd_uq_floor.md`](./432_vfd_uq_floor.md)

**Corpora:**
- (a) AR(1): φ=0.7, σ=0.5, N=400 (200 train + 200 test, 168 scored). Two members with φ̂ estimated via least-squares on disjoint training halves.
- (b) 1D Bimodal: markov-switching between ±2, σ=0.5, switch_prob=0.05, N=400. Two fixed-attractor members (+2, −2).

**VFD-bearing forecaster design:**
```
point = ensemble_mean(y)
variance = σ_ale² + λ · max(u_e(y), 0)
interval = [point ± z(α) · sqrt(variance)]
```
λ grid-search-calibrated on training CRPS. Grid `{0, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0}` INCLUDES λ=0 (pure ensemble baseline).

**Results:**

| Corpus | λ\* | VFD CRPS | Floor CRPS | Harness Verdict | Honest Assessment |
|---|---|---|---|---|---|
| AR(1) | **0** | 2.0478 | 2.0794 | ✅ BeatsFloor (Winkler WIN) | Win inherited from ensemble point forecast, NOT VFD |
| Bimodal | **0** | 7.9470 | 3.2671 | ❌ LosesToFloor | Loss due to bad point forecast (ensemble mean=0 vs data at ±2) |

**Why G2 fails for VFD's epistemic-UQ claim:**
1. **Optimal λ\*=0 on both corpora.** VFD's epistemic scaling does not improve training CRPS on either corpus. Any positive λ adds variance without reducing CRPS.
2. **AR(1) — homoscedastic residuals.** VFD's disagreement scales with y², but the actual AR(1) prediction error is homoscedastic (constant variance). VFD's per-target widening is uncorrelated with actual error.
3. **Bimodal — constant disagreement.** The two attractors disagree by a constant amount regardless of y. VFD provides a constant variance inflation, which cannot adapt to the markov-switching structure.
4. **The AR(1) `BeatsFloor` is NOT from VFD.** It's from the ensemble's point-forecast advantage (ridge-fit φ ≈ 0.73 vs the floor's implicit φ=1) — the exact same mechanism Plan 376 Phase 6 demonstrated. VFD contributes nothing on top (λ\*=0).

**When would VFD's epistemic scaling help?** When disagreement correlates with actual error — e.g., misspecified linear members on a nonlinear process with heteroscedastic noise. Neither plan-specified corpus satisfies this condition.

### G3 (no regression) — ✅ PASS

| Check | Result |
|---|---|
| `cargo clippy -p katgpt-core --features velocity_field_disagreement --lib -- -D warnings` | Clean |
| `cargo clippy -p katgpt-core --all-features --lib -- -D warnings` | Clean (combo check) |
| `cargo clippy -p katgpt-core --features velocity_field_disagreement,conformal_predictive_intervals --test velocity_field_disagreement_uq_floor -- -D warnings` | Clean |
| Zero-alloc re-check (bench `bench_432_vfd_goat` G3 gate) | 0 allocs / 0 deallocs on 1000 score calls |

### G4 (latency) — ✅ PASS

**Bench:** `benches/bench_432_vfd_goat.rs`

**Config:** M=2, D=8 (HLA dim), N_s=10, B=5 (paper defaults).

**Result:** `vfd_score_into` p50 = **10.43 µs** (target ≤ 50 µs; **4.8× margin**).

| Metric | Value |
|---|---|
| p50 latency | 10,434 ns (10.43 µs) |
| Target | 50,000 ns (50 µs) |
| Margin | 4.8× under target |
| Samples | 200 (each = 50 batched calls) |
| Profile | release (`--release`) |

**Note:** The latency is well within the plasma-tier budget. At 10K NPCs × 2Hz per-NPC VFD sampling (T_vfd=10 ticks), this is 20K inferences/sec × 10.43µs = 0.21 sec/sec of compute — feasible on a single core. Per Plan 432 Risk Note #2, VFD is still NOT a per-tick per-NPC primitive (it should run off the hot path or at reduced frequency), but the per-call cost is comfortably under budget.

### G5 (QGF integration) — ✅ PASS

**Test:** `test_qgf_bridge_smoke` (Phase 1 unit test, gated on `qgf_adaptive`).

**What it checks:** `VfdVarianceSignal` implements `QgfVarianceSignal`, and feeding a VFD score into the QGF adaptive guidance weight pipeline produces the expected response (higher VFD → lower confidence → lower guidance weight). Closes the "ensemble KL" open item in `qgf/adaptive.rs`.

---

## Promotion Decision (Plan 432 T3.2 / T3.3)

**Decision: DO NOT promote to default-on.**

Per Plan 432 T3.3 (the G2-fails path): `velocity_field_disagreement` stays **opt-in**. The primitive ships as a **non-UQ disagreement score** with the following honest scope:

- ✅ **Valid use cases:** CLR L1 evidence gating (P307), sleep-time anticipator prioritization (P334), runtime failure detection (paper §6.4), QGF adaptive guidance weight modulation (when `qgf_adaptive` is also enabled).
- ❌ **NOT a calibrated UQ claim:** VFD's epistemic scaling does not beat the conformal-naive floor on the tested corpora. It is a heuristic disagreement signal, not a calibrated probability distribution.
- ❌ **Does NOT upgrade the velocity-field ensemble's UQ claim:** Plan 376 Phase 6 demonstrated the ensemble is UQ-bearing on its own (via its point-forecast advantage). VFD does not add calibrated epistemic UQ on top.

**Plan 376 Phase 6 G7 status:** Still **deferred**. VFD did not promote to default-on, so the ensemble's G7 gate is not activated by VFD. The ensemble's UQ claim stands independently (Plan 376 Phase 6 / `.benchmarks/376_uq_floor.md`).

---

## Cross-references

- **UQ floor benchmark (full detail):** [`.benchmarks/432_vfd_uq_floor.md`](./432_vfd_uq_floor.md)
- **UQ floor test file:** `crates/katgpt-core/tests/velocity_field_disagreement_uq_floor.rs`
- **Latency + alloc bench:** `crates/katgpt-core/benches/bench_432_vfd_goat.rs`
- **Primitive source:** `crates/katgpt-core/crates/katgpt-core/src/velocity_field_disagreement.rs` (918 lines, 20 tests)
- **Plan:** `.plans/432_vfd_velocity_field_disagreement_primitive.md`
- **Research:** `.research/420_VFD_Velocity_Field_Disagreement_Epistemic_UQ.md`
- **Ensemble's UQ gate (Plan 376 Phase 6, PASSED):** `.benchmarks/376_uq_floor.md`
- **Issue 010 (the "Report the Floor" rule):** `.benchmarks/010_report_the_floor_consolidated.md`
- **Floor (Plan 340):** `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` (m=1, default-on)
- **Source paper:** arXiv:2606.18043 — Römer et al., §4 (VFD estimator + Theorem 4.1)

---

## Reproduction

```sh
cd /Users/katopz/git/katgpt-rs

# Phase 1 unit tests (G1 mechanics + G5 QGF smoke)
cargo test -p katgpt-core --features velocity_field_disagreement --lib

# G2 UQ floor benchmark (both corpora)
cargo test -p katgpt-core \
  --features velocity_field_disagreement,conformal_predictive_intervals \
  --test velocity_field_disagreement_uq_floor -- --ignored --nocapture

# G3 (combo check)
cargo clippy -p katgpt-core --all-features --lib -- -D warnings

# G4 latency + G3 alloc-free
CARGO_TARGET_DIR=/tmp/vfd_432 cargo build --release -p katgpt-core \
    --features velocity_field_disagreement --bench bench_432_vfd_goat
/tmp/vfd_432/release/deps/bench_432_vfd_goat-* --nocapture
rm -rf /tmp/vfd_432
```
