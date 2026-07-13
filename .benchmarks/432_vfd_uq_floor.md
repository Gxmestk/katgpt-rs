# Benchmark 432 — VFD UQ Conformal Floor (Plan 432 Phase 2 T2.2)

**Date:** 2026-07-13
**Plan:** [katgpt-rs/.plans/432_vfd_velocity_field_disagreement_primitive.md](../.plans/432_vfd_velocity_field_disagreement_primitive.md) Phase 2 T2.2
**Rule:** "Report the Floor" (Research 322, Plan 340, Issue 010) — any UQ-bearing primitive MUST beat `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` (m=1) on CRPS / coverage / Winkler.
**Test file:** `crates/katgpt-core/tests/velocity_field_disagreement_uq_floor.rs`

---

## TL;DR

**Verdict: ⚠️ MIXED — VFD's epistemic scaling does NOT add calibrated UQ value (λ\*=0 on both corpora), but the VFD-wrapped ensemble forecaster beats the floor on AR(1) via point-forecast quality (inherited from the ensemble, not from VFD).**

The GOAT gate G2 ("Report the Floor") passes on AR(1) (the harness returns `BeatsFloor`), but the win is entirely due to the **ensemble's point-forecast advantage** (the ridge-fit `(φ_0+φ_1)/2·y` beats the floor's seasonal-naive `y`) — which is the same result Plan 376 Phase 6 already demonstrated. VFD's epistemic scaling contributes **zero** to the win: the optimal λ is 0 on both corpora. On the bimodal corpus, VFD loses decisively because the ensemble mean (0) is a poor point forecast for bimodal data, and no amount of interval widening can compensate.

**Implication:** VFD should NOT be promoted to default-on as a UQ-bearing primitive. It ships as an **opt-in non-UQ disagreement score** — still useful for CLR L1 gating, sleep-time prioritization, and runtime failure detection (the paper's §6.4 use case), but with no calibrated-UQ claim. The velocity-field ensemble (Plan 376) remains the UQ-bearing primitive; VFD does not upgrade it.

---

## Setup

### The VFD-bearing forecaster design

VFD produces a scalar disagreement score `u_e(y) ∈ [0, +∞)`, NOT samples or intervals. To compare it as a UQ primitive against the conformal floor, we construct a VFD-calibrated interval forecaster:

```
point_forecast = ensemble_mean_prediction(y_t)
total_variance  = σ_ale² + λ · max(u_e(y_t), 0)
interval        = [point ± z(α) · sqrt(total_variance)]
```

where:
- `σ_ale` is the training residual std (aleatoric uncertainty).
- `λ` is a scaling factor, calibrated **modellessly** via grid search on training CRPS. The grid `{0, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0}` INCLUDES λ=0 (pure ensemble baseline), so VFD can only help or tie.
- `z(α) = 1.959964` (Gaussian 97.5th percentile for α=0.05).

The velocity fields are closures conditioned on the current observation `y_t`, so `u_e(y_t)` varies per-target.

### Corpora

#### (a) AR(1) corpus

- **Process:** `x_{t+1} = φ·x_t + ε`, `ε ~ N(0, σ²)`, `φ=0.7`, `σ=0.5`, seed `0x1234_5678_9ABC_DEF0`.
- **N_TRAIN = 200, N_TEST = 200** (first 32 of test are warmup; n_scored = 168).
- **Two members:** φ̂₀ estimated via least-squares on first 100 training pairs; φ̂₁ on last 100. The VFD score varies with `y²` — disagreement = `(φ₀−φ₁)²·y²·C` where `C = (1/N_s)·Σ κ_{s_ℓ}`.
- **Point forecast:** ensemble mean = `(φ₀+φ₁)/2 · y`.

#### (b) 1D Bimodal flow-matching toy (simplified from paper Appendix C.1 2D setup)

- **Process:** Markov-switching `x_{t+1} = μ_{s_t} + ε`, `μ₀=+2, μ₁=−2`, `ε ~ N(0, 0.5²)`, regime switches with prob 0.05/step.
- **Two members:** fixed attractors `v₀(x) = 2−x` (toward +2), `v₁(x) = −2−x` (toward −2). Disagreement = 16 (constant).
- **Point forecast:** ensemble mean = `(2 + (−2))/2 = 0` (constant — a deliberately poor forecast for bimodal data).

### Floor

`ConformalIntervalCalibrator<SeasonalNaiveForecaster>` with `m=1`, `exp_lambda=0.0`, `HStep` residual mode, capacity 256 (the Issue 010 canonical config). Adapts online.

---

## Results

### Corpus (a): AR(1)

**Setup:** φ̂₀ = 0.6549, φ̂₁ = 0.8142, |φ₀−φ₁| = 0.1593, σ_ale = 0.5193.
**λ calibration:** Best λ = **0** (pure ensemble baseline; VFD epistemic scaling does NOT reduce training CRPS).

```
=== Floor Comparison: VFD (2 linear closure members, VFD-scaled Gaussian interval) ===
Corpus: ar1_phi0.7_sigma0.5_n200 (n_scored=168, n_unscorable=0, α=0.05)

Metric             | Primitive  | Floor      | Ratio (prim/floor) | Verdict
-------------------|------------|------------|--------------------|---------
Mean CRPS          |     2.0478 |     2.0794 |             0.9848 | tie
Mean Winkler       |     2.2820 |     2.4616 |             0.9270 | WIN
Coverage (nom=0.95) |     0.9643 |     0.9583 | err 0.0143 vs 0.0083 | tie

Overall: ✅ BEATS FLOOR — primitive adds UQ value
```

| Metric | VFD (λ=0) | Floor | Ratio | Verdict |
|---|---|---|---|---|
| Mean CRPS | 2.0478 | 2.0794 | 0.9848 | **tie** (within ±5%) |
| Mean Winkler | 2.2820 | 2.4616 | 0.9270 | **WIN** (7.3% better) |
| Coverage (nom 0.95) | 0.9643 | 0.9583 | err 0.0143 vs 0.0083 | **tie** (within ±0.02) |

**Verdict: ✅ BeatsFloor** (driven by Winkler win; CRPS and coverage tie).

### Corpus (b): Bimodal

**Setup:** σ_ale = 2.0273 (large — data spread across ±2). Ensemble mean point forecast = 0.
**λ calibration:** Best λ = **0** (pure ensemble baseline; VFD epistemic scaling does NOT reduce training CRPS).

```
=== Floor Comparison: VFD (bimodal fixed attractors, VFD-scaled Gaussian interval) ===
Corpus: bimodal_mode2_sigma0.5_switch0.05_n200 (n_scored=168, n_unscorable=0, α=0.05)

Metric             | Primitive  | Floor      | Ratio (prim/floor) | Verdict
-------------------|------------|------------|--------------------|---------
Mean CRPS          |     7.9470 |     3.2671 |             2.4324 | LOSE
Mean Winkler       |     7.9470 |     7.0873 |             1.1213 | LOSE
Coverage (nom=0.95) |     1.0000 |     0.9524 | err 0.0500 vs 0.0024 | LOSE

Overall: ❌ LOSES TO FLOOR — primitive does not add UQ value
```

| Metric | VFD (λ=0) | Floor | Ratio | Verdict |
|---|---|---|---|---|
| Mean CRPS | 7.9470 | 3.2671 | 2.4324 | **LOSE** (2.4× worse) |
| Mean Winkler | 7.9470 | 7.0873 | 1.1213 | **LOSE** |
| Coverage (nom 0.95) | 1.0000 | 0.9524 | err 0.0500 vs 0.0024 | **LOSE** (over-covers) |

**Verdict: ❌ LosesToFloor** (dominated by the bad point forecast — ensemble mean 0 vs actuals at ±2).

---

## Analysis

### Why λ\* = 0 on both corpora

The grid search selects λ=0 on BOTH corpora. This means VFD's epistemic scaling does not improve training CRPS on either corpus:

1. **AR(1) — homoscedastic residuals.** The true AR(1) innovations are Gaussian with constant variance σ². VFD's disagreement signal scales with `y²` (high when |y| is large), but the actual prediction error is homoscedastic (constant variance regardless of y). VFD's per-target widening is therefore **uncorrelated with actual error** — it widens intervals for extreme y where errors are no larger than average. The optimal λ is 0 because any positive λ adds variance without reducing CRPS.

2. **Bimodal — constant disagreement.** The two attractors (+2, −2) disagree by a constant amount (16 in velocity space) regardless of the observation y. VFD provides a constant variance inflation, which cannot adapt to the markov-switching structure (where the actual uncertainty depends on whether a regime switch is imminent, which VFD cannot detect from the current observation alone).

### Why VFD still "wins" on AR(1)

The AR(1) win is entirely due to the **ensemble's point-forecast quality**, not VFD:

- The ensemble mean forecast `(φ₀+φ₁)/2 · y ≈ 0.73·y` is a better one-step prediction than the floor's seasonal-naive `y` (which corresponds to φ=1).
- On AR(1) with true φ=0.7, the optimal one-step forecast is `0.7·y`. The floor cannot learn this slope — it always predicts `y` (φ=1). The ensemble's ridge-fit φ ≈ 0.73 is much closer to the true 0.7.
- This is the **exact same mechanism** Plan 376 Phase 6 demonstrated: the velocity-field ensemble beats the floor because its point forecast learns the AR(1) slope.
- VFD adds nothing on top — the λ=0 case (pure ensemble, no VFD scaling) achieves the win.

### Why VFD loses badly on bimodal

The bimodal loss is dominated by **point-forecast failure**:

- The ensemble mean is always 0 (average of +2 and −2 attractors).
- The actual observations are at ±2, so every residual is ≈2.
- CRPS ≈ 7.95 is dominated by this constant point-forecast error.
- The floor (seasonal-naive: predict last observation ≈ ±2) has much better point forecasts — its residuals are either ≈0 (same regime, 95% of the time) or ≈4 (regime switch, 5% of the time).
- No amount of VFD-based interval widening can fix a point forecast that's always 0 when the data is at ±2.

### When would VFD's epistemic scaling actually help?

VFD's epistemic scaling would add calibrated UQ value in a regime where:

1. **Disagreement correlates with actual error.** The members must disagree MORE in regions where the prediction error is LARGER. This happens when the members are misspecified (e.g., linear members on a nonlinear process) and the misspecification is worse in some regions than others.
2. **The error structure is heteroscedastic.** If the actual noise variance varies across the input space AND the members' disagreement tracks that variation, then VFD's per-target widening correctly matches the actual error structure.

Neither AR(1) (homoscedastic) nor the constant-disagreement bimodal toy satisfies these conditions. A better test corpus would be a **heteroscedastic nonlinear process** (e.g., `x_{t+1} = sin(x_t) + (0.3 + 0.5|x_t|)·ε`) with linear members — but that's a different benchmark. The plan's specified corpora (AR(1) + bimodal) are what we tested, and VFD does not pass on either via its own epistemic-scaling contribution.

---

## Verdict per the "Report the Floor" Rule (Issue 010)

| Corpus | Harness Verdict | VFD Epistemic Scaling | Honest Assessment |
|---|---|---|---|
| AR(1) | ✅ BeatsFloor | λ\*=0 (does NOT help) | Win is inherited from ensemble point forecast, NOT from VFD |
| Bimodal | ❌ LosesToFloor | λ\*=0 (does NOT help) | Loss due to bad point forecast; VFD cannot compensate |

**The plan's literal PASS criterion** ("beats the floor on ≥2 of 3 metrics on ≥1 corpus") is met on AR(1) (Winkler WIN + CRPS/coverage tie → `BeatsFloor`). **However, the win is not attributable to VFD** — it's the ensemble's point-forecast advantage, already demonstrated in Plan 376 Phase 6.

**The honest UQ verdict:** VFD does not add calibrated UQ value on the tested corpora. The GOAT gate G2 **fails for VFD's epistemic-UQ claim**. Per Plan 432 T3.3: VFD ships as an **opt-in non-UQ disagreement score** — no calibrated-UQ claim, no promotion to default-on.

---

## Caveats

1. **Single seed, two synthetic corpora.** The result is demonstrated on AR(1) + bimodal, not exhaustively characterized. A heteroscedastic nonlinear corpus (where disagreement correlates with error) might show VFD adding value — but that's not the plan's specified corpus.

2. **λ grid is coarse.** The grid `{0, 0.001, ..., 100.0}` may miss the optimal λ. However, the fact that λ=0 beats all positive λ values on BOTH corpora is strong evidence that VFD's signal doesn't correlate with actual error on these corpora — a finer grid wouldn't change the qualitative finding.

3. **Point forecast dominates CRPS.** On both corpora, the point-forecast quality dominates the CRPS metric. VFD (which only scales interval width, not the point forecast) is structurally disadvantaged — it can only help when the point forecast is already good AND the per-target variance signal matches the actual error structure. AR(1) satisfies the first condition but not the second.

4. **Gaussian interval assumption.** The VFD adapter constructs Gaussian intervals `[point ± z·sqrt(variance)]`. For multimodal data (like the bimodal corpus), Gaussian intervals are fundamentally misspecified — the true predictive distribution is bimodal, not unimodal Gaussian. A samples-based approach (generate samples from each member's trajectory, take empirical quantiles) might do better, but that's a different adapter design.

5. **The ensemble already provides UQ (Plan 376).** The velocity-field ensemble IS a UQ-bearing primitive that beats the floor on AR(1) (demonstrated in Plan 376 Phase 6 / `.benchmarks/376_uq_floor.md`). VFD's job was to ADD calibrated epistemic UQ on top — it does not. The ensemble's UQ claim stands; VFD does not upgrade it.

---

## Reproduction

```sh
cd /Users/katopz/git/katgpt-rs
cargo test -p katgpt-core \
  --features velocity_field_disagreement,conformal_predictive_intervals \
  --test velocity_field_disagreement_uq_floor -- --ignored --nocapture
```

---

## Cross-References

- **Plan 432 Phase 2 T2.2:** `.plans/432_vfd_velocity_field_disagreement_primitive.md`
- **Plan 376 Phase 6 (the ensemble's UQ gate, which PASSED):** `.benchmarks/376_uq_floor.md`
- **Issue 010 (the "Report the Floor" rule):** `.benchmarks/010_report_the_floor_consolidated.md`
- **Plan 340 (the floor):** `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` (m=1, default-on)
- **Floor harness:** `crates/katgpt-core/src/conformal/floor_harness.rs`
- **Research 420 (the VFD research note):** `.research/420_VFD_Velocity_Field_Disagreement_Epistemic_UQ.md`
- **Source paper:** arXiv:2606.18043 — Römer et al., §4 (VFD estimator + Theorem 4.1)
