# Conformal Predictive Intervals — Modelless UQ Overlay + "Report the Floor" Rule

**Plan:** [340](../../.plans/340_conformal_predictive_intervals_primitive.md) (primitive) + [468](../../.plans/468_conformal_predictive_intervals_default_promotion.md) (default-on promotion)
**Research:** [322](../../.research/322_Conformal_Seasonal_Pools_Calibrated_UQ_Overlay.md)
**Source papers:** [arxiv 2605.03789](https://arxiv.org/abs/2605.03789) CSP (Angelopoulos et al.) + [arxiv 2606.09473](https://arxiv.org/abs/2606.09473) "Report the Floor" (Tibshirani et al.)
**Status:** Plan 340 Phase 1 + Phase 2 shipped. Plan 468 promoted to **DEFAULT-ON** on 2026-07-20.
**Feature flag:** `conformal_predictive_intervals` (**DEFAULT-ON** since Plan 468 promotion, 2026-07-20).

---

## What it is

A **modelless conformal uncertainty-quantification overlay** that wraps any
`PointForecaster` and produces **coverage-guaranteed predictive intervals**
`[point + q_{α/2}, point + q_{1−α/2}]` from a per-channel × per-horizon-bucket
exp-recency-weighted residual ring buffer. No training, no learned parameters,
no gradient descent — the math is empirical-quantile calibration over a residual
reservoir (split conformal).

The primitive also ships the **canonical conformal-naive UQ floor**:
`ConformalIntervalCalibrator<SeasonalNaiveForecaster>` with `m=1`. Per
[`AGENTS.md` §"UQ-bearing primitive GOAT gate extension"](../../AGENTS.md)
(the "Report the Floor" rule, adopted 2026-06-28), **every UQ-bearing
primitive's GOAT gate MUST benchmark against this floor** on CRPS / coverage /
Winkler score. If a UQ primitive cannot beat the floor, its GOAT gate FAILs.
The rule was codified in response to Issue 010 / Research 322; it is now
enforceable because the floor ships here.

---

## The three-layer split pattern

Conformal predictive intervals follow the canonical **three-layer split** that
governs primitive-vs-consumer promotions in this repo:

| Layer | Surface | Status |
|---|---|---|
| Primitive (this crate, `katgpt-core`) | `conformal_predictive_intervals` | **DEFAULT-ON** (Plan 468, 2026-07-20) |
| Consumer bridge (riir-engine) | `karc_conformal_width`, `salience_conformal_width` | **Opt-in** |
| Probe / diagnostic (katgpt-core) | 4 probe features (BOM, sleep-time, best-belief, KARC-overlay floor tests) | **Opt-in** (per-probe) |

**Why the split:** the primitive is a generic math overlay (empirical-quantile
calibration over any forecaster's residuals). It carries no game IP, no NPC
semantics — pure modelless inference substrate. The consumer bridges carry
runtime integration (per-NPC KARC collapse τ, salience tri-gate Delegate nudge,
etc.); each consumer has its own GOAT gate and stays opt-in until that gate
clears default-on. **Promoting the primitive to default-on does not auto-enable
any consumer** — it only removes the katgpt-core-level re-forward friction.

This is the **append-only anti-pattern defense**: when a primitive ships
DEFAULT-ON but a consumer STAYS opt-in, that is not a missed propagation — it
is a deliberate layer-split where the consumer gates a different concern than
the primitive (cf. the feature-gate-audit skill).

---

## API surface

All under `katgpt_core::conformal::`, gated behind `feature =
"conformal_predictive_intervals"` (default-on since Plan 468).

### `ConformalIntervalCalibrator<F: PointForecaster>`

The calibrator. Generic over any `PointForecaster`. Per-channel ×
per-horizon-bucket exp-recency-weighted residual ring buffer.

| Method | Purpose |
|---|---|
| `new(forecaster, cfg)` | Construct with config (α, horizon buckets, ring capacity, recency half-life). |
| `observe(channel, horizon, residual)` | Push a new residual (warm path). |
| `interval_into(&out, channel, horizon, point)` | Compute `[lo, hi]` from the residual reservoir + the forecaster's point estimate. Zero-alloc hot path. |
| `interval(channel, horizon, point) -> [f32; 2]` | Convenience wrapper returning `[lo, hi]`. |

### `PointForecaster` trait

```rust
pub trait PointForecaster {
    fn forecast_into(&self, out: &mut [f32], channel: usize, horizon: usize);
}
```

Any type implementing this trait can be wrapped by the calibrator. Shipped
forecasters:

- **`SeasonalNaiveForecaster`** (always-on, with `m=1` as the canonical
  conformal-naive floor instance — see the [Report the Floor](#the-canonical-uq-floor)
  section below).
- **`KarcChannelForecaster`** adapter (`karc_forecaster` feature, Plan 308 +
  Plan 340 Phase 2) — wraps the KARC reservoir-computing ridge forecaster as
  a `PointForecaster` for conformal overlay.

### Metrics module (`conformal::metrics`)

CRPS (continuous ranked probability score), empirical coverage, Winkler score.
These are the metrics every UQ-bearing primitive MUST report against the floor
per the "Report the Floor" rule.

### Floor harness (`ConformalIntervalCalibrator<SeasonalNaiveForecaster>`)

The canonical conformal-naive UQ floor. Lives in `tests/conformal_floor_harness.rs`
(Issue 010 T2) — the integration-test harness every UQ primitive uses for its
"Report the Floor" GOAT gate.

---

## The canonical UQ floor ("Report the Floor" rule)

Per `AGENTS.md` §"UQ-bearing primitive GOAT gate extension" (adopted
2026-06-28 per Research 322 / Plan 340):

> Any primitive that claims a probability distribution, predictive interval,
> quantile, coverage guarantee, confidence score, or calibrated uncertainty
> (collectively: **UQ-bearing**) MUST benchmark against the **conformal-naive
> floor** — `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` (Plan 340
> with `m=1`, plain split conformal) — on CRPS / coverage / Winkler score.
> If the primitive cannot beat the floor, the GOAT gate FAILs.

The floor is the modelless analog of "always compare against the
seasonal-naive forecasting baseline" — a long-standing rule in the forecasting
literature (Hyndman, Athanasopoulos) that prevents a sophisticated primitive
from claiming a gain over a degenerate baseline that was never measured.

### Grandfathered UQ primitives

Existing UQ-bearing primitives that predate the rule (BoMSampler Plan 281,
Sleep-Time Anticipator Plan 334, Best-Belief Beta Selector Plan 336, KARC +
overlay) were grandfathered at rule adoption. Each must include the floor at
its next re-gate:

- **T3 BoMSampler** — floor test shipped, EXCLUDED (same structural class —
  diversity-for-exploration, not calibrated UQ).
- **T4 Sleep-Time Anticipator** — floor test shipped, FAIL (wider than 5× EMA).
- **T5 Best-Belief Beta Selector** — floor test shipped (selection-quality
  comparison, not interval calibration — different metric axis).
- **T7 KARC + overlay** — floor test shipped 2026-07-20 (Issue 010 T7 closure);
  SCOPE-LIMITED to chaotic regimes (BEATS on Lorenz-x at crps_ratio 0.0047 with
  K=4; LOSES on stationary seasonal at crps_ratio 5.74 with K=4). K-sweep
  refuted the prior "K=4 too shallow" hypothesis: K=12 LOSES WORSE on seasonal
  (CRPS 5.74→20.26) and WINS HARDER on Lorenz (CRPS 0.0047→0.0018). The
  scope-limit is **structural** (KARC's Chebyshev basis + ridge-fit cannot fit
  periodic data regardless of K), not parametric. Coverage stays calibrated on
  both — no false-confidence signature.

Issue 010 is **FULLY CLOSED** (T1–T7 all complete). See
[`.benchmarks/010_report_the_floor_consolidated.md`](../../.benchmarks/010_report_the_floor_consolidated.md)
for the cross-primitive summary.

---

## GOAT gate status

Honest assessment of what is measured.

### Primitive-level GOAT gate (Plan 340 / Bench 340) — ALL PASS

| Gate | Target | Result | Status |
|---|---|---|---|
| **G1** Coverage | empirical coverage at α=0.05 ∈ [0.93, 0.97] over 10K ticks for ALL `m ∈ {12, 24, 48}` + HStep | **[0.9445, 0.9493]** across the sweep | ✅ PASS |
| **G1** Alpha sweep | coverage tracks nominal α across {0.01, 0.05, 0.10, 0.20} | 0.9842 / 0.9463 / 0.8966 / 0.7954 (targets 0.99/0.95/0.90/0.80) | ✅ PASS |
| **G2** Latency | `interval_into` ≤ 1µs at H=1, ≤ 100µs at H=8×8 | H=1 **642 ns** (36% headroom); H=8×8 well under budget | ✅ PASS |
| **G3** Allocation | 0 allocs / 100 calls (CountingAllocator) | 0 allocs | ✅ PASS |
| **G4** Reproducibility | bit-identical output for bit-identical input + ring buffer state | verified by deterministic replay test | ✅ PASS |

**Bench file:** [`benchmarks/340_conformal_goat.md`](../../.benchmarks/340_conformal_goat.md).

**G2 perf note (the win that landed it in default):** the initial
implementation called `weighted_quantile` twice per `interval_into` — once for
`q_{α/2}`, once for `q_{1−α/2}` — recomputing the full O(n) `exp()` weight scan
on each call (**4n `exp()` calls per interval**). This put H=1 at 1.15µs, 15%
over the 1µs budget. The fix: `weighted_quantile_pair` computes the weights
once into a 4KB stack buffer (`WEIGHTS_BUF_LEN = 1024`) and reuses them for
both quantile lookups — **n `exp()` calls per interval**, a 4× reduction.
Result: H=1 dropped to **642ns** (44% faster, 36% headroom under budget). This
is the "Don't recompute unchanged values" optimization rule applied at the
micro-level.

### Runtime-consumer promotion gate (Plan 468) — PASS

Plan 340 T1.14 deferred promotion *"pending a runtime consumer that
demonstrably beats its simpler heuristic counterpart."* Four consumers were
probed across Benches 562 / 563 / 564 / 565; **two PASSed**:

| Bench | Consumer | Verdict | Headline metric |
|---|---|---|---|
| 562 | Curiosity gate | ❌ FAIL | conformal interval width wider than 5× EMA — no gain |
| 563 | Sleep-time anticipator | ❌ FAIL | distribution-level summary loses cycle info |
| **564** | **MCTS collapse** | ✅ **PASS** | per-NPC calibrated τ beats fixed magic number on collapse-detection F1 |
| **565** | **Salience Tri-Gate Delegate nudge** | ✅ **PASS** | dF1 = +0.3145 at 6.3× gate margin, dFP = −0.8155 |

The Cargo.toml language required only one consumer PASS; two landed. Bench 565
was vindicated bit-identically by Plan 513 — the width-definition semantic bug
(`KarcConformalSidecar::interval_width()` was using half-width instead of full
interval width `iv.upper − iv.lower`) was fixed and the G3 PASS reproduced.

**Two consumer-side gates stay opt-in** because their own G2 perf gate FAILED
default-on promotion:

- `karc_conformal_width` (riir-engine): Plan 512 measured **+113.9% overhead**
  per KarcCollapseConfig construction → stays opt-in.
- `salience_conformal_width` (riir-engine): consumer-specific gain, no
  production default-on promotion.

---

## Why this is in `katgpt-rs` (the public engine)

The primitive is pure modelless math — empirical-quantile calibration over a
residual reservoir, no game IP, no NPC semantics. The moat (the *consumer*
integrations — KARC collapse τ, salience tri-gate Delegate nudge, etc.) lives
in `riir-ai` / `riir-engine`. Per the 7-repo commercial strategy
(Research 003), the generic math stays in the public MIT repo; the consumer
integrations stay private.

The canonical UQ floor is **load-bearing for the whole 7-repo stack** — every
UQ-bearing primitive's GOAT gate depends on it (the "Report the Floor" rule).
Keeping it in `katgpt-rs` means the rule is enforceable across all repos.

---

## Usage

### Basic (wrap any forecaster)

```rust
use katgpt_core::conformal::{
    ConformalIntervalCalibrator, ConformalConfig, SeasonalNaiveForecaster,
};

let forecaster = SeasonalNaiveForecaster::new(m = 24);
let mut cal = ConformalIntervalCalibrator::new(forecaster, ConformalConfig::default());

// Warm path: observe residuals as they arrive
cal.observe(channel = 0, horizon = 1, residual = 0.3);

// Query: get a coverage-guaranteed interval around the point forecast
let point = cal.forecaster().forecast(channel = 0, horizon = 1);
let [lo, hi] = cal.interval(channel = 0, horizon = 1, point);
// At α = 0.05, empirical coverage will track 0.95 over a long enough run.
```

### As the canonical UQ floor (for other primitives' GOAT gates)

See `tests/conformal_floor_harness.rs` (Issue 010 T2) for the canonical
adapter pattern. The harness exposes `UqPrimitiveUnderTest` trait; you
implement it on your primitive, then call `run_floor_comparison` to get the
CRPS / coverage / Winkler delta vs the floor.

```rust
// Pseudo-code — see the harness for the full trait surface
struct MyUqPrimitive { /* ... */ }
impl UqPrimitiveUnderTest for MyUqPrimitive { /* ... */ }

let result = run_floor_comparison(&MyUqPrimitive::new(/* ... */));
assert!(result.crps_ratio < 1.0, "must beat the conformal-naive floor on CRPS");
```

### Example binaries

```text
cargo run --example conformal_airpassengers --features conformal_predictive_intervals
cargo run --example conformal_karc_overlay --features conformal_predictive_intervals,karc_forecaster
```

`conformal_airpassengers` demonstrates the calibrator on the classic Box &
Jenkins airline dataset. `conformal_karc_overlay` demonstrates the documented
KARC integration pattern (Plan 340 Phase 2 T2.2).

---

## What is NOT in this crate

- **KARC forecaster** (`karc_forecaster` feature, Plan 308) — separate
  primitive, opt-in. Conformal wraps it via the `KarcChannelForecaster`
  adapter when both features are enabled.
- **Per-NPC consumer wiring** (KARC collapse τ, salience Delegate nudge, etc.)
  — lives in `riir-engine` (`karc_conformal_width`, `salience_conformal_width`
  features, opt-in per the three-layer split pattern).
- **Training-dependent UQ** (e.g., learned quantile heads) — out of scope per
  the modelless-first mandate. Conformal calibration is empirical-quantile;
  no learned parameters, no gradient descent.
- **Cross-channel correlation modeling** — the calibrator is per-channel ×
  per-horizon-bucket; it does not model cross-channel correlation. Each
  channel is calibrated independently. (Bench 568 closed the door on
  per-channel HLA conformal calibration for the current consumer — distribution-level summary loses cycle info.)

---

## References

- Plan: [`katgpt-rs/.plans/340_conformal_predictive_intervals_primitive.md`](../../.plans/340_conformal_predictive_intervals_primitive.md)
- Promotion Plan: [`katgpt-rs/.plans/468_conformal_predictive_intervals_default_promotion.md`](../../.plans/468_conformal_predictive_intervals_default_promotion.md)
- Research: [`katgpt-rs/.research/322_Conformal_Seasonal_Pools_Calibrated_UQ_Overlay.md`](../../.research/322_Conformal_Seasonal_Pools_Calibrated_UQ_Overlay.md)
- Bench (primitive GOAT): [`katgpt-rs/.benchmarks/340_conformal_goat.md`](../../.benchmarks/340_conformal_goat.md)
- Bench (floor rule consolidated, includes KARC+overlay T7 section): [`katgpt-rs/.benchmarks/010_report_the_floor_consolidated.md`](../../.benchmarks/010_report_the_floor_consolidated.md)
- Test (KARC+overlay floor, Issue 010 T7): [`katgpt-rs/crates/katgpt-core/tests/conformal_floor_karc_overlay.rs`](../../crates/katgpt-core/tests/conformal_floor_karc_overlay.rs)
- Bench (MCTS collapse consumer, PASS): `riir-ai/.benchmarks/564_*`
- Bench (Salience Tri-Gate consumer, PASS): `riir-ai/.benchmarks/565_*`
- Bench (Curiosity consumer, FAIL): `riir-ai/.benchmarks/562_*`
- Bench (Sleep-Time consumer, FAIL): `riir-ai/.benchmarks/563_*`
- Bench (Per-channel HLA, MIXED): `riir-ai/.benchmarks/568_*`
- Bench (`karc_conformal_width` G2 overhead, FAIL promotion): `riir-ai/.benchmarks/512_*`
- Source papers:
  - [arxiv 2605.03789](https://arxiv.org/abs/2605.03789) — Conformal Standardized Predictions (Angelopoulos et al.)
  - [arxiv 2606.09473](https://arxiv.org/abs/2606.09473) — "Report the Floor" (Tibshirani et al.)
- Cross-repo consumer bridges:
  - `riir-engine` `karc_conformal_width` (Plan 509 wiring + Plan 513 width fix)
  - `riir-engine` `salience_conformal_width` (Plan 510 + Plan 511 wiring)
